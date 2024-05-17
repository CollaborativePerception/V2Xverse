import torch
from torch import nn
import numpy as np
from opencood.loss.ciassd_loss import CiassdLoss, weighted_smooth_l1_loss
from icecream import ic 

class HeterceptionReDetLoss(nn.Module):
    def __init__(self, args):
        super(HeterceptionReDetLoss, self).__init__()
        # self.ciassd_loss = CiassdLoss(args['stage1'])
        self.ciassd_loss = CiassdLoss(args['shared_head_out'], keyname='shared_head_out')
        self.redet_loss = CiassdLoss(args['stage2'], keyname='stage2_out')


        self.kd = args['stage2']['kd']
        self.kd_fn = nn.MSELoss(reduce='mean')

        self.loss_dict = {}

    def forward(self, output_dict, label_dict):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        ciassd_loss = self.ciassd_loss(output_dict, label_dict['stage1'])

        # only update ciassd if no bbox is detected in the first stage
        if 'stage2_out' not in output_dict:
            self.loss_dict = {
                'loss': ciassd_loss,
            }
            return ciassd_loss
        
        output_dict['batch_size'] = len(output_dict['record_len'])
        output_dict.pop('record_len')

        redet_loss = self.redet_loss(output_dict, label_dict['stage2'])
        loss = redet_loss + ciassd_loss

        # knowledge distillation
        if 'kd_items' in output_dict:
            lidar_features = output_dict['kd_items']["lidar_roi_features"] # [C, sum(bev_grids)]
            camera_features = output_dict['kd_items']["camera_roi_features"] # [C, sum(bev_grids)]
            kd_loss_reduced = self.kd_fn(lidar_features, camera_features) * self.kd['weight']
            loss += kd_loss_reduced
            self.loss_dict.update({'kd_loss': kd_loss_reduced})

        self.loss_dict.update({
            'loss': loss,
            'redet_loss': redet_loss,
        })

        return loss

    def logging(self, epoch, batch_id, batch_len, writer=None):
        """
        Print out  the loss function for current iteration.
        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        ciassd_loss_dict = self.ciassd_loss.loss_dict
        ciassd_total_loss = ciassd_loss_dict['total_loss']
        reg_loss = ciassd_loss_dict['reg_loss']
        cls_loss = ciassd_loss_dict['cls_loss']
        dir_loss = ciassd_loss_dict['dir_loss']


        if (batch_id + 1) % 1 == 0:
            str_to_print = "[epoch %d][%d/%d], || Loss: %.4f || Ciassd: %.4f " \
                           "|| Cls1: %.4f || Loc1: %.4f || Dir1: %.4f " % (
                               epoch, batch_id + 1, batch_len, self.loss_dict['loss'],
                               ciassd_total_loss.item(), cls_loss.item(), reg_loss.item(),
                               dir_loss.item()
                               )
            if 'redet_loss' in self.loss_dict:
                str_to_print += " || redet_loss: %.4f || Cls2: %.4f || Loc2: %.4f || Dir2: %.4f" % (
                        self.loss_dict['redet_loss'].item(),
                        self.redet_loss.loss_dict['cls_loss'].item(),
                        self.redet_loss.loss_dict['reg_loss'].item(),
                        self.redet_loss.loss_dict['dir_loss'].item(),
                    )
            if 'kd_loss' in self.loss_dict:
                str_to_print += " || Heter kd: %.4f " % (
                        self.loss_dict['kd_loss'].item(),
                    )


            print(str_to_print)
            
        if writer:
            writer.add_scalar('Ciassd_regression_loss', reg_loss.item(),
                            epoch * batch_len + batch_id)
            writer.add_scalar('Ciassd_Confidence_loss', cls_loss.item(),
                            epoch * batch_len + batch_id)
            writer.add_scalar('Ciassd_Direction_loss', dir_loss.item(),
                            epoch * batch_len + batch_id)
            writer.add_scalar('Ciassd_loss', ciassd_total_loss.item(),
                            epoch * batch_len + batch_id)
                            
            if 'redet_loss' in self.loss_dict:
                writer.add_scalar('ReDet_loss',
                                self.loss_dict['redet_loss'].item(),
                                epoch * batch_len + batch_id)
                writer.add_scalar('ReDet_Confidence_loss',
                                self.redet_loss.loss_dict['cls_loss'].item(),
                                epoch * batch_len + batch_id)
                writer.add_scalar('ReDet_regression_loss',
                                self.redet_loss.loss_dict['reg_loss'].item(),
                                epoch * batch_len + batch_id)
                writer.add_scalar('ReDet_direction_loss', 
                                self.redet_loss.loss_dict['dir_loss'].item(),
                                epoch * batch_len + batch_id)
                writer.add_scalar('Total_loss', self.loss_dict['loss'].item(),
                                epoch * batch_len + batch_id)

            if 'kd_loss' in self.loss_dict:
                writer.add_scalar('Heter_kd_loss',
                                self.loss_dict['kd_loss'].item(),
                                epoch * batch_len + batch_id)



def weighted_sigmoid_binary_cross_entropy(preds, tgts, weights=None,
                                          class_indices=None):
    if weights is not None:
        weights = weights.unsqueeze(-1)
    if class_indices is not None:
        weights *= (
            indices_to_dense_vector(class_indices, preds.shape[2])
                .view(1, 1, -1)
                .type_as(preds)
        )
    per_entry_cross_ent = nn.functional.binary_cross_entropy_with_logits(preds,
                                                                         tgts,
                                                                         weights)
    return per_entry_cross_ent


def indices_to_dense_vector(
        indices, size, indices_value=1.0, default_value=0, dtype=np.float32
):
    """Creates dense vector with indices set to specific value and rest to zeros.
    This function exists because it is unclear if it is safe to use
        tf.sparse_to_dense(indices, [size], 1, validate_indices=False)
    with indices which are not ordered.
    This function accepts a dynamic size (e.g. tf.shape(tensor)[0])
    Args:
        indices: 1d Tensor with integer indices which are to be set to
            indices_values.
        size: scalar with size (integer) of output Tensor.
        indices_value: values of elements specified by indices in the output vector
        default_value: values of other elements in the output vector.
        dtype: data type.
    Returns:
        dense 1D Tensor of shape [size] with indices set to indices_values and the
            rest set to default_value.
    """
    dense = torch.zeros(size).fill_(default_value)
    dense[indices] = indices_value

    return dense