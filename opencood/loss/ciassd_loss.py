import torch
import torch.nn as nn
import numpy as np
from opencood.utils.common_utils import limit_period
from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import aligned_boxes_iou3d_gpu
from icecream import ic

class CiassdLoss(nn.Module):
    def __init__(self, args, keyname='stage1_out'):
        super(CiassdLoss, self).__init__()
        self.pos_cls_weight = args['pos_cls_weight']
        self.encode_rad_error_by_sin = args['encode_rad_error_by_sin']
        self.cls = args['cls']
        self.reg = args['reg']
        self.dir = args['dir']
        self.iou = None if 'iou' not in args else args['iou']
        self.keyname = keyname
        self.loss_dict = {}
        ##
        self.num_cls = 2
        self.box_codesize = 7

    def forward(self, output_dict, label_dict):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        preds_dict = output_dict[self.keyname]

        if 'stage1' in label_dict.keys():
            target_dict = label_dict['stage1']
        else: # for PointPillars
            target_dict = label_dict

        if 'record_len' in output_dict:
            batch_size = int(output_dict['record_len'].sum())
        else:
            batch_size = output_dict['batch_size']

        cls_labls = target_dict['pos_equal_one'].view(batch_size, -1,  self.num_cls - 1)
        positives = cls_labls > 0
        negatives = target_dict['neg_equal_one'].view(batch_size, -1,  self.num_cls - 1) > 0
        cared = torch.logical_or(positives, negatives)
        cls_labls = cls_labls * cared.type_as(cls_labls)
        # num_normalizer = cared.sum(1, keepdim=True)
        pos_normalizer = positives.sum(1, keepdim=True).float()

        # cls loss
        cls_preds = preds_dict["cls_preds"].permute(0, 2, 3, 1).contiguous() \
                    .view(batch_size, -1,  self.num_cls - 1)
        cls_weights = positives * self.pos_cls_weight + negatives * 1.0
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_loss = sigmoid_focal_loss(cls_preds, cls_labls, weights=cls_weights, **self.cls)
        cls_loss_reduced = cls_loss.sum() * self.cls['weight'] / batch_size

        # reg loss
        reg_weights = positives / torch.clamp(pos_normalizer, min=1.0)
        reg_preds = preds_dict['reg_preds'].permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.box_codesize)
        reg_targets = target_dict['targets'].view(batch_size, -1, self.box_codesize)
        if self.encode_rad_error_by_sin:
            reg_preds, reg_targets = add_sin_difference(reg_preds, reg_targets)
        reg_loss = weighted_smooth_l1_loss(reg_preds, reg_targets, weights=reg_weights, sigma=self.reg['sigma'])
        reg_loss_reduced = reg_loss.sum() * self.reg['weight'] / batch_size


        # dir loss
        dir_targets = self.get_direction_target(target_dict['targets'].view(batch_size, -1, self.box_codesize))
        dir_logits = preds_dict[f"dir_preds"].permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2) # [N, H*W*#anchor, 2]

        dir_loss = softmax_cross_entropy_with_logits(dir_logits.view(-1, self.anchor_num), dir_targets.view(-1, self.anchor_num)) 
        dir_loss = dir_loss.flatten() * reg_weights.flatten() # [N, H*W*anchor_num]
        dir_loss_reduced = dir_loss.sum() * self.dir['weight'] / batch_size

        loss = cls_loss_reduced + reg_loss_reduced + dir_loss_reduced

        # iou loss
        if self.iou is not None:
            iou_preds = preds_dict["iou_preds"].permute(0, 2, 3, 1).contiguous()
            pos_pred_mask = reg_weights.squeeze(dim=-1) > 0 # (4, 70400)
            iou_pos_preds = iou_preds.view(batch_size, -1)[pos_pred_mask]
            boxes3d_pred = VoxelPostprocessor.delta_to_boxes3d(preds_dict['reg_preds'].permute(0, 2, 3, 1).contiguous().detach(),
                                                            output_dict['anchor_box'])[pos_pred_mask]
            boxes3d_tgt = VoxelPostprocessor.delta_to_boxes3d(target_dict['targets'],
                                                            output_dict['anchor_box'])[pos_pred_mask]

            iou_weights = reg_weights[pos_pred_mask].view(-1)
            iou_pos_targets = aligned_boxes_iou3d_gpu(boxes3d_pred.float()[:, [0, 1, 2, 5, 4, 3, 6]],
                                                    boxes3d_tgt.float()[:, [0, 1, 2, 5, 4, 3, 6]]).detach().squeeze()
            iou_pos_targets = 2 * iou_pos_targets.view(-1) - 1
            iou_loss = weighted_smooth_l1_loss(iou_pos_preds, iou_pos_targets, weights=iou_weights, sigma=self.iou['sigma'])
            iou_loss_reduced = iou_loss.sum() * self.iou['weight'] / batch_size

            loss += iou_loss_reduced
            self.loss_dict.update({
                'iou_loss': iou_loss_reduced
            })
        
        
        self.loss_dict.update({
            'total_loss': loss,
            'cls_loss': cls_loss_reduced,
            'reg_loss': reg_loss_reduced,
            'dir_loss': dir_loss_reduced,
        })

        return loss

    def logging(self, epoch, batch_id, batch_len, writer = None):
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
        total_loss = self.loss_dict['total_loss']
        reg_loss = self.loss_dict['reg_loss']
        cls_loss = self.loss_dict['cls_loss']
        dir_loss = self.loss_dict['dir_loss']
        if 'iou_loss' in self.loss_dict:
            iou_loss = self.loss_dict['iou_loss']
        if (batch_id + 1) % 10 == 0:
            print("[epoch %d][%d/%d], || Loss: %.4f || Cls: %.4f"
                  " || Loc: %.4f || Dir: %.4f || Iou: %.4f" % (
                      epoch, batch_id + 1, batch_len,
                      total_loss.item(), cls_loss.item(), reg_loss.item(), dir_loss.item(), iou_loss.item()))
        if writer is not None:
            writer.add_scalar('Regression_loss', reg_loss.item(),
                            epoch*batch_len + batch_id)
            writer.add_scalar('Confidence_loss', cls_loss.item(),
                            epoch*batch_len + batch_id)
            writer.add_scalar('Direction_loss', dir_loss.item(),
                            epoch*batch_len + batch_id)
            if 'iou_loss' in self.loss_dict:
                writer.add_scalar('Iou_loss', iou_loss.item(),
                                epoch*batch_len + batch_id)


    def get_direction_target(self, reg_targets):
        """
        Args:
            reg_targets:  [N, H * W * #anchor_num, 7]
                The last term is (theta_gt - theta_a)
        
        Returns:
            dir_targets:
                theta_gt: [N, H * W * #anchor_num, NUM_BIN] 
                NUM_BIN = 2
        """
        num_bins = self.dir['args']['num_bins']
        dir_offset = self.dir['args']['dir_offset']
        anchor_yaw = np.deg2rad(np.array(self.dir['args']['anchor_yaw']))  # for direction classification
        self.anchor_yaw_map = torch.from_numpy(anchor_yaw).view(1,-1,1)  # [1,2,1]
        self.anchor_num = self.anchor_yaw_map.shape[1]

        H_times_W_times_anchor_num = reg_targets.shape[1]
        anchor_map = self.anchor_yaw_map.repeat(1, H_times_W_times_anchor_num//self.anchor_num, 1).to(reg_targets.device) # [1, H * W * #anchor_num, 1]

        rot_gt = reg_targets[..., -1] + anchor_map[..., -1] # [N, H*W*anchornum]
        offset_rot = limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()  # [N, H*W*anchornum]
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
        # one_hot:
        # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
        dir_cls_targets = one_hot_f(dir_cls_targets, num_bins)
        return dir_cls_targets



def add_sin_difference(boxes1, boxes2):
    rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(boxes2[..., -1:])   # ry -> sin(pred_ry)*cos(gt_ry)
    rad_gt_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[..., -1:])     # ry -> cos(pred_ry)*sin(gt_ry)
    res_boxes1 = torch.cat([boxes1[..., :-1], rad_pred_encoding], dim=-1)
    res_boxes2 = torch.cat([boxes2[..., :-1], rad_gt_encoding], dim=-1)
    return res_boxes1, res_boxes2


def get_direction_target(reg_targets, anchors, one_hot=True, dir_offset=0.0):
    """
    Generate targets for bounding box direction classification.

    Parameters
    ----------
    anchors: torch.Tensor
        shape as (H*W*2, 7) or (H, W, 2, 7)
    reg_targets: torch.Tensor
        shape as (B, H*W*2, 7)

    Returns
    -------
    dir_cls_targets : torch.Tensor
        [batch_size, w*h*num_anchor_per_pos, 2]
    """
    batch_size = reg_targets.shape[0]
    anchors = anchors.view(1,  -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
    rot_gt = reg_targets[..., -1] + anchors[..., -1]  # [4, 70400]
    dir_cls_targets = ((rot_gt - dir_offset) > 0).long()  # [4, 70400]
    if one_hot:
        dir_cls_targets = one_hot_f(dir_cls_targets, 2, dtype=anchors.dtype)
    return dir_cls_targets


def one_hot_f(tensor, depth, dim=-1, on_value=1.0, dtype=torch.float32):
    tensor_onehot = torch.zeros(*list(tensor.shape), depth, dtype=dtype, device=tensor.device) # [4, 70400, 2]
    tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)                        # [4, 70400, 2]
    return tensor_onehot


def sigmoid_focal_loss(preds, targets, weights=None, **kwargs):
    assert 'gamma' in kwargs and 'alpha' in kwargs
    # sigmoid cross entropy with logits
    # more details: https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    per_entry_cross_ent = torch.clamp(preds, min=0) - preds * targets.type_as(preds)
    per_entry_cross_ent += torch.log1p(torch.exp(-torch.abs(preds)))
    # focal loss
    prediction_probabilities = torch.sigmoid(preds)
    p_t = (targets * prediction_probabilities) + ((1 - targets) * (1 - prediction_probabilities))
    modulating_factor = torch.pow(1.0 - p_t, kwargs['gamma'])
    alpha_weight_factor = targets * kwargs['alpha'] + (1 - targets) * (1 - kwargs['alpha'])

    loss = modulating_factor * alpha_weight_factor * per_entry_cross_ent
    if weights is not None:
        loss *= weights
    return loss


def softmax_cross_entropy_with_logits(logits, labels):
    param = list(range(len(logits.shape)))
    transpose_param = [0] + [param[-1]] + param[1:-1]
    logits = logits.permute(*transpose_param)
    loss_ftor = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_ftor(logits, labels.max(dim=-1)[1])
    return loss


def weighted_smooth_l1_loss(preds, targets, sigma=3.0, weights=None):
    diff = preds - targets
    abs_diff = torch.abs(diff)
    abs_diff_lt_1 = torch.le(abs_diff, 1 / (sigma ** 2)).type_as(abs_diff)
    loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * sigma, 2) + \
               (abs_diff - 0.5 / (sigma ** 2)) * (1.0 - abs_diff_lt_1)
    if weights is not None:
        loss *= weights
    return loss