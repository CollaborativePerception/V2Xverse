# -*- coding: utf-8 -*-
# Author: Yifan Lu
# Add direction classification loss
# The originally point_pillar_loss.py, can not determine if the box heading is opposite to the GT.
# Add depth loss (optional for camera based perception)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from opencood.utils.common_utils import limit_period
from icecream import ic

class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar to enforce numerical stability. This is no longer
          used.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Example:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(self, alpha, gamma = 2.0, reduction= 'none', smooth_target = False , eps = None) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.smooth_target = smooth_target
        self.eps = eps
        if self.smooth_target:
            self.smooth_kernel = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
            self.smooth_kernel.weight = torch.nn.Parameter(torch.tensor([[[0.2, 0.9, 0.2]]]), requires_grad=False)
            self.smooth_kernel = self.smooth_kernel.to(torch.device("cuda"))

    def forward(self, input, target):
        n = input.shape[0]
        out_size = (n,) + input.shape[2:]

        # compute softmax over the classes axis
        input_soft = input.softmax(1)
        log_input_soft = input.log_softmax(1)

        # create the labels one hot tensor
        D = input.shape[1]
        if self.smooth_target:
            target_one_hot = F.one_hot(target, num_classes=D).to(input).view(-1, D) # [N*H*W, D]
            target_one_hot = self.smooth_kernel(target_one_hot.float().unsqueeze(1)).squeeze(1) # [N*H*W, D]
            target_one_hot = target_one_hot.view(*target.shape, D).permute(0, 3, 1, 2)
        else:
            target_one_hot = F.one_hot(target, num_classes=D).to(input).permute(0, 3, 1, 2)
        # compute the actual focal loss
        weight = torch.pow(-input_soft + 1.0, self.gamma)

        focal = -self.alpha * weight * log_input_soft
        loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))

        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")
        return loss

class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor,
                target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

            #anchors = H * W * anchor_num

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss



class PointPillarDirDepthLoss(nn.Module):
    def __init__(self, args):
        super(PointPillarDirDepthLoss, self).__init__()
        self.reg_loss_func = WeightedSmoothL1Loss()
        self.alpha = 0.25
        self.gamma = 2.0

        self.cls_weight = args['cls_weight']
        self.reg_coe = args['reg']

        self.dir_weight = args['dir_args']['dir_weight']
        self.dir_offset = args['dir_args']['args']['dir_offset']
        self.num_bins = args['dir_args']['args']['num_bins']
        anchor_yaw = np.deg2rad(np.array(args['dir_args']['anchor_yaw']))  # for direction classification
        self.anchor_yaw_map = torch.from_numpy(anchor_yaw).view(1,-1,1)  # [1,2,1]
        self.anchor_num = self.anchor_yaw_map.shape[1]

        self.depth_weight = args['depth_weight']
        self.smooth_target = True if 'smooth_target' in args and args['smooth_target'] else False
        self.use_fg_mask = True if 'use_fg_mask' in args and args['use_fg_mask'] else False
        self.fg_weight = 3.25
        self.bg_weight = 0.25
        if self.smooth_target:
            self.depth_loss_func = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction="none", smooth_target=True)
        else:
            self.depth_loss_func = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction="none")

        self.loss_dict = {}

    def forward(self, output_dict, target_dict, suffix=""):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        self.loss_dict = {}
        rm = output_dict[f'rm{suffix}']  # [B, 14, 50, 176]
        psm = output_dict[f'psm{suffix}'] # [B, 2, 50, 176]
        targets = target_dict['targets']

        cls_preds = psm.permute(0, 2, 3, 1).contiguous() # N, C, H, W -> N, H, W, C

        box_cls_labels = target_dict['pos_equal_one']  # [B, 50, 176, 2]
        """
        Visualize
        """
        box_cls_labels = box_cls_labels.view(psm.shape[0], -1).contiguous()  # [B, 50*176*2]

        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0) # [N, H*W*anchor_num]
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels
        cls_targets = cls_targets.unsqueeze(dim=-1) 

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), 2,
            dtype=cls_preds.dtype, device=cls_targets.device
        ) # [B, H*W*C, 2], C=#anchor=2
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(psm.shape[0], -1, 1) # [B, H*W*C, 1], C=#anchor=2
        one_hot_targets = one_hot_targets[..., 1:]
        
        cls_loss_src = self.cls_loss_func(cls_preds,
                                          one_hot_targets,
                                          weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / psm.shape[0]
        conf_loss = cls_loss * self.cls_weight

        # regression
        rm = rm.permute(0, 2, 3, 1).contiguous()
        rm = rm.view(rm.size(0), -1, 7)
        targets = targets.view(targets.size(0), -1, 7)
        box_preds_sin, reg_targets_sin = self.add_sin_difference(rm,
                                                                 targets)
        loc_loss_src =\
            self.reg_loss_func(box_preds_sin,
                               reg_targets_sin,
                               weights=reg_weights)
        reg_loss = loc_loss_src.sum() / rm.shape[0]
        reg_loss *= self.reg_coe

        ######## direction ##########
        dir_targets = self.get_direction_target(targets)
        N =  output_dict[f"dm{suffix}"].shape[0]
        dir_logits = output_dict[f"dm{suffix}"].permute(0, 2, 3, 1).contiguous().view(N, -1, 2) # [N, H*W*#anchor, 2]


        dir_loss = softmax_cross_entropy_with_logits(dir_logits.view(-1, self.anchor_num), dir_targets.view(-1, self.anchor_num)) 
        dir_loss = dir_loss.view(dir_logits.shape[:2]) * reg_weights # [N, H*W*anchor_num]
        dir_loss = dir_loss.sum() * self.dir_weight / N

        total_loss = reg_loss + conf_loss + dir_loss

        ########## depth #############
        if f"depth_items{suffix}" in output_dict and output_dict[f'depth_items{suffix}'] is not None:
            # depth logdit: [N, D, H, W]
            # depth gt indices: [N, H, W]
            # fg_mask: [N, H, W]
            depth_logit, depth_gt_indices = output_dict[f'depth_items{suffix}'][0], output_dict[f'depth_items{suffix}'][1]
            depth_loss = self.depth_loss_func(depth_logit, depth_gt_indices) 
            if self.use_fg_mask:
                fg_mask = output_dict[f'depth_items{suffix}'][-1]
                weight_mask = (fg_mask > 0) * self.fg_weight + (fg_mask == 0) * self.bg_weight
                depth_loss *= weight_mask

            depth_loss = depth_loss.mean() * self.depth_weight 

            total_loss += depth_loss
            self.loss_dict.update({'depth_loss': depth_loss})

        self.loss_dict.update({'total_loss': total_loss,
                               'reg_loss': reg_loss,
                               'conf_loss': conf_loss,
                               'dir_loss': dir_loss})

        return total_loss

    def cls_loss_func(self, input: torch.Tensor,
                      target: torch.Tensor,
                      weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
                            torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * \
                          torch.sin(boxes2[..., dim:dim + 1])

        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding,
                            boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding,
                            boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

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
        # (1, 2, 1)
        H_times_W_times_anchor_num = reg_targets.shape[1]
        anchor_map = self.anchor_yaw_map.repeat(1, H_times_W_times_anchor_num//self.anchor_num, 1).to(reg_targets.device) # [1, H * W * #anchor_num, 1]
        rot_gt = reg_targets[..., -1] + anchor_map[..., -1] # [N, H*W*anchornum]
        offset_rot = limit_period(rot_gt - self.dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / self.num_bins)).long()  # [N, H*W*anchornum]
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=self.num_bins - 1)
        # one_hot:
        # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
        dir_cls_targets = one_hot_f(dir_cls_targets, self.num_bins)
        return dir_cls_targets



    def logging(self, epoch, batch_id, batch_len, writer = None, suffix=""):
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
        conf_loss = self.loss_dict['conf_loss']
        dir_loss = self.loss_dict['dir_loss']
        depth_loss = 0 if 'depth_loss' not in self.loss_dict else self.loss_dict['depth_loss'].item()

        print("[epoch %d][%d/%d]%s || Loss: %.4f || Conf Loss: %.4f"
              " || Loc Loss: %.4f || Dir Loss: %.4f || depth loss %.4f" % (
                  epoch, batch_id + 1, batch_len, suffix,
                  total_loss.item(), conf_loss.item(), reg_loss.item(), dir_loss.item(), depth_loss))

        if not writer is None:
            writer.add_scalar('Regression_loss'+suffix, reg_loss.item(),
                            epoch*batch_len + batch_id)
            writer.add_scalar('Confidence_loss'+suffix, conf_loss.item(),
                            epoch*batch_len + batch_id)
            writer.add_scalar('Dir_loss'+suffix, dir_loss.item(),
                            epoch*batch_len + batch_id)
            writer.add_scalar('depth_loss'+suffix, depth_loss,
                            epoch*batch_len + batch_id)

def one_hot_f(tensor, num_bins, dim=-1, on_value=1.0, dtype=torch.float32):
    tensor_onehot = torch.zeros(*list(tensor.shape), num_bins, dtype=dtype, device=tensor.device) 
    tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)                    
    return tensor_onehot

def softmax_cross_entropy_with_logits(logits, labels):
    param = list(range(len(logits.shape)))
    transpose_param = [0] + [param[-1]] + param[1:-1]
    logits = logits.permute(*transpose_param)
    loss_ftor = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_ftor(logits, labels.max(dim=-1)[1])
    return loss
