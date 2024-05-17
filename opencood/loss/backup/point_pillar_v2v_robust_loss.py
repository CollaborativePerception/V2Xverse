# -*- coding: utf-8 -*-
# Author: OpenPCDet, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


from icecream import ic
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from opencood.models.sub_modules.v2v_robust_module import regroup
from opencood.utils.transformation_utils import tfm_to_pose, tfm_to_pose_torch
torch.set_printoptions(precision=3, sci_mode=False)

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


class PointPillarV2VRobustLoss(nn.Module):
    def __init__(self, args):
        super(PointPillarV2VRobustLoss, self).__init__()
        self.reg_loss_func = WeightedSmoothL1Loss()
        self.score_loss_func = nn.BCELoss(reduce=True, reduction="mean")
        self.pose_loss_func = nn.SmoothL1Loss(reduce=True, reduction="mean", beta=1.0/9)
        self.alpha = 0.25
        self.gamma = 2.0

        self.cls_weight = args['cls_weight']
        self.reg_coe = args['reg']
        self.score_weight = args['score_weight']
        self.pose_weight = args['pose_weight']
        self.loss_dict = {}

    def forward(self, output_dict, target_dict):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        stage = output_dict['stage']

        if stage == 0 or stage == 2:
            rm = output_dict['rm']  # [B, 14, 50, 176]
            psm = output_dict['psm'] # [B, 2, 50, 176]
            targets = target_dict['targets']

            cls_preds = psm.permute(0, 2, 3, 1).contiguous() # N, C, H, W -> N, H, W, C

            box_cls_labels = target_dict['pos_equal_one']  # [B, 50, 176, 2]
            box_cls_labels = box_cls_labels.view(psm.shape[0], -1).contiguous()

            positives = box_cls_labels > 0
            negatives = box_cls_labels == 0
            negative_cls_weights = negatives * 1.0
            cls_weights = (negative_cls_weights + 1.0 * positives).float()
            reg_weights = positives.float()

            pos_normalizer = positives.sum(1, keepdim=True).float()
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)
            cls_targets = box_cls_labels
            cls_targets = cls_targets.unsqueeze(dim=-1)

            cls_targets = cls_targets.squeeze(dim=-1)
            one_hot_targets = torch.zeros(
                *list(cls_targets.shape), 2,
                dtype=cls_preds.dtype, device=cls_targets.device
            )
            one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
            cls_preds = cls_preds.view(psm.shape[0], -1, 1)
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

            total_loss = reg_loss + conf_loss

            self.loss_dict.update({'total_loss': total_loss,
                                'reg_loss': reg_loss,
                                'conf_loss': conf_loss})
        else:
            total_loss = 0

        

        # robust v2vnet part
        record_len = target_dict['record_len'] # we can also put this in output_dict
        if stage == 0:
            scores = output_dict['scores']
            choice = output_dict['choice']
            
            score_loss = self.attention_loss(scores, choice, record_len)
            total_loss += self.score_weight * score_loss
            self.loss_dict.update({'total_loss': total_loss,
                                    'score_loss': score_loss})
                                    
        elif stage == 1 or stage == 2:
            pairwise_corr = output_dict['pairwise_corr']
            pairwise_t_matrix = output_dict['pairwise_t_matrix']
            pairwise_t_matrix_gt = target_dict['pairwise_t_matrix']

            pose_loss = self.pose_loss(pairwise_corr, pairwise_t_matrix, pairwise_t_matrix_gt, record_len)
            total_loss += self.pose_weight * pose_loss
            self.loss_dict.update({'total_loss': total_loss,
                                    'pose_loss': pose_loss})
        

        return total_loss

    def attention_loss(self, scores, choices, record_len):
        """
        Args:
            scores: (B, L, L)
                scores[b,i,i] is already 0. 
            choices: (sum(N_cav), 1)
                0 is strong noise, 1 is weak noise
            record_len: 
                list, shape [B]
        """
        # first build gt label from choice
        B = scores.shape[0]
        choice_split = regroup(choices, record_len)
        label = torch.zeros_like(scores, device=scores.device)
        mask = torch.zeros_like(scores, device=scores.device)
        for b in range(B):
            N = record_len[b]
            choice = choice_split[b].float() # [N, 1]
            choice = choice @ choice.T # [N, N]
            
            gamma = 0.85
            label[b,:N,:N] = choice * gamma + (1-choice) * (1-gamma) # [N, N]

            mask[b,:N,:N] = 1
            mask[b,range(N),range(N)] = 0

        mask = mask.bool()

        input = torch.masked_select(scores, mask) 
        target = torch.masked_select(label, mask)
        print("input:", input)
        print("target:", target)

        return self.score_loss_func(input, target)

    def pose_loss(self, pairwise_corr,  pairwise_t_matrix, pairwise_t_matrix_gt, record_len):
        """
        Args:
            pairwise_corr: [B, L, L, 3]
            pairwise_t_matrix/pairwise_t_matrix_gt: [B,L,L,4,4]
            record_len: list, shape [B]
        """

        pairwise_t_matrix_gt = pairwise_t_matrix_gt.float()
        B, L = pairwise_t_matrix.shape[:2]
        mask = torch.zeros((B, L, L), device = pairwise_t_matrix.device)

        for b in range(B):
            N = record_len[b]
            mask[b,:N,:N] = 1
            mask[b,range(N), range(N)] = 0
        
        pair_corr_gt = torch.linalg.solve(pairwise_t_matrix.transpose(-2,-1), pairwise_t_matrix_gt.transpose(-2,-1)).transpose(-2,-1)
 
        yaw = pairwise_corr[..., 2] # [B,L,L]
        yaw_gt = torch.rad2deg(torch.atan2(pair_corr_gt[...,1,0], pair_corr_gt[...,0,0])) # [B,L,L]

        x = pairwise_corr[..., 0] # [B,L,L]
        x_gt = pair_corr_gt[..., 0,3]

        y = pairwise_corr[..., 1] # [B,L,L]
        y_gt = pair_corr_gt[..., 1,3]

        mask = mask.bool()
        mask = mask.view(B,L,L) # [B, L, L, ]

        input_x = torch.masked_select(x, mask)
        target_x = torch.masked_select(x_gt, mask)

        input_y = torch.masked_select(y, mask)
        target_y = torch.masked_select(y_gt, mask)

        input_yaw = torch.masked_select(yaw, mask)
        target_yaw = torch.masked_select(yaw_gt, mask)

        loss_x = self.pose_loss_func(input_x, target_x)
        loss_y = self.pose_loss_func(input_y, target_y)
        loss_yaw = self.pose_loss_func(input_yaw, target_yaw)

        lambda_trans = 2/3
        lambda_rot = 1/3
        print("x:\n", input_x, "\n", target_x)
        print("y:\n", input_y, "\n", target_y)
        print("yaw:\n", input_yaw, "\n", target_yaw)

        return lambda_trans * (loss_x + loss_y) + lambda_rot * loss_yaw



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
        total_loss = self.loss_dict['total_loss'].item()
        if 'reg_loss' in self.loss_dict:
            reg_loss = self.loss_dict['reg_loss'].item()
        else:
            reg_loss = 0
        if 'conf_loss' in self.loss_dict:
            conf_loss = self.loss_dict['conf_loss'].item()
        else:
            conf_loss = 0
        if "score_loss" in self.loss_dict:
            score_loss = self.loss_dict['score_loss']
        else:
            score_loss = 0
        if "pose_loss" in self.loss_dict:
            pose_loss = self.loss_dict['pose_loss']
        else:
            pose_loss = 0

        print("[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
              " || Loc Loss: %.4f || Score Loss: %.4f || Pose Loss: %.4f" % (
                  epoch, batch_id + 1, batch_len,
                  total_loss, conf_loss, reg_loss, score_loss, pose_loss))

        if not writer is None:
            writer.add_scalar('Regression_loss', reg_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Confidence_loss', conf_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Score_loss', score_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Pose_loss', pose_loss,
                            epoch*batch_len + batch_id)