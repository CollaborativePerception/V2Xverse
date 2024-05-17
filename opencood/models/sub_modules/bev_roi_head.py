import torch
import torch.nn as nn
from mmcv.ops import RoIAlignRotated
from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from opencood.utils import box_utils
from opencood.utils import common_utils
import numpy as np
from icecream import ic

class BEVRoIHead(nn.Module):
    def __init__(self, model_cfg, pc_range):
        super().__init__()
        self.model_cfg = model_cfg
        self.pc_range = pc_range
        self.roi_align_size = 3
        self.code_size = 7
        self.enlarge_ratio = model_cfg.get("enlarge_ratio", 1)
        self.roialign_rotated = RoIAlignRotated(output_size=self.roi_align_size, spatial_scale=1, clockwise=True)
        
        c_out = self.model_cfg['in_channels'] # 128
        pre_channel = self.roi_align_size * self.roi_align_size * c_out # 3*3*128
        fc_layers = [self.model_cfg['n_fc_neurons']] * 2
        self.shared_fc_layers, pre_channel = self._make_fc_layers(pre_channel,
                                                                  fc_layers)

        self.cls_layers, pre_channel = self._make_fc_layers(pre_channel,
                                                            fc_layers,
                                                            output_channels=
                                                            self.model_cfg[
                                                                'num_cls'])
        self.iou_layers, _ = self._make_fc_layers(pre_channel, fc_layers,
                                                  output_channels=
                                                  self.model_cfg['num_cls'])
        self.reg_layers, _ = self._make_fc_layers(pre_channel, fc_layers,
                                                  output_channels=
                                                  self.model_cfg[
                                                      'num_cls'] * 7)

        self._init_weights(weight_init='xavier')

    def _init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def _make_fc_layers(self, input_channels, fc_list, output_channels=None):
        fc_layers = []
        pre_channel = input_channels
        for k in range(len(fc_list)):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                # nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if self.model_cfg['dp_ratio'] > 0:
                fc_layers.append(nn.Dropout(self.model_cfg['dp_ratio']))
        if output_channels is not None:
            fc_layers.append(
                nn.Conv1d(pre_channel, output_channels, kernel_size=1,
                          bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers, pre_channel

    def forward(self, batch_dict):
        batch_dict = self.assign_targets(batch_dict)

        # put roi back to dense feature map for rotated roi align.
        batch_size = batch_dict['batch_size_2stage']
        # [[RoI_H0*RoI_W0, C], [RoI_H1*RoI_W1, C], ...]
        feature_of_proposals_ego_list = batch_dict['feature_of_proposals_ego_list'] 
        C = feature_of_proposals_ego_list[0].shape[1]
        device = feature_of_proposals_ego_list[0].device
        
        H, W = batch_dict['feature_shape']
        grid_size_H = (self.pc_range[4] - self.pc_range[1]) / H 
        grid_size_W = (self.pc_range[3] - self.pc_range[0]) / W 

        # dense feature map
        feature_map = torch.zeros((batch_size, C, H, W), device=device)
        roi_cnt = 0
        for batch_idx, roi_fused in enumerate(batch_dict['roi_fused']): # per scene
            for roi in roi_fused:
                feature_map[batch_idx, :, roi[2]:roi[3], roi[0]:roi[1]] = \
                    feature_of_proposals_ego_list[roi_cnt].permute(1,0).view(C, roi[3]-roi[2], roi[1]-roi[0]) 
                roi_cnt += 1

        # proposal to rotated roi input, 
        # (batch_index, center_x, center_y, w, h, angle). The angle is in radian.
        roi_input = torch.zeros((len(feature_of_proposals_ego_list), 6), device=device)
        
        box_cnt = 0
        for batch_idx, box_fused in enumerate(batch_dict['boxes_fused']): # per scene
            # box_fused is [n_boxes, 7], x, y, z, h, w, l, yaw -> (center_x, center_y, w, h)
            roi_input[box_cnt:box_cnt+box_fused.shape[0], 0] = batch_idx
            roi_input[box_cnt:box_cnt+box_fused.shape[0], 1] = (box_fused[:, 0] - self.pc_range[0]) / grid_size_W
            roi_input[box_cnt:box_cnt+box_fused.shape[0], 2] = (box_fused[:, 1] - self.pc_range[1]) / grid_size_H
            roi_input[box_cnt:box_cnt+box_fused.shape[0], 3] = box_fused[:, 5] / grid_size_W * self.enlarge_ratio  # box's l -> W
            roi_input[box_cnt:box_cnt+box_fused.shape[0], 4] = box_fused[:, 4] / grid_size_H * self.enlarge_ratio # box's w -> H
            roi_input[box_cnt:box_cnt+box_fused.shape[0], 5] = box_fused[:, 6] 
            box_cnt += box_fused.shape[0]

        # roi align
        N_proposals = roi_input.shape[0]
        # [sum(proposal), C, self.roi_align_size, self.roi_align_size]
        pooled_feature = self.roialign_rotated(feature_map, roi_input) 
        # [sum(proposal), self.roi_align_size * self.roi_align_size * C, 1]
        pooled_feature = pooled_feature.flatten(start_dim=2).permute(0,2,1).flatten(start_dim=1).unsqueeze(-1)
        shared_features = self.shared_fc_layers(pooled_feature)
        
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)
        rcnn_iou = self.iou_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)

        batch_dict['stage2_out'] = {
            'rcnn_cls': rcnn_cls,
            'rcnn_iou': rcnn_iou,
            'rcnn_reg': rcnn_reg,
        }

        return batch_dict


    def assign_targets(self, batch_dict):
        batch_dict['rcnn_label_dict'] = {
            'rois': [],
            'gt_of_rois': [],
            'gt_of_rois_src': [],
            'cls_tgt': [],
            'reg_tgt': [],
            'iou_tgt': [],
            'rois_anchor': [],
            'record_len': [],
            'rois_scores_stage1': []
        }
        pred_boxes = batch_dict['boxes_fused']
        pred_scores = batch_dict['scores_fused']
        gt_boxes = [b[m][:, [0, 1, 2, 5, 4, 3, 6]].float() for b, m in
                    zip(batch_dict['object_bbx_center'],
                        batch_dict['object_bbx_mask'].bool())]  # hwl -> lwh order
        for rois, scores, gts in zip(pred_boxes, pred_scores,  gt_boxes): # each frame
            rois = rois[:, [0, 1, 2, 5, 4, 3, 6]]  # hwl -> lwh
            if gts.shape[0] == 0:
                gts = rois.clone()

            ious = boxes_iou3d_gpu(rois, gts)
            max_ious, gt_inds = ious.max(dim=1)
            gt_of_rois = gts[gt_inds]
            rcnn_labels = (max_ious > 0.3).float()
            mask = torch.logical_not(rcnn_labels.bool())

            # set negative samples back to rois, no correction in stage2 for them
            gt_of_rois[mask] = rois[mask]
            gt_of_rois_src = gt_of_rois.clone().detach()

            # canoical transformation
            roi_center = rois[:, 0:3]
            # TODO: roi_ry > 0 in pcdet
            roi_ry = rois[:, 6] % (2 * np.pi)
            gt_of_rois[:, 0:3] = gt_of_rois[:, 0:3] - roi_center
            gt_of_rois[:, 6] = gt_of_rois[:, 6] - roi_ry

            # transfer LiDAR coords to local coords
            gt_of_rois = common_utils.rotate_points_along_z(
                points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]),
                angle=-roi_ry.view(-1)
            ).view(-1, gt_of_rois.shape[-1])

            # flip orientation if rois have opposite orientation
            heading_label = (gt_of_rois[:, 6] + (
                    torch.div(torch.abs(gt_of_rois[:, 6].min()),
                              (2 * np.pi), rounding_mode='trunc')
                    + 1) * 2 * np.pi) % (2 * np.pi)  # 0 ~ 2pi
            opposite_flag = (heading_label > np.pi * 0.5) & (
                    heading_label < np.pi * 1.5)

            # (0 ~ pi/2, 3pi/2 ~ 2pi)
            heading_label[opposite_flag] = (heading_label[
                                                opposite_flag] + np.pi) % (
                                                   2 * np.pi)
            flag = heading_label > np.pi
            heading_label[flag] = heading_label[
                                      flag] - np.pi * 2  # (-pi/2, pi/2)
            heading_label = torch.clamp(heading_label, min=-np.pi / 2,
                                        max=np.pi / 2)
            gt_of_rois[:, 6] = heading_label

            # generate regression target
            rois_anchor = rois.clone().detach().view(-1, self.code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0

            reg_targets = box_utils.box_encode(
                gt_of_rois.view(-1, self.code_size), rois_anchor
            )

            batch_dict['rcnn_label_dict']['rois'].append(rois)
            batch_dict['rcnn_label_dict']['rois_scores_stage1'].append(scores)
            batch_dict['rcnn_label_dict']['gt_of_rois'].append(gt_of_rois)
            batch_dict['rcnn_label_dict']['gt_of_rois_src'].append(
                gt_of_rois_src)
            batch_dict['rcnn_label_dict']['cls_tgt'].append(rcnn_labels)
            batch_dict['rcnn_label_dict']['reg_tgt'].append(reg_targets)
            batch_dict['rcnn_label_dict']['iou_tgt'].append(max_ious)
            batch_dict['rcnn_label_dict']['rois_anchor'].append(rois_anchor)
            batch_dict['rcnn_label_dict']['record_len'].append(rois.shape[0])
            

        # cat list to tensor
        for k, v in batch_dict['rcnn_label_dict'].items():
            if k == 'record_len':
                continue
            batch_dict['rcnn_label_dict'][k] = torch.cat(v, dim=0)

        return batch_dict