# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
from numpy import record
import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.v2v_robust_module import AttentionWrapper, PoseRegressionWraper, WeightedEM, get_intersection, regroup
from opencood.utils.pose_utils import generate_noise_torch
from opencood.utils.transformation_utils import get_pairwise_transformation_torch
from opencood.models.fuse_modules.v2v_fuse import V2VNetFusion

from icecream import ic
from opencood.utils.model_utils import weight_init

class PointPillarV2VNetRobust(nn.Module):
    def __init__(self, args):
        super(PointPillarV2VNetRobust, self).__init__()

        self.max_cav = args['max_cav']
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.fusion_net = V2VNetFusion(args['v2vfusion'])

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)


        self.downsample_rate = args['robust']['downsample_rate']
        self.discrete_ratio = args['robust']['discrete_ratio']
        self.H = args['robust']['H']
        self.W = args['robust']['W']

        self.affine_parameter = {"H":self.H, "W": self.W, "downsample_rate": self.downsample_rate, "discrete_ratio": self.discrete_ratio}
        learnable_alpha = True if 'learnable_alpha' not in args['robust'] else args['robust']['learnable_alpha']

        self.pose_reg_net = PoseRegressionWraper(args['robust']['feature_dim']*2, 
                                                args['robust']['hidden_dim'],
                                                self.affine_parameter
                                                )   

        self.attention_net = AttentionWrapper(args['robust']['feature_dim']*2, 
                                              args['robust']['hidden_dim'],
                                              self.affine_parameter,
                                              learnable_alpha,
                                              )

        self.stage = args['stage'] # [0, 1, 2]

        self.apply(weight_init)

        if self.stage == 1:
            self.backbone_fix()
        if self.stage == 2:
            self.backbone_unfix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone for stage 1 
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.fusion_net.parameters():
            p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

        for p in self.attention_net.parameters():
            p.requires_grad = False

    def backbone_unfix(self):
        """
        unfix for stage 2
        """

        for p in self.pillar_vfe.parameters():
            p.requires_grad = True

        for p in self.scatter.parameters():
            p.requires_grad = True

        for p in self.backbone.parameters():
            p.requires_grad = True

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = True
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = True

        for p in self.fusion_net.parameters():
            p.requires_grad = True

        for p in self.cls_head.parameters():
            p.requires_grad = True
        for p in self.reg_head.parameters():
            p.requires_grad = True

        for p in self.attention_net.parameters():
            p.requires_grad = True

    def pose_correction(self, features, record_len, pairwise_t_matrix):
        """ use pose regression module to correct relative pose
        Args:

        Returns:
            pairwise_t_matrix_new: 
                [B, L, L, 4, 4], the relative pose after correction.
        """
        return self.pose_reg_net(features, record_len, pairwise_t_matrix)

    def global_correction(self, lidar_pose, pairwise_t_matrix, record_len):
        """
        Args:
            lidar_pose: [N, 3]
                input noisy lidar pose
            pairwise_t_matrix: [B, L, L, 4, 4]
                relative pose after pose regression module
            record_len: list,
                shape [B]
        
        Returns:
            lidar_pose_new: [N, 3]
                refined lidar pose
        """

        B = len(record_len)
        lidar_pose_new = []

        # [[N1,3], [N2, 3], ...]
        lidar_pose_split = regroup(lidar_pose, record_len)

        for b in range(B):
            if record_len[b] == 1:
                lidar_pose_new.append(lidar_pose_split[b])
                continue
            lidar_pose = lidar_pose_split[b]
            intersection_matrix = get_intersection(pairwise_t_matrix[b], self.affine_parameter)
            lidar_pose_corrected = WeightedEM(lidar_pose, pairwise_t_matrix[b],intersection_matrix)

            lidar_pose_new.append(lidar_pose_corrected)

        lidar_pose_new = torch.cat(lidar_pose_new, dim=0)

        return lidar_pose_new
        

    def noise_generator(self, lidar_pose, all_strong=False):
        noise_s = generate_noise_torch(lidar_pose, pos_std=0.4, rot_std=4)  # (N, 6)
        noise_w = generate_noise_torch(lidar_pose, pos_std=0.01, rot_std=0.1)  # (N, 6)
        N = lidar_pose.shape[0]
        
        if all_strong:
            choice = torch.zeros((N, 1), device=lidar_pose.device) # (N, 1) 0 choose strong, 1 choose weak
            noise = noise_s
        else:
            choice = torch.randint(0, 2, (N, 1), device=lidar_pose.device) # (N, 1) 0 choose strong, 1 choose weak
            noise = choice * noise_w + (1-choice) * noise_s
        
        return noise, choice


    def train_forward(self, spatial_features_2d, record_len, lidar_pose, pairwise_t_matrix, stage):
        """
        stage = 0, only training attentive_aggregation and v2vnet, strong noise and weak noise are used.
        stage = 1, only training pose correction module. all strong noise
        stage = 2, all component are used. all strong noise

        Args:
            spatial_features_2d: (N, C, H, W)
            record_len: list
            lidar_pose: (N, 6), it will turn to [N, 3] quickly
        """
        if stage == 0:
            noise, choice = self.noise_generator(lidar_pose, all_strong=False)

        if stage == 1 or stage == 2:
            noise, choice = self.noise_generator(lidar_pose, all_strong=True)

        lidar_pose += noise
        lidar_pose = lidar_pose[:,[0,1,4]] # [N, 3]
        pairwise_t_matrix = get_pairwise_transformation_torch(lidar_pose, self.max_cav, record_len, dof=3)
        
        # when training pairwise_t_matrix, pairwise_t_matrix carries given noise.
        if self.stage == 0:
            scores, weight = self.attention_net(spatial_features_2d, record_len, pairwise_t_matrix)
            fused_feature = self.fusion_net(spatial_features_2d, record_len, pairwise_t_matrix, weight)
            psm = self.cls_head(fused_feature)
            rm = self.reg_head(fused_feature)
            print("scores:", scores)
            print("weight:", weight)
            print("alpha:", self.attention_net.alpha)

            output_dict = {'stage': stage,
                            'scores': scores,
                            'choice': choice,
                            'psm': psm,
                            'rm': rm}

        if self.stage == 1:
            pairwise_corr, _ = self.pose_correction(spatial_features_2d, record_len, pairwise_t_matrix)
            output_dict = {'stage': stage,
                            'pairwise_corr' : pairwise_corr,
                            'pairwise_t_matrix': pairwise_t_matrix}
        
        if self.stage == 2:
            pairwise_corr, pairwise_t_matrix_new = self.pose_correction(spatial_features_2d, record_len, pairwise_t_matrix)
            lidar_pose_corrected = self.global_correction(lidar_pose, pairwise_t_matrix_new, record_len) # [N, 3]

            pairwise_t_matrix_corrected = get_pairwise_transformation_torch(lidar_pose_corrected, self.max_cav, record_len, dof=3)
            scores, weight = self.attention_net(spatial_features_2d, record_len, pairwise_t_matrix_corrected)
            fused_feature = self.fusion_net(spatial_features_2d, record_len, pairwise_t_matrix_corrected, weight)
            psm = self.cls_head(fused_feature)
            rm = self.reg_head(fused_feature)

            output_dict = {'stage': stage,
                            'scores': scores,
                            'psm': psm,
                            'rm': rm,
                            'pairwise_corr' : pairwise_corr,
                            'pairwise_t_matrix': pairwise_t_matrix}

        return output_dict



    def eval_forward(self, spatial_features_2d, record_len, lidar_pose, pairwise_t_matrix, stage):
        """
            same as stage=2 in training, but no noise added.
        """
        lidar_pose = lidar_pose[:,[0,1,4]] # [N, 3]
        pairwise_t_matrix = get_pairwise_transformation_torch(lidar_pose, self.max_cav, record_len, dof=3)

        pairwise_corr, pairwise_t_matrix = self.pose_correction(spatial_features_2d, record_len, pairwise_t_matrix)

        lidar_pose_corrected = self.global_correction(lidar_pose, pairwise_t_matrix, record_len) # [N, 3]
        pairwise_t_matrix_corrected = get_pairwise_transformation_torch(lidar_pose_corrected, self.max_cav, record_len, dof=3)


        scores, weight = self.attention_net(spatial_features_2d, record_len, pairwise_t_matrix_corrected)
        fused_feature = self.fusion_net(spatial_features_2d, record_len, pairwise_t_matrix_corrected, weight)
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'stage': stage,
                        'scores': scores,
                        'psm': psm,
                        'rm': rm,
                        'pairwise_t_matrix': pairwise_t_matrix}

        return output_dict


    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        lidar_pose = data_dict['lidar_pose'] # [sum(cav), 6]

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)
        # N, C, H', W'. [N, 256, 50, 176]
        spatial_features_2d = batch_dict['spatial_features_2d']

        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        # spatial_features_2d is [sum(cav_num), 256, 50, 176]
        # output only contains ego
        # [B, 256, 50, 176]

        # lidar_pose -> pairwise_t_matrix , same content

        # if self.training:
        #     return self.train_forward(spatial_features_2d, record_len, lidar_pose, pairwise_t_matrix, self.stage)
        # else:
        #     return self.eval_forward(spatial_features_2d, record_len, lidar_pose, pairwise_t_matrix, self.stage)

        return self.train_forward(spatial_features_2d, record_len, lidar_pose, pairwise_t_matrix, self.stage)