# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


from numpy import record
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.mash_utils import QueryEncoder, KeyEncoder, SmoothingNetwork

from icecream import ic


class PointPillarMash(nn.Module):
    def __init__(self, args):
        super(PointPillarMash, self).__init__()

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

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)

        mash_args = args['mash']
        self.query_encoder = QueryEncoder(mash_args["feature_dim"], mash_args['query_dim'])
        self.key_encoder = KeyEncoder(mash_args["feature_dim"], mash_args['key_dim'])
        self.queryKeySim = nn.Conv2d(mash_args['query_dim'],  mash_args['key_dim'], 1, 1)
        self.smoothing_net = SmoothingNetwork(in_ch=mash_args['H'] * mash_args['W'] + 1)
        self.H = mash_args['H']
        self.W = mash_args['W']
        self.downsample_rate = mash_args['downsample_rate']
        self.discrete_ratio = args['voxel_size'][0]

        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
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

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        pairwise_t_matrix = data_dict['pairwise_t_matrix']

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

        B = len(record_len)

        querys = self.queryKeySim(self.query_encoder(spatial_features_2d))
        keys = self.key_encoder(spatial_features_2d)

        split_query = self.regroup(querys, record_len)
        split_key = self.regroup(keys, record_len)
        split_feature = self.regroup(spatial_features_2d, record_len)
        
        fuse_features = []
        estimate_volumes = []
        for b in range(B):
            # N, C, H, W
            feature = split_feature[b]
            key = split_key[b]
            query = split_query[b]

            ego = 0
            fuse_feature = [feature[ego]]
            N = record_len[b]

            for i in range(1, N):
                corr_volume = self.computeCorrespondenceVolume(query[ego], key[i])
                corr_volume_decoded = self.smoothCorrespondenceVolume(corr_volume) # (Hs*Ws+1, Ht, Wt)
                grid, mask = self.idx2grid(corr_volume_decoded) # (1, H, W, 2)
                weight = torch.max(corr_volume_decoded, dim=0, keepdim=True)[0]
                estimate_volumes.append(corr_volume_decoded)
                
                warp_feature = F.grid_sample(feature[i].unsqueeze(0), grid).squeeze()
                warp_feature *= weight
                warp_feature *= mask
                fuse_feature.append(warp_feature)
            
            # max / sum
            fuse_features.append(torch.max(torch.stack(fuse_feature), dim = 0)[0])
        
        # B,C,H,W
        out_feature = torch.stack(fuse_features)
        if estimate_volumes:
            corr_vol = torch.stack(estimate_volumes)
        else:
            corr_vol = None

        psm = self.cls_head(out_feature)
        rm = self.reg_head(out_feature)

        output_dict = {'psm': psm,
                       'rm': rm,
                       'corr_vol': corr_vol}

        return output_dict
    
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def computeCorrespondenceVolume(self, featA, featB):
        """compute the similarity volume

        Args:
            featA: [C, H, W], the query vectors of target agent 
            featB: [C, H, W], the key vectors of support agent

        Returns:
            distAB: [Hs*Ws+1, Ht, Wt]
        """
        C, H, W = featA.shape
        
        distAB = torch.zeros((self.H * self.W +1, self.H, self.W), device=featA.device)
        fA = featA.permute(1,2,0).reshape(-1,C) # (H*W, C)
        fB = featB.permute(1,2,0).reshape(-1,C) # (H*W, C)
        
        fA2 = torch.pow(torch.norm(fA,dim=-1),2).view(-1,1).repeat(1,fA.shape[0]) # (H*W, H*W)
        fB2 = torch.pow(torch.norm(fB,dim=-1),2).view(-1,1).repeat(1,fB.shape[0]) # (H*W, H*W)

 
        normA = torch.pow( fA2 + fB2.t() - 2.*torch.matmul(fA,fB.t()), 0.5 ) # (H*W, H*W)


        distAB[:-1,...] = normA.permute(1,0).reshape(-1, H, W)
        distAB[-1,:,:] = torch.norm(featA,p=2,dim=0)
        distAB = -distAB # two pixel is similar, then distAB[pixel1,pixel2] is low. We want it high 

        return distAB

    def smoothCorrespondenceVolume(self, distAB):
        """ smooth the correspondence Volume

        Args:
            distAB: (Hs*Ws+1, Ht, Wt)
        Returns:
            smoothed distAB
        """
        distAB = distAB.unsqueeze(0)
        output = self.smoothing_net(distAB)
        output.squeeze_(0)

        return output

    def idx2grid(self, matches):
        """
        Args:
            matches: (Hs*Ws + 1, Ht, Wt)
        """
        # should rewrite because H!=W
        # matches = matches.unsqueeze(0) # [1, Hs*Ws + 1, Ht, Wt]


        H, W = matches.shape[-2:]
        X = torch.arange(W).view(1,-1).repeat(H,1).type(torch.long).view(-1).to(matches.device) # (Ht * Wt)
        Y = torch.arange(H).view(-1,1).repeat(1,W).type(torch.long).view(-1).to(matches.device) # (Ht * Wt)
        X = torch.cat([X,torch.tensor([0],device=matches.device)],0)
        Y = torch.cat([Y,torch.tensor([0],device=matches.device)],0)

        idx = torch.argmax(matches.detach(),0).view(-1) # (Ht*Wt), the value is the index in supporting map
        
        # idx has no gradient
        # mask select those have no correspondence.
        # that means, ego's feature is used.
        mask = (idx == (matches.shape[0] - 1)).view(H, W).to(matches.device)

        x = torch.index_select(X,0,idx).view(1,H,W) # x_src in affine_grid
        y = torch.index_select(Y,0,idx).view(1,H,W) # y_src in affine_grid
        x = 2*((1.*x/W)-0.5) # (1, H, W)
        y = 2*((1.*y/H)-0.5) # (1, H, W)

        grid = torch.cat([x.unsqueeze(-1),y.unsqueeze(-1)],-1) # (1, 32, 32, 2)

        return grid, mask
