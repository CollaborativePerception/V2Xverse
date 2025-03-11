# -*- coding: utf-8 -*-
# Author: Yue Hu <phyllis1sjtu@outlook.com>, Genjia Liu <LGJ1zed@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import numpy as np
import copy
import random

from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.utils.waypoint2map import waypoints2map_radius 

class Communication(nn.Module):
    def __init__(self, args):
        super(Communication, self).__init__()
        
        self.smooth = False
        self.thre = args['thre']
        if 'gaussian_smooth' in args:
            # Gaussian Smooth
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
        self.det_range = args['cav_lidar_range']
        self.use_driving_request = args['driving_request']

        self.args = args
        
    def init_gaussian_filter(self, k_size=5, sigma=1):
        def _gen_gaussian_kernel(k_size=5, sigma=1):
            center = k_size // 2
            x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
            g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
            return g
        gaussian_kernel = _gen_gaussian_kernel(k_size, sigma)
        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def forward(self, batch_confidence_maps, record_len, pairwise_t_matrix, waypoints=None):
        # batch_confidence_maps:[(L1, H, W), (L2, H, W), ...]
        # pairwise_t_matrix: (B,L,L,2,3)
        # thre: threshold of objectiveness
        # a_ji = (1 - q_i)*q_ji
        B, L, _, _, _ = pairwise_t_matrix.shape
        _, _, H, W = batch_confidence_maps[0].shape
        
        ### get matrix for inverse transform
        pairwise_t_matrix_inverse = pairwise_t_matrix.clone()

        pairwise_t_matrix_inverse[...,0,1] = pairwise_t_matrix_inverse[...,0,1] / (H / W)
        pairwise_t_matrix_inverse[...,1,0] = pairwise_t_matrix_inverse[...,1,0] / (W / H)

        pairwise_t_matrix_inverse[...,0,2] *= -1
        pairwise_t_matrix_inverse[...,1,2] *= -1

        pairwise_t_matrix_inverse_2 = pairwise_t_matrix_inverse.clone()

        pairwise_t_matrix_inverse[...,0,1] = pairwise_t_matrix_inverse_2[...,1,0]
        pairwise_t_matrix_inverse[...,1,0] = pairwise_t_matrix_inverse_2[...,0,1]

        pairwise_t_matrix_inverse[...,0,1] = pairwise_t_matrix_inverse[...,0,1] * (H / W)
        pairwise_t_matrix_inverse[...,1,0] = pairwise_t_matrix_inverse[...,1,0] * (W / H)        

        communication_masks = []
        communication_rates = []
        batch_communication_maps = []
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            # t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

            ori_communication_maps = batch_confidence_maps[b].sigmoid().max(dim=1)[0].unsqueeze(1) # dim1=2 represents the confidence of two anchors

            if False: # self.smooth:
                processed_communication_maps = self.gaussian_filter(ori_communication_maps)
                # normalize to 0-1
                if processed_communication_maps.max() >0:
                    processed_communication_maps = processed_communication_maps/processed_communication_maps.max()*ori_communication_maps.max()
            else:
                processed_communication_maps = ori_communication_maps

            ########## driving request ############
            if waypoints is not None: # only used with waypoints prediction model
                # assert B==1 # waypoints.size(0)==len(record_len)

                # radius=40  sigma_reverse=5
                bev_grad_cam = waypoints2map_radius( waypoints.cpu().numpy(), radius=self.args.get('radius',160), sigma_reverse=self.args.get('sigma_reverse',2), \
                                                    grid_coord=[batch_confidence_maps[b].size(2),batch_confidence_maps[b].size(3), \
                                                                self.det_range[4]/(self.det_range[4]-self.det_range[1]),\
                                                                self.det_range[3]/(self.det_range[3]-self.det_range[0])] \
                                                    , det_range=self.det_range) # (1,10,2) -> (1,192,576)

                bev_grad_cam_tensor = torch.tensor(bev_grad_cam).to(batch_confidence_maps[0].device)
                # warp request map
                N = record_len[b].item()
                grad_cam_repeat = bev_grad_cam_tensor[0][None, None].repeat(N,1,1,1) # bev_grad_cam_tensor[b][None, None].repeat(N,1,1,1)
                t_matrix = pairwise_t_matrix_inverse[b][:N, :N, :, :]
                warpped_grad_cam = warp_affine_simple(grad_cam_repeat,
                                                        t_matrix[0, :, :, :],
                                                        (H, W)).clamp(0,1)

                processed_communication_maps = processed_communication_maps * torch.clamp((warpped_grad_cam.to(processed_communication_maps.dtype)*5/(warpped_grad_cam.max()+1e-7)), min=1e-4, max=1 - 1e-4)

            ############################################

            communication_maps = processed_communication_maps

            ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
            zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)

            if self.args.get('random_thre',False):
                thre_list = [0.001,0.003,0.01,0.02,0.1]
                thre = random.choice(thre_list)
                thre = np.random.uniform(0.5*thre, 1.5*thre)
            else:
                thre = self.thre


            communication_mask = torch.where(communication_maps>= thre, ones_mask, zeros_mask)

            communication_rate = communication_mask[1:N].sum()/(H*W)

            # communication_mask = warp_affine_simple(communication_mask,
            #                                 t_matrix[0, :, :, :],
            #                                 (H, W))
            
            communication_mask_nodiag = communication_mask.clone()
            ones_mask = torch.ones_like(communication_mask).to(communication_mask.device)
            communication_mask_nodiag[0] = ones_mask[0]

            communication_masks.append(communication_mask_nodiag)
            communication_rates.append(communication_rate)
            batch_communication_maps.append(ori_communication_maps*communication_mask_nodiag)
        communication_rates = sum(communication_rates)/B
        # communication_masks = torch.stack(communication_masks, dim=0)  ## torch.concat
        communication_masks = torch.concat(communication_masks, dim=0)
        
        return batch_communication_maps, communication_masks, communication_rates