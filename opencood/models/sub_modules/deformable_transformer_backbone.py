import numpy as np
import torch
import math
import torch.nn as nn
from opencood.models.sub_modules.resblock import ResNetModified, BasicBlock, Bottleneck
from opencood.models.sub_modules.detr_module import PositionEmbeddingSine, \
                DeformableTransformerEncoderLayer, DeformableTransformerEncoder
from opencood.models.fuse_modules.self_attn import AttFusion
from opencood.models.fuse_modules.deform_fuse import DeformFusion
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple


DEBUG = True


"""
    Different from MaxFusion in max_fuse.py
    This is a simplified version.
    pairwise_t_matrix is already scaled.
"""
def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x

class MaxFusion(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, record_len, pairwise_t_matrix):
        """
        pairwise_t_matrix is already normalized [B, L, L, 2, 3]
        """
        split_x = regroup(x, record_len)
        batch_size = len(record_len)
        C, H, W = split_x[0].shape[1:]  # C, W, H before
        out = []
        for b, xx in enumerate(split_x):
            N = xx.shape[0]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            i = 0
            xx = warp_affine_simple(xx, t_matrix[i, :, :, :], (H, W))

            h = torch.max(xx, dim=0)[0]  # C, W, H before
            out.append(h)
        return torch.stack(out, dim=0)




class DeformableTransformerBackbone(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.compress = False

        
        self.proj_first = True
        if ('proj_first' in model_cfg) and (model_cfg['proj_first'] is False):
            self.proj_first = False
            self.discrete_ratio = model_cfg['voxel_size'][0]
            self.downsample_rate = 1

        self.level_num = len(model_cfg['layer_nums']) # exactly 3 now

        layer_nums = model_cfg['layer_nums']
        num_filters = model_cfg['num_filters']
        layer_strides = model_cfg['layer_strides']
        hidden_dim = model_cfg['hidden_dim']
        upsample_strides = model_cfg['upsample_strides']
        num_upsample_filters = model_cfg['num_upsample_filter']

        self.resnet = ResNetModified(BasicBlock, 
                                        layer_nums,
                                        layer_strides,
                                        num_filters)

        self.position_embedding = PositionEmbeddingSine(hidden_dim//2)

        self.hidden_dim = hidden_dim

        if model_cfg['fusion'] == 'max':
            self.fuse_net = [MaxFusion() for _ in range(self.level_num)]
        elif model_cfg['fusion'] == 'self_att':
            self.fuse_net = [AttFusion(n_filter) for n_filter in num_filters]
        elif model_cfg['fusion'] == 'deform':
            self.fuse_net = DeformFusion(num_filters[0], model_cfg['deform_method'])
        elif model_cfg['fusion'] == 'deform_w_cycle':
            assert self.proj_first is False
            assert model_cfg['deform_method'] == 'rigid'
            self.fuse_net = DeformFusion(num_filters[0], model_cfg['deform_method'], cycle_consist_loss=True)
        else:
            raise

        input_proj_list = []
        for i in range(self.level_num):
            proj_in_channels = num_filters[i]
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(proj_in_channels, self.hidden_dim, kernel_size=1),
                nn.GroupNorm(32, self.hidden_dim),
            ))

        self.input_proj = nn.ModuleList(input_proj_list)
        self.level_embed = nn.Parameter(torch.Tensor(self.level_num, self.hidden_dim))
        self.upsample_strides = model_cfg['upsample_strides']

        encoder_layer = DeformableTransformerEncoderLayer(self.hidden_dim, model_cfg['dim_feedforward'],
                                                          model_cfg['dropout'], model_cfg['activation'],
                                                          self.level_num, model_cfg['n_head'], model_cfg['enc_n_points'])
        self.encoder = DeformableTransformerEncoder(encoder_layer, model_cfg['num_encoder_layers'])

        self.deblocks = nn.ModuleList()
        for idx in range(self.level_num):
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(
                    self.hidden_dim, num_upsample_filters[idx],
                    upsample_strides[idx],
                    stride=upsample_strides[idx], bias=False
                ),
                nn.BatchNorm2d(num_upsample_filters[idx],
                                eps=1e-3, momentum=0.01),
                nn.ReLU()
                ))

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        if DEBUG:
            origin_features = torch.clone(spatial_features)

        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        lidar_pose = data_dict['lidar_pose'] # (sum(cav),6 )

        ups = []
        ret_dict = {}
        x = spatial_features

        B = len(record_len)
        H, W = x.shape[2:]  ## this is original feature map [200, 704], not downsampled
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]

        if not self.proj_first:
            pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
            pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
            pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
            pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2


        features = self.resnet(x)  # feature[i] is (sum(cav), C, H, W), different i, different C, H, W

        if self.model_cfg['fusion'].startswith('deform'):
            x_fuseds = self.fuse_net(features, record_len, pairwise_t_matrix, lidar_pose)
        else:
            x_fuseds = [self.fuse_net[i](features[i], record_len, pairwise_t_matrix) for i in range(len(features))]

        pos_embeds = [self.position_embedding(x_fused) for x_fused in x_fuseds]
        srcs = [self.input_proj[i](x_fuseds[i]) for i in range(len(x_fuseds))]


        # srcs = []
        # pos_embeds = []
        # for i, feat in enumerate(features):
        #     x_fused = self.fuse_net[i](feat, record_len, pairwise_t_matrix)
        #     x_pos = self.position_embedding(x_fused)
        #     x_fused = self.input_proj[i](x_fused)
        #     srcs.append(x_fused)  # (B, hidden_dim, H1, W1)
        #     pos_embeds.append(x_pos)


        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.zeros(src_flatten.shape[:2], device=src_flatten.device, dtype=torch.bool)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in srcs], 1)


        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        flatten_length = [h*w for (h,w) in spatial_shapes]
        output_split = torch.split(memory, flatten_length, dim=1)
        output_features = [output.reshape(bs,spatial_shapes[i][0], spatial_shapes[i][1],self.hidden_dim).permute(0,3,1,2) for i, output in enumerate(output_split)]
        
        ups = []
        for i, feat in enumerate(output_features):
            feat = self.deblocks[i](feat)
            ups.append(feat)

        ups = torch.cat(ups, dim=1)

        x = ups

        data_dict['spatial_features_2d'] = x
        return data_dict


    def get_valid_ratio(self, x):
        N, _, H, W = x.shape
        mask = torch.zeros((N,H,W),dtype=torch.bool,device=x.device)
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio