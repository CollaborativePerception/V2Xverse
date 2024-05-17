import torch
import torch.nn as nn
from opencood.models.fuse_modules.fusion_in_one import regroup, warp_feature
from opencood.models.fuse_modules.att_fuse import ScaledDotProductAttention
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple

# TODO
# https://github.com/microsoft/Swin-Transformer/tree/f92123a0035930d89cf53fcb8257199481c4428d/kernels/window_process


class MAttFusion(nn.Module):
    def __init__(self, feature_dims):
        super().__init__()
        print(feature_dims)
        print(type(feature_dims))
        self.att = ScaledDotProductAttention(feature_dims)
    
    def forward(self, x, record_len, pairwise_t_matrix, lidar_agent_indicator):
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(x, record_len)
        split_lidar_indicator = regroup(lidar_agent_indicator, record_len)

        batch_node_features = split_x
        batch_node_lidar_agent = split_lidar_indicator

        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            lidar_agent = batch_node_lidar_agent[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            
            # update each node i
            i = 0 # ego
            neighbor_feature = warp_affine_simple(batch_node_features[b],
                                            t_matrix[i, :, :, :],
                                            (H, W))
            if sum(lidar_agent) !=0 and sum(lidar_agent) != N:
                # multi modality aware
                lidar_feature = torch.max(neighbor_feature[lidar_agent], dim=0)[0] # [C, H, W]
                camera_feature = torch.max(neighbor_feature[1-lidar_agent], dim=0)[0] # [C, H, W]
                N_lidar = sum(lidar_agent)
                N_camera = N - N_lidar

                # spatial attention 3x3
                camera_feature_3x3 = []
                x_offsets = [-1, 0, 1]
                y_offsets = [-1, 0, 1]
                for x_offset in x_offsets:
                    for y_offset in y_offsets:
                        camera_feature_3x3.append(torch.roll(camera_feature, (x_offset, y_offset), (0,1)))
                camera_feature_3x3 = torch.stack(camera_feature_3x3, dim=0) # 9, C, H, W

                key = lidar_feature.view(1, C, -1).permute(2, 0, 1) #  [H*W, 1, C]
                query = camera_feature_3x3.view(9, C, -1).permute(2, 0, 1) # [H*W, N_camera, C]
                value = query
                h = self.att(key, query, value)
                h = h.permute(1, 2, 0).view(1, C, H, W)[0, ...] # [C, H, W]
                out.append(torch.maximum(h, lidar_feature))

            else:
                # single modality
                cav_num = neighbor_feature.shape[0]
                x = neighbor_feature.view(cav_num, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
                h = self.att(x, x, x)
                h = h.permute(1, 2, 0).view(cav_num, C, H, W)[0, ...]  # C, W, H before
                out.append(h)


        out = torch.stack(out)
        
        return out
