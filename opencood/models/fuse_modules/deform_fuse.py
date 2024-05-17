"""
Implementation of deformable fusion

The design is: for ego agent f_0 and collaborative agent f_1.

f_0[x0,y0] may not correspond to f_1[x0,y0]

So it will learn an offset (delta_x and delta_y) for this pixel position.
Then f_0[x0,y0] will fuse with f_1[x0+delta_x, y0+delta_y]
"""

from this import d
import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from icecream import ic

def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x


class RigidOffset(nn.Module):
    """ Learn a rigid transformation grid for the whole feature map
    """

    def __init__(self, in_ch, hidden_ch=32):
        super(RigidOffset, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_features=hidden_ch, out_features=hidden_ch, bias=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(in_features=hidden_ch, out_features=3, bias=True),
        )

    def forward(self, x, return_M=False):
        """
        Args:
            x.shape:(sum(record_len_minus1), 2C, H, W)
        Returns:
            out.shape: (sum(record_len_minus1), H, W, 2)
        """
        N, _, H, W = x.shape
        xytheta = self.model(x)  # [sum(record_len_minus1), 3], 3 corresponds to x, y, theta


        cos = torch.cos(xytheta[:, 2])
        sin = torch.sin(xytheta[:, 2])

        M = torch.zeros((N, 2, 3), device=x.device)
        M[:, 0, 0] = cos
        M[:, 0, 1] = sin
        M[:, 1, 0] = -sin
        M[:, 1, 1] = cos
        M[:, 0, 2] = xytheta[:, 0]
        M[:, 1, 2] = xytheta[:, 1]

        grid = F.affine_grid(M, size=x.shape)

        if return_M:
            return grid, M

        return grid


class ArbitraryOffset(nn.Module):
    """ Learn a offset/residual grid for each pixel
    """

    def __init__(self, in_ch, out_ch=2, hidden_ch=32):
        """
        Args:
            in_ch: is 2 times feature channel, since they concat together
        """
        super(ArbitraryOffset, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, 1, 1),
            nn.InstanceNorm2d(hidden_ch),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(hidden_ch, hidden_ch // 2, 3, 1, 1),
            nn.InstanceNorm2d(hidden_ch // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(hidden_ch // 2, hidden_ch // 4, 1, 1, 0),
            nn.InstanceNorm2d(hidden_ch // 4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(hidden_ch // 4, 2, 1, 1, 0)
        )

    def forward(self, x):
        """
        Args:
            x.shape:(sum(record_len_minus1), 2C, H, W)
        Returns:
            out.shape: (sum(record_len_minus1), H, W, 2)
        """
        N, _, H, W = x.shape

        x = self.model(x)

        grid_residual = x.reshape(N, H, W, 2)

        M_origin = torch.Tensor([[[1, 0, 0], [0, 1, 0]]])
        grid_origin = F.affine_grid(M_origin, size=(1, 1, H, W)).to(x.device)

        grid = grid_residual + grid_origin
        return grid


class DeformFusion(nn.Module):
    """ deformable fusion for multiscale feature map
        For each pixel in ego agent's feature map,
        it will learn a offset to fuse the feature.
    """

    def __init__(self, in_ch, deform_method, cycle_consist_loss=False):
        """
        Args:
            in_ch: channels num of one agent's feature map.
        """
        super(DeformFusion, self).__init__()
        self.cycle_consistency_loss = cycle_consist_loss

        if deform_method == "rigid":
            self.grid_net = RigidOffset(in_ch * 2)
        elif deform_method == "arbitrary":
            self.grid_net = ArbitraryOffset(in_ch * 2)
        

    def forward(self, features, record_len, pairwise_t_matrix, lidar_pose=None):
        """
        Args:
            features: List[torch.Tensor]
                multiscale features. features[i] is (sum(cav), C, H, W), different i, different C, H, W
            record_len: torch.tensor
                record cav number
            pairwise_t_matrix: torch.Tensor,
                already normalized. shape [B, N_max, N_max, 2, 3]
            lidar_pose: torch.Tensor
                shape [(sum(cav), 6)], this is only used to calculate intersection. If proj_first=False, then equal to pairwise_t_matrix
        """

        ##### first align them to ego coordinate, espeically when proj_first = False.
        device = features[0].device
        record_len_minus1 = record_len - 1

        if(torch.sum(record_len_minus1)==0):
            return features

        ms_split_x = [regroup(features[i], record_len) for i in range(len(features))]
        ms_split_x_warp = []

        for split_x in ms_split_x:  # different scale
            split_x_warp = []
            H, W = split_x[0].shape[2:]
            for b, xx in enumerate(split_x):  # different samples
                N = xx.shape[0]
                t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                i = 0  # ego
                split_x_warp.append(warp_affine_simple(xx, t_matrix[i, :, :, :], (H, W)))  # [N_,C,H,W], N_ varies
            ms_split_x_warp.append(split_x_warp)


        ##### we caculate the grid by scale=1 feature, and share it with all scales.
        split_x = ms_split_x_warp[0]  # first scale
        H, W = split_x[0].shape[:2]

        cat_features = []
        for b, xx in enumerate(split_x):
            N = xx.shape[0]
            cat_feature = torch.cat([xx[0:1].expand(N - 1, -1, -1, -1), xx[1:]], dim=1)  # (N-1, 2C, H, W)
            cat_features.append(cat_feature)

        cat_feature = torch.cat(cat_features, dim=0)  # (sum(record_len_minus1), 2C, H, W)

        grid_offset = self.grid_net(cat_feature)  # (sum(record_len_minus1), H, W, 2)

        grid = grid_offset  # (sum(record_len_minus1),H,W,2)
        ms_grid = [grid[:,::2**i,::2**i,:] for i in range(len(features))]

        ms_split_grid = [regroup(grid, record_len_minus1) for grid in ms_grid]  # [[N1-1,H,W,2], [N2-1,H,W,2],...], shared for all scales.

        #####  fusion
        ms_fused_features = []
        for scale, split_x in enumerate(ms_split_x_warp):
            fused_features = []
            for b, xx in enumerate(split_x):
                if xx.shape[0] == 1:
                    fused_features.append(xx[0])
                else:
                    neighbor_feature_deform = torch.cat([F.grid_sample(xx[1:], ms_split_grid[scale][b]), xx[0:1]], dim=0) # (N-1, C, H, W)
                    fuesd_feature = torch.max(neighbor_feature_deform, dim=0)[0]  
                    fused_features.append(fuesd_feature)
            ms_fused_features.append(torch.stack(fused_features))


        if self.cycle_consistency_loss:
            split_x = ms_split_x[0]  # before warping to the ego agent, scale = 1
            H, W = split_x[0].shape[2:]

            cat_features = []
            for b, xx in enumerate(split_x):
                N = xx.shape[0]
                t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                """
                    [agent0, agent1] -> M_0
                    [agent1, agent2] -> M_1
                    ...
                    [agentN-1,agent0] -> M_N-1
                    M_0@M_1@...@M_N-1 = I

                    The latter should align to the former agent.
                """
                latter_agent = torch.cat([xx[1:],xx[:1]], dim=0) # [agent1,agent2,..., agent0]
                t_matrix_adj = torch.stack([t_matrix[i,(i+1)%N] for i in range(N)])
                latter_agent_warp = warp_affine_simple(latter_agent, t_matrix_adj, dsize=(H,W))
                cat_feature = torch.cat([xx, latter_agent_warp], dim=1)
                cat_features.append(cat_feature)

            cat_feature = torch.cat(cat_features, dim=0)  # (sum(record_len), 2C, H, W)
            _, M = self.grid_net(cat_feature, return_M=True)  # (sum(record_len)*H*W, 2)

            M_homo = F.pad(M, (0, 0, 0, 1), "constant", 0)  # pad 2nd to last by (0, 1)
            M_homo[:, 2, 2] = 1

            split_M = regroup(M_homo, record_len)

        return ms_fused_features


if __name__ == "__main__":
    features = [torch.randn(4,64,200,704), torch.randn(4,128,100,352), torch.randn(4,256,50,176)]
    record_len = torch.tensor([1,3])
    pairwise_t_matirx = torch.eye(4).view(1,1,1,4,4).expand(2,5,5,4,4)

    model = DeformFusion(in_ch=64, deform_method='rigid', cycle_consist_loss=True)

    out = model(features, record_len, pairwise_t_matirx)
    for xx in out:
        print(xx.shape)