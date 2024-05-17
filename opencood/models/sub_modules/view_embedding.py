import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
import numpy as np
from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu

def all_pair_l2(A, B):
    """ All pair L2 distance for A and B
    Args:
        A : np.ndarray
            shape [N_A, D]
        B : np.ndarray
            shape [N_B, D]
    Returns:
        C : np.ndarray
            shape [N_A, N_B]
    """
    TwoAB = 2*A@B.T  # [N_A, N_B]
    C = torch.sqrt(
              torch.sum(A * A, 1, keepdim=True).repeat_interleave(TwoAB.shape[1], dim=1) \
            + torch.sum(B * B, 1, keepdim=True).T.repeat_interleave(TwoAB.shape[0], dim=0) \
            - TwoAB
        )
    return C

def bilinear_interpolate_torch(im, x, y):
    """
    .--------> x
    |
    |
    |
    v y

    x0y0 ------ x1
    |           |
    |           |
    |           |
    |           |
    y1 ------- x1y1

    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)
    Returns:
    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans

def boxes_to_tfm(box3d):
    with torch.no_grad():
        cos_theta = torch.cos(box3d[:, -1])
        sin_theta = torch.sin(box3d[:, -1])
        pos_x = box3d[:, 0]
        pos_y = box3d[:, 1]
        T_ego_obj_row1 = torch.stack([cos_theta, -sin_theta, pos_x], dim=-1) # [N, 3]
        T_ego_obj_row2 = torch.stack([sin_theta, cos_theta, pos_y], dim=-1)
        T_ego_obj_row3 = torch.tensor([0,0,1.], device=T_ego_obj_row1.device).expand(T_ego_obj_row1.shape)
        T_ego_obj = torch.stack([T_ego_obj_row1, T_ego_obj_row2, T_ego_obj_row3], dim=1)
        return T_ego_obj

def get_poi(pred_box3d_list, order, num_of_sample):
    """
        get point of interest

    Frist, Divide the area of ego agent
    .--------> x
    |
    |
    |
    v y

       0   |   1     |   2 
    -------+---------+-------
       3   | (obj) 4 |   5
    -------+---------+-------
       6   |   7     |   8


    Inputs:
        pred_box3d_list: [[shape: N1, 7], [shape: N2, 7], ...], angle in rad
    Returns
        ego_partition_list: [[shape: N1], [shape: N2], ...]
    """
    poi_list = []
    poi_norm_in_obj = []
    poi_valid_mask_list = []
    for box3d in pred_box3d_list:
        T_ego_obj = boxes_to_tfm(box3d) # [N_box, 3, 3]
        
        T_obj_ego = torch.linalg.inv(T_ego_obj)
        x_obj_ego = T_obj_ego[:, 0, 2]
        y_obj_ego = T_obj_ego[:, 1, 2]

        hwl = box3d[:, 3:6] if order == "hwl" else box3d[:, [5,4,3]]
        ego_in_left = (x_obj_ego < - hwl[:, 2]/2).int().view(-1,1) # [N_box, 1]
        ego_in_right = (x_obj_ego > - hwl[:, 2]/2).int().view(-1,1)
        ego_in_up = (y_obj_ego < - hwl[:, 1]/2).int().view(-1,1)
        ego_in_down = (y_obj_ego > hwl[:, 1]/2).int().view(-1,1)
        
        poi_norm = torch.rand((box3d.shape[0], num_of_sample, 2), device=box3d.device) * 2 - 1 # range [-1, 1]

        ego_in_left_poi_deprecated_mask = (poi_norm[..., 0] > 0.6).int() # [N_box, num_of_sample]
        ego_in_right_poi_deprecated_mask = (poi_norm[..., 0] < -0.6).int()
        ego_in_up_poi_deprecated_mask = (poi_norm[..., 1] > 0.6).int()
        ego_in_down_poi_deprecated_mask = (poi_norm[..., 1] < -0.6).int()

        # filter poi
        ego_in_left = (x_obj_ego < - hwl[:, 2]/2).int().view(-1,1) # [N_box, 1]
        # [N_box, num_of_sample]
        poi_deprecated_mask = ego_in_left * ego_in_left_poi_deprecated_mask + \
                              ego_in_right * ego_in_right_poi_deprecated_mask + \
                              ego_in_up * ego_in_up_poi_deprecated_mask + \
                              ego_in_down * ego_in_down_poi_deprecated_mask 
        poi_deprecated_mask = poi_deprecated_mask > 1
        poi_valid_mask = poi_deprecated_mask == 0

        poi_exact_pos_in_obj_coor = poi_norm * hwl[:, [2,1]].view(box3d.shape[0], 1, 2) # [N_box, num_of_sample ,2]
        poi_exact_pos_in_obj_coor_homo = F.pad(poi_exact_pos_in_obj_coor, (0,1), 'constant', 1) # [N_box, num_of_sample, 3]
        poi_exact_pos_in_ego_coor = torch.bmm(T_ego_obj, poi_exact_pos_in_obj_coor_homo.permute(0, 2, 1)) # [N_box, 3, num_of_sample]
        poi_exact_pos_in_ego_coor = poi_exact_pos_in_ego_coor.permute(0, 2, 1)  # [N_box, num_of_sample, 3]
        poi_exact_pos_in_ego_coor = poi_exact_pos_in_ego_coor[..., :2] # [N_box, num_of_sample, 2]

        poi_list.append(poi_exact_pos_in_ego_coor)
        poi_valid_mask_list.append(poi_valid_mask)
        poi_norm_in_obj.append(poi_norm)

    return poi_list, poi_norm_in_obj, poi_valid_mask_list

class PoiExtractor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pc_range = args['pc_range']
        self.bev_stride = args['stride']
        self.voxel_size= args['voxel_size'][0]
        self.grid_size = self.voxel_size * self.bev_stride
        self.order = args['order']
        self.sample_num = args['sample_num'] # 20 may be ok
        self.feat_dim = args['feat_dim'] # 64

        # learn from relative position (poi_norm) to feature
        self.emb = Embedding(2, self.feat_dim, args['N_freqs'])
        self.alpha = nn.Parameter(torch.tensor([0.5]))

        # preset grids
        grid_x = torch.linspace(self.pc_range[0] + self.grid_size/2, self.pc_range[3] - self.grid_size/2, steps = int((self.pc_range[3]-self.pc_range[0])//self.grid_size), device='cuda')
        grid_y = torch.linspace(self.pc_range[1] + self.grid_size/2, self.pc_range[4] - self.grid_size/2, steps = int((self.pc_range[4]-self.pc_range[1])//self.grid_size), device='cuda')

        self.grid_x_idx = torch.arange(int((self.pc_range[3]-self.pc_range[0])//self.grid_size), device='cuda')
        self.grid_y_idx = torch.arange(int((self.pc_range[4]-self.pc_range[1])//self.grid_size), device='cuda')
        self.bev_grid_idx = torch.cartesian_prod(self.grid_x_idx, self.grid_y_idx) # [num_of_grid, 2]

        self.bev_grid_points = torch.cartesian_prod(grid_x, grid_y) # [num_of_grid, 2]
        self.bev_grid_points_xyz = F.pad(self.bev_grid_points, (0,1), mode='constant', value=1) # x,y,z, [num_of_grid, 3]
        

    def forward(self, heter_feature_2d, pred_box3d_list, lidar_agent_indicator, inferring=False):
        bs = heter_feature_2d.shape[0]
        # poi_list              [[N_box1, num_of_sample, 2], ...]
        # poi_norm_in_obj       [[N_box1, num_of_sample, 2], ...]
        # poi_valid_mask_list   [[N_box1, num_of_sample]]
        
        lidar_pred_box3d_list = [x for i, x in enumerate(pred_box3d_list) if lidar_agent_indicator[i]]
        poi_list, poi_norm_in_obj, poi_valid_mask_list = get_poi(lidar_pred_box3d_list, self.order, self.sample_num)

        # learning. only within lidar agent
        poi_feature_pred, poi_feature_gt = \
            self.learning(heter_feature_2d[lidar_agent_indicator==1], poi_list, poi_norm_in_obj, poi_valid_mask_list)

        if inferring:
            heter_feature_2d_pred, heter_feature_2d_pred_mask = self.inferring(heter_feature_2d, pred_box3d_list)
            heter_feature_2d = heter_feature_2d * (1 - heter_feature_2d_pred_mask) + \
                               heter_feature_2d * (heter_feature_2d_pred_mask) * self.alpha + \
                               heter_feature_2d_pred * (heter_feature_2d_pred_mask) * (1 - self.alpha)

        return heter_feature_2d, poi_feature_pred, poi_feature_gt


    def learning(self, lidar_feature_2d, poi_list, poi_norm_in_obj, poi_valid_mask_list):
        poi_feature_list = []
        poi_norm_valid_list = []

        # learning
        for i, (poi, poi_norm, mask) in enumerate(zip(poi_list, poi_norm_in_obj, poi_valid_mask_list)):
            x_idxs = (poi[..., 0] - self.pc_range[0]) / self.grid_size + 0.5 # [N_box1, num_of_sample]
            y_idxs = (poi[..., 1] - self.pc_range[1]) / self.grid_size + 0.5 # [N_box1, num_of_sample]
            cur_x_idxs = x_idxs[mask == 1] # [N_poi, ]
            cur_y_idxs = y_idxs[mask == 1] # [N_poi, ]

            cur_bev_feature = lidar_feature_2d[i].permute(1, 2, 0) # [H, W, C]
            poi_feature = bilinear_interpolate_torch(cur_bev_feature, cur_x_idxs, cur_y_idxs) # [N_poi, C]
            poi_norm_valid = poi_norm[mask == 1] # [N_poi, 2]
            
            poi_feature_list.append(poi_feature)
            poi_norm_valid_list.append(poi_norm_valid)

        poi_feature_gt = torch.cat(poi_feature_list) # [sum(N_poi), C]
        poi_norm = torch.cat(poi_norm_valid_list) # [sum(N_poi), 2]
        poi_feature_pred = self.emb(poi_norm)

        return poi_feature_pred, poi_feature_gt


    def inferring(self, heter_feature_2d, pred_box3d_list):
        max_len = max([len(pred_box3d) for pred_box3d in pred_box3d_list])
        pred_box3d_tensor = torch.zeros((heter_feature_2d.shape[0], max_len, 7), device=heter_feature_2d.device) # [B, max_box_num, 7]
        heter_feature_pred = torch.zeros_like(heter_feature_2d, device=heter_feature_2d.device)
        heter_feature_pred_mask = torch.zeros((heter_feature_2d.shape[0], 1, heter_feature_2d.shape[2], heter_feature_2d.shape[3]), \
                                               device=heter_feature_2d.device)

        for i, pred_box3d in enumerate(pred_box3d_list):
            pred_box3d_copy = pred_box3d.clone()
            pred_box3d_copy[:, 2] = 1 # move the z center to 1
            if self.order == "hwl":
                pred_box3d_copy[:, [3,4,5]] = pred_box3d_copy[:, [5,4,3]] # -> dx dy dz

            pred_box3d_tensor[i,:len(pred_box3d)] = pred_box3d_copy

        bev_grid_points = self.bev_grid_points_xyz.expand(heter_feature_2d.shape[0], -1, -1) # [B, num_of_grid, 3]
        masks = points_in_boxes_gpu(bev_grid_points, pred_box3d_tensor) # [B, num_of_grid]

        for i, mask in enumerate(masks):
            pred_box3d = pred_box3d_list[i]
            if pred_box3d.shape[0] == 0 or sum(mask > 0) == 0:
                continue
            T_ego_objs = boxes_to_tfm(pred_box3d)
            T_objs_ego = torch.linalg.inv(T_ego_objs) # [N_box, 3, 3]
            object_xy_coor = pred_box3d[:, :2] # [N_box, 2]
            bev_grid_xy_coor = bev_grid_points[i][..., :2][mask > 0] # [num_of_valid_grid, 2]

            # assign grid to object
            grid_object_l2dis = all_pair_l2(bev_grid_xy_coor, object_xy_coor)
            grid_in_which_object = torch.argmin(grid_object_l2dis, dim=1) # shape [num_of_valid_grid,], value within [0, N_box)
            T_objs_ego_for_the_grid = T_objs_ego[grid_in_which_object] # [num_of_valid_grid, 3, 3]

            object_size_for_the_grid = pred_box3d[grid_in_which_object][:,[5,4]] if self.order=='hwl' \
                                  else pred_box3d[grid_in_which_object][:,[3,4]] # [num_of_valid_grid, 2]

            # get pos in object coord.
            bev_grid_xy_homo = F.pad(bev_grid_xy_coor, (0,1), 'constant', 1).unsqueeze(-1) # [num_of_valid_grid, 3, 1]
            grid_in_obj_coor = torch.bmm(T_objs_ego_for_the_grid, bev_grid_xy_homo) # [num_of_valid_grid, 3, 1]
            grid_in_obj_xy_coor = grid_in_obj_coor[:,:2,0] # [num_of_valid_grid, 2]
            grid_in_obj_xy_norm = grid_in_obj_xy_coor / object_size_for_the_grid # [num_of_valid_grid, 2]

            feature_idx = self.bev_grid_idx[mask > 0] # [num_of_valid_grid, 2]
            features = self.emb(grid_in_obj_xy_norm) # [num_of_valid_grid, 64]
            
            heter_feature_pred[i, :, feature_idx[:, 1], feature_idx[:, 0]] = features.T
            heter_feature_pred_mask[i, 0, feature_idx[:, 1], feature_idx[:, 0]] = 1
        
        return heter_feature_pred, heter_feature_pred_mask
        





class Embedding(nn.Module):
    def __init__(self, in_channels, out_channels=64, N_freqs=8, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.mlp_in_channels = in_channels*(len(self.funcs)*N_freqs + 1) # 2 * 8 * 2 + 2 = 34
        self.mlp_inter_channels = out_channels * 2
        self.mlp_out_channels = out_channels

        self.mlp_layers = [nn.Linear(self.mlp_in_channels, self.mlp_inter_channels)]
        for i in range(4):
            self.mlp_layers.append(nn.ReLU(inplace=True))
            self.mlp_layers.append(nn.Linear(self.mlp_inter_channels, self.mlp_inter_channels))
        self.mlp_layers.append(nn.ReLU(inplace=True))
        self.mlp_layers.append(nn.Linear(self.mlp_inter_channels, self.mlp_out_channels))
        
        self.mlp_layers = nn.Sequential(*self.mlp_layers)

 
        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)
 
    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
 
        Inputs:
            x: (B, self.in_channels)
 
        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        out = torch.cat(out, -1)
        out = self.mlp_layers(out)
 
        return out

