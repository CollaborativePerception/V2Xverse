import random, os

import torch
from torch import nn
import numpy as np
from icecream import ic
from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.cia_ssd_utils import SSFA, Head
from opencood.models.sub_modules.matcher_v2 import MatcherV2
from opencood.models.sub_modules.voxel_rcnn_head import VoxelRCNNHead
from opencood.data_utils.post_processor.fpvrcnn_postprocessor import \
    FpvrcnnPostprocessor
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

class FVoxelRCNN(nn.Module):
    def __init__(self, args):
        super(FVoxelRCNN, self).__init__()
        lidar_range = np.array(args['lidar_range'])
        grid_size = np.round((lidar_range[3:6] - lidar_range[:3]) /
                             np.array(args['voxel_size'])).astype(np.int64)
        self.vfe = MeanVFE(args['mean_vfe'],
                           args['mean_vfe']['num_point_features'])
        self.spconv_block = VoxelBackBone8x(args['spconv'],
                                            input_channels=args['spconv'][
                                                'num_features_in'],
                                            grid_size=grid_size)
        self.map_to_bev = HeightCompression(args['map2bev'])
        # set experiment to validate the ssfa module
        self.ssfa = SSFA(args['ssfa'])
        self.head = Head(**args['head'])
        self.post_processor = FpvrcnnPostprocessor(args['post_processer'],
                                                   train=self.training)
        self.matcher = MatcherV2(args['matcher'], args['lidar_range'])
        self.roi_head = VoxelRCNNHead(args['roi_head'], self.spconv_block.backbone_channels)
        self.train_stage2 = args['activate_stage2']

    def forward(self, batch_dict):
        # lidar
        voxel_features = batch_dict['processed_lidar']['voxel_features']
        voxel_coords = batch_dict['processed_lidar']['voxel_coords']
        voxel_num_points = batch_dict['processed_lidar']['voxel_num_points']
        # cemera 

        # save memory
        batch_dict.pop('processed_lidar')
        batch_dict.update({'voxel_features': voxel_features,
                           'voxel_coords': voxel_coords,
                           'voxel_num_points': voxel_num_points,
                           'batch_size': int(batch_dict['record_len'].sum()),
                           'proj_first': batch_dict['proj_first'],
                           'lidar_pose': batch_dict['lidar_pose']})

        batch_dict = self.vfe(batch_dict)
        batch_dict = self.spconv_block(batch_dict)
        batch_dict = self.map_to_bev(batch_dict)

        out = self.ssfa(batch_dict['spatial_features'])
        batch_dict['stage1_out'] = self.head(out) 
        ### stage 1 ### finished

        data_dict, output_dict = {}, {}
        data_dict['ego'], output_dict['ego'] = batch_dict, batch_dict

        pred_box3d_list, scores_list = \
            self.post_processor.post_process(data_dict, output_dict,
                                             stage1=True)
  
        batch_dict['det_boxes'] = pred_box3d_list
        batch_dict['det_scores'] = scores_list

        if pred_box3d_list is not None and self.train_stage2:
            batch_dict = self.matcher(batch_dict)
            batch_dict = self.roi_head(batch_dict)
        return batch_dict



if __name__ == "__main__":
    model = SSFA(None)
    print(model)