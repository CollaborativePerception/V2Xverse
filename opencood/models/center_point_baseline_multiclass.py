import torch.nn as nn
import numpy as np
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
# from opencood.models.sub_modules.compress_core import CompressCore
from opencood.models.sub_modules.naive_compress import NaiveCompressor
# from opencood.models.sub_modules.dcn_net import DCNNet
# from opencood.models.fuse_modules.where2comm import Where2comm
from opencood.models.fuse_modules.where2comm_attn import Where2comm
import torch
import torch.nn.functional as F

class centerpointbaselinemulticlass(nn.Module):
    def __init__(self, args):
        super(centerpointbaselinemulticlass, self).__init__()
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        
        
        if 'resnet' in args['base_bev_backbone']:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        
        self.voxel_size = args['voxel_size']
        self.out_size_factor = args['out_size_factor']
        self.cav_lidar_range  = args['lidar_range']

        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]
        
        self.compression = False
        if 'compression' in args and args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(self.out_channel, args['compression'])

        self.dcn = False
        if 'dcn' in args:
            self.dcn = True
            self.dcn_net = DCNNet(args['dcn'])

        # self.fusion_net = TransformerFusion(args['fusion_args'])
        self.fusion_net = Where2comm(args['fusion_args'])
        self.multi_scale = args['fusion_args']['multi_scale']

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 8 * args['anchor_number'],
                                  kernel_size=1)
        if 'backbone_fix' in args.keys() and args['backbone_fix']:
            self.backbone_fix()
        
        self.init_weight()
    
    def init_weight(self):
        pi = 0.01
        nn.init.constant_(self.cls_head.bias, -np.log((1 - pi) / pi) )
        nn.init.normal_(self.reg_head.weight, mean=0, std=0.001)

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
    
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, data_dict):
        if type(data_dict) == dict:
            voxel_features = data_dict['processed_lidar']['voxel_features']
            voxel_coords = data_dict['processed_lidar']['voxel_coords']
            voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
            record_len = data_dict['record_len']
            # pairwise_t_matrix = data_dict['pairwise_t_matrix']
            batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
            batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
            batch_dict = self.scatter(batch_dict)
            batch_dict = self.backbone(batch_dict)
        # N, C, H', W'. [N, 384, 100, 352]
            spatial_features_2d = batch_dict['spatial_features_2d']
            # print(spatial_features_2d)
        elif type(data_dict) == list:
            spatial_features_2d = []
            for data in data_dict:
                voxel_features = data['processed_lidar']['voxel_features']
                voxel_coords = data['processed_lidar']['voxel_coords']
                voxel_num_points = data['processed_lidar']['voxel_num_points']
                record_len = data['record_len']
                # pairwise_t_matrix = data_dict['pairwise_t_matrix']
                batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
                # n, 4 -> n, c
                batch_dict = self.pillar_vfe(batch_dict)
                # n, c -> N, C, H, W
                batch_dict = self.scatter(batch_dict)
            
                batch_dict = self.backbone(batch_dict)
                # N, C, H', W'. 
                spatial_feature_2d = batch_dict['spatial_features_2d']
                spatial_features_2d.append(spatial_feature_2d)
            spatial_features_2d = torch.cat(spatial_features_2d)
        else:
            print("wrong type of data_dict")


        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        # dcn
        if self.dcn:
            spatial_features_2d = self.dcn_net(spatial_features_2d)
        # spatial_features_2d is [sum(cav_num), 256, 50, 176]
        # output only contains ego
        # [B, 256, 50, 176]
        psm_single = self.cls_head(spatial_features_2d)
        # print(spatial_features_2d)
        rm_single = self.reg_head(spatial_features_2d)
        
        fused_feature = spatial_features_2d
        cls = self.cls_head(fused_feature) # fused_feature [B, 128, 96, 288] -> [B, 3, 96, 288]
        bbox = self.reg_head(fused_feature) # fused_feature [B, 128, 96, 288] -> [B, 24, 96, 288]

        if not self.training:
            _, C, H, W = cls.shape
            cls = psm_single[0].unsqueeze(0).contiguous().view(1, -1, H, W)
            bbox = rm_single[0].unsqueeze(0).contiguous().view(1, -1, H, W)


        box_preds_for_infer = bbox.permute(0, 2, 3, 1).contiguous()
        bbox_temp_list = []
        num_class = int(box_preds_for_infer.shape[3]/8)
        box_preds_for_infer = box_preds_for_infer.view(box_preds_for_infer.shape[0], box_preds_for_infer.shape[1], box_preds_for_infer.shape[2], num_class, 8)
        for i in range(num_class):
            box_preds_for_infer_singleclass = box_preds_for_infer[:,:,:,i,:]
            box_preds_for_infer_singleclass = box_preds_for_infer_singleclass.permute(0, 3, 1, 2)
            _, bbox_temp = self.generate_predicted_boxes(cls[:, i, :, :], box_preds_for_infer_singleclass)
            bbox_temp_list.append(bbox_temp)
        bbox_temp_list = torch.stack(bbox_temp_list, dim=1)


        _, bbox_temp = self.generate_predicted_boxes(cls, bbox)

        feature_list = []
        feature_regroup = self.regroup(spatial_features_2d, record_len)
        for ego_id in range(len(feature_regroup)):
            feature_list.append(feature_regroup[ego_id][0:1])
        feature_egos = torch.cat(feature_list, dim=0)
        result_dict = {'fused_feature':feature_egos}

        output_dict = {'cls_preds': cls,
                       'reg_preds': bbox_temp,
                       'reg_preds_multiclass': bbox_temp_list,
                       'bbox_preds': bbox
                       }
        output_dict.update(result_dict)
        
        

        _, bbox_temp_single = self.generate_predicted_boxes(psm_single, rm_single)

        output_dict.update({'cls_preds_single': psm_single,
                       'reg_preds_single': bbox_temp_single,
                       'bbox_preds_single': rm_single,
                       # 'comm_rate': communication_rates
                       })
        

        return output_dict

    def generate_predicted_boxes(self, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        
        batch, H, W, code_size = box_preds.size()   ## code_size 表示的是预测的尺寸
        
        box_preds = box_preds.reshape(batch, H*W, code_size)

        batch_reg = box_preds[..., 0:2]
        # batch_hei = box_preds[..., 2:3] 
        # batch_dim = torch.exp(box_preds[..., 3:6])
        
        h = box_preds[..., 3:4] * self.out_size_factor * self.voxel_size[0]
        w = box_preds[..., 4:5] * self.out_size_factor * self.voxel_size[1]
        l = box_preds[..., 5:6] * self.out_size_factor * self.voxel_size[2]
        batch_dim = torch.cat([h,w,l], dim=-1)
        batch_hei = box_preds[..., 2:3] * self.out_size_factor * self.voxel_size[2] + self.cav_lidar_range[2]

        batch_rots = box_preds[..., 6:7]
        batch_rotc = box_preds[..., 7:8]

        rot = torch.atan2(batch_rots, batch_rotc)

        ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        ys = ys.view(1, H, W).repeat(batch, 1, 1).to(cls_preds.device)
        xs = xs.view(1, H, W).repeat(batch, 1, 1).to(cls_preds.device)

        xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
        ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

        xs = xs * self.out_size_factor * self.voxel_size[0] + self.cav_lidar_range[0]   ## 基于feature_map 的size求解真实的坐标
        ys = ys * self.out_size_factor * self.voxel_size[1] + self.cav_lidar_range[1]


        batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, rot], dim=2)

        return cls_preds, batch_box_preds
