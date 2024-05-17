# -*- coding: utf-8 -*-
# Author: Yangheng Zhao <zhaoyangheng-sjtu@sjtu.edu.cn>

import os
from torch.utils.data import DataLoader, Subset
from opencood.data_utils import datasets
import torch
from opencood.tools import train_utils, inference_utils
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.visualization import simple_vis_v2, simple_vis
from opencood.data_utils.datasets import build_dataset
import numpy as np

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    np.random.seed(100)


    current_path = os.path.dirname(os.path.realpath(__file__))
    params = load_yaml('./opencood/hypes_yaml/v2xverse/where2comm_debug_config.yaml')
    output_path = "./data_vis"
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    opencda_dataset = build_dataset(params, visualize=True, train=False)
    len = len(opencda_dataset)
    sampled_indices = np.random.permutation(len)[:100]
    subset = Subset(opencda_dataset, sampled_indices)
    
    data_loader = DataLoader(subset, batch_size=1, num_workers=2,
                             collate_fn=opencda_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False)
    vis_gt_box = True
    vis_pred_box = False
    hypes = params

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i, batch_data in enumerate(data_loader):
        # try:
        print(i)
        batch_data = train_utils.to_device(batch_data, device)
        gt_box_tensor = opencda_dataset.post_processor.generate_gt_bbx(batch_data)
        img_dict = {'img_front':batch_data['ego']['img_front'][0], 
                    'img_left':batch_data['ego']['img_left'][0], 
                    'img_right':batch_data['ego']['img_right'][0], 
                    'BEV':batch_data['ego']['BEV'][0]}

        vis_save_path = os.path.join(output_path, 'all_%05d.png' % i)
        simple_vis_v2.visualize({'gt_box_tensor':gt_box_tensor},  # 
                            batch_data['ego']['origin_lidar'][0],
                            batch_data['ego']['lidar_len'][0],
                            hypes['postprocess']['gt_range'],
                            vis_save_path,
                            method='all',
                            left_hand=True,
                            img_dict=img_dict)

        vis_save_path = os.path.join(output_path, 'bev_%05d.png' % i)
        simple_vis.visualize({},
                            batch_data['ego']['origin_lidar'][0],
                            hypes['postprocess']['gt_range'],
                            vis_save_path,
                            method='bev',
                            left_hand=True)

        # except:
        #     print('error batch')
        #     pass
    