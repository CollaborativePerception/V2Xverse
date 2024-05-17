# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time

import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method',  type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty']

    hypes = yaml_utils.load_yaml(None, opt)

    hypes['validate_dir'] = hypes['test_dir']
    assert "test" in hypes['validate_dir']
    left_hand = True if "OPV2V" in hypes['test_dir'] else False
    print(f"Left hand visualizing: {left_hand}")

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    opencood_dataset_tiny = Subset(opencood_dataset, range(0,150))
    data_loader = DataLoader(opencood_dataset_tiny,
                             batch_size=1,
                             num_workers=4,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0},
                   0.5: {'tp': [], 'fp': [], 'gt': 0},
                   0.7: {'tp': [], 'fp': [], 'gt': 0}}


    for i, batch_data in enumerate(data_loader):
        print(i)
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_late_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_early_fusion(batch_data,
                                                           model,
                                                           opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_intermediate_fusion(batch_data,
                                                                  model,
                                                                  opencood_dataset)
            elif opt.fusion_method == 'no':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_no_fusion(batch_data,
                                                                  model,
                                                                  opencood_dataset)
            elif opt.fusion_method == 'no_w_uncertainty':
                    pred_box_tensor, pred_score, gt_box_tensor, uncertainty_tensor = \
                        inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                    model,
                                                                    opencood_dataset)
            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')


            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.7)

            vis_save_path = os.path.join(opt.model_dir, 'vis')
            if not os.path.exists(vis_save_path):
                os.makedirs(vis_save_path)
            
            vis_save_path = os.path.join(opt.model_dir, 'vis/3d_%05d.png' % i)
            simple_vis.visualize(pred_box_tensor,
                                gt_box_tensor,
                                batch_data['ego'][
                                    'origin_lidar'][0],
                                hypes['postprocess']['gt_range'],
                                vis_save_path,
                                method='3d',
                                left_hand=left_hand)
            
            vis_save_path = os.path.join(opt.model_dir, 'vis/bev_%05d.png' % i)
            simple_vis.visualize(pred_box_tensor,
                                gt_box_tensor,
                                batch_data['ego'][
                                    'origin_lidar'][0],
                                hypes['postprocess']['gt_range'],
                                vis_save_path,
                                method='bev',
                                left_hand=left_hand)



    eval_utils.eval_final_results(result_stat,
                                  opt.model_dir)


if __name__ == '__main__':
    main()
