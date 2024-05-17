# -*- coding: utf-8 -*-
# Author: Yifan Lu
# This function loops all modality to be ego.


import argparse
import os
import time
from typing import OrderedDict
import importlib
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis
from opencood.utils.common_utils import update_dict, read_json
torch.multiprocessing.set_sharing_strategy('file_system')

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=100,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--range', type=str, default="140.8,40",
                        help="detection range is [-140.8,+140.8m, -40m, +40m]")
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--bn_train', action='store_true',
                        help="use the running mean and variance instead of stored stats.")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 

    hypes = yaml_utils.load_yaml(None, opt)

    if 'heter' in hypes:
        # hypes['heter']['lidar_channels'] = 16
        # opt.note += "_16ch"

        x_min, x_max = -eval(opt.range.split(',')[0]), eval(opt.range.split(',')[0])
        y_min, y_max = -eval(opt.range.split(',')[1]), eval(opt.range.split(',')[1])
        opt.note += f"_{x_max}_{y_max}"

        new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2], \
                            x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]

        # replace all appearance
        hypes = update_dict(hypes, {
            "cav_lidar_range": new_cav_range,
            "lidar_range": new_cav_range,
            "gt_range": new_cav_range
        })

        # reload anchor
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                parser_func = func
        hypes = parser_func(hypes)

        
    
    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    
    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']


    # setting noise
    np.random.seed(303)
    
    # Create the dictionary for evaluation
    result_stat_all = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

   
    if opt.bn_train:
        opt.note += '_BNtrain'
    infer_info_all = opt.fusion_method + opt.note

    """
    Loop m1 m2 m3 m4 modality.
    In each loop, select the agent of the modality to be ego. 
    """
    for modality in ['m1', 'm2', 'm3', 'm4']:
    # for modality in ['m1', 'm2']:
        hypes = update_dict(hypes, {
            "ego_modality": modality
        })

        print('Creating Model')
        model = train_utils.create_model(hypes)
        # we assume gpu is necessary
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print('Loading Model from checkpoint')
        saved_path = opt.model_dir
        resume_epoch, model = train_utils.load_saved_model(saved_path, model)
        print(f"resume from {resume_epoch} epoch.")
        opt.note += f"_epoch{resume_epoch}"
        
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        # use running mean and variance !
        if opt.bn_train:
            def bn_train(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.train()
            model.apply(bn_train)

        print('Dataset Building')
        opencood_dataset = build_dataset(hypes, visualize=True, train=False)
        data_loader = DataLoader(opencood_dataset,
                                batch_size=1,
                                num_workers=4,
                                collate_fn=opencood_dataset.collate_batch_test,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=False)
                                

        result_stat_modality = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
            0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
            0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
        infer_info_modality = infer_info_all + "_" + modality
        
        for i, batch_data in enumerate(data_loader):
            print(f"{infer_info_modality}_{i}")

            if batch_data is None:
                continue
            if hypes['fusion']['core_method'] == 'intermediateheter' and modality != batch_data['ego']['agent_modality_list'][0]:
                continue
            if hypes['fusion']['core_method'] == 'lateheter' and modality != batch_data['ego']['modality_name']:
                print(batch_data['ego']['modality_name'])
                continue
            
            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, device)

                if opt.fusion_method == 'late':
                    infer_result = inference_utils.inference_late_fusion(batch_data,
                                                            model,
                                                            opencood_dataset)
                elif opt.fusion_method == 'early':
                    infer_result = inference_utils.inference_early_fusion(batch_data,
                                                            model,
                                                            opencood_dataset)
                elif opt.fusion_method == 'intermediate':
                    infer_result = inference_utils.inference_intermediate_fusion(batch_data,
                                                                    model,
                                                                    opencood_dataset)
                elif opt.fusion_method == 'no':
                    infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                    model,
                                                                    opencood_dataset)
                elif opt.fusion_method == 'no_w_uncertainty':
                    infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                    model,
                                                                    opencood_dataset)
                elif opt.fusion_method == 'single':
                    infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                    model,
                                                                    opencood_dataset,
                                                                    single_gt=True)
                else:
                    raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                            'fusion is supported.')

                pred_box_tensor = infer_result['pred_box_tensor']
                gt_box_tensor = infer_result['gt_box_tensor']
                pred_score = infer_result['pred_score']
                
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat_modality,
                                        0.3)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat_modality,
                                        0.5)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat_modality,
                                        0.7)
                if opt.save_npy:
                    npy_save_path = os.path.join(opt.model_dir, 'npy')
                    if not os.path.exists(npy_save_path):
                        os.makedirs(npy_save_path)
                    inference_utils.save_prediction_gt(pred_box_tensor,
                                                    gt_box_tensor,
                                                    batch_data['ego'][
                                                        'origin_lidar'][0],
                                                    i,
                                                    npy_save_path)

                if not opt.no_score:
                    infer_result.update({'score_tensor': pred_score})

                if getattr(opencood_dataset, "heterogeneous", False):
                    cav_box_np, agent_modality_list = inference_utils.get_cav_box(batch_data)
                    infer_result.update({"cav_box_np": cav_box_np, \
                                        "agent_modality_list": agent_modality_list})

                if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None or gt_box_tensor is not None):
                    vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info_all}')
                    if not os.path.exists(vis_save_path_root):
                        os.makedirs(vis_save_path_root)

                    # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d_%s.png' % (i, modality))
                    # simple_vis.visualize(infer_result,
                    #                     batch_data['ego'][
                    #                         'origin_lidar'][0],
                    #                     hypes['postprocess']['gt_range'],
                    #                     vis_save_path,
                    #                     method='3d',
                    #                     left_hand=left_hand)
                    
                    vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d_%s.png' % (i, modality))
                    simple_vis.visualize(infer_result,
                                        batch_data['ego'][
                                            'origin_lidar'][0],
                                        hypes['postprocess']['gt_range'],
                                        vis_save_path,
                                        method='bev',
                                        left_hand=left_hand)
            torch.cuda.empty_cache()

        _, ap50, ap70 = eval_utils.eval_final_results(result_stat_modality,
                                    opt.model_dir, infer_info_modality)

        # merge result_stat_modality to result_stat_all
        # {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
        # 0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
        # 0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
        for iou_thres, stats in result_stat_modality.items():
            for k, v in stats.items():
                if k != 'gt':
                    result_stat_all[iou_thres][k].extend(v) 
                else:
                    result_stat_all[iou_thres][k] += v

    _, ap50, ap70 = eval_utils.eval_final_results(result_stat_all,
                                opt.model_dir, infer_info_all)


if __name__ == '__main__':
    main()
