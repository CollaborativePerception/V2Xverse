# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Genjia Liu <LGJ1zed@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time
from typing import OrderedDict
import importlib
import torch
import open3d as o3d
from torch.utils.data import DataLoader
import numpy as np

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis_multiclass
from opencood.utils.common_utils import update_dict
torch.multiprocessing.set_sharing_strategy('file_system')

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--also_laplace', action='store_true',
                        help="whether to use laplace to simulate noise. Otherwise Gaussian")
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=200,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')    
    parser.add_argument('--range', type=str, default="140.8,40",
                        help="detection range is [-140.8,+140.8m, -40m, +40m]")
    parser.add_argument('--modal', type=int, default=0,
                        help='used in heterogeneous setting, 0 lidaronly, 1 camonly, 2 ego_lidar_other_cam, 3 ego_cam_other_lidar')
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single']

    hypes = yaml_utils.load_yaml(None, opt)
    
    if 'heter' in hypes:
        if opt.modal == 0:
            hypes['heter']['lidar_ratio'] = 1
            hypes['heter']['ego_modality'] = 'lidar'
            opt.note += '_lidaronly' 

        if opt.modal == 1:
            hypes['heter']['lidar_ratio'] = 0
            hypes['heter']['ego_modality'] = 'camera'
            opt.note += '_camonly' 
            
        if opt.modal == 2:
            hypes['heter']['lidar_ratio'] = 0
            hypes['heter']['ego_modality'] = 'lidar'
            opt.note += 'ego_lidar_other_cam'

        if opt.modal == 3:
            hypes['heter']['lidar_ratio'] = 1
            hypes['heter']['ego_modality'] = 'camera'
            opt.note += '_ego_cam_other_lidar'

        x_min, x_max = -140.8, 140.8
        y_min, y_max = -40, 40
        opt.note += f"_{x_max}_{y_max}"
        hypes['fusion']['args']['grid_conf']['xbound'] = [x_min, x_max, hypes['fusion']['args']['grid_conf']['xbound'][2]]
        hypes['fusion']['args']['grid_conf']['ybound'] = [y_min, y_max, hypes['fusion']['args']['grid_conf']['ybound'][2]]
        hypes['model']['args']['grid_conf'] = hypes['fusion']['args']['grid_conf']

        new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2], \
                            x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]
        
        hypes['preprocess']['cav_lidar_range'] =  new_cav_range
        hypes['postprocess']['anchor_args']['cav_lidar_range'] = new_cav_range
        hypes['postprocess']['gt_range'] = new_cav_range
        hypes['model']['args']['lidar_args']['lidar_range'] = new_cav_range
        if 'camera_mask_args' in hypes['model']['args']:
            hypes['model']['args']['camera_mask_args']['cav_lidar_range'] = new_cav_range

        # reload anchor
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                parser_func = func
        hypes = parser_func(hypes)


    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    left_hand = True if ("OPV2V" in hypes['test_dir'] or 'V2XSET' in hypes['test_dir']) else False
    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # add noise to pose.
    pos_std_list = [0, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6]
    rot_std_list = [0, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6]
    pos_mean_list = [0, 0, 0, 0, 0, 0, 0]
    rot_mean_list = [0, 0, 0, 0, 0, 0, 0]

    
    if opt.also_laplace:
        use_laplace_options = [False, True]
    else:
        use_laplace_options = [False]

    file_path = os.path.join(opt.model_dir,"AP_noise.txt")
    log_file = open(file_path, 'w')
    print(f'Text has been saved to {file_path}')

    for use_laplace in use_laplace_options:
        
        AP_all = {}
        for (pos_mean, pos_std, rot_mean, rot_std) in zip(pos_mean_list, pos_std_list, rot_mean_list, rot_std_list):
            # setting noise
            np.random.seed(30330)
            torch.manual_seed(10000)
            noise_setting = OrderedDict()
            noise_args = {'pos_std': pos_std,
                          'rot_std': rot_std,
                          'pos_mean': pos_mean,
                          'rot_mean': rot_mean}

            noise_setting['add_noise'] = True
            noise_setting['args'] = noise_args

            suffix = "_noise"
            if use_laplace:
                noise_setting['args']['laplace'] = True
                suffix = "_laplace"

            # build dataset for each noise setting
            print('Dataset Building')
            print(f"Noise Added: {pos_std}/{rot_std}/{pos_mean}/{rot_mean}.")
            hypes.update({"noise_setting": noise_setting})
            opencood_dataset = build_dataset(hypes, visualize=True, train=False)
            data_loader = DataLoader(opencood_dataset,
                                    batch_size=1,
                                    num_workers=4,
                                    collate_fn=opencood_dataset.collate_batch_test,
                                    shuffle=False,
                                    pin_memory=False,
                                    drop_last=False)
            
            # Create the dictionary for evaluation
            result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                           0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                           0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
            if hypes['model']['args']['multi_class']:
                result_stat = {}
                class_list = [0,1,3]
                for c in class_list:
                    result_stat[c] = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                                    0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                                    0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    
            noise_level = f"{pos_std}_{rot_std}_{pos_mean}_{rot_mean}_" + opt.fusion_method + suffix + opt.note


            for i, batch_data in enumerate(data_loader):

                try:
                    print(f"{noise_level}_{i}")
                    if batch_data is None:
                        continue
                    with torch.no_grad():
                        batch_data = train_utils.to_device(batch_data, device)
                        
                        if opt.fusion_method == 'late':
                            if 'multiclass' in hypes['fusion']['core_method']:
                                infer_result = inference_utils.inference_late_fusion_multiclass(batch_data,
                                                                            model,
                                                                            opencood_dataset)
                            else:
                                infer_result = inference_utils.inference_late_fusion(batch_data,
                                                                    model,
                                                                    opencood_dataset)
                        elif opt.fusion_method == 'early':
                            if 'multiclass' in hypes['fusion']['core_method']:
                                infer_result = inference_utils.inference_early_fusion_multiclass(batch_data,
                                                                model,
                                                                opencood_dataset)
                            else:
                                infer_result = inference_utils.inference_early_fusion(batch_data,
                                                                    model,
                                                                    opencood_dataset)
                        elif opt.fusion_method == 'intermediate':
                            if 'multiclass' in hypes['fusion']['core_method']:
                                infer_result = inference_utils.inference_intermediate_fusion_multiclass(batch_data,
                                                                            model,
                                                                            opencood_dataset)
                            else:
                                infer_result = inference_utils.inference_intermediate_fusion(batch_data,
                                                                            model,
                                                                            opencood_dataset)
                        elif opt.fusion_method == 'no':
                            if 'multiclass' in hypes['fusion']['core_method']:
                                infer_result = inference_utils.inference_no_fusion_multiclass(batch_data,
                                                                            model,
                                                                            opencood_dataset,
                                                                            single_gt=True)
                            else:
                                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                            model,
                                                                            opencood_dataset)
                        elif opt.fusion_method == 'no_w_uncertainty':
                            infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                            model,
                                                                            opencood_dataset)
                        elif opt.fusion_method == 'single':
                            if 'multiclass' in hypes['fusion']['core_method']:
                                infer_result = inference_utils.inference_no_fusion_multiclass(batch_data,
                                                                            model,
                                                                            opencood_dataset,
                                                                            single_gt=True)
                            else:
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


                        eval_utils.caluclate_tp_fp_multiclass(pred_box_tensor,
                                                pred_score,
                                                gt_box_tensor,
                                                result_stat,
                                                0.3)
                        eval_utils.caluclate_tp_fp_multiclass(pred_box_tensor,
                                                pred_score,
                                                gt_box_tensor,
                                                result_stat,
                                                0.5)
                        eval_utils.caluclate_tp_fp_multiclass(pred_box_tensor,
                                                pred_score,
                                                gt_box_tensor,
                                                result_stat,
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
                            cav_box_np, lidar_agent_record = inference_utils.get_cav_box(batch_data)
                            infer_result.update({"cav_box_np": cav_box_np, \
                                            "lidar_agent_record": lidar_agent_record})
                        
                        if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None or gt_box_tensor is not None) and (use_laplace is False):
                            vis_save_path_root = os.path.join(opt.model_dir, f'vis_{noise_level}')
                            if not os.path.exists(vis_save_path_root):
                                os.makedirs(vis_save_path_root)

                            """ If you want to 3d vis, uncomment lines below """
                            # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                            # simple_vis_multiclass.visualize(infer_result,
                            #                     batch_data['ego'][
                            #                         'origin_lidar'][0],
                            #                     hypes['postprocess']['gt_range'],
                            #                     vis_save_path,
                            #                     method='3d',
                            #                     left_hand=left_hand)
                            
                            vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                            simple_vis_multiclass.visualize(infer_result,
                                                batch_data['ego'][
                                                    'origin_lidar'][0],
                                                hypes['postprocess']['gt_range'],
                                                vis_save_path,
                                                method='bev',
                                                left_hand=left_hand)

                    torch.cuda.empty_cache()
                except:
                    print('skip')
                    raise

            all_class_results, ap30, ap50, ap70 = eval_utils.eval_final_results_multiclass(result_stat,
                                        opt.model_dir, noise_level)
            for tpe in all_class_results:
                if not tpe in AP_all:
                    AP_all[tpe] = {'ap30': [] ,'ap50': [], 'ap70': []}
                AP_all[tpe]['ap30'].append(all_class_results[tpe]['ap30'])
                AP_all[tpe]['ap50'].append(all_class_results[tpe]['ap50'])
                AP_all[tpe]['ap70'].append(all_class_results[tpe]['ap70'])
            yaml_utils.save_yaml(AP_all, os.path.join(opt.model_dir, f'AP030507{suffix}.yaml'))

            log_file.write('veh_ap30: {} veh_ap50: {} veh_ap70: {} '\
                            'ped_ap30: {} ped_ap50: {} ped_ap70: {} '\
                            'bicy_ap30: {} bicy_ap50: {} bicy_ap70: {} '\
                            'pos_std: {} rot_std: {}\n'.format(all_class_results[0]['ap30'],all_class_results[0]['ap50'],all_class_results[0]['ap70'], \
                                                                all_class_results[1]['ap30'],all_class_results[1]['ap50'],all_class_results[1]['ap70'], \
                                                                all_class_results[3]['ap30'],all_class_results[3]['ap50'],all_class_results[3]['ap70'], \
                                                                pos_std, rot_std))
            log_file.close()
            log_file = open(file_path, 'a')

        

if __name__ == '__main__':
    main()
