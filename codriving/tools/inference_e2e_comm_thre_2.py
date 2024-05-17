# -*- coding: utf-8 -*-
# Author: Genjia Liu <lgj1zed@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time
from typing import OrderedDict
import importlib
import logging
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis_multiclass
from opencood.utils.occ_render import box2occ
torch.multiprocessing.set_sharing_strategy('file_system')

import warnings
warnings.filterwarnings('ignore')

import copy

from common.io import load_config_from_yaml
from codriving.utils.torch_helper import \
    move_dict_data_to_device, build_dataloader
from codriving import CODRIVING_REGISTRY
from codriving.models.model_decoration import decorate_model
from common.registry import build_object_within_registry_from_config
from common.detection import warp_image
from common.torch_helper import load_checkpoint
from codriving.utils import initialize_root_logger

logger = logging.getLogger("test")

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=False, default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
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
    parser.add_argument('--skip_frames', default=1, type=int, help="frame gap in planning data")
    parser.add_argument(
        "-c",
        "--config-file",
        default="",
        type=str,
        metavar="FILE",
        help="Config file for training",
    )
    parser.add_argument(
        "--planner_resume",
        default="",
        type=str,
        help="Path of the checkpoint from which the training resumes",
        )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="#workers for dataloader (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=73,
        help="Random seed",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Log interval",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        type=str,
        help="directory to output model/log/etc.",
        )
    parser.add_argument(
        "--log-filename",
        default="log.txt",
        type=str,
        help="log filename",
        )
    parser.add_argument(
        "--using_driving_request",
        default=1,
        type=int,
        help="whether use driving request map",
        )
    parser.add_argument(
        "--thre",
        default=None,
        type=float,
        help="confidence thre",
        )
    parser.add_argument(
        "--radius",
        default=160,
        type=float,
        help="draw gaussian radius",
        )
    parser.add_argument(
        "--sigma_reverse",
        default=2,
        type=float,
        help="draw gaussian sigma reverse",
        )
    parser.add_argument(
        "--k_size",
        default=161,
        type=float,
        help="confidence gaussian radius",
        )
    parser.add_argument(
        "--c_sigma",
        default=5,
        type=float,
        help="confidence gaussian sigma reverse",
        )
    opt = parser.parse_args()
    return opt

def inference_single_from_intermediate_fusion_multiclass(batch_data, model, dataset, online_eval_only=False):
    output_dict = OrderedDict()
    # cav_content = batch_data['ego']
    # output_dict['ego'] = model(cav_content)

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)
        output_dict[cav_id]['cls_preds'] = output_dict[cav_id]['cls_preds_single_ego']
        output_dict[cav_id]['reg_preds_multiclass'] = output_dict[cav_id]['reg_preds_multiclass_single_ego']
        output_dict[cav_id]['bbox_preds'] = output_dict[cav_id]['bbox_preds_single_ego']

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process_multiclass(batch_data,
                             output_dict, online_eval_only)

    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    if "depth_items" in output_dict['ego']:
        return_dict.update({"depth_items" : output_dict['ego']['depth_items']})

    return return_dict, output_dict

def inference_intermediate_fusion_multiclass(batch_data, model, dataset, online_eval_only=False):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    # cav_content = batch_data['ego']
    # output_dict['ego'] = model(cav_content)

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)
    
    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process_multiclass(batch_data,
                             output_dict, online_eval_only)
    
    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    if "depth_items" in output_dict['ego']:
        return_dict.update({"depth_items" : output_dict['ego']['depth_items']})
    if "comm_rate" in output_dict['ego']:
        return_dict.update({"comm_rate" : output_dict['ego']['comm_rate']})
    return return_dict, output_dict

def inference_intermediate_fusion_multiclass_driving_request(batch_data, model, dataset, online_eval_only=False, waypoints=None):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset
    waypoints : points to rend request map

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    # cav_content = batch_data['ego']
    # output_dict['ego'] = model(cav_content)

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content, waypoints)
    
    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process_multiclass(batch_data,
                             output_dict, online_eval_only)
    
    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    if "depth_items" in output_dict['ego']:
        return_dict.update({"depth_items" : output_dict['ego']['depth_items']})
    if "comm_rate" in output_dict['ego']:
        return_dict.update({"comm_rate" : output_dict['ego']['comm_rate']})
    return return_dict, output_dict


def main():
    opt = test_parser()
    print('using_driving_request',opt.using_driving_request)
    initialize_root_logger(path=f'{opt.out_dir}/{opt.log_filename}')
    logging.info(f'Using perception checkpoint: {opt.model_dir}')

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
    
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False
    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    # setting noise
    np.random.seed(30330)
    torch.manual_seed(10000)
    
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build dataset for each noise setting
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    # opencood_dataset_subset = Subset(opencood_dataset, range(640,2100))
    # data_loader = DataLoader(opencood_dataset_subset,
    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False, # False
                            pin_memory=False,
                            drop_last=False)

    if opt.using_driving_request:
        log_flag = '_' + opt.note + "_DS_" + f'C_sigma{opt.c_sigma}_W_sigma{opt.sigma_reverse}'
    else:
        log_flag = '_' + opt.note + f"_S_C_sigma{opt.c_sigma}"

    if opt.thre is not None:
        log_flag += '_{}'.format(opt.thre)

    # ap_file_path = os.path.join(opt.model_dir,"AP_comm_thre{}.txt".format(log_flag))
    # AP_log_file = open(ap_file_path, 'w')

    ade_fde_file_path = os.path.join(opt.model_dir,"ADE_FDE_comm_thre{}.txt".format(log_flag))
    ADE_FDE_log_file = open(ade_fde_file_path, 'w')

    # load planning dataset
    config = load_config_from_yaml(opt.config_file)
    data_config = config['data']
    test_data_config = data_config['test_comm_thre']

    # test_data_config['dataset']['perception_hypes'] = hypes

    dataset_config = test_data_config['dataset']
    planning_dataset = build_object_within_registry_from_config(
        CODRIVING_REGISTRY,
        dataset_config,
    )

    # build planner NN model
    model_config = config['model']
    planning_model = build_object_within_registry_from_config(
        CODRIVING_REGISTRY,
        model_config,
    )
    model_decoration_config = config['model_decoration']
    decorate_model(planning_model, **model_decoration_config)
    planning_model.to(device)
    last_epoch_idx = load_checkpoint(opt.planner_resume, device, planning_model)

    # planner metric
    metric_config = config['test_metric']
    metric_func = build_object_within_registry_from_config(
        CODRIVING_REGISTRY,
        metric_config,
    )

    planning_model.eval()

    ##  different gaussian parameters
    if opt.radius is not None:
         hypes['model']['args']['fusion_args']['communication']['radius'] = opt.radius
    if opt.sigma_reverse is not None:
         hypes['model']['args']['fusion_args']['communication']['sigma_reverse'] = opt.sigma_reverse
    if opt.k_size is not None:
         hypes['model']['args']['fusion_args']['communication']['gaussian_smooth']['k_size'] = opt.k_size
    if opt.c_sigma is not None:
         hypes['model']['args']['fusion_args']['communication']['gaussian_smooth']['c_sigma'] = opt.c_sigma

    if opt.thre is not None:
        comm_thre_list = [opt.thre]
    else:
        comm_thre_list = [0,0.001,0.003,0.006,0.01,0.03,0.10,0.20,0.40,0.60,0.80,1.0]

    comm_thre_list = [0,1.0000001,0.001,0.003,0.006,0.01,0.03,0.10,0.20,0.40,0.60,0.80]

    for comm_thre in comm_thre_list:

        hypes['model']['args']['fusion_args']['communication']['thre'] = comm_thre

        print('Creating Model')
        model = train_utils.create_model(hypes)


        print('Loading Model from checkpoint')
        saved_path = opt.model_dir
        resume_epoch, model = train_utils.load_saved_model(saved_path, model)
        print(f"resume from {resume_epoch} epoch.")

        if opt.using_driving_request:
            note = opt.note + f"_epoch{resume_epoch}_DS_C_sigma{opt.c_sigma}_W_sigma{opt.sigma_reverse}"
        else:
            note = opt.note + f"_epoch{resume_epoch}_S_C_sigma{opt.c_sigma}"
        
        if torch.cuda.is_available():
            model.cuda()
        model.eval()

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
        
        infer_info = 'comm_thre_{}'.format(comm_thre) + note

        AP_all = {}

        ADEs = list()
        FDEs = list()    
        comm_rates = []

        logging.debug('Testing...')

        cur_data_dir = None
        occ_map_list = {}
        feature_list = {}

        for i, batch_data in enumerate(data_loader):
            try:
                # if i >10:
                #     break
                print(f"comm_thre_{comm_thre}_{log_flag}_{i}")
                if batch_data is None:
                    continue
                with torch.no_grad():
                    batch_data = train_utils.to_device(batch_data, device)
                        
                    data_root, frame_id = batch_data['ego']['dict_list'][0]
                    data_dir=(data_root, frame_id)
                    perception_memory_bank = [{}]

                    ref_bank = {}
                    for his_id in range(0, 17*opt.skip_frames, opt.skip_frames):
                        exist_flag = False
                        for ref_root, ref_frame in opencood_dataset.route_frames:
                            if data_root['ego'] == ref_root['ego']:
                                if ref_frame==frame_id+his_id:
                                    ref_bank[frame_id+his_id] = ref_root
                                    exist_flag = True
                                    break
                        if not exist_flag:
                            break

                    test_flag = True
                    if len(ref_bank)==17:
                        for his_id in range(0, 17*opt.skip_frames, opt.skip_frames):
                            # test_flag *= (os.path.exists(os.path.join(data_loader.dataset.root_dir,ref_bank[frame_id+his_id]['ego'],'measurements',"%04d.json" % (frame_id+his_id))))
                            test_flag *= ((ref_bank[frame_id+his_id],frame_id+his_id) in opencood_dataset.route_frames)
                    else:
                        test_flag = False

                    if test_flag:
                        print('pnp test frame')

                        if len(occ_map_list.keys())==0:
                            print('initialize occ_map_list!')
                            occ_map_list[data_root['ego']] = {}    
                            feature_list[data_root['ego']] = {}                        
                        elif data_root['ego'] in occ_map_list:
                            pass
                        elif not data_root['ego'] in occ_map_list:
                            print('change route!')
                            del occ_map_list[list(occ_map_list.keys())[0]]
                            del feature_list[list(feature_list.keys())[0]]
                            occ_map_list[data_root['ego']] = {}  
                            feature_list[data_root['ego']] = {} 
                                                  

                        cur_data_dir = data_root['ego']

                        for his_id in range(0, 5*opt.skip_frames, opt.skip_frames):
                            
                            if frame_id+his_id in occ_map_list[cur_data_dir]:
                                continue

                            # print('update:', frame_id+his_id)

                            data_dir_his=(ref_bank[frame_id+his_id], frame_id+his_id)
                            data_history = opencood_dataset.__getitem__(idx=None, data_dir=data_dir_his, plan_without_perception_gt=False)
                            batch_data_perception = [data_history]
                            batch_data_perception = opencood_dataset.collate_batch_test(batch_data_perception, online_eval_only=False)

                            batch_data_perception = train_utils.to_device(batch_data_perception, device)

                            if his_id==4*opt.skip_frames:
                                batch_data_plan_moment = batch_data_perception

                            if opt.using_driving_request:
                                if len(occ_map_list[cur_data_dir].keys()) >= 5:
                                    infer_result, output_dict = inference_single_from_intermediate_fusion_multiclass(batch_data_perception,
                                                                            model,
                                                                            opencood_dataset)
                                else:
                                    infer_result, output_dict = inference_intermediate_fusion_multiclass(batch_data_perception,
                                                                            model,
                                                                            opencood_dataset)        
                            else:
                                infer_result, output_dict = inference_intermediate_fusion_multiclass(batch_data_perception,
                                                                        model,
                                                                        opencood_dataset)
                                if "comm_rate" in infer_result:
                                    comm_rates.append(infer_result["comm_rate"].cpu().numpy())

                                pred_box_tensor = infer_result['pred_box_tensor']  # torch.Size([21, 8, 3])
                                gt_box_tensor = infer_result['gt_box_tensor']  # torch.Size([20, 8, 3])
                                pred_score = infer_result['pred_score'] # torch.Size([21]

                                # if his_id==4*opt.skip_frames:
                                #     eval_utils.caluclate_tp_fp_multiclass(pred_box_tensor,
                                #                                 pred_score,
                                #                                 gt_box_tensor,
                                #                                 result_stat,
                                #                                 0.3)
                                #     eval_utils.caluclate_tp_fp_multiclass(pred_box_tensor,
                                #                                 pred_score,
                                #                                 gt_box_tensor,
                                #                                 result_stat,
                                #                                 0.5)
                                #     eval_utils.caluclate_tp_fp_multiclass(pred_box_tensor,
                                #                                 pred_score,
                                #                                 gt_box_tensor,
                                #                                 result_stat,
                                #                                 0.7)

                            #################### filte ego car ###########################
                            # if infer_result['pred_box_tensor'][0] is not None:
                            box_filte_ego_car_list = []
                            score_filte_ego_car_list = []

                            if infer_result['pred_box_tensor'][0] is not None:
                                num_car = infer_result['pred_box_tensor'][0].shape[0]
                                for car_actor_id in range(num_car):
                                    car_box = copy.deepcopy(infer_result['pred_box_tensor'][0][car_actor_id])
                                    car_box = car_box.cpu().numpy()
                                    car_box[:,0] += 1.3
                                    location_box = np.mean(car_box[:4,:2], 0)
                                    if np.linalg.norm(location_box) < 1.4:
                                        continue
                                    box_filte_ego_car_list.append(infer_result['pred_box_tensor'][0][car_actor_id])
                                    score_filte_ego_car_list.append(infer_result['pred_score'][0][car_actor_id])
                                if len(box_filte_ego_car_list) > 0:
                                    infer_result['pred_box_tensor'][0] = torch.stack(box_filte_ego_car_list, dim=0)
                                else:
                                    infer_result['pred_box_tensor'][0] = None
                                if len(score_filte_ego_car_list) > 0:
                                    infer_result['pred_score'][0] = torch.stack(score_filte_ego_car_list, dim=0)
                                else:
                                    infer_result['pred_score'][0] = None
                            ##############################################


                            ################## visualization check ######################
                            # vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                            # if not os.path.exists(vis_save_path_root):
                            #     os.makedirs(vis_save_path_root)

                            # # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                            # # simple_vis_multiclass.visualize(infer_result,
                            # #                         batch_data['ego'][
                            # #                             'origin_lidar'][0],
                            # #                         hypes['postprocess']['gt_range'],
                            # #                         vis_save_path,
                            # #                         method='3d',
                            # #                         left_hand=left_hand)
                                
                            # vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % (i*10+his_id))
                            # simple_vis_multiclass.visualize(infer_result,
                            #                         batch_data_perception['ego'][
                            #                             'origin_lidar'][0],
                            #                         hypes['postprocess']['gt_range'],
                            #                         vis_save_path,
                            #                         method='bev',
                            #                         left_hand=left_hand)
                            ###############################################################

                            occ_map_list[cur_data_dir][frame_id+his_id] = box2occ(copy.deepcopy(infer_result))
                            if len(occ_map_list[cur_data_dir].keys()) > 5:
                                del occ_map_list[cur_data_dir][frame_id-opt.skip_frames]

                            

                            perception_results = output_dict['ego']
                            fused_feature_2 = perception_results['fused_feature'].permute(0,1,3,2) #  ([2, 128, 96, 288]) -> ([2, 128, 288, 96])
                            fused_feature_3 = torch.flip(fused_feature_2, dims=[2])
                            w = fused_feature_2.shape[3]//2
                            feature_list[cur_data_dir][frame_id+his_id] = fused_feature_3[:,:,:192,w-48:w+48]




                        data_plan = planning_dataset.__getitem__(idx=None, data_dir=data_dir)

                        feature_dim = perception_results['fused_feature'].shape[1] # 128,256
                        feature_to_warp = torch.zeros(1, 5, feature_dim, 192, 96).to(device).float()

                        occ_to_warp = torch.zeros(1, 5, 1, 192, 96).cuda().float()
                        det_map_pose = torch.zeros(1, 5, 3).cuda().float()

                        for t in range(5):
                            feature_to_warp[0, t, :] = feature_list[cur_data_dir][frame_id+t*opt.skip_frames]
                            occ_to_warp[0, t, 0:1] = occ_map_list[cur_data_dir][frame_id+t*opt.skip_frames]
                            det_map_pose[:, t] = torch.tensor(data_plan['detmap_pose'][t]) # N, 3
                        
                        feature_warped = warp_image(det_map_pose, feature_to_warp)
                        occ_warped = warp_image(det_map_pose, occ_to_warp)
                        extra_source = {'occ':occ_warped}
                        batch_data_planning = planning_dataset.collate_fn([data_plan], extra_source=extra_source)

                        batch_data_planning['feature_warpped_list'] = [feature_warped]

                        
                        move_dict_data_to_device(batch_data_planning, device)
                        planner_output = planning_model(batch_data_planning)


                        if opt.using_driving_request:

                            # Round 2, update by request map
                            pred_waypoints = planner_output['future_waypoints'] # torch.Size([1, 10, 2])
                            batch_data_plan_moment = batch_data_plan_moment # frame_id + 4 的数据

                            # update perception evaluation results using predicted waypoints
                            infer_result, output_dict = inference_intermediate_fusion_multiclass_driving_request(batch_data_plan_moment,
                                                                    model,
                                                                    opencood_dataset,
                                                                    waypoints=pred_waypoints)
                            pred_box_tensor = infer_result['pred_box_tensor']  # torch.Size([21, 8, 3])
                            gt_box_tensor = infer_result['gt_box_tensor']  # torch.Size([20, 8, 3])
                            pred_score = infer_result['pred_score'] # torch.Size([21]
                            if "comm_rate" in infer_result:
                                comm_rates.append(infer_result["comm_rate"].cpu().numpy())


                            # eval_utils.caluclate_tp_fp_multiclass(pred_box_tensor,
                            #                             pred_score,
                            #                             gt_box_tensor,
                            #                             result_stat,
                            #                             0.3)
                            # eval_utils.caluclate_tp_fp_multiclass(pred_box_tensor,
                            #                             pred_score,
                            #                             gt_box_tensor,
                            #                             result_stat,
                            #                             0.5)
                            # eval_utils.caluclate_tp_fp_multiclass(pred_box_tensor,
                            #                             pred_score,
                            #                             gt_box_tensor,
                            #                             result_stat,
                            #                             0.7)

                            #################### filte ego car ###########################
                            # if infer_result['pred_box_tensor'][0] is not None:
                            box_filte_ego_car_list = []
                            score_filte_ego_car_list = []

                            if infer_result['pred_box_tensor'][0] is not None:
                                num_car = infer_result['pred_box_tensor'][0].shape[0]
                                for car_actor_id in range(num_car):
                                    car_box = copy.deepcopy(infer_result['pred_box_tensor'][0][car_actor_id])
                                    car_box = car_box.cpu().numpy()
                                    car_box[:,0] += 1.3
                                    location_box = np.mean(car_box[:4,:2], 0)
                                    if np.linalg.norm(location_box) < 1.4:
                                        continue
                                    box_filte_ego_car_list.append(infer_result['pred_box_tensor'][0][car_actor_id])
                                    score_filte_ego_car_list.append(infer_result['pred_score'][0][car_actor_id])
                                if len(box_filte_ego_car_list) > 0:
                                    infer_result['pred_box_tensor'][0] = torch.stack(box_filte_ego_car_list, dim=0)
                                else:
                                    infer_result['pred_box_tensor'][0] = None
                                if len(score_filte_ego_car_list) > 0:
                                    infer_result['pred_score'][0] = torch.stack(score_filte_ego_car_list, dim=0)
                                else:
                                    infer_result['pred_score'][0] = None
                            ##############################################

                            # # if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None):
                            # vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}_comm_thre')
                            # if not os.path.exists(vis_save_path_root):
                            #     os.makedirs(vis_save_path_root)
                                
                            # vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                            # simple_vis_multiclass.visualize(infer_result,
                            #                         batch_data_plan_moment['ego'][
                            #                             'origin_lidar'][0],
                            #                         hypes['postprocess']['gt_range'],
                            #                         vis_save_path,
                            #                         method='bev',
                            #                         left_hand=left_hand)


                            # update the latest item of occ_bank
                            occ_map_list[cur_data_dir][frame_id+4*opt.skip_frames] = box2occ(copy.deepcopy(infer_result))

                            perception_results = output_dict['ego']
                            fused_feature_2 = perception_results['fused_feature'].permute(0,1,3,2) #  ([2, 128, 96, 288]) -> ([2, 128, 288, 96])
                            fused_feature_3 = torch.flip(fused_feature_2, dims=[2])
                            w = fused_feature_2.shape[3]//2
                            feature_list[cur_data_dir][frame_id+4*opt.skip_frames] = fused_feature_3[:,:,:192,w-48:w+48]

                            occ_to_warp = torch.zeros(1, 5, 1, 192, 96).cuda().float()
                            det_map_pose = torch.zeros(1, 5, 3).cuda().float()
                            feature_dim = perception_results['fused_feature'].shape[1] # 128,256
                            feature_to_warp = torch.zeros(1, 5, feature_dim, 192, 96).to(device).float()


                            for t in range(5):
                                feature_to_warp[0, t, :] = feature_list[cur_data_dir][frame_id+t*opt.skip_frames]
                                occ_to_warp[0, t, 0:1] = occ_map_list[cur_data_dir][frame_id+t*opt.skip_frames]
                                det_map_pose[:, t] = torch.tensor(data_plan['detmap_pose'][t]) # N, 3
                            occ_warped_2 = warp_image(det_map_pose, occ_to_warp)
                            feature_warped = warp_image(det_map_pose, feature_to_warp)
                            extra_source_2 = {'occ':occ_warped_2}
                            batch_data_planning = planning_dataset.collate_fn([data_plan], extra_source=extra_source_2)

                            batch_data_planning['feature_warpped_list'] = [feature_warped]

                            move_dict_data_to_device(batch_data_planning, device)
                            planner_output = planning_model(batch_data_planning)

                        ADE, FDE = metric_func(batch_data_planning, planner_output)
                        ADEs.append(ADE)
                        FDEs.append(FDE)
                    
                    else:
                        infer_result = inference_utils.inference_intermediate_fusion_multiclass(batch_data,
                                                                model,
                                                                opencood_dataset)
                        if "comm_rate" in infer_result:
                            comm_rates.append(infer_result["comm_rate"].cpu().numpy())

                        pred_box_tensor = infer_result['pred_box_tensor']  # torch.Size([21, 8, 3])
                        gt_box_tensor = infer_result['gt_box_tensor']  # torch.Size([20, 8, 3])
                        pred_score = infer_result['pred_score'] # torch.Size([21]

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


                #break
                torch.cuda.empty_cache()

            except:
                raise
                print('error! skip')

        if len(comm_rates) > 0:
            comm_rates = sum(comm_rates) / len(comm_rates)
        else:
            comm_rates = 0.0

        # all_class_results, _, _, _ = eval_utils.eval_final_results_multiclass(result_stat,
        #                             opt.model_dir, infer_info)
        # for tpe in all_class_results:
        #     if not tpe in AP_all:
        #         AP_all[tpe] = {'ap30': [] ,'ap50': [], 'ap70': []}
        #     AP_all[tpe]['ap30'].append(all_class_results[tpe]['ap30'])
        #     AP_all[tpe]['ap50'].append(all_class_results[tpe]['ap50'])
        #     AP_all[tpe]['ap70'].append(all_class_results[tpe]['ap70'])
        # yaml_utils.save_yaml(AP_all, os.path.join(opt.model_dir, 'AP030507.yaml'))

        # AP_log_file.write('veh_ap30: {} veh_ap50: {} veh_ap70: {} '\
        #                 'ped_ap30: {} ped_ap50: {} ped_ap70: {} '\
        #                 'bicy_ap30: {} bicy_ap50: {} bicy_ap70: {} '\
        #                         'comm_thre: {} comm_rate: {}\n'.format(all_class_results[0]['ap30'],all_class_results[0]['ap50'],all_class_results[0]['ap70'], \
        #                                                     all_class_results[1]['ap30'],all_class_results[1]['ap50'],all_class_results[1]['ap70'], \
        #                                                     all_class_results[3]['ap30'],all_class_results[3]['ap50'],all_class_results[3]['ap70'], \
        #                                                         comm_thre, comm_rates))
        # AP_log_file.close()
        # AP_log_file = open(ap_file_path, 'a')

        ADE_tensor = torch.cat(ADEs, dim=0)
        FDE_tensor = torch.cat(FDEs, dim=0)

        ADE_FDE_log_file.write('ADE: {} FDE: {} comm_thre: {} comm_rate: {}\n'.format(torch.mean(ADE_tensor, dim=0),torch.mean(FDE_tensor, dim=0), comm_thre, comm_rates))
        ADE_FDE_log_file.close()
        ADE_FDE_log_file = open(ade_fde_file_path, 'a')

        logging.info(f'average ADE: {torch.mean(ADE_tensor, dim=0)}, average FDE: {torch.mean(FDE_tensor, dim=0)}')




if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    main()
