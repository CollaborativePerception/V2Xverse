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

import warnings
warnings.filterwarnings('ignore')

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
        default="./out_dir/",
        type=str,
        help="directory to output model/log/etc.",
        )
    parser.add_argument(
        "--log-filename",
        default="log.txt",
        type=str,
        help="log filename",
        )
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    log_file_name = opt.log_filename.split('.')[0]+'_'+opt.planner_resume.split('.')[0].split('_')[-1]+'.txt'
    initialize_root_logger(path=f'{opt.out_dir}/{log_file_name}')
    logging.info(f'Using perception checkpoint: {opt.model_dir}')

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 

    hypes = yaml_utils.load_yaml(None, opt)
    hypes['validate_dir'] = hypes['test_dir']
    
    print('Creating Model')
    perception_model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, perception_model = train_utils.load_saved_model(saved_path, perception_model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    
    if torch.cuda.is_available():
        perception_model.cuda()
    perception_model.eval()
    
    infer_info = opt.fusion_method + opt.note

    # load planning dataset
    config = load_config_from_yaml(opt.config_file)
    data_config = config['data']
    test_data_config = data_config['test']
    # test_data_config['dataset']['perception_hypes'] = hypes
    # test_dataloader = build_dataloader(test_data_config, is_distributed=False)

    # build planner NN model
    model_config = config['model']
    planning_model = build_object_within_registry_from_config(
        CODRIVING_REGISTRY,
        model_config,
    )
    model_decoration_config = config['model_decoration']
    decorate_model(planning_model, **model_decoration_config)
    planning_model.to(device)
    last_epoch_idx = load_checkpoint(opt.planner_resume, device, planning_model, strict=True)

    # planner metric
    metric_config = config['test_metric']
    metric_func = build_object_within_registry_from_config(
        CODRIVING_REGISTRY,
        metric_config,
    )

    planning_model.eval()


    logging.info('Testing...')

    # add noise to pose.
    pos_std_list = [0, 0.2, 0.4, 0.6]
    rot_std_list = [0, 0.2, 0.4, 0.6]
    pos_mean_list = [0, 0, 0, 0]
    rot_mean_list = [0, 0, 0, 0]

    use_laplace_options = [False]
    file_path = os.path.join(opt.model_dir,"ADE_FDE_e2e_noise.txt")
    log_file = open(file_path, 'w')
    print(f'Text has been saved to {file_path}')    
    for use_laplace in use_laplace_options:
        
        for (pos_mean, pos_std, rot_mean, rot_std) in zip(pos_mean_list, pos_std_list, rot_mean_list, rot_std_list):
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
            noise_level = f"{pos_std}_{rot_std}_{pos_mean}_{rot_mean}_" + opt.fusion_method + suffix + opt.note

            print('Dataset Building')
            print(f"Noise Added: {pos_std}/{rot_std}/{pos_mean}/{rot_mean}.")
            hypes.update({"noise_setting": noise_setting})

            test_data_config['dataset']['perception_hypes'] = hypes
            test_dataloader = build_dataloader(test_data_config, is_distributed=False)

            ADEs = list()
            FDEs = list()    

            for i, batch_data in enumerate(test_dataloader):
                try:
                    with torch.no_grad():
                        print(f"{noise_level}_{i}")

                        pred_batch_data, perce_batch_data_dict = batch_data
                        move_dict_data_to_device(pred_batch_data, device)

                        pred_batch_data.update({'fused_feature':[],
                                                'features_before_fusion':[],})

                        ############ before prediction ##########
                        # perception model inference
                        frame_list = list(perce_batch_data_dict.keys())
                        frame_list.sort()
                        perception_results_list = []
                        occ_map_list = []

                        for frame in frame_list:
                            perce_batch_data_dict[frame] = train_utils.to_device(perce_batch_data_dict[frame], device)

                            # perception_results = perception_model(perce_batch_data_dict[frame]['ego'])

                            output_dict = OrderedDict()
                            for cav_id, cav_content in perce_batch_data_dict[frame].items():
                                output_dict[cav_id] = perception_model(cav_content)
                            pred_box_tensor, pred_score, gt_box_tensor = \
                                test_dataloader.dataset.perception_dataset.post_process_multiclass(perce_batch_data_dict[frame],
                                                    output_dict, online_eval_only=True)
                            infer_result = {"pred_box_tensor" : pred_box_tensor, \
                                            "pred_score" : pred_score, \
                                            "gt_box_tensor" : gt_box_tensor}
                            if "comm_rate" in output_dict['ego']:
                                infer_result.update({"comm_rate" : output_dict['ego']['comm_rate']})
                            #################### filte ego car ###########################
                            if infer_result['pred_box_tensor'][0] is not None:
                                box_filte_ego_car_list = []
                                score_filte_ego_car_list = []
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

                            occ_map_list.append(box2occ(infer_result))

                            perception_results = output_dict['ego']
                            fused_feature_2 = perception_results['fused_feature'].permute(0,1,3,2)
                            fused_feature_3 = torch.flip(fused_feature_2, dims=[2])
                            w = fused_feature_2.shape[3]//2
                            pred_batch_data['fused_feature'].append(fused_feature_3[:,:,:192,w-48:w+48])

                            perception_results_list.append(perception_results)


                        
                        # warp feature in time sequence
                        pred_batch_data['feature_warpped_list'] = []

                        for b in range(len(perception_results_list[0]['fused_feature'])):
                            feature_dim = perception_results_list[0]['fused_feature'].shape[1] # 128,256
                            feature_to_warp = torch.zeros(1, 5, feature_dim, 192, 96).to(device).float()
                            det_map_pose = torch.zeros(1, 5, 3).to(device).float()
                            occ_to_warp = torch.zeros(1, 5, 1, 192, 96).cuda().float()


                            for t in range(5):
                                feature_to_warp[0, t, :] = pred_batch_data['fused_feature'][t][b] # occ_map_list[t]
                                det_map_pose[:, t] = torch.tensor(pred_batch_data['detmap_pose'][b,t]) # N, 3
                                occ_to_warp[0, t, 0:1] = occ_map_list[t]
                            
                            feature_warped = warp_image(det_map_pose, feature_to_warp)
                            pred_batch_data['feature_warpped_list'].append(feature_warped)

                            occ_warped = warp_image(det_map_pose, occ_to_warp)
                            extra_source = {'occ':occ_warped}

                            pred_batch_data['occupancy'][:,:,0,:,:] = occ_warped[:,:,0,:,:]
                        ##########################################

                        model_output = planning_model(pred_batch_data)

                        ADE, FDE = metric_func(pred_batch_data, model_output)
                        ADEs.append(ADE)
                        FDEs.append(FDE)

                        torch.cuda.empty_cache()
                except:
                    logging.info('test error batch, skip!')

            ADE_tensor = torch.cat(ADEs, dim=0)
            FDE_tensor = torch.cat(FDEs, dim=0)

            log_file.write('ADE: {} FDE: {} pos_std: {} rot_std: {}\n'.format(torch.mean(ADE_tensor, dim=0),torch.mean(FDE_tensor, dim=0), pos_std, rot_std))
            log_file.close()
            log_file = open(file_path, 'a')

            logging.info(f'average ADE: {torch.mean(ADE_tensor, dim=0)}, average FDE: {torch.mean(FDE_tensor, dim=0)}')

if __name__ == '__main__':
    main()
