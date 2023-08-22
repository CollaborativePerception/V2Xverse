import os

import argparse
import opencood.hypes_yaml.yaml_utils as yaml_utils
import importlib

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=False, default='/GPFS/data/gjliu/Auto-driving/OpenCOODv2/opencood/logs/v2xverse_centerpoint_where2comm_withshrinkhead_multiclass_none_fusion_2023_07_13_23_36_58',
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
    opt = parser.parse_args()
    return opt

class GlobalConfig:
    """base architecture configurations"""

    # Controller
    turn_KP = 1.0
    turn_KI = 0.2
    turn_KD = 0.1
    turn_n = 5  # buffer size

    speed_KP = 5.0
    speed_KI = 1
    speed_KD = 0.1
    speed_n = 20  # buffer size

    max_throttle = 0.75  # upper limit on throttle signal value in dataset
    brake_speed = 0.1  # desired speed below which brake is triggered
    brake_ratio = 1.1  # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.35  # maximum change in speed input to logitudinal controller

    max_speed = 5
    collision_buffer = [2.5, 1.2]
    momentum = 0
    skip_frames = 4
    detect_threshold = 0.04
    target_point_distance = 10

    ego_num = 1

    fusion_mode = 'inter'  # ['none', 'early', 'inter', 'late', 'cheat']
    if fusion_mode=='inter':
        perception_model = {
            'name': 'cop3_point_pillar_multiclass',
            'path': '/GPFS/public/InterFuser/results_close/checkpoints/detection_epoch_8.pth.tar'
            # 'path': '/GPFS/public/InterFuser/results/cop3/pointpillar/multiclass_inter/checkpoint-9.pth.tar'
        }
    elif fusion_mode=='none':
        perception_model = {
            'name': 'cop3_point_pillar_multiclass',
            'path': '/GPFS/public/InterFuser/results/planner/20230418-154655-cop3_planner-224-nan_debug/checkpoint-18.pth.tar'
        }
    elif fusion_mode=='cheat':
        perception_model = {
            'name': 'cop3_point_pillar_multiclass',
            'path': '/GPFS/public/InterFuser/results/cop3/pointpillar/none/20230418-184051-cop3_point_pillar_multiclass-224-multiclass_try1/checkpoint-28.pth.tar'
        }
        fusion_mode = 'none'

    # planner model with drivable area.
    planning_model = {
        'name': 'cop3_planner',
        'path': '/GPFS/public/InterFuser/results/planner/20230421-121022-cop3_planner-224-drivable_area_debug/last.pth.tar'
    }


    def __init__(self, **kwargs):  
        for k, v in kwargs.items():
            setattr(self, k, v)
