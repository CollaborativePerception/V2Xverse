import os


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
    # planning_model = {
    #     'name': 'cop3_planner',
    #     # 'path': '/GPFS/public/InterFuser/results/planner/20230426-215325-cop3_planner-224-drivable_area_downsample_2_try1/last.pth.tar'
    #     'path': '/GPFS/public/InterFuser/results/planner/20230426-225154-cop3_planner-224-drivable_area_downsample_4_try1/last.pth.tar'
    # }

    def __init__(self, **kwargs):  
        for k, v in kwargs.items():
            setattr(self, k, v)