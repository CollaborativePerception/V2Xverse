import os


class GlobalConfig:
    """base architecture configurations"""

    # Controller
    turn_KP = 1.25
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 40  # buffer size

    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40  # buffer size

    max_throttle = 1.25  # upper limit on throttle signal value in dataset (0.75)
    brake_speed = 0.1  # desired speed below which brake is triggered
    brake_ratio = 1.1  # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.35  # maximum change in speed input to logitudinal controller

    max_speed = 12
    collision_buffer = [2.5, 1.2]

    visible_mode = 'all' # 'single+rsu'  'all' 'single'
    det_range = [18,2,10,10,1]

    fusion_mode = 'cheat'  # ['none', 'early', 'inter', 'late', 'cheat']
    if fusion_mode=='inter':
        model = "interfuser_cop3"
        model_path = "/GPFS/public/InterFuser/results/cop3/20230320-180918-interfuser_cop3_inter/model_best.pth.tar"
    elif fusion_mode=='none':
        model = "interfuser_baseline"
        model_path = "/GPFS/data/gjliu/Auto-driving/Cop3/leaderboard/team_code/interfuser.pth.tar"
    elif fusion_mode=='cheat':
        model = "interfuser_baseline"
        model_path = "/GPFS/data/gjliu/Auto-driving/Cop3/leaderboard/team_code/interfuser.pth.tar"
    momentum = 0
    skip_frames = 4
    detect_threshold = 0.04

    ego_num = 1

    def __init__(self, **kwargs):  
        for k, v in kwargs.items():
            setattr(self, k, v)
