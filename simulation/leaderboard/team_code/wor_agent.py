import copy
import os
import json
import datetime
import pathlib
import time
import imp
import cv2
import carla
from collections import deque

import torch
import carla
import numpy as np
from PIL import Image
from easydict import EasyDict
from team_code.v2x_controller import V2X_Controller
from torchvision import transforms

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track


# from interfuser.timm.models import create_model
# from team_code.utils import lidar_to_histogram_features, transform_2d_points
from team_code.planner import RoutePlanner
from team_code.interfuser_controller import InterfuserController
# from team_code.render import render, render_self_car, render_waypoints
# from team_code.tracker import Tracker
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import math
import yaml


from team_code.wor_models import EgoModel, CameraModel
from team_code.wor_models.waypointer import Waypointer

# from team_code.lav_models.lidar import LiDARModel
# from team_code.lav_models.uniplanner import UniPlanner
# from team_code.lav_models.bev_planner import BEVPlanner
# from team_code.lav_models.rgb import RGBSegmentationModel, RGBBrakePredictionModel

# from team_code.lav_utils.pid import PIDController
# from team_code.lav_utils.ekf import EKF
# from team_code.lav_utils.point_painting import CoordConverter, point_painting
# from team_code.lav_utils.planner import RoutePlanner as RoutePlanner_lav
# from team_code.lav_utils.waypointer import Waypointer
# from team_code.lav_utils.model_inference import InferModel

try:
    import pygame
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")


SAVE_PATH = os.environ.get("SAVE_PATH", 'eval')
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
CAMERA_YAWS = [-60,0,60]
NUM_REPEAT = 4
GAP = NUM_REPEAT + 1
FPS = 20.
PIXELS_PER_METER = 4

os.environ["SDL_VIDEODRIVER"] = "dummy"

class DisplayInterface(object):
    def __init__(self):
        self._width = 256*3
        self._height = 288
        self._surface = None

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode(
            (self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("Human Agent")

    def run_interface(self, input_data):
        rgbs = input_data['rgbs']
        # print(rgbs[0].shape)
        # 288, 256, 3
        # raise ValueError

        surface = np.zeros((288, 256*3, 3),np.uint8)

        surface[:, :256] = rgbs[0]
        surface[:, 256:256*2] = rgbs[1]
        surface[:, 256*2:256*3] = rgbs[2]

        surface[:, 254:258] = 0
        surface[:, 510:514] = 0
        
        
        # surface[:150,198:202]=0
        # surface[:150,323:327]=0
        # surface[:150,473:477]=0
        # surface[:150,598:602]=0
        # surface[148:152, :200] = 0
        # surface[148:152, 325:475] = 0
        # surface[148:152, 600:800] = 0
        # surface[430:600, 998:1000] = 255
        # surface[0:600, 798:800] = 255
        # surface[0:600, 1198:1200] = 255
        # surface[0:2, 800:1200] = 255
        # surface[598:600, 800:1200] = 255
        # surface[398:400, 800:1200] = 255


        # display image
        self._surface = pygame.surfarray.make_surface(surface.swapaxes(0, 1))
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))

        pygame.display.flip()
        pygame.event.get()
        return surface

    def _quit(self):
        pygame.quit()


def get_entry_point():
    return "ImageAgent"


class Resize2FixedSize:
    def __init__(self, size):
        self.size = size

    def __call__(self, pil_img):
        pil_img = pil_img.resize(self.size)
        return pil_img


def create_carla_rgb_transform(
    input_size, need_scale=True, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
):

    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size
    tfl = []

    if isinstance(input_size, (tuple, list)):
        input_size_num = input_size[-1]
    else:
        input_size_num = input_size

    if need_scale:
        if input_size_num == 112:
            tfl.append(Resize2FixedSize((170, 128)))
        elif input_size_num == 128:
            tfl.append(Resize2FixedSize((195, 146)))
        elif input_size_num == 224:
            tfl.append(Resize2FixedSize((341, 256)))
        elif input_size_num == 256:
            tfl.append(Resize2FixedSize((288, 288)))
        else:
            raise ValueError("Can't find proper crop size")
    tfl.append(transforms.CenterCrop(img_size))
    tfl.append(transforms.ToTensor())
    tfl.append(transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))

    return transforms.Compose(tfl)


class ImageAgent(AutonomousAgent):
    
    """
    Trained image agent
    """
    
    def setup(self, path_to_conf_file, ego_vehicles_num=1, max_speed=10, threshold=0):
        """
        Setup the agent parameters
        """

        self.agent_name='WOR'

        self.lane_changed = None
        self.prev_control = dict()
        self.skip_frames = 4
        self.ego_vehicles_num = ego_vehicles_num

        self.track = Track.SENSORS
        self.num_frames = 0

        with open(path_to_conf_file, 'r') as f:
            config = yaml.safe_load(f)
        self.config = config

        for key, value in config.items():
            setattr(self, key, value)

        self.device = torch.device('cuda')

        self.image_model = CameraModel(config).to(self.device)
        # print(self.main_model_dir)
        # print(torch.load(self.main_model_dir))
        self.image_model.load_state_dict(torch.load(self.main_model_dir))
        self.image_model.eval()

        # self.vizs = []

        self.waypointer = None
            
        self.steers = torch.tensor(np.linspace(-self.max_steers,self.max_steers,self.num_steers)).float().to(self.device)
        self.throts = torch.tensor(np.linspace(0,self.max_throts,self.num_throts)).float().to(self.device)

        self.prev_steer = 0
        self.lane_change_counter = 0
        self.stop_counter = 0

        self.initialized = False

        self.save_path = None
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ["ROUTES"]).stem + "_"
            string += "_".join(
                map(
                    lambda x: "%02d" % x,
                    (now.month, now.day, now.hour, now.minute, now.second),
                )
            )

            print(string)

            self.save_path = pathlib.Path(SAVE_PATH) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / "meta").mkdir(parents=True, exist_ok=False)


    def destroy(self):
        # if len(self.vizs) == 0:
        #     return

        # self.flush_data()
        self.prev_steer = 0
        self.lane_change_counter = 0
        self.stop_counter = 0
        self.lane_changed = None
        
        del self.waypointer
        del self.image_model
    
    # def flush_data(self):
            
    #     self.vizs.clear()

    def sensors(self):
        sensors = [
            # {'type': 'sensor.collision', 'id': 'COLLISION'},
            {'type': 'sensor.speedometer', 'id': 'EGO'},
            {'type': 'sensor.other.gnss', 'x': 0., 'y': 0.0, 'z': self.camera_z, 'id': 'GPS'},
            {'type': 'sensor.camera.rgb', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 160, 'height': 240, 'fov': 60, 'id': f'Wide_RGB'},
            {'type': 'sensor.camera.rgb', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 384, 'height': 240, 'fov': 50, 'id': f'Narrow_RGB'},
        ]
        
        return sensors


    def _get_position(self, gps):
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps

    def _init(self):
        self._route_planner = RoutePlanner(4.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True


    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        if self.num_frames % self.skip_frames != 0 and self.num_frames > 4:
            self.num_frames += 1
            return self.prev_control
        control_all = []
        for ego_id in range(self.ego_vehicles_num):
            _, wide_rgb = input_data.get(f'Wide_RGB_{ego_id}')
            _, narr_rgb = input_data.get(f'Narrow_RGB_{ego_id}')

            # Crop images
            _wide_rgb = wide_rgb[self.wide_crop_top:,:,:3]
            _narr_rgb = narr_rgb[:-self.narr_crop_bottom,:,:3]

            _wide_rgb = _wide_rgb[...,::-1].copy()
            _narr_rgb = _narr_rgb[...,::-1].copy()

            _, ego = input_data.get(f'EGO_{ego_id}')
            _, gps = input_data.get(f'GPS_{ego_id}')

            # print(gps, gps.shape)

            pos = self._get_position(gps[:2])


            if self.waypointer is None:
                self.waypointer = Waypointer(self._global_plan[ego_id], gps)

            _, _, cmd = self.waypointer.tick(gps)

            spd = ego.get('speed')
            
            cmd_value = cmd.value-1
            cmd_value = 3 if cmd_value < 0 else cmd_value

            if cmd_value in [4,5]:
                if self.lane_changed is not None and cmd_value != self.lane_changed:
                    self.lane_change_counter = 0

                self.lane_change_counter += 1
                self.lane_changed = cmd_value if self.lane_change_counter > {4:200,5:200}.get(cmd_value) else None
            else:
                self.lane_change_counter = 0
                self.lane_changed = None

            if cmd_value == self.lane_changed:
                cmd_value = 3

            _wide_rgb = torch.tensor(_wide_rgb[None]).float().permute(0,3,1,2).to(self.device)
            _narr_rgb = torch.tensor(_narr_rgb[None]).float().permute(0,3,1,2).to(self.device)
            
            if self.all_speeds:
                steer_logits, throt_logits, brake_logits = self.image_model.policy(_wide_rgb, _narr_rgb, cmd_value)
                # Interpolate logits
                steer_logit = self._lerp(steer_logits, spd)
                throt_logit = self._lerp(throt_logits, spd)
                brake_logit = self._lerp(brake_logits, spd)
            else:
                steer_logit, throt_logit, brake_logit = self.image_model.policy(_wide_rgb, _narr_rgb, cmd_value, spd=torch.tensor([spd]).float().to(self.device))

            
            action_prob = self.action_prob(steer_logit, throt_logit, brake_logit)

            brake_prob = float(action_prob[-1])

            steer = float(self.steers @ torch.softmax(steer_logit, dim=0))
            throt = float(self.throts @ torch.softmax(throt_logit, dim=0))

            steer, throt, brake = self.post_process(steer, throt, brake_prob, spd, cmd_value)
            control_cur = carla.VehicleControl(steer=steer, throttle=throt, brake=brake)
            self.prev_control[ego_id] = control_cur
            control_all.append(control_cur)
            
            planning_record = {
                'steer': steer,
                'throttle': throt,
                'brake': brake,
                'pose_x': pos[0],
                'pose_y': pos[1],
            }

            save_path_tmp = self.save_path / pathlib.Path("ego_vehicle_{}".format(ego_id))
            folder_path = save_path_tmp / pathlib.Path("meta")
            if not os.path.exists(save_path_tmp):
                os.mkdir(save_path_tmp)
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            # print(narr_rgb.shape)
            # print(narr_rgb)
            Image.fromarray(narr_rgb[:, :, :3]).save(
                save_path_tmp / ("%04d.jpg" % self.num_frames)
            )
            with open(save_path_tmp / ("%04d.json" % self.num_frames), 'w') as f:
                json.dump(planning_record, f, indent=4)

        self.num_frames += 1
        # print(self.num_frames)

        return control_all
    
    def _lerp(self, v, x):

        D = v.shape[0]

        min_val = self.min_speeds
        max_val = self.max_speeds

        # print(v, x, min_val, max_val, D)
        x = (x - min_val)/(max_val - min_val)*(D-1)

        x0, x1 = max(min(math.floor(x), D-1),0), max(min(math.ceil(x), D-1),0)
        w = x - x0

        return (1-w) * v[x0] + w * v[x1]

    def action_prob(self, steer_logit, throt_logit, brake_logit):

        steer_logit = steer_logit.repeat(self.num_throts)
        throt_logit = throt_logit.repeat_interleave(self.num_steers)

        action_logit = torch.cat([steer_logit, throt_logit, brake_logit[None]])

        return torch.softmax(action_logit, dim=0)

    def post_process(self, steer, throt, brake_prob, spd, cmd):
        
        if brake_prob > 0.5:
            steer, throt, brake = 0, 0, 1
        else:
            brake = 0
            throt = max(0.4, throt)

        # # To compensate for non-linearity of throttle<->acceleration
        # if throt > 0.1 and throt < 0.4:
        #     throt = 0.4
        # elif throt < 0.1 and brake_prob > 0.3:
        #     brake = 1

        if spd > {0:10,1:10}.get(cmd, 20)/3.6: # 10 km/h for turning, 15km/h elsewhere
            throt = 0

        # if cmd == 2:
        #     steer = min(max(steer, -0.2), 0.2)

        # if cmd in [4,5]:
        #     steer = min(max(steer, -0.4), 0.4) # no crazy steerings when lane changing

        return steer, throt, brake
   

