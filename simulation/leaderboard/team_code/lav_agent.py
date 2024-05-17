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
from leaderboard.autoagents import autonomous_agent
# from interfuser.timm.models import create_model
# from team_code.utils import lidar_to_histogram_features, transform_2d_points
from team_code.planner import RoutePlanner
from team_code.interfuser_controller import InterfuserController
# from team_code.render import render, render_self_car, render_waypoints
# from team_code.tracker import Tracker
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import math
import yaml

from team_code.lav_models.lidar import LiDARModel
from team_code.lav_models.uniplanner import UniPlanner
from team_code.lav_models.bev_planner import BEVPlanner
from team_code.lav_models.rgb import RGBSegmentationModel, RGBBrakePredictionModel

from team_code.lav_utils.pid import PIDController
from team_code.lav_utils.ekf import EKF
from team_code.lav_utils.point_painting import CoordConverter, point_painting
from team_code.lav_utils.planner import RoutePlanner as RoutePlanner_lav
from team_code.lav_utils.waypointer import Waypointer
from team_code.lav_utils.model_inference import InferModel

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
    return "LAVAgent"


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


class LAVAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file, ego_vehicles_num, max_speed=10, threshold=0):

        self.agent_name='LAV'

        self._hic = DisplayInterface()
        self.lidar_processed = list()
        self.track = autonomous_agent.Track.SENSORS
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        self.rgb_front_transform = create_carla_rgb_transform(224)
        self.rgb_left_transform = create_carla_rgb_transform(128)
        self.rgb_right_transform = create_carla_rgb_transform(128)
        self.rgb_center_transform = create_carla_rgb_transform(128, need_scale=False)


        self.prev_control = dict()

        with open(path_to_conf_file, 'r') as f:
            config = yaml.safe_load(f)
        self.config = config

        for key, value in config.items():
            setattr(self, key, value)
        
        self.device = torch.device('cuda')

        self.waypointer = None
        self.planner    = None

        # Setup models
        self.lidar_model = LiDARModel(
            num_input=len(self.seg_channels)+10+self.num_frame_stack if self.point_painting else 10,
            backbone=self.backbone,
            num_features=self.num_features,
            min_x=self.min_x, max_x=self.max_x,
            min_y=self.min_y, max_y=self.max_y,
            pixels_per_meter=self.pixels_per_meter,
        ).to(self.device)

        bev_planner = BEVPlanner(
            pixels_per_meter=self.pixels_per_meter,
            crop_size=self.crop_size,
            feature_x_jitter=self.feature_x_jitter,
            feature_angle_jitter=self.feature_angle_jitter,
            x_offset=0, y_offset=1+self.min_x/((self.max_x-self.min_x)/2),
            num_cmds=self.num_cmds,
            num_plan=self.num_plan,
            num_plan_iter=self.num_plan_iter,
            num_frame_stack=self.num_frame_stack,
        ).to(self.device)

        self.uniplanner = UniPlanner(
            bev_planner,
            pixels_per_meter=self.pixels_per_meter,
            crop_size=self.crop_size,
            feature_x_jitter=self.feature_x_jitter,
            feature_angle_jitter=self.feature_angle_jitter,
            x_offset=0, y_offset=1+self.min_x/((self.max_x-self.min_x)/2),
            num_cmds=self.num_cmds,
            num_plan=self.num_plan,
            num_input_feature=self.num_features[-1]*6,
            num_plan_iter=self.num_plan_iter,
        ).to(self.device)

        self.bra_model = torch.jit.load(self.bra_model_trace_dir)
        self.seg_model = torch.jit.load(self.seg_model_trace_dir)

        # Load the models
        # print(self.lidar_model_dir)
        lidar_model_dict = torch.load(self.lidar_model_dir)
        # wandb.log(lidar_model_dict)
        self.lidar_model.load_state_dict(torch.load(self.lidar_model_dir))
        self.uniplanner.load_state_dict(torch.load(self.uniplanner_dir))

        self.lidar_model.eval()
        self.uniplanner.eval()
        self.bra_model.eval()
        self.seg_model.eval()

        self.infer_model = InferModel(self.lidar_model, self.uniplanner, self.camera_x, self.camera_z).to(self.device)

        # Coordinate converters for point-painting
        self.coord_converters = [CoordConverter(
            cam_yaw, lidar_xyz=[0,0,self.camera_z], cam_xyz=[self.camera_x,0,self.camera_z],
            rgb_h=288, rgb_w=256, fov=64
        ) for cam_yaw in CAMERA_YAWS]

        # Setup tracker TODO: update 1 to actual cos0
        self.ekf = EKF(1, 1.477531, 1.393600)
        self.ekf_initialized = False

        # FIFO
        self.lidars = deque([])
        self.locs = deque([])
        self.oris = deque([])

        # Book-keeping
        self.vizs = []
        # self.num_frames = 0

        self.prev_lidar = None
        self.num_frame_keep = (self.num_frame_stack + 1) * GAP

        self.turn_controller = PIDController(K_P=self.turn_KP, K_I=self.turn_KI, K_D=self.turn_KD, n=self.turn_n)
        self.speed_controller = PIDController(K_P=self.speed_KP, K_I=self.speed_KI, K_D=self.speed_KD, n=self.speed_n)

        self.lane_change_counter = 0
        self.stop_counter = 0
        self.force_move = 0
        self.lane_changed = None


        self.ego_vehicles_num = 1        
        self.skip_frames = 4
        

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

    def _init(self):
        self._route_planner = RoutePlanner(4.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        self.vehicle_num = 0
        self.tick_data = {}


    def _get_position(self, tick_data):
        gps = tick_data["gps"]
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps

    def sensors(self):
        sensors = [
            {'type': 'sensor.speedometer', 'id': 'EGO'},
            {'type': 'sensor.other.gnss', 'x': 0., 'y': 0., 'z': self.camera_z, 'id': 'GPS'},
            {'type': 'sensor.other.imu',  'x': 0., 'y': 0., 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,'sensor_tick': 0.05, 'id': 'IMU'},
            
        ]

        # Add LiDAR
        sensors.append({
            'type': 'sensor.lidar.ray_cast', 'x': 0.0, 'y': 0.0, 'z': self.camera_z, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 
            'id': 'LIDAR'
        })

        # Add cameras
        for i, yaw in enumerate(CAMERA_YAWS):
            sensors.append({'type': 'sensor.camera.rgb', 'x': self.camera_x, 'y': 0.0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': yaw,
            'width': 256, 'height': 288, 'fov': 64, 'id': f'RGB_{i}'})

        sensors.append({'type': 'sensor.camera.rgb', 'x': self.camera_x, 'y': 0.0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 480, 'height': 288, 'fov': 40, 'id': 'TEL_RGB'})

        return sensors

    def tick(self, input_data, vehicle_num):
        self.tick_data = {}
        
        _, lidar = input_data.get('LIDAR_{}'.format(vehicle_num))
        _, gps   = input_data.get('GPS_{}'.format(vehicle_num))
        _, imu   = input_data.get('IMU_{}'.format(vehicle_num))
        _, ego   = input_data.get('EGO_{}'.format(vehicle_num))
        spd      = ego.get('speed')

        compass = imu[-1]
        
        # Let's hope this only happens when compass == 0 or 2pi
        # https://discord.com/channels/444206285647380482/551506571608326156/769089544103919626
        if np.isnan(compass):
            compass = 0.

        if not self.ekf_initialized:
            self.ekf.init(*gps[:2], compass-math.pi/2)
            self.ekf_initialized = True

        loc, ori = self.ekf.x[:2], self.ekf.x[2]

        if spd < 0.1:
            self.stop_counter += 1
        else:
            self.stop_counter = 0

        lidar = torch.tensor(lidar, dtype=torch.float, device=self.device)

        if self.step <= 1:
            self.prev_lidar = lidar
            return carla.VehicleControl()


        if self.prev_lidar is not None:
            cur_lidar = torch.cat([lidar, self.prev_lidar])
        else:
            cur_lidar = lidar

        self.prev_lidar = lidar

        
        cur_lidar = self.preprocess(cur_lidar)

        # Paint the lidars
        rgbs_data = []

        for i in range(len(CAMERA_YAWS)):
            _, rgb = input_data.get(f'RGB_{i}_{vehicle_num}')
            rgbs_data.append(rgb[...,:3][...,::-1])

        rgb = np.concatenate(rgbs_data, axis=1)
        all_rgb = np.stack(rgbs_data, axis=0)

        _, tel_rgb = input_data.get(f'TEL_RGB_{vehicle_num}')
        tel_rgb = tel_rgb[...,:3][...,::-1].copy()
        tel_rgb = tel_rgb[:-self.crop_tel_bottom]

        all_rgbs = torch.tensor(all_rgb).permute(0,3,1,2).float().to(self.device)
        pred_sem = torch.softmax(self.seg_model(all_rgbs), dim=1)

        fused_lidar = self.infer_model.forward_paint(cur_lidar, pred_sem)

        # EKF updates and bookeepings
        self.lidars.append(fused_lidar)
        self.locs.append(loc)
        self.oris.append(ori)
        if len(self.lidars) > self.num_frame_keep:
            self.lidars.popleft()
            self.locs.popleft()
            self.oris.popleft()

        lidar_points = self.get_stacked_lidar()

        # High-level commands
        if self.waypointer is None:

            self.waypointer = Waypointer(
                self._global_plan[vehicle_num], gps, pop_lane_change=True
            )

            self.planner = RoutePlanner_lav(self._global_plan[vehicle_num])
        
        _, _, cmd = self.waypointer.tick(gps)
        wx, wy = self.planner.run_step(gps)

        cmd_value = cmd.value - 1
        cmd_value = 3 if cmd_value < 0 else cmd_value
        
        if cmd_value in [4,5]:
            if self.lane_changed is not None and cmd_value != self.lane_changed:
                self.lane_change_counter = 0

            self.lane_change_counter += 1
            self.lane_changed = cmd_value if self.lane_change_counter > {4:300,5:300}.get(cmd_value) else None
        else:
            self.lane_change_counter = 0
            self.lane_changed = None

        if cmd_value == self.lane_changed:
            cmd_value = 3

        # Transform to ego-coordinates
        wx, wy = _rotate(wx, wy, -imu[-1]+np.pi/2)

        # Predict brakes from images
        rgbs = torch.tensor(rgb[None]).permute(0,3,1,2).float().to(self.device)
        tel_rgbs= torch.tensor(tel_rgb[None]).permute(0,3,1,2).float().to(self.device)

        nxps       = torch.tensor([-wx,-wy]).float().to(self.device)

        # Motion forecast & planning
        ego_embd, ego_plan_locs, ego_cast_locs, other_cast_locs, other_cast_cmds, pred_bev, det = self.infer_model(lidar_points, nxps, cmd_value)
        ego_plan_locs = to_numpy(ego_plan_locs)
        ego_cast_locs = to_numpy(ego_cast_locs)
        other_cast_locs = to_numpy(other_cast_locs)
        other_cast_cmds = to_numpy(other_cast_cmds)

        pred_bra = self.bra_model(rgbs, tel_rgbs)

        if cmd_value in [4,5]:
            ego_plan_locs = ego_cast_locs

        if not np.isnan(ego_plan_locs).any():
            steer, throt, brake = self.pid_control(ego_plan_locs, spd, cmd_value)
        else:
            steer, throt, brake = 0, 0, 0

        if not np.isnan(ego_plan_locs).any():
            steer, throt, brake = self.pid_control(ego_plan_locs, spd, cmd_value)
        else:
            steer, throt, brake = 0, 0, 0

        self.ekf.step(spd, steer, *gps[:2], compass-math.pi/2)

        if float(pred_bra) > 0.1:
            throt, brake = 0, 1
        # elif self.plan_collide(ego_plan_locs, other_cast_locs, other_cast_cmds):
        #     throt, brake = 0, 1
        if spd * 3.6 > self.max_speed:
            throt = 0

        if self.stop_counter >= 20: # Creep forward
            self.force_move = 20

        if self.force_move > 0:
            throt, brake = max(0.4, throt), 0
            self.force_move -= 1

        # viz = self.visualize(rgb, tel_rgb, lidar_points, float(pred_bra), to_numpy(torch.sigmoid(pred_bev[0])), ego_plan_locs, other_cast_locs, other_cast_cmds, det, [-wx, -wy], cmd_value, spd, steer, throt, brake)
        # self.vizs.append(viz)

        # if len(self.vizs) >= 12000:
        #     self.flush_data()
        # print('Getting tick_data')
        self.tick_data = {
            'rgbs': rgbs_data,
            'planning': {
                'steer': steer,
                'throttle': throt,
                'brake': brake,
                'pose_x': loc[0],
                'pose_y': loc[1],
            }
        }
        # print('Running interface.')
        surface = self._hic.run_interface(self.tick_data)
    
        self.tick_data["surface"] = surface
        if SAVE_PATH is not None:
            # print('Saving the data')
            self.save(self.tick_data, vehicle_num)
        return carla.VehicleControl(steer=steer, throttle=throt, brake=brake)

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        # print('Running step!')
        if not self.initialized:
            self._init()
        
        # print(input_data.keys())
        # dict_keys(['GPS_0', 'LIDAR_0', 'IMU_0', 'EGO_0', 'TEL_RGB_0', 'RGB_2_0', 'RGB_1_0', 'RGB_0_0'])
        # raise ValueError
        self.step += 1
        if self.step % self.skip_frames != 0 and self.step > 4:
                return self.prev_control
        # print('Running control.')
        # print(self.ego_vehicles_num)
        control_all = []
        input_data_copy = copy.copy(input_data) 
        for vehicle_num in range(self.ego_vehicles_num):
            # if not CarlaDataProvider.get_hero_actor(hero_id=vehicle_num).is_alive:
            # if CarlaDataProvider.get_hero_actor(hero_id=vehicle_num) is None:
            #     continue
            input_data_tmp = copy.deepcopy(input_data_copy)
            # self.vehicle_num = vehicle_num
            control_cur = self.tick(input_data_tmp, vehicle_num)

            self.prev_control[vehicle_num] = control_cur
            control_all.append(control_cur)
            
                        
        return control_all

    def save(self, tick_data, vehicle_num):
        if self.step % self.skip_frames != 0:
            return
        # frame = self.step // self.skip_frames
        frame = self.step
        save_path_tmp = self.save_path / pathlib.Path("ego_vehicle_{}".format(vehicle_num))
        folder_path = save_path_tmp / pathlib.Path("meta")
        if not os.path.exists(save_path_tmp):
            os.mkdir(save_path_tmp)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        Image.fromarray(tick_data["surface"]).save(
            save_path_tmp / ("%04d.jpg" % frame)
        )
        with open(save_path_tmp / ("%04d.json" % frame), 'w') as f:
            json.dump(tick_data['planning'], f, indent=4)
        return

    def destroy(self):
        pass
    


    def get_stacked_lidar(self):

        loc0, ori0 = self.locs[-1], self.oris[-1]

        rel_lidars = []
        for i, t in enumerate(range(len(self.lidars)-1, -1, -GAP)):
            loc, ori = self.locs[t], self.oris[t]
            lidar = self.lidars[t]

            lidar_xyz = lidar[:,:3]
            lidar_f = lidar[:,3:]

            lidar_xyz = move_lidar_points(lidar_xyz, loc - loc0, ori0, ori)
            lidar_t = torch.zeros((len(lidar_xyz), self.num_frame_stack+1), dtype=lidar_xyz.dtype, device=self.device)
            lidar_t[:,i] = 1      # Be extra careful on this.

            rel_lidar = torch.cat([lidar_xyz, lidar_f, lidar_t], dim=-1)

            rel_lidars.append(rel_lidar)

        return torch.cat(rel_lidars)

    def plan_collide(self, ego_plan_locs, other_cast_locs, other_cast_cmds, dist_threshold_static=1.0, dist_threshold_moving=2.5):
        # TODO: Do a proper occupancy map?
        for other_trajs, other_cmds in zip(other_cast_locs, other_cast_cmds):
            init_x, init_y = other_trajs[0,0]
            if init_y > 0.5*self.pixels_per_meter:
                continue
            for other_traj, other_cmd in zip(other_trajs, other_cmds):
                if other_cmd < self.cmd_thresh:
                    continue

                spd = np.linalg.norm(other_traj[1:]-other_traj[:-1], axis=-1).mean()
                dist_threshold = dist_threshold_static if spd < self.brake_speed else dist_threshold_moving
                dist = np.linalg.norm(other_traj-ego_plan_locs, axis=-1).min() # TODO: outer norm?
                if dist < dist_threshold:
                    return True

        return False


    def pid_control(self, waypoints, speed, cmd):

        waypoints = np.copy(waypoints) * self.pixels_per_meter
        waypoints[:,1] *= -1

        # desired_speed = np.linalg.norm(waypoints[1:]-waypoints[:-1], axis=1).mean()
        # desired_speed = np.mean((waypoints[1:]-waypoints[:-1])@[0,1])
        desired_speed = np.linalg.norm(waypoints[1:]-waypoints[:-1], axis=1).mean()

        aim = waypoints[self.aim_point[cmd]]
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        # Below: experimental
        # steer = steer if desired_speed > self.brake_speed * self.pixels_per_meter * 2 else 0.

        brake = desired_speed < self.brake_speed * self.pixels_per_meter
        delta = np.clip(desired_speed * self.speed_ratio[cmd] - speed, 0.0, self.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.max_throttle)
        throttle = throttle if not brake else 0.0

        return float(steer), float(throttle), float(brake)


    def det_inference(self, heatmaps, sizemaps, orimaps, **kwargs):

        dets = []
        for i, c in enumerate(heatmaps):
            det = []
            for s, x, y in extract_peak(c, **kwargs):
                w, h = float(sizemaps[0,y,x]),float(sizemaps[1,y,x])
                cos, sin = float(orimaps[0,y,x]), float(orimaps[1,y,x])
                
                if i==1 and w < 0.1*self.pixels_per_meter or h < 0.2*self.pixels_per_meter:
                    continue
                
                # TODO: remove hardcode
                if np.linalg.norm([x-160,y-280]) <= 2:
                    continue

                det.append((x,y,w,h,cos,sin))
            dets.append(det)
        
        return dets

    def preprocess(self, lidar_xyzr, lidar_painted=None):

        idx = (lidar_xyzr[:,0] > -2.4)&(lidar_xyzr[:,0] < 0)&(lidar_xyzr[:,1]>-0.8)&(lidar_xyzr[:,1]<0.8)&(lidar_xyzr[:,2]>-1.5)&(lidar_xyzr[:,2]<-1)

        if lidar_painted is None:
            return lidar_xyzr[~idx]
        else:
            return lidar_xyzr[~idx], lidar_painted[~idx]

    



def _rotate(x, y, theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    return R @ [x, y]

def to_numpy(x):
    return x.detach().cpu().numpy()

def extract_peak(heatmap, max_pool_ks=7, min_score=0.1, max_det=15, break_tie=False):
    
    # Credit: Prof. Philipp Krähenbühl in CS394D

    if break_tie:
        heatmap = heatmap + 1e-7*torch.randn(*heatmap.size(), device=heatmap.device)

    max_cls = F.max_pool2d(heatmap[None, None], kernel_size=max_pool_ks, padding=max_pool_ks//2, stride=1)[0, 0]
    possible_det = heatmap - (max_cls > heatmap).float() * 1e5
    if max_det > possible_det.numel():
        max_det = possible_det.numel()
    score, loc = torch.topk(possible_det.view(-1), max_det)

    return [(float(s), int(l) % heatmap.size(1), int(l) // heatmap.size(1))
            for s, l in zip(score.cpu(), loc.cpu()) if s > min_score]

def move_lidar_points(lidar, dloc, ori0, ori1):


    dloc = dloc @ [
        [ np.cos(ori0), -np.sin(ori0)],
        [ np.sin(ori0), np.cos(ori0)]
    ]

    ori = ori1 - ori0
    lidar = lidar @ torch.tensor([
        [np.cos(ori), np.sin(ori),0],
        [-np.sin(ori), np.cos(ori),0],
        [0,0,1],
    ], dtype=torch.float, device=lidar.device)

    lidar[:,0] += dloc[0]
    lidar[:,1] += dloc[1]
    
    return lidar

def lidar_to_bev(lidar, min_x=-10,max_x=70,min_y=-40,max_y=40, pixels_per_meter=4, hist_max_per_pixel=10):
    xbins = np.linspace(
        min_x, max_x+1,
        (max_x - min_x) * pixels_per_meter + 1,
    )
    ybins = np.linspace(
        min_y, max_y+1,
        (max_y - min_y) * pixels_per_meter + 1,
    )

    hist = np.histogramdd(lidar[..., :2], bins=(xbins, ybins))[0]
    hist[hist > hist_max_per_pixel] = hist_max_per_pixel

    overhead_splat = hist / hist_max_per_pixel * 255.
    return overhead_splat[::-1,:]