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

import random

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

# from interfuser.timm.models import create_model
# from team_code.utils import lidar_to_histogram_features, transform_2d_points
from team_code.planner import RoutePlanner
from team_code.interfuser_controller import InterfuserController
# from team_code.render import render, render_self_car, render_waypoints
# from team_code.tracker import Tracker
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import math
import yaml

import argparse
# from team_code.wor_models import EgoModel, CameraModel
# from team_code.wor_models.waypointer import Waypointer
from team_code.coop_models import CooperativePointTransformer


from team_code.AVR.autocast_agents.new_agent import NewAgent
# from team_code.AVR.PCProcess import LidarPreprocessor
from team_code.AVR.DataLogger import DataLogger
# from team_code.AVR import Collaborator
from team_code.AVR import Utils


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


class LidarProcessorConfig():
    X_max = 70
    X_min = -70
    Y_max = 70
    Y_min = -70
    Z_max = 2.5
    Z_min = -2.5
    dX = 0.5
    dY = 0.5
    dZ = 0.5

    X_SIZE = int((X_max - X_min) / dX)
    Y_SIZE = int((Y_max - Y_min) / dY)
    Z_SIZE = int((Z_max - Z_min) / dZ)

    lidar_dim = [X_SIZE, Y_SIZE, Z_SIZE]
    lidar_depth_dim = [X_SIZE, Y_SIZE, 1]


class ImageAgent(AutonomousAgent):
    
    """
    Trained image agent
    """
    
    def setup(self, path_to_conf_file, ego_vehicles_num=1, max_speed=10, threshold=0):
        """
        Setup the agent parameters
        """
        config = yaml.load(open(path_to_conf_file),Loader=yaml.FullLoader)
        self.config = config

        self.num_commands = config['num_commands']['value']
        self.ego_only = config['ego_only']['value']
        self.shared = config['shared']['value']
        self.num_hidden = config['num_hidden']['value']
        self.max_num_neighbors = config['max_num_neighbors']['value']
        self.npoints = config['npoints']['value']
        self.nblocks = config['nblocks']['value']
        self.nneighbor = config['nneighbor']['value']
        self.num_output = config['num_output']['value']
        self.input_dim = config['input_dim']['value']
        self.transformer_dim = config['transformer_dim']['value']
        self.cpt = config['cpt']['value']

        self.ego_vehicles_num = 1 
       
        self.earlyfusion = False
        num_checkpoint = 105
        self.model_checkpoint = 'model-{}.th'.format(num_checkpoint)
        print("Loading Checkpoint", self.model_checkpoint)
        self.model_path = '/GPFS/data/gjliu/Auto-driving/Coopernaut/ckpts/model-105.th' # '/GPFS/data/gjliu/Auto-driving/V2Xverse/simulation/leaderboard/team_code/coop_models/8-cpt-dagger-run2/files/model-105.th' # os.path.join(os.path.split(path_to_conf_file)[0], self.model_checkpoint)
        self.num_output = 3
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        print("Using Device:", self.device) 
        #self.frame_stack = config["frame_stack"]["value"]

        config = argparse.ArgumentParser()
        config.num_hidden = self.num_hidden
        config.num_commands = self.num_commands
        config.npoints = self.npoints
        config.nblocks = self.nblocks
        config.nneighbor = self.nneighbor
        config.num_output = self.num_output
        config.input_dim = self.input_dim
        #config.frame_stack = self.frame_stack
        config.transformer_dim = self.transformer_dim
        config.max_num_neighbors = self.max_num_neighbors 
        # if self.cpt:
        print("Using Cooperative Point Transformer")
        model = CooperativePointTransformer(config).to(self.device)
       
        model.eval()
        print("Loading model and initializing ", self.model_path)
        model.load_state_dict(torch.load(self.model_path))
        self.model = model
        self.count = 0
        self.agent_trajectory_points_timestamp = []
        self.collider_trajectory_points_timestamp = []
        self.next_target_location = None
        self.drawing_object_list = []
        self.transform = transforms.ToTensor()

        self._agent_control = None
        self._target_radius = 2.0
        self._final_goal = None
        self._agent = None
        self._route_assigned = False
        self._target_speed = 20  # default 20 km/h
        # if Utils.EvalEnv.ego_speed_kmph is not None:
        #     self._target_speed = Utils.EvalEnv.ego_speed_kmph
        #     Utils.target_speed_kmph = self._target_speed
        #     Utils.target_speed_mps = self._target_speed / 3.6
        
        self.other_states = {} 
        for i in range(self.max_num_neighbors):
            self.other_states[i] = {}
            self.other_states[i]['lidar'] = deque()
            self.other_states[i]['speed'] = 0.0
            self.other_states[i]['transform'] = np.eye(4)
        self.other_ids = []

        self.lidar_dim = LidarProcessorConfig.lidar_dim
        self.lidar_depth_dim = LidarProcessorConfig.lidar_depth_dim
        self.count = 0
        ##### End of setup


        self.waypointer = None
            
        # self.steers = torch.tensor(np.linspace(-self.max_steers,self.max_steers,self.num_steers)).float().to(self.device)
        # self.throts = torch.tensor(np.linspace(0,self.max_throts,self.num_throts)).float().to(self.device)

        self.prev_steer = 0
        self.lane_change_counter = 0
        self.stop_counter = 0

        self.initialized = False

        # self.set_global_plan_from_parent(self._global_plan, self._global_plan_world_coord)

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
        del self.model
    
    def sensors(self):
        """
        Define the sensor suite required by the agent
        :return: a list containing the required sensors in the following format:
        """
        sensors = [
            {'type': 'sensor.camera.rgb', 
             'x': 0, 'y': 0.0, 'z': 50,
             'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
             'width': 720, 'height': 720, 'fov': 90, 
             'id': 'RGB'},  
             # use same width height to align with lidar display
            {'type': 'sensor.lidar.ray_cast', 
             'x': 0, 'y': 0.0,'z': 0.5,  
             # the spawn function will add this on top of bbox.extent.z
             'yaw': 90.0, 'pitch': 0.0, 'roll': 0.0,
             'range': 50,
             # set same as camera height, cuz camera fov is 90 deg, HUD can visualize in same dimension
             'rotation_frequency': 20, 
             'channels': 64,
             'upper_fov': 4, 
             'lower_fov': -20, 
             'points_per_second': 2304000,
             'id': 'LIDAR'},
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
        agent_id = 0
        if not self._agent:
            hero_actor = CarlaDataProvider.get_hero_actor(hero_id=agent_id)
            if hero_actor:
                self._agent = NewAgent(hero_actor, self._target_speed)
        # Use NPC agent result
        control = super().run_step(input_data, timestamp)
        # Obtain ego Lidar data and tranform it to be BEV
        # import pdb
        # pdb.set_trace()
        lidar = input_data['LIDAR_'+str(agent_id)][1]
        # Obtain Fused Lidar data
        # fused_points = input_data[str(self._agent.id)+Collaborator.FusedLidarSensorName][1]
        # if fused_points.id != -1 and self.shared:
        #     try:
        #         fused_lidar = fused_points.pc 
        #         assert len(fused_lidar) > 0
        #         lidar = fused_lidar
        #         print("Using fused Lidar")
        #     except:
        #         print("Using non-sharing Lidar")
        lidar = np.array(lidar)[:,:3]
        # fused_sensor_id = str(self._agent.id) + Collaborator.FusedLidarSensorName
        # frame_id = input_data[fused_sensor_id][0]
        frame_id = 100
        JSONState = DataLogger.compile_actor_state(self, frame_id)[0]
        pred_location, pred_brake, speed, pred_control = None, None, None, None
        if JSONState is not None:
            ego_id = self._agent.id
            ego_actor_info = JSONState['other_actors'][ego_id]
            current_transform = ego_actor_info['transform']
            ego_z_compensation = 2*abs(ego_actor_info['bounding_box']['extent_z'])+Utils.LidarRoofTopDistance
            ego_transform = Utils.convert_json_to_transform(JSONState['ego_vehicle_transform'])
            ego_location = JSONState['ego_vehicle_transform']
            ego_location_z = ego_transform.location.z
            ego_transform.location.z = 0
            ego_transform = Utils.TransformMatrix_WorldCoords(ego_transform)
            ego_transform = np.array(ego_transform.inversematrix())
            lidar[:,2] = lidar[:,2] + abs(ego_z_compensation)
            lidar = Utils.pc_to_car_alignment(lidar)
            if not self.earlyfusion:
                # lidar = LidarPreprocessor.Sparse_Quantize(lidar)
                lidar = np.unique(lidar, axis=0)
                ego_lidar = self._pad_or_truncate(lidar, self.npoints)
            else:
                ego_lidar = self._pad_or_truncate(lidar, 65536)
            #ego_lidar = torch.from_numpy(np.array([ego_lidar])).float().to(self.device)

            ego_speed = torch.tensor(np.array([ego_actor_info['velocity']/30.0])).float().to(self.device)
            ego_command = torch.tensor(np.array([3])).to(self.device)
            
            if self.cpt or self.earlyfusion:
                other_actors = JSONState['other_actors']
                other_actors_ids = []
                for key in input_data.keys():
                    current_id = int(key.split("_")[1]) 
                    if current_id not in other_actors_ids:
                        other_actors_ids.append(current_id)
                if int(ego_id) in other_actors_ids:
                    other_actors_ids.remove(int(ego_id))
                # other_agents_keys_dist = []
                # for o_agent in other_actors_ids:
                #     o_location = other_actors[o_agent]['transform']
                #     dist = self._cal_dist(ego_location, o_location)
                #     if dist<40:
                #         other_agents_keys_dist.append((str(o_agent), dist))
                # other_agents_keys_dist = sorted(other_agents_keys_dist, key=lambda x:x[1])
                # other_agents_keys_dist = other_agents_keys_dist[:2*self.max_num_neighbors]
                # # random.shuffle(other_agents_keys_dist)
                # other_lidar = []
                # other_transform = []
                # other_speed = []
                num_valid_neighbors = 0
                # for key in sorted(self.other_states.keys()):
                #     # try:
                #     #     other_agent_id = int(other_agents_keys_dist[int(key)][0])
                #     # except:
                #     other_agent_id = None
                #     if other_agent_id in other_actors_ids:
                #         lidar = np.array(input_data[str(other_agent_id)+"_LIDAR"][1])[:,:3]
                #         other_z_compensation = 2*abs(other_actors[other_agent_id]['bounding_box']['extent_z'])+Utils.LidarRoofTopDistance
                #         transform = Utils.convert_json_to_transform(other_actors[other_agent_id]['transform'])
                #         transform.location.z = 0#ego_location_z
                #         transform = Utils.TransformMatrix_WorldCoords(transform)
                #         transform = np.array(transform.matrix)
                #         transform = np.matmul(ego_transform, transform) 
                #         self.other_states[key]['transform'] = transform
                #         lidar[:,2] = lidar[:,2]+abs(other_z_compensation)
                #         lidar = Utils.pc_to_car_alignment(lidar)
                #         if not self.earlyfusion:
                #             # lidar = LidarPreprocessor.Sparse_Quantize(lidar)
                #             lidar = np.unique(lidar, axis=0)
                #             lidar = self._pad_or_truncate(lidar, self.npoints)
                #         else:
                #             #lidar = LidarPreprocessor.Sparse_Quantize(lidar)
                #             lidar = np.transpose(np.matmul(transform[:3,:3], np.transpose(lidar))) + np.tile(transform[:3,3],(len(lidar),1)) 
                #             lidar = self._pad_or_truncate(lidar, 65536)
                #         self.other_states[key]['lidar'] = lidar
                #         print("SUCCESSFULLY LOADED", other_agent_id, sum(lidar))
                #         speed = other_actors[other_agent_id]['velocity']/30.0
                #         self.other_states[key]['speed'] = speed
                #         print("sharing vehicles", ego_id, other_agent_id)
                #     else:
                #         self.other_states[key]['lidar'] = np.zeros((self.npoints,3))
                #         self.other_states[key]['transform'] = np.eye(4) 
                #         self.other_states[key]['speed'] = 0.0
                #     other_lidar.append(list(self.other_states[key]['lidar']))
                #     other_speed.append(self.other_states[key]['speed'])
                #     other_transform.append(self.other_states[key]['transform'])
                
                ego_lidar = torch.from_numpy(np.array([ego_lidar])).float().to(self.device)

                if self.cpt:
                    # other_lidar = torch.from_numpy(np.array([other_lidar])).float().to(self.device)
                    # other_transform = torch.from_numpy(np.array([other_transform])).float().to(self.device)
                    # other_lidar = torch.from_numpy(np.array([ego_lidar])).float().to(self.device)
                    other_transform = torch.from_numpy(np.array([ego_transform])).float().to(self.device)
                    # import pdb
                    # pdb.set_trace()
                    pred_throttle, pred_brake, pred_steer = self.model(ego_lidar, ego_speed, ego_lidar.unsqueeze(0), other_transform.unsqueeze(0))
                # elif self.earlyfusion:
                #     pred_throttle, pred_brake, pred_steer = self.model(ego_lidar, ego_speed)
            # else:
            #     ego_lidar = torch.from_numpy(np.array([ego_lidar])).float().to(self.device)
            #     pred_throttle, pred_brake, pred_steer = self.model(ego_lidar, ego_speed)
            pred_throttle = pred_throttle.cpu().detach().numpy()[0]
            pred_brake = pred_brake.cpu().detach().numpy()[0]
            pred_steer = pred_steer.cpu().detach().numpy()[0]
        
        
        control = carla.VehicleControl()
        pid_control = self._agent._local_planner.run_step(goal = 'Forward')
        if pred_throttle is not None:
            control.throttle = np.float(pred_throttle)*1.5
            control.brake = np.float(pred_brake)
            control.steer = np.float(pred_steer)
            print("Model's Output:\t[{:.2f},\t{:.2f},\t{:.2f}]".format(control.throttle, control.brake, control.steer))
            if control.brake >= control.throttle or control.brake>0.7:
                control.throttle = 0.0
                #control.brake = 0.75
            else:
                control.brake = 0.0
            if pid_control.brake > 0:
                control.throttle = 0.0
                control.brake = pid_control.brake
            control.throttle = np.clip(control.throttle,0.0,0.75)
            control.brake = np.clip(control.brake,0.0,0.75)
            control.steer = np.clip(control.steer,-1.0,1.0)

        pred_control = control
        
        #FIXME: What will happen if the PID coeff are different
        print("Student's Action:\t[{:.2f},\t{:.2f},\t{:.2f}]".format(pred_control.throttle, pred_control.brake, pred_control.steer))
        
        if not self._route_assigned:
            print("Setting up global plan")
            if self._global_plan:
                plan = []
                for transform, road_option in self._global_plan_world_coord[0]:
                    print(transform.location, road_option)
                    wp = CarlaDataProvider.get_map().get_waypoint(transform.location)
                    plan.append((wp, road_option))
                    self._final_goal = np.array([transform.location.x,
                                                 transform.location.y,
                                                 transform.location.z])
                self._agent._local_planner.set_global_plan(plan)  # pylint: disable=protected-access
                self._route_assigned = True
                print("Global Plan set")
        
        self._agent_control = pred_control
        return [pred_control]
    

    def postprocess(self, steer, throttle, brake):
        control = carla.VehicleControl()
        control.steer = np.clip(steer, -1.0, 1.0)
        control.throttle = np.clip(throttle, 0.0, 1.0)
        control.brake = np.clip(brake, 0.0, 1.0)
        control.manual_gear_shift = False

        return control

    def _pad_or_truncate(self, points, npoints):
        if points is None:                                                                                                                                       
            return None
        diff = npoints - len(points)                                                                                                           
        if diff > 0:   
            #pad npoints - len(points)           
            points = np.concatenate((points,np.array(random.choices(points, k=abs(diff)))), axis=0)                
            #points = np.concatenate((points,np.zeros((diff,3))),axis=0)
        else:                
            #points = points[:self.npoints]     
            points = np.array(random.sample(list(points), k=abs(npoints)))   
        #points = random.choices(points, self.npoints)     
        return points 
       
    def set_global_plan_from_parent(self, global_plan, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """
        self._global_plan_world_coord = global_plan_world_coord[0]
        self._global_plan = global_plan[0]