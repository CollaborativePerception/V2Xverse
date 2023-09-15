import imp
import time
import torch
import math
import cv2
import carla
import numpy as np
from PIL import Image
import pdb
import sys
import os
# sys.path.append('/GPFS/data/gjliu/Auto-driving/Cop3')

from codriving.utils.torch_helper import \
        move_dict_data_to_device, build_dataloader
from common.torch_helper import load_checkpoint as load_planning_model_checkpoint
from team_code.planner_pnp import RoutePlanner

from leaderboard.autoagents import autonomous_agent

from team_code.utils.carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions

from team_code.pnp_infer_action import PnP_infer
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from leaderboard.sensors.fixed_sensors import RoadSideUnit, get_rsu_point

import team_code.utils.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.tools.train_utils import create_model as create_perception_model
from opencood.tools.train_utils import load_saved_model as load_perception_model
from opencood.data_utils.datasets import build_dataset

from common.registry import build_object_within_registry_from_config as build_planning_model
from codriving import CODRIVING_REGISTRY
from codriving.models.model_decoration import decorate_model

def get_entry_point():
    return "PnP_Agent"

def get_camera_intrinsic(sensor):
    """
    Retrieve the camera intrinsic matrix.
    Parameters
    ----------
    sensor : carla.sensor
        Carla rgb camera object.
    Returns
    -------
    matrix_x : list
        The 2d intrinsic matrix.
    """
    VIEW_WIDTH = int(sensor['width'])
    VIEW_HEIGHT = int(sensor['height'])
    VIEW_FOV = int(float(sensor['fov']))
    # VIEW_WIDTH = int(sensor.attributes['image_size_x'])
    # VIEW_HEIGHT = int(sensor.attributes['image_size_y'])
    # VIEW_FOV = int(float(sensor.attributes['fov']))

    matrix_k = np.identity(3)
    matrix_k[0, 2] = VIEW_WIDTH / 2.0
    matrix_k[1, 2] = VIEW_HEIGHT / 2.0
    matrix_k[0, 0] = matrix_k[1, 1] = VIEW_WIDTH / \
        (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))

    return matrix_k.tolist()

def get_camera_extrinsic(cur_extrin, ref_extrin):
    """
    Args:
        cur_extrin (carla.Transform): current extrinsic
        ref_extrin (carla.Transform): reference extrinsic
    Returns:
        extrin (list): 4x4 extrinsic matrix with respect to reference coordinate system 
    """
    extrin = np.array(ref_extrin.get_inverse_matrix()) @ np.array(cur_extrin.get_matrix())
    return extrin.tolist()


#### agents
class PnP_Agent(autonomous_agent.AutonomousAgent):
    """
    Navigated by a pipline with perception then prediction, from sensor data to control signal
    """
    def setup(self, path_to_conf_file, ego_vehicles_num):
    
        self.agent_name = "pnp"
        self.wall_start = time.time()
        self.initialized = False
        self._rgb_sensor_data = {"width": 800, "height": 600, "fov": 100}
        self._sensor_data = self._rgb_sensor_data.copy()
        self.ego_vehicles_num = ego_vehicles_num   

        # load agent config
        self.config = imp.load_source("MainModel", path_to_conf_file).GlobalConfig()
        
        # load perception model and dataloader
        perception_hypes = yaml_utils.load_yaml(self.config.perception_model_dir)
        self.config.perception_hypes = perception_hypes
        perception_dataloader = build_dataset(perception_hypes, visualize=True, train=False)
        print('Creating perception Model')
        self.perception_model = create_perception_model(perception_hypes)
        print('Loading perception Model from checkpoint')
        resume_epoch, self.perception_model = load_perception_model(self.config.perception_model_dir, self.perception_model)
        print(f"resume from {resume_epoch} epoch.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.perception_model.to(device)
        self.perception_model.eval()

        # load planning model
        planning_model_config = self.config.planner_config['model']
        print('Creating planning Model')
        planning_model = build_planning_model(
            CODRIVING_REGISTRY,
            planning_model_config,
        )
        print('Loading planning Model from checkpoint')
        load_planning_model_checkpoint(self.config.planner_model_checkpoint, device, planning_model)
        model_decoration_config = self.config.planner_config['model_decoration']
        decorate_model(planning_model, **model_decoration_config)
        planning_model.to(device)
        planning_model.eval()

        # core module, infer the action from sensor data
        self.infer = PnP_infer(config=self.config,
                               ego_vehicles_num=self.ego_vehicles_num,
                               perception_model=self.perception_model,
                               planning_model=planning_model,
                               perception_dataloader=perception_dataloader,
                               device=device)


    def _init(self):
        """
        initialization before the first step of simulation
        """
        self._route_planner = RoutePlanner(2.0, 10.0)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        self.first_generate_rsu = True
        self.vehicle_num = 0
        self.step = -1

        print('Set BEV producers')
        self.birdview_producer = BirdViewProducer(
            CarlaDataProvider.get_client(),  # carla.Client
            target_size=PixelDimensions(width=400, height=400),
            pixels_per_meter=5,
            crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
        )

    def _get_position(self, tick_data):
        # GPS coordinate!
        gps = tick_data["gps"]
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps
    
    def get_save_path(self):
        return  self.infer.save_path   
    
    def pose_def(self):
        """
        location of sensor related to vehicle
        """
        self.lidar_pose = carla.Transform(carla.Location(x=1.3,y=0.0,z=1.85),
                                        carla.Rotation(roll=0.0,pitch=0.0,yaw=-90.0))
        self.camera_front_pose = carla.Transform(carla.Location(x=1.3,y=0.0,z=2.3),
                                        carla.Rotation(roll=0.0,pitch=0.0,yaw=0.0))
        self.camera_rear_pose = carla.Transform(carla.Location(x=-1.3,y=0.0,z=2.3),
                                        carla.Rotation(roll=0.0,pitch=0.0,yaw=180.0))
        self.camera_left_pose = carla.Transform(carla.Location(x=1.3,y=0.0,z=2.3),
                                        carla.Rotation(roll=0.0,pitch=0.0,yaw=-60.0))
        self.camera_right_pose = carla.Transform(carla.Location(x=1.3,y=0.0,z=2.3),
                                        carla.Rotation(roll=0.0,pitch=0.0,yaw=60.0))
        return

    def sensors(self):
        """
        Return the sensor list.
        """
        self.pose_def()
        sensors_list = [
            {
                "type": "sensor.camera.rgb",
                "x": self.camera_front_pose.location.x,
                "y": self.camera_front_pose.location.y,
                "z": self.camera_front_pose.location.z,
                "roll": self.camera_front_pose.rotation.roll,
                "pitch": self.camera_front_pose.rotation.pitch,
                "yaw": self.camera_front_pose.rotation.yaw,
                "width": self._rgb_sensor_data["width"],
                "height": self._rgb_sensor_data["height"],
                "fov": self._rgb_sensor_data["fov"],
                "id": "rgb_front",
            },
            {
                "type": "sensor.camera.rgb",
                "x": self.camera_rear_pose.location.x,
                "y": self.camera_rear_pose.location.y,
                "z": self.camera_rear_pose.location.z,
                "roll": self.camera_rear_pose.rotation.roll,
                "pitch": self.camera_rear_pose.rotation.pitch,
                "yaw": self.camera_rear_pose.rotation.yaw,
                "width": self._rgb_sensor_data["width"],
                "height": self._rgb_sensor_data["height"],
                "fov": self._rgb_sensor_data["fov"],
                "id": "rgb_rear",
            },
            {
                "type": "sensor.camera.rgb",
                "x": self.camera_left_pose.location.x,
                "y": self.camera_left_pose.location.y,
                "z": self.camera_left_pose.location.z,
                "roll": self.camera_left_pose.rotation.roll,
                "pitch": self.camera_left_pose.rotation.pitch,
                "yaw": self.camera_left_pose.rotation.yaw,
                "width": self._rgb_sensor_data["width"],
                "height": self._rgb_sensor_data["height"],
                "fov": self._rgb_sensor_data["fov"],
                "id": "rgb_left",
            },
            {
                "type": "sensor.camera.rgb",
                "x": self.camera_right_pose.location.x,
                "y": self.camera_right_pose.location.y,
                "z": self.camera_right_pose.location.z,
                "roll": self.camera_right_pose.rotation.roll,
                "pitch": self.camera_right_pose.rotation.pitch,
                "yaw": self.camera_right_pose.rotation.yaw,
                "width": self._rgb_sensor_data["width"],
                "height": self._rgb_sensor_data["height"],
                "fov": self._rgb_sensor_data["fov"],
                "id": "rgb_right",
            },
            {
                "type": "sensor.lidar.ray_cast",
                "x": self.lidar_pose.location.x,
                "y": self.lidar_pose.location.y,
                "z": self.lidar_pose.location.z,
                "roll": self.lidar_pose.rotation.roll,
                "pitch": self.lidar_pose.rotation.pitch,
                "yaw": self.lidar_pose.rotation.yaw,
                "id": "lidar",
            },
            {
                "type": "sensor.other.imu",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                # "sensor_tick": 0.05,
                "id": "imu",
            },
            {
                "type": "sensor.other.gnss",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                # "sensor_tick": 0.01,
                "id": "gps",
            },
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
        ]
        return sensors_list

    def tick(self, input_data: dict, vehicle_num: int):
        """
        Pre-load raw sensor data
        """
        _vehicle = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)

        # bev drivable area image
        bev_image = Image.fromarray(BirdViewProducer.as_rgb(
            self.birdview_producer.produce(agent_vehicle=_vehicle, actor_exist=False)
        )) #  (400, 400, 3), Image
        img_c = bev_image.crop([140, 20, 260, 260])
        img_r = np.array(img_c.resize((96, 192))) # (96, 192, 3)
        drivable_area = np.where(img_r.sum(axis=2)>200, 1, 0) # size (192, 96), only 0/1.
        
        # rgb camera data
        rgb_front = cv2.cvtColor(
            input_data["rgb_front_{}".format(vehicle_num)][1][:, :, :3], cv2.COLOR_BGR2RGB
        )
        rgb_left = cv2.cvtColor(
            input_data["rgb_left_{}".format(vehicle_num)][1][:, :, :3], cv2.COLOR_BGR2RGB
        )
        rgb_right = cv2.cvtColor(
            input_data["rgb_right_{}".format(vehicle_num)][1][:, :, :3], cv2.COLOR_BGR2RGB
        )
        rgb_rear = cv2.cvtColor(
            input_data["rgb_rear_{}".format(vehicle_num)][1][:, :, :3], cv2.COLOR_BGR2RGB
        )

        # lidar sensor data
        lidar = input_data["lidar_{}".format(vehicle_num)][1]

        # measurements
        gps = input_data["gps_{}".format(vehicle_num)][1][:2]
        speed = input_data["speed_{}".format(vehicle_num)][1]["move_state"]["speed"]
        compass = input_data["imu_{}".format(vehicle_num)][1][-1]
        if (
            math.isnan(compass) == True
        ):  # It can happen that the compass sends nan for a few frames
            compass = 0.0

        # compute lidar pose in world coordinate
        cur_lidar_pose = carla.Location(x=self.lidar_pose.location.x,
                                            y=self.lidar_pose.location.y,
                                            z=self.lidar_pose.location.z)
        _vehicle.get_transform().transform(cur_lidar_pose)

        # compute camera intrinsic and extrinsic params
        self.intrinsics = get_camera_intrinsic(self._rgb_sensor_data)
        self.lidar2front = get_camera_extrinsic(self.camera_front_pose, self.lidar_pose)
        self.lidar2left = get_camera_extrinsic(self.camera_left_pose, self.lidar_pose)
        self.lidar2right = get_camera_extrinsic(self.camera_right_pose, self.lidar_pose)
        self.lidar2rear = get_camera_extrinsic(self.camera_rear_pose, self.lidar_pose)

        # target waypoint produced by global planner
        pos = self._get_position({"gps": gps})
        next_wp, next_cmd = self._route_planner.run_step(pos, vehicle_num)
        theta = compass + np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
        local_command_point = R.T.dot(local_command_point)
        local_command_point = np.clip(local_command_point, a_min=[-self.config.detection_range[1], -self.config.detection_range[0]],
                                                                     a_max=[self.config.detection_range[2], self.config.detection_range[3]])

        # record measurements
        mes = {
            "gps_x": pos[0],
            "gps_y": pos[1],
            "x": pos[1],
            "y": -pos[0],
            "theta": compass,
            "lidar_pose_x": cur_lidar_pose.x,
            "lidar_pose_y": cur_lidar_pose.y,
            "lidar_pose_z": cur_lidar_pose.z,
            "lidar_pose_gps_x": -cur_lidar_pose.y,
            "lidar_pose_gps_y": cur_lidar_pose.x,
            "camera_front_intrinsics": self.intrinsics,
            "camera_front_extrinsics": self.lidar2front,
            "camera_left_intrinsics": self.intrinsics,
            "camera_left_extrinsics": self.lidar2left,
            "camera_right_intrinsics": self.intrinsics,
            "camera_right_extrinsics": self.lidar2right,
            "camera_rear_intrinsics": self.intrinsics,
            "camera_rear_extrinsics": self.lidar2rear,
            "speed": speed,
            "compass": compass,
            "command": next_cmd.value,
            "target_point": local_command_point
        }

        # return pre-loaded sensor data
        result = {
            "rgb_front": rgb_front,
            "rgb_left": rgb_left,
            "rgb_right": rgb_right,
            "rgb_rear": rgb_rear,
            "lidar": lidar,
            "measurements": mes,
            "bev": bev_image,
            "drivable_area": drivable_area
        }
        return result

    def spawn_rsu(self):
        """
        spawn rsu and delete old ones
        """
        if self.step % self.config.change_rsu_frame == 0:
            if self.first_generate_rsu:
                self.first_generate_rsu = False
            else:
                # destroy previous rsu sensor except the first entry
                [self.rsu[vehicle_num].cleanup() for vehicle_num in range(self.ego_vehicles_num)]
                self.rsu=[]
            # handle the initialization issue
            for vehicle_num in range(self.ego_vehicles_num):
                # generate rsu
                vehicle = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
                # spawn rsu to capture data without saving
                self.rsu.append(RoadSideUnit(save_path=None,
                                            id=int((self.step/self.config.change_rsu_frame+1)*1000+vehicle_num),
                                            is_None=(vehicle is None)))
                if vehicle is not None:
                    spawn_loc = get_rsu_point(vehicle, height=self.config.rsu_height, lane_side=self.config.rsu_lane_side, distance=self.config.rsu_distance)
                    self.rsu[vehicle_num].setup_sensors(parent=None, spawn_point=spawn_loc)
        return
    
    @torch.no_grad()
    def run_step(self, input_data: dict, timestamp):
        """
        Execute one step of navigation.
        Args:
            input_data: raw sensor data from ego vehicle
        Returns: 
            control_all: control_all[i] represents control signal of the ith vehicle
        """

        # initialization before starting
        if not self.initialized:
            self._init()
        self.step += 1

        # spawn the rsu.
        self.spawn_rsu()
        rsu_data = []

        # capture a list of sensor data from rsu
        if self.step % self.config.change_rsu_frame != 0:
            for vehicle_num in range(self.ego_vehicles_num):
                if CarlaDataProvider.get_hero_actor(hero_id=vehicle_num) is not None:
                    rsu_data.append(
                        self.rsu[vehicle_num].process(self.rsu[vehicle_num].tick(),is_train=True)
                    )
                else:
                    rsu_data.append(None)
        
        # If the frame is skipped.
        if self.step % self.config.skip_frames != 0 and self.step > self.config.skip_frames:
            # return the previous control signal.   
            return self.infer.prev_control

        control_all = []

        # capture a list of sensor data from ego vehicles 
        ego_data = [
            self.tick(input_data, vehicle_num)
            if CarlaDataProvider.get_hero_actor(hero_id=vehicle_num) is not None
            else None
            for vehicle_num in range(self.ego_vehicles_num)
        ]
        
        # infer control signal from sensor data
        control_all = self.infer.get_action_from_list_inter(car_data_raw=ego_data,
                                                        rsu_data_raw=rsu_data,
                                                        step=self.step,
                                                        timestamp=timestamp)

        ### return the control signal in list format.
        return control_all

    def destroy(self):
        del self.perception_model
        del self.planning_model

    def clean_rsu_single(self,vehicle_num):
        self.rsu[vehicle_num].cleanup()
        return
      
    def clean_rsu(self):
        [self.clean_rsu_single(vehicle_num) for vehicle_num in range(self.ego_vehicles_num)]
        return