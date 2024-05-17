import time
import os
import datetime
import pathlib
import json
import yaml
from collections import deque
import math
import cv2
import carla
import copy
from leaderboard.autoagents import autonomous_agent
from team_code.planner import RoutePlanner
from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions
from eval_utils import get_yaw_angle, boxes_to_corners_3d, get_points_in_rotated_box_3d, process_lidar_visibility
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from team_code.utils.map_drawing import \
    cv2_subpixel, draw_agent, draw_road, \
    draw_lane, road_exclude, draw_crosswalks, draw_city_objects

from team_code.utils.map_utils import \
    world_to_sensor, lateral_shift, list_loc2array, list_wpt2array, \
    convert_tl_status, exclude_off_road_agents, retrieve_city_object_info, \
    obj_in_range

import numpy as np
from PIL import Image, ImageDraw


SAVE_PATH = os.environ.get("SAVE_PATH", None)

WEATHERS = {
    "ClearNoon": carla.WeatherParameters.ClearNoon,
    "ClearSunset": carla.WeatherParameters.ClearSunset,
    "CloudyNoon": carla.WeatherParameters.CloudyNoon,
    "CloudySunset": carla.WeatherParameters.CloudySunset,
    "WetNoon": carla.WeatherParameters.WetNoon,
    "WetSunset": carla.WeatherParameters.WetSunset,
    "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
    "MidRainSunset": carla.WeatherParameters.MidRainSunset,
    "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
    "WetCloudySunset": carla.WeatherParameters.WetCloudySunset,
    "HardRainNoon": carla.WeatherParameters.HardRainNoon,
    "HardRainSunset": carla.WeatherParameters.HardRainSunset,
    "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
    "SoftRainSunset": carla.WeatherParameters.SoftRainSunset,
    "ClearNight": carla.WeatherParameters(5.0,0.0,0.0,10.0,-1.0,-90.0,60.0,75.0,1.0,0.0),
    "CloudyNight": carla.WeatherParameters(60.0,0.0,0.0,10.0,-1.0,-90.0,60.0,0.75,0.1,0.0),
    "WetNight": carla.WeatherParameters(5.0,0.0,50.0,10.0,-1.0,-90.0,60.0,75.0,1.0,60.0),
    "WetCloudyNight": carla.WeatherParameters(60.0,0.0,50.0,10.0,-1.0,-90.0,60.0,0.75,0.1,60.0),
    "SoftRainNight": carla.WeatherParameters(60.0,30.0,50.0,30.0,-1.0,-90.0,60.0,0.75,0.1,60.0),
    "MidRainyNight": carla.WeatherParameters(80.0,60.0,60.0,60.0,-1.0,-90.0,60.0,0.75,0.1,80.0),
    "HardRainNight": carla.WeatherParameters(100.0,100.0,90.0,100.0,-1.0,-90.0,100.0,0.75,0.1,100.0),
}
WEATHERS_IDS = list(WEATHERS)

TPE = {
    carla.CityObjectLabel.Buildings: "Building", 
    carla.CityObjectLabel.Vegetation: "Vegetation", 
    carla.CityObjectLabel.Poles: "Poles",
    carla.CityObjectLabel.Pedestrians: "Pedestrian",
    carla.CityObjectLabel.TrafficSigns: "TrafficSign",
    carla.CityObjectLabel.TrafficLight: "TrafficLight",
    carla.CityObjectLabel.Static: "Static",
    carla.CityObjectLabel.Fences: "Fence",
    carla.CityObjectLabel.Walls: "Wall"
}

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


class BaseAgent(autonomous_agent.AutonomousAgent):
    """
    AutonomousAgent -> BaseAgent
    This class mainly define:
        the format and path for the data to be saved during data collection
        sensor parameters
        sensor data pre-process before saving
    Args:
        self.config: dict()
        self.step: int, simulation step
        self._rgb_sensor_data: dict(), rgb data parameters
        self.weather_id: int, weather parameters
        self.save_skip_frames: int (default: 4), simulation_frequency/data_saving_frequency
        self.change_rsu_frame: int (default: 100), simulation_frequency/changing_rsu_position_frequency
        self.ego_vehicles_num: int, numbers of communicating vehicles


    """
    def setup(self, path_to_conf_file: str, ego_vehicles_num: int):
        """
        Initialize everything needed by your agent and set the track attribute to the right type:
            Track.SENSORS : CAMERAS, LIDAR, RADAR, GPS and IMU sensors are allowed
            Track.MAP : OpenDRIVE map is also allowed
        """
        self.track = autonomous_agent.Track.SENSORS

        # load config file, default type for data collection: yaml
        if path_to_conf_file.endswith("yaml"):
            self.config = yaml.load(open(path_to_conf_file, "r"), Loader=yaml.FullLoader)
        else:
            self.config = {}
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        # load key params related to sensor data saving
        self._rgb_sensor_data = {"width": 800, "height": 600, "fov": 100}
        self._sensor_data = self._rgb_sensor_data.copy()
        self._3d_bb_distance = 50
        self.weather_id = self.config.get("weather", None)
        self.waypoint_disturb = self.config.get("waypoint_disturb", 0)
        self.waypoint_disturb_seed = self.config.get("waypoint_disturb_seed", 2021)
        self.rgb_only = self.config.get("rgb_only", True)
        self.ego_vehicles_num = ego_vehicles_num

        # initialize and create data saving directory
        self.save_path = None
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ["ROUTES"]).stem + "_"

            if self.weather_id is None:
                string += "_".join(
                    map(
                        lambda x: "%02d" % x,
                        (now.month, now.day, now.hour, now.minute, now.second),
                    )
                )
            else:
                weather = WEATHERS[WEATHERS_IDS[self.weather_id]]
                string += "w%d_" % self.weather_id + "_".join(
                    map(
                        lambda x: "%02d" % x,
                        (now.month, now.day, now.hour, now.minute, now.second),
                    )
                )
            print('data saving path:',string)
            self.save_path = pathlib.Path(os.environ["SAVE_PATH"]) / string
            self.save_path.mkdir(parents=True, exist_ok=False)

            for i in range(self.ego_vehicles_num):
                self.save_path_tmp =self.save_path / pathlib.Path("ego_vehicle_{}".format(i))
                for sensor in self.sensors():
                    if hasattr(sensor, "save") and sensor["save"]:
                        (self.save_path / sensor["id"]).mkdir()

                (self.save_path_tmp / "3d_bbs").mkdir(parents=True, exist_ok=True)
                (self.save_path_tmp / "affordances").mkdir(parents=True, exist_ok=True)
                (self.save_path_tmp / "measurements").mkdir(parents=True, exist_ok=True)
                (self.save_path_tmp / "actors_data").mkdir(parents=True, exist_ok=True)
                (self.save_path_tmp / "env_actors_data").mkdir(parents=True, exist_ok=True)
                (self.save_path_tmp / "lidar").mkdir(parents=True, exist_ok=True)
                (self.save_path_tmp / "topdown").mkdir(parents=True, exist_ok=True)
                (self.save_path_tmp / "birdview").mkdir(parents=True, exist_ok=True)
                (self.save_path_tmp / "bev_visibility").mkdir(parents=True, exist_ok=True)

                for pos in ["front"]:
                    for sensor_type in ["rgb", "seg", "depth", "2d_bbs", "lidar_semantic"]:
                        name = sensor_type + "_" + pos
                        (self.save_path_tmp / name).mkdir()

                for pos in ["left", "right"]:
                    for sensor_type in ["rgb", "seg", "depth", "2d_bbs"]:
                        name = sensor_type + "_" + pos
                        (self.save_path_tmp / name).mkdir()

                for pos in ["rear"]:
                    for sensor_type in ["rgb"]:
                        name = sensor_type + "_" + pos
                        (self.save_path_tmp / name).mkdir()

    def _init(self):

        # initialize route planner
        self._command_planner = RoutePlanner(7.5, 25.0, 257)
        self._command_planner.set_route(self._global_plan, True)

        # agent only initialize once
        self.initialized = True

        self._sensor_data["calibration"] = self._get_camera_to_car_calibration(
            self._sensor_data
        )
        self._sensors = self.sensor_interface._sensors_objects
        
        # start from vehicle 0
        self.vehicle_num = 0

    def _get_position(self, tick_data, vehicle_num):
        gps = tick_data["gps_{}".format(self.vehicle_num)]

        '''
        self.mean = np.array([0.0, 0.0])  # for carla 9.10
        self.scale = np.array([111324.60662786, 111319.490945])  # for carla 9.10
        '''
        
        gps = (gps - self._command_planner.mean) * self._command_planner.scale

        return gps

    def sensors(self):
        self.lidar_pose = carla.Transform(carla.Location(x=1.3,y=0.0,z=1.85),  # z=2.5 z=1.85
                                        carla.Rotation(roll=0.0,pitch=0.0,yaw=-90.0))
        self.camera_front_pose = carla.Transform(carla.Location(x=1.3,y=0.0,z=2.3),
                                        carla.Rotation(roll=0.0,pitch=0.0,yaw=0.0))
        self.camera_rear_pose = carla.Transform(carla.Location(x=-1.3,y=0.0,z=2.3),
                                        carla.Rotation(roll=0.0,pitch=0.0,yaw=180.0))
        self.camera_left_pose = carla.Transform(carla.Location(x=1.3,y=0.0,z=2.3),
                                        carla.Rotation(roll=0.0,pitch=0.0,yaw=-60.0))
        self.camera_right_pose = carla.Transform(carla.Location(x=1.3,y=0.0,z=2.3),
                                        carla.Rotation(roll=0.0,pitch=0.0,yaw=60.0))
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
                "type": "sensor.camera.semantic_segmentation",
                "x": self.camera_front_pose.location.x,
                "y": self.camera_front_pose.location.y,
                "z": self.camera_front_pose.location.z,
                "roll": self.camera_front_pose.rotation.roll,
                "pitch": self.camera_front_pose.rotation.pitch,
                "yaw": self.camera_front_pose.rotation.yaw,
                "width": self._sensor_data["width"],
                "height": self._sensor_data["height"],
                "fov": self._sensor_data["fov"],
                "id": "seg_front",
            },
            {
                "type": "sensor.camera.depth",
                "x": self.camera_front_pose.location.x,
                "y": self.camera_front_pose.location.y,
                "z": self.camera_front_pose.location.z,
                "roll": self.camera_front_pose.rotation.roll,
                "pitch": self.camera_front_pose.rotation.pitch,
                "yaw": self.camera_front_pose.rotation.yaw,
                "width": self._sensor_data["width"],
                "height": self._sensor_data["height"],
                "fov": self._sensor_data["fov"],
                "id": "depth_front",
            },
            {
                "type": "sensor.lidar.ray_cast_semantic",
                "x": self.camera_front_pose.location.x,
                "y": self.camera_front_pose.location.y,
                "z": self.camera_front_pose.location.z,
                "roll": self.camera_front_pose.rotation.roll,
                "pitch": self.camera_front_pose.rotation.pitch,
                "yaw": self.camera_front_pose.rotation.yaw,
                "id": "lidar_semantic_front",
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
                "type": "sensor.camera.semantic_segmentation",
                "x": self.camera_rear_pose.location.x,
                "y": self.camera_rear_pose.location.y,
                "z": self.camera_rear_pose.location.z,
                "roll": self.camera_rear_pose.rotation.roll,
                "pitch": self.camera_rear_pose.rotation.pitch,
                "yaw": self.camera_rear_pose.rotation.yaw,
                "width": self._sensor_data["width"],
                "height": self._sensor_data["height"],
                "fov": self._sensor_data["fov"],
                "id": "seg_rear",
            },
            {
                "type": "sensor.camera.depth",
                "x": self.camera_rear_pose.location.x,
                "y": self.camera_rear_pose.location.y,
                "z": self.camera_rear_pose.location.z,
                "roll": self.camera_rear_pose.rotation.roll,
                "pitch": self.camera_rear_pose.rotation.pitch,
                "yaw": self.camera_rear_pose.rotation.yaw,
                "width": self._sensor_data["width"],
                "height": self._sensor_data["height"],
                "fov": self._sensor_data["fov"],
                "id": "depth_rear",
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
                "type": "sensor.camera.semantic_segmentation",
                "x": self.camera_left_pose.location.x,
                "y": self.camera_left_pose.location.y,
                "z": self.camera_left_pose.location.z,
                "roll": self.camera_left_pose.rotation.roll,
                "pitch": self.camera_left_pose.rotation.pitch,
                "yaw": self.camera_left_pose.rotation.yaw,
                "width": self._sensor_data["width"],
                "height": self._sensor_data["height"],
                "fov": self._sensor_data["fov"],
                "id": "seg_left",
            },
            {
                "type": "sensor.camera.depth",
                "x": self.camera_left_pose.location.x,
                "y": self.camera_left_pose.location.y,
                "z": self.camera_left_pose.location.z,
                "roll": self.camera_left_pose.rotation.roll,
                "pitch": self.camera_left_pose.rotation.pitch,
                "yaw": self.camera_left_pose.rotation.yaw,
                "width": self._sensor_data["width"],
                "height": self._sensor_data["height"],
                "fov": self._sensor_data["fov"],
                "id": "depth_left",
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
                "type": "sensor.camera.semantic_segmentation",
                "x": self.camera_right_pose.location.x,
                "y": self.camera_right_pose.location.y,
                "z": self.camera_right_pose.location.z,
                "roll": self.camera_right_pose.rotation.roll,
                "pitch": self.camera_right_pose.rotation.pitch,
                "yaw": self.camera_right_pose.rotation.yaw,
                "width": self._sensor_data["width"],
                "height": self._sensor_data["height"],
                "fov": self._sensor_data["fov"],
                "id": "seg_right",
            },
            {
                "type": "sensor.camera.depth",
                "x": self.camera_right_pose.location.x,
                "y": self.camera_right_pose.location.y,
                "z": self.camera_right_pose.location.z,
                "roll": self.camera_right_pose.rotation.roll,
                "pitch": self.camera_right_pose.rotation.pitch,
                "yaw": self.camera_right_pose.rotation.yaw,
                "width": self._sensor_data["width"],
                "height": self._sensor_data["height"],
                "fov": self._sensor_data["fov"],
                "id": "depth_right",
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
        if self.rgb_only:
            sensors_list = [
                x
                for x in sensors_list
                if x["type"]
                in [
                    "sensor.camera.rgb",
                    "sensor.other.imu",
                    "sensor.other.gnss",
                    "sensor.speedometer",
                ]
            ]
        return sensors_list

    def tick(self, input_data, vehicle_num):

        if not self.rgb_only:
            affordances = self._get_affordances()

            traffic_lights = self._find_obstacle("*traffic_light*")
            stop_signs = self._find_obstacle("*stop*")

            depth = {}
            seg = {}

            bb_3d = self._get_3d_bbs(max_distance=self._3d_bb_distance)

            bb_2d = {}

            for pos in ["front", "left", "right"]:
                seg_cam = "seg_" + pos + "_{}".format(self.vehicle_num)
                depth_cam = "depth_" + pos + "_{}".format(self.vehicle_num)
                _segmentation = np.copy(input_data[seg_cam][1][:, :, 2])

                depth[pos] = self._get_depth(input_data[depth_cam][1][:, :, :3])
                self._change_seg_tl(_segmentation, depth[pos], traffic_lights)
                self._change_seg_stop(_segmentation, depth[pos], stop_signs, seg_cam)

                bb_2d[pos] = self._get_2d_bbs(
                    seg_cam, affordances, bb_3d, _segmentation
                )

                seg[pos] = _segmentation

            depth_front = cv2.cvtColor(
                input_data["depth_front_{}".format(self.vehicle_num)][1][:, :, :3], cv2.COLOR_BGR2RGB
            )
            depth_left = cv2.cvtColor(
                input_data["depth_left_{}".format(self.vehicle_num)][1][:, :, :3], cv2.COLOR_BGR2RGB
            )
            depth_right = cv2.cvtColor(
                input_data["depth_right_{}".format(self.vehicle_num)][1][:, :, :3], cv2.COLOR_BGR2RGB
            )

        rgb_front = cv2.cvtColor(
            input_data["rgb_front_{}".format(self.vehicle_num)][1][:, :, :3], cv2.COLOR_BGR2RGB
        )
        rgb_rear = cv2.cvtColor(input_data["rgb_rear_{}".format(self.vehicle_num)][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data["rgb_left_{}".format(self.vehicle_num)][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(
            input_data["rgb_right_{}".format(self.vehicle_num)][1][:, :, :3], cv2.COLOR_BGR2RGB
        )
        gps = input_data["gps_{}".format(self.vehicle_num)][1][:2]
        move_state = input_data["speed_{}".format(self.vehicle_num)][1]["move_state"]
        compass = input_data["imu_{}".format(self.vehicle_num)][1][-1]
        imu = input_data["imu_{}".format(self.vehicle_num)][1][:]

        weather = self._weather_to_dict(self._world.get_weather())

        if self.rgb_only:
            return {
                "rgb_front_{}".format(self.vehicle_num): rgb_front,
                "rgb_rear_{}".format(self.vehicle_num): rgb_rear,
                "rgb_left_{}".format(self.vehicle_num): rgb_left,
                "rgb_right_{}".format(self.vehicle_num): rgb_right,
                "gps_{}".format(self.vehicle_num): gps,
                "move_state_{}".format(self.vehicle_num): move_state,
                "compass_{}".format(self.vehicle_num): compass,
                "weather": weather,
            }
        else:
            return {
                "rgb_front_{}".format(self.vehicle_num): rgb_front,
                "seg_front_{}".format(self.vehicle_num): seg["front"],
                "depth_front_{}".format(self.vehicle_num): depth_front,
                "2d_bbs_front_{}".format(self.vehicle_num): bb_2d["front"],
                "rgb_rear_{}".format(self.vehicle_num): rgb_rear,
                "rgb_left_{}".format(self.vehicle_num): rgb_left,
                "seg_left_{}".format(self.vehicle_num): seg["left"],
                "depth_left_{}".format(self.vehicle_num): depth_left,
                "2d_bbs_left_{}".format(self.vehicle_num): bb_2d["left"],
                "rgb_right_{}".format(self.vehicle_num): rgb_right,
                "seg_right_{}".format(self.vehicle_num): seg["right"],
                "depth_right_{}".format(self.vehicle_num): depth_right,
                "2d_bbs_right_{}".format(self.vehicle_num): bb_2d["right"],
                "lidar_{}".format(self.vehicle_num): input_data["lidar_{}".format(self.vehicle_num)][1],
                "lidar_semantic_front_{}".format(self.vehicle_num): input_data["lidar_semantic_front_{}".format(self.vehicle_num)][1],
                "gps_{}".format(self.vehicle_num): gps,
                "move_state_{}".format(self.vehicle_num): move_state,
                "compass_{}".format(self.vehicle_num): compass,
                "imu_{}".format(self.vehicle_num): imu,
                "weather": weather,
                "affordances_{}".format(self.vehicle_num): affordances,
                "3d_bbs_{}".format(self.vehicle_num): bb_3d,
            }

    def save(
        self,
        near_node,
        far_node,
        near_command,
        steer,
        throttle,
        brake,
        target_speed,
        tick_data,
        mode='all'
    ):
        """
        save sensor data
        """
        frame = self.step  // self.save_skip_frames

        # save measurements of ego vehicle
        ego_pos = self._get_position(tick_data, self.vehicle_num)
        theta = tick_data["compass_{}".format(self.vehicle_num)]
        imu = {} 
        imu['accelerometer_x'] = tick_data["imu_{}".format(self.vehicle_num)][0]
        imu['accelerometer_y'] = tick_data["imu_{}".format(self.vehicle_num)][1]
        imu['accelerometer_z'] = tick_data["imu_{}".format(self.vehicle_num)][2]
        imu['gyroscope_x'] = tick_data["imu_{}".format(self.vehicle_num)][3]
        imu['gyroscope_y'] = tick_data["imu_{}".format(self.vehicle_num)][4]
        imu['gyroscope_z'] = tick_data["imu_{}".format(self.vehicle_num)][5]
        imu['compass'] = tick_data["imu_{}".format(self.vehicle_num)][6]
        move_state = tick_data["move_state_{}".format(self.vehicle_num)]
        weather = tick_data["weather"]
        cur_lidar_pose = carla.Location(x=self.lidar_pose.location.x,
                                            y=self.lidar_pose.location.y,
                                            z=self.lidar_pose.location.z)
        # transform pose from ego coordinate to world coordinate
        self._vehicle.get_transform().transform(cur_lidar_pose)
        self.cur_camera_front_pose = carla.Location(x=self.camera_front_pose.location.x,
                                            y=self.camera_front_pose.location.y,
                                            z=self.camera_front_pose.location.z)        
        self._vehicle.get_transform().transform(self.cur_camera_front_pose)
        if int(frame) == 0:
            self.intrinsics = get_camera_intrinsic(self._rgb_sensor_data)
            self.lidar2front = get_camera_extrinsic(self.camera_front_pose, self.lidar_pose)
            self.lidar2left = get_camera_extrinsic(self.camera_left_pose, self.lidar_pose)
            self.lidar2right = get_camera_extrinsic(self.camera_right_pose, self.lidar_pose)
            self.lidar2rear = get_camera_extrinsic(self.camera_rear_pose, self.lidar_pose)
        data = {
            "gps_x": ego_pos[0],
            "gps_y": ego_pos[1],
            "x": self._loc[0],
            "y": self._loc[1],
            "theta": theta,
            "imu": imu,
            "lidar_pose_x": cur_lidar_pose.x,
            "lidar_pose_y": cur_lidar_pose.y,
            "lidar_pose_z": cur_lidar_pose.z,
            "lidar_pose_gps_x": -cur_lidar_pose.y,
            "lidar_pose_gps_y": cur_lidar_pose.x,
            "move_state": move_state,
            "target_speed": target_speed,
            "x_command": far_node[0],
            "y_command": far_node[1],
            "command": near_command.value,
            "gt_command": self._command_planner.route[self.vehicle_num][0][1].value,
            "steer": steer,
            "throttle": throttle,
            "brake": brake,
            "near_node_x": near_node[0],
            "near_node_y": near_node[1],
            "far_node_x": far_node[0],
            "far_node_y": far_node[1],
            "is_junction": self.is_junction,
            "is_vehicle_present": self.is_vehicle_present,
            "is_bike_present": self.is_bike_present,
            "is_lane_vehicle_present": self.is_lane_vehicle_present,
            "is_junction_vehicle_present": self.is_junction_vehicle_present,
            "is_pedestrian_present": self.is_pedestrian_present,
            "is_red_light_present": self.is_red_light_present,
            "is_stop_sign_present": self.is_stop_sign_present,
            "should_slow": int(self.should_slow),
            "should_brake": int(self.should_brake),
            "future_waypoints": self._waypoint_planner.get_future_waypoints(self.vehicle_num,50),
            "camera_front_intrinsics": self.intrinsics,
            "camera_front_extrinsics": self.lidar2front,
            "camera_left_intrinsics": self.intrinsics,
            "camera_left_extrinsics": self.lidar2left,
            "camera_right_intrinsics": self.intrinsics,
            "camera_right_extrinsics": self.lidar2right,
            "camera_rear_intrinsics": self.intrinsics,
            "camera_rear_extrinsics": self.lidar2rear,
            "weather": weather,
            "weather_id": self.weather_id,
            "affected_light_id": self.affected_light_id,
        }
        self.save_path_tmp = self.save_path / pathlib.Path("ego_vehicle_{}".format(self.vehicle_num))
        measurements_file = self.save_path_tmp / "measurements" / ("%04d.json" % frame)
        f = open(measurements_file, "w")
        json.dump(data, f, indent=4)
        f.close()

        # other data
        if mode == 'all':
            # collect actors data from carla world
            self.actors_data = self.collect_actor_data()
            lidar_pose = {"lidar_pose_x": data["lidar_pose_x"],
                          "lidar_pose_y": data["lidar_pose_y"],
                          "lidar_pose_z": data["lidar_pose_z"],
                          "theta": data["theta"]}
            # add lidar visibility information for actors data
            self.actors_data, _ = process_lidar_visibility(self.actors_data, tick_data["lidar" + "_{}".format(self.vehicle_num)], lidar_pose, change_actor_file=True)
            actors_data_file = self.save_path_tmp / "actors_data" / ("%04d.json" % frame)
            f = open(actors_data_file, "w")
            json.dump(self.actors_data, f, indent=4)
            f.close()

            # collect static environment actors
            self.env_actors_data = self.collect_env_actor_data()
            actors_data_file = self.save_path_tmp / "env_actors_data" / ("%04d.json" % frame)
            f = open(actors_data_file, "w")
            json.dump(self.env_actors_data, f, indent=4)
            f.close()

            # BEV map
            self.birdview = BirdViewProducer.as_rgb(
            self.birdview_producer.produce(agent_vehicle=self._vehicle )
            )
            Image.fromarray(self.birdview).save(
                self.save_path_tmp / "birdview" / ("%04d.jpg" % frame)
            )

            # camera rgb, depth, segmentation, 2d bounding box
            for pos in ["front", "left", "right", "rear"]:
                name = "rgb_" + pos + "_{}".format(self.vehicle_num)
                name_save = "rgb_" + pos
                Image.fromarray(tick_data[name]).save(
                    self.save_path_tmp / name_save / ("%04d.jpg" % frame)
                )
                if not self.rgb_only and pos != "rear":
                    for sensor_type in ["depth", "seg"]:  # , 
                        name = sensor_type + "_" + pos + "_{}".format(self.vehicle_num)
                        name_save = sensor_type + "_" + pos
                        Image.fromarray(tick_data[name]).save(
                            self.save_path_tmp / name_save / ("%04d.jpg" % frame)
                        )
                    for sensor_type in ["2d_bbs"]:
                        name = sensor_type + "_" + pos + "_{}".format(self.vehicle_num)
                        name_save = sensor_type + "_" + pos
                        np.save(
                            self.save_path_tmp / name_save / ("%04d.npy" % frame),
                            tick_data[name],
                            allow_pickle=True,
                        )

            if not self.rgb_only:
                Image.fromarray(tick_data["topdown" + "_{}".format(self.vehicle_num)]).save(
                    self.save_path_tmp / "topdown" / ("%04d.jpg" % frame)
                )
                np.save(
                    self.save_path_tmp / "affordances" / ("%04d.npy" % frame),
                    tick_data["affordances" + "_{}".format(self.vehicle_num)],
                    allow_pickle=True,
                )
                np.save(
                    self.save_path_tmp / "lidar" / ("%04d.npy" % frame),
                    tick_data["lidar" + "_{}".format(self.vehicle_num)],
                    allow_pickle=True,
                )
                np.save(
                    self.save_path_tmp / "lidar_semantic_front" / ("%04d.npy" % frame),
                    tick_data["lidar_semantic_front" + "_{}".format(self.vehicle_num)],
                    allow_pickle=True,
                )

                # bev_visiblity for camera detection
                bev_map = self.sg_lidar_2_bevmap(tick_data)
                save_visibility_name = os.path.join(self.save_path_tmp,
                                                    'bev_visibility',
                                                    "%04d.png" % frame)
                cv2.imwrite(save_visibility_name, bev_map)

                # 3d bounding box
                np.save(
                    self.save_path_tmp / "3d_bbs" / ("%04d.npy" % frame),
                    tick_data["3d_bbs" + "_{}".format(self.vehicle_num)],
                    allow_pickle=True,
                )
        return frame

    def collect_actor_data(self):
        """
        collect actors data from carla world within a radius of 100 meters, 
        including car for type 0, walker for type 1, traffic light for type 2, bicycle for type 3
        """
        data = {}
        vehicles = self._world.get_actors().filter("*vehicle*")
        for actor in vehicles:
            loc = actor.get_location()
            if loc.distance(self._vehicle.get_location()) > 100:
                continue
            _id = actor.id
            data[_id] = {}
            data[_id]["loc"] = [loc.x, loc.y, loc.z]
            ori = actor.get_transform().rotation.get_forward_vector()
            data[_id]["ori"] = [ori.x, ori.y, ori.z]
            box = actor.bounding_box.extent
            data[_id]["box"] = [box.x, box.y, box.z]
            vel = actor.get_velocity()
            data[_id]["vel"] = [vel.x, vel.y, vel.z]
            angle = actor.get_transform().rotation
            data[_id]["angle"] = [angle.roll, angle.yaw, angle.pitch]
            if actor.type_id=="vehicle.diamondback.century":
                data[_id]["tpe"] = 3
            else:
                data[_id]["tpe"] = 0

        walkers = self._world.get_actors().filter("*walker.*")
        for actor in walkers:
            loc = actor.get_location()
            if loc.distance(self._vehicle.get_location()) > 100:
                continue
            _id = actor.id
            data[_id] = {}
            data[_id]["loc"] = [loc.x, loc.y, loc.z]
            ori = actor.get_transform().rotation.get_forward_vector()
            data[_id]["ori"] = [ori.x, ori.y, ori.z]
            box = actor.bounding_box.extent
            data[_id]["box"] = [box.x, box.y, box.z]
            vel = actor.get_velocity()
            data[_id]["vel"] = [vel.x, vel.y, vel.z]
            angle = actor.get_transform().rotation
            data[_id]["angle"] = [angle.roll, angle.yaw, angle.pitch]
            data[_id]["tpe"] = 1

            # walker's location is at the center of bounding box, not like other actors' location at the bottom,
            # therefore transform the location from the center to the bottom
            data[_id]["loc"][2] -= data[_id]["box"][2]

        lights = self._world.get_actors().filter("*traffic_light*")
        for actor in lights:
            loc = actor.get_location()
            if loc.distance(self._vehicle.get_location()) > 100:
                continue
            _id = actor.id
            data[_id] = {}
            data[_id]["loc"] = [loc.x, loc.y, loc.z]
            ori = actor.get_transform().rotation.get_forward_vector()
            data[_id]["ori"] = [ori.x, ori.y, ori.z]
            vel = actor.get_velocity()
            data[_id]["sta"] = int(actor.state)
            data[_id]["tpe"] = 2

            trigger = actor.trigger_volume
            box = trigger.extent
            loc = trigger.location
            ori = trigger.rotation.get_forward_vector()
            data[_id]["taigger_loc"] = [loc.x, loc.y, loc.z]
            data[_id]["trigger_ori"] = [ori.x, ori.y, ori.z]
            data[_id]["trigger_box"] = [box.x, box.y, box.z]
        return data
    
    def collect_env_actor_data(self):
        """
        collect static environment actors data from carla world within a radius of 100 meters
        """
        data = {}
        _id = 0
        for actor_type in list(TPE.keys()):
            actors = CarlaDataProvider.get_world().get_level_bbs(actor_type=actor_type)
            for actor in actors:
                loc = actor.location
                if loc.distance(self._vehicle.get_location()) > 100:
                    continue
                _id += 1
                data[_id] = {}
                data[_id]["loc"] = [loc.x, loc.y, loc.z]
                ori = actor.rotation.get_forward_vector()
                data[_id]["ori"] = [ori.x, ori.y, ori.z]
                box = actor.extent
                data[_id]["box"] = [box.x, box.y, box.z]
                data[_id]["tpe"] = TPE[actor_type]
        return data

    def sg_lidar_2_bevmap(self, tick_data):
        config = {'thresh':5,
                  'radius_meter':50,
                  'raster_size':[256, 256]
                  }

        # data = np.frombuffer(semantic_lidar, dtype=np.dtype([
        #     ('x', np.float32), ('y', np.float32), ('z', np.float32),
        #     ('CosAngle', np.float32), ('ObjIdx', np.uint32),
        #     ('ObjTag', np.uint32)]))

        # # (x, y, z, intensity)
        # points = np.array([data['x'], data['y'], data['z']]).T
        # obj_tag = np.array(data['ObjTag'])
        # obj_idx = np.array(data['ObjIdx'])
        # thresh = config['thresh']
        # # self.data = data
        # # self.frame = event.frame
        # # self.timestamp = event.timestamp

        # while obj_idx is None or obj_tag is None or \
        #         obj_idx.shape[0] != obj_tag.shape[0]:
        #     continue

        # # label 10 is the vehicle
        # vehicle_idx = obj_idx[obj_tag == 10]
        # # each individual instance id
        # vehicle_unique_id = list(np.unique(vehicle_idx))
        # vehicle_id_filter = []

        # for veh_id in vehicle_unique_id:
        #     if vehicle_idx[vehicle_idx == veh_id].shape[0] > thresh:
        #         vehicle_id_filter.append(veh_id)


        lidar_pose = {"lidar_pose_x": self.cur_camera_front_pose.x,
                        "lidar_pose_y": self.cur_camera_front_pose.y,
                        "lidar_pose_z": self.cur_camera_front_pose.z,
                        "theta": (self.camera_front_pose.rotation.yaw + 90)/180 * np.pi + tick_data["compass_{}".format(self.vehicle_num)]}
        _, vehicle_id_filter = process_lidar_visibility(self.actors_data, tick_data["lidar_semantic_front" + "_{}".format(self.vehicle_num)], lidar_pose, change_actor_file=True, mode='camera', thresh = 5)


        self.radius_meter = config['radius_meter']
        self.raster_size = np.array(config['raster_size'])

        self.pixels_per_meter = self.raster_size[0] / (self.radius_meter * 2)

        meter_per_pixel = 1 / self.pixels_per_meter
        raster_radius = \
            float(np.linalg.norm(self.raster_size *
                                 np.array([meter_per_pixel,
                                           meter_per_pixel]))) / 2
        dynamic_agents = self.load_agents_world(raster_radius)
        final_agents = dynamic_agents

        # final_agents = agents_in_range(raster_radius,
        #                                     dynamic_agents)
        
        corner_list = []
        for agent_id, agent in final_agents.items():
            # in case we don't want to draw the cav itself
            if agent_id == self._vehicle.id:
                continue
            if False: # not agent_id in vehicle_id_filter:
                continue
            agent_corner = self.generate_agent_area(agent['corners'])
            corner_list.append(agent_corner)

        self.vis_mask = 255 * np.zeros(
            shape=(self.raster_size[1], self.raster_size[0], 3),
            dtype=np.uint8)

        self.vis_mask = draw_agent(corner_list, self.vis_mask)

        return self.vis_mask

    def load_agents_world(self, max_distance=50):
        """
        Load all the dynamic agents info from server directly
        into a  dictionary.

        Returns
        -------
        The dictionary contains all agents info in the carla world.
        """

        vehicle_list = self._world.get_actors().filter('vehicle.*')
        walker_list = self._world.get_actors().filter("*walker.*")
        
        dynamic_agent_info = {}

        def read_actors_corner(agent_list, agent_info):
            for agent in agent_list:
                loc = agent.get_location()
                if loc.distance(self._vehicle.get_location()) > max_distance:
                    continue

                agent_id = agent.id

                agent_transform = agent.get_transform()

                type_id = agent.type_id
                if not 'walker' in type_id:
                    agent_loc = [agent_transform.location.x,
                                agent_transform.location.y,
                                agent_transform.location.z - agent.bounding_box.extent.z, ]
                else:
                    agent_loc = [agent_transform.location.x,
                                agent_transform.location.y,
                                agent_transform.location.z, ]                

                agent_yaw = agent_transform.rotation.yaw

                # calculate 4 corners
                bb = agent.bounding_box.extent
                corners = [
                    carla.Location(x=-bb.x, y=-bb.y),
                    carla.Location(x=-bb.x, y=bb.y),
                    carla.Location(x=bb.x, y=bb.y),
                    carla.Location(x=bb.x, y=-bb.y)
                ]
                # corners are originally in ego coordinate frame, convert to
                # world coordinate
                agent_transform.transform(corners)
                corners_reformat = [[x.x, x.y, x.z] for x in corners]

                agent_info[agent_id] = {'location': agent_loc,
                                                'yaw': agent_yaw,
                                                'corners': corners_reformat}
            return agent_info

        dynamic_agent_info = read_actors_corner(vehicle_list, dynamic_agent_info)
        dynamic_agent_info = read_actors_corner(walker_list, dynamic_agent_info)            

        return dynamic_agent_info
    
    def generate_agent_area(self, corners):
        """
        Convert the agent's bbx corners from world coordinates to
        rasterization coordinates.

        Parameters
        ----------
        corners : list
            The four corners of the agent's bbx under world coordinate.

        Returns
        -------
        agent four corners in image.
        """
        # (4, 3) numpy array
        corners = np.array(corners)
        # for homogeneous transformation
        corners = corners.T
        corners = np.r_[
            corners, [np.ones(corners.shape[1])]]
        # convert to ego's coordinate frame
        corners = world_to_sensor(corners, self._vehicle.get_transform()).T
        corners = corners[:, :2]

        # switch x and y
        corners = corners[..., ::-1]
        # y revert
        corners[:, 1] = -corners[:, 1]

        corners[:, 0] = corners[:, 0] * self.pixels_per_meter + \
                        self.raster_size[0] // 2
        corners[:, 1] = corners[:, 1] * self.pixels_per_meter + \
                        self.raster_size[1] // 2

        # to make more precise polygon
        corner_area = cv2_subpixel(corners[:, :2])

        return corner_area

    def _weather_to_dict(self, carla_weather):
        weather = {
            "cloudiness": carla_weather.cloudiness,
            "precipitation": carla_weather.precipitation,
            "precipitation_deposits": carla_weather.precipitation_deposits,
            "wind_intensity": carla_weather.wind_intensity,
            "sun_azimuth_angle": carla_weather.sun_azimuth_angle,
            "sun_altitude_angle": carla_weather.sun_altitude_angle,
            "fog_density": carla_weather.fog_density,
            "fog_distance": carla_weather.fog_distance,
            "wetness": carla_weather.wetness,
            "fog_falloff": carla_weather.fog_falloff,
        }

        return weather

    def _create_bb_points(self, bb):
        """
        Returns 3D bounding box world coordinates.
        """

        cords = np.zeros((8, 4))
        extent = bb[1]
        loc = bb[0]
        cords[0, :] = np.array(
            [loc[0] + extent[0], loc[1] + extent[1], loc[2] - extent[2], 1]
        )
        cords[1, :] = np.array(
            [loc[0] - extent[0], loc[1] + extent[1], loc[2] - extent[2], 1]
        )
        cords[2, :] = np.array(
            [loc[0] - extent[0], loc[1] - extent[1], loc[2] - extent[2], 1]
        )
        cords[3, :] = np.array(
            [loc[0] + extent[0], loc[1] - extent[1], loc[2] - extent[2], 1]
        )
        cords[4, :] = np.array(
            [loc[0] + extent[0], loc[1] + extent[1], loc[2] + extent[2], 1]
        )
        cords[5, :] = np.array(
            [loc[0] - extent[0], loc[1] + extent[1], loc[2] + extent[2], 1]
        )
        cords[6, :] = np.array(
            [loc[0] - extent[0], loc[1] - extent[1], loc[2] + extent[2], 1]
        )
        cords[7, :] = np.array(
            [loc[0] + extent[0], loc[1] - extent[1], loc[2] + extent[2], 1]
        )
        return cords

    def _translate_tl_state(self, state):

        if state == carla.TrafficLightState.Red:
            return 0
        elif state == carla.TrafficLightState.Yellow:
            return 1
        elif state == carla.TrafficLightState.Green:
            return 2
        elif state == carla.TrafficLightState.Off:
            return 3
        elif state == carla.TrafficLightState.Unknown:
            return 4
        else:
            return None

    def _get_affordances(self):

        # affordance tl
        affordances = {}
        affordances["traffic_light"] = None

        affecting = self._vehicle.get_traffic_light()
        if affecting is not None:
            for light in self._traffic_lights:
                if light.id == affecting.id:
                    affordances["traffic_light"] = self._translate_tl_state(
                        self._vehicle.get_traffic_light_state()
                    )

        affordances["stop_sign"] = self._affected_by_stop[self.vehicle_num]

        return affordances

    def _get_3d_bbs(self, max_distance=50):

        bounding_boxes = {
            "traffic_lights": [],
            "stop_signs": [],
            "vehicles": [],
            "pedestrians": [],
        }

        bounding_boxes["traffic_lights"] = self._find_obstacle_3dbb(
            "*traffic_light*", max_distance
        )
        bounding_boxes["stop_signs"] = self._find_obstacle_3dbb("*stop*", max_distance)
        bounding_boxes["vehicles"] = self._find_obstacle_3dbb("*vehicle*", max_distance)
        bounding_boxes["pedestrians"] = self._find_obstacle_3dbb(
            "*walker.*", max_distance
        )

        return bounding_boxes

    def _get_2d_bbs(self, seg_cam, affordances, bb_3d, seg_img):
        """Returns a dict of all 2d boundingboxes given a camera position, affordances and 3d bbs

        Args:
            seg_cam ([type]): [description]
            affordances ([type]): [description]
            bb_3d ([type]): [description]

        Returns:
            [type]: [description]
        """

        bounding_boxes = {
            "traffic_light": list(),
            "stop_sign": list(),
            "vehicles": list(),
            "pedestrians": list(),
        }

        if affordances["stop_sign"]:
            if self._target_stop_sign[self.vehicle_num] is not None:
                baseline = self._get_2d_bb_baseline(self._target_stop_sign[self.vehicle_num])
                bb = self._baseline_to_box(baseline, seg_cam)

                if bb is not None:
                    bounding_boxes["stop_sign"].append(bb)

        if affordances["traffic_light"] is not None:
            baseline = self._get_2d_bb_baseline(
                self._vehicle.get_traffic_light(), distance=8
            )

            tl_bb = self._baseline_to_box(baseline, seg_cam, height=0.5)

            if tl_bb is not None:
                bounding_boxes["traffic_light"].append(
                    {
                        "bb": tl_bb,
                        "state": self._translate_tl_state(
                            self._vehicle.get_traffic_light_state()
                        ),
                    }
                )

        for vehicle in bb_3d["vehicles"]:

            trig_loc_world = self._create_bb_points(vehicle).T
            cords_x_y_z = self._world_to_sensor(
                trig_loc_world, self._get_sensor_position(seg_cam), False
            )

            cords_x_y_z = np.array(cords_x_y_z)[:3, :]
            veh_bb = self._coords_to_2d_bb(cords_x_y_z)

            if veh_bb is not None:
                if np.any(
                    seg_img[veh_bb[0][1] : veh_bb[1][1], veh_bb[0][0] : veh_bb[1][0]]
                    == 10
                ):
                    bounding_boxes["vehicles"].append(veh_bb)

        for pedestrian in bb_3d["pedestrians"]:

            trig_loc_world = self._create_bb_points(pedestrian).T
            cords_x_y_z = self._world_to_sensor(
                trig_loc_world, self._get_sensor_position(seg_cam), False
            )

            cords_x_y_z = np.array(cords_x_y_z)[:3, :]

            ped_bb = self._coords_to_2d_bb(cords_x_y_z)

            if ped_bb is not None:
                if np.any(
                    seg_img[ped_bb[0][1] : ped_bb[1][1], ped_bb[0][0] : ped_bb[1][0]]
                    == 4
                ):
                    bounding_boxes["pedestrians"].append(ped_bb)

        return bounding_boxes

    def _draw_2d_bbs(self, seg_img, bbs):
        """For debugging only

        Args:
            seg_img ([type]): [description]
            bbs ([type]): [description]
        """

        for bb_type in bbs:

            _region = np.zeros(seg_img.shape)

            if bb_type == "traffic_light":
                for bb in bbs[bb_type]:
                    _region = np.zeros(seg_img.shape)
                    box = bb["bb"]
                    _region[box[0][1] : box[1][1], box[0][0] : box[1][0]] = 1
                seg_img[(_region == 1)] = 23
            else:

                for bb in bbs[bb_type]:

                    _region[bb[0][1] : bb[1][1], bb[0][0] : bb[1][0]] = 1

                if bb_type == "stop_sign":
                    seg_img[(_region == 1)] = 26
                elif bb_type == "vehicles":
                    seg_img[(_region == 1)] = 10
                elif bb_type == "pedestrians":
                    seg_img[(_region == 1)] = 4

    def _find_obstacle_3dbb(self, obstacle_type, max_distance=50):
        """Returns a list of 3d bounding boxes of type obstacle_type.
        If the object does have a bounding box, this is returned. Otherwise a bb
        of size 0.5,0.5,2 is returned at the origin of the object.

        Args:
            obstacle_type (String): Regular expression
            max_distance (int, optional): max search distance. Returns all bbs in this radius. Defaults to 50.

        Returns:
            List: List of Boundingboxes
        """
        obst = list()
        # TODO: Check it! Could conceal hidden
        if obstacle_type == "*traffic_light*" or obstacle_type == "*stop*":
            # Retrieve all bounding boxes for traffic lights within the level
            if obstacle_type == "*traffic_light*":
                bounding_box_set = self._world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
            else:
                bounding_box_set = self._world.get_level_bbs(carla.CityObjectLabel.TrafficSigns)
            # Filter the list to extract bounding boxes within a 50m radius
            for bbox in bounding_box_set:
                distance_to_car = bbox.location.distance(self._vehicle.get_location())
                if 0 < distance_to_car <= max_distance:
                    loc = bbox.location
                    ori = bbox.rotation
                    extent = bbox.extent
                    # _rotation_matrix = self.get_matrix(
                    #     carla.Transform(
                    #         carla.Location(0, 0, 0), bbox.rotation
                    #     )
                    # )

                    # rotated_extent = np.squeeze(
                    #     np.array(
                    #         (
                    #             np.array([[extent.x, extent.y, extent.z, 1]])
                    #             @ _rotation_matrix
                    #         )[:3]
                    #     )
                    # )
                    bb = np.array(
                        [
                            [loc.x, loc.y, loc.z],
                            [ori.roll, ori.pitch, ori.yaw],
                            [extent.x, extent.y, extent.z]
                            # [rotated_extent[0], rotated_extent[1], rotated_extent[2]],
                        ]
                    )

                    obst.append(bb)
        else: # original
            _actors = self._world.get_actors()
            _obstacles = _actors.filter(obstacle_type)

            for _obstacle in _obstacles:
                distance_to_car = _obstacle.get_transform().location.distance(
                    self._vehicle.get_location()
                )

                if 0 < distance_to_car <= max_distance:

                    if hasattr(_obstacle, "bounding_box"):
                        loc = _obstacle.get_location()
                        ori = _obstacle.get_transform().rotation.get_forward_vector()
                        extent = _obstacle.bounding_box.extent
                        # loc = _obstacle.bounding_box.location
                        # _obstacle.get_transform().transform(loc)
                        # _rotation_matrix = self.get_matrix(
                        #     carla.Transform(
                        #         carla.Location(0, 0, 0), _obstacle.get_transform().rotation
                        #     )
                        # )

                        # rotated_extent = np.squeeze(
                        #     np.array(
                        #         (
                        #             np.array([[extent.x, extent.y, extent.z, 1]])
                        #             @ _rotation_matrix
                        #         )[:3]
                        #     )
                        # )
                        bb = np.array(
                            [
                                [loc.x, loc.y, loc.z],
                                [ori.x, ori.y, ori.z],
                                [extent.x, extent.y, extent.z]
                                # [rotated_extent[0], rotated_extent[1], rotated_extent[2]],
                            ]
                        )

                    else:
                        loc = _obstacle.get_transform().location
                        bb = np.array([[loc.x, loc.y, loc.z], [0.5, 0.5, 2]])

                    obst.append(bb)

        return obst

    def _get_2d_bb_baseline(self, obstacle, distance=2, cam="seg_front"):
        """Returns 2 coordinates for the baseline for 2d bbs in world coordinates
        (distance behind trigger volume, as seen from camera)

        Args:
            obstacle (Actor): obstacle with
            distance (int, optional): Distance behind trigger volume. Defaults to 2.

        Returns:
            np.ndarray: Baseline
        """
        cam = cam + "_{}".format(self.vehicle_num)
        trigger = obstacle.trigger_volume
        bb = self._create_2d_bb_points(trigger)
        trig_loc_world = self._trig_to_world(bb, obstacle, trigger)
        # self._draw_line(trig_loc_world[:,0], trig_loc_world[:,3], 0.7, color=(0, 255, 255))

        cords_x_y_z = np.array(
            self._world_to_sensor(trig_loc_world, self._get_sensor_position(cam))
        )
        indices = (-cords_x_y_z[0]).argsort()

        # check crooked up boxes
        if self._get_dist(
            cords_x_y_z[:, indices[0]], cords_x_y_z[:, indices[1]]
        ) < self._get_dist(cords_x_y_z[:, indices[0]], cords_x_y_z[:, indices[2]]):
            cords = cords_x_y_z[:, [indices[0], indices[2]]] + np.array(
                [[distance], [0], [0], [0]]
            )
        else:
            cords = cords_x_y_z[:, [indices[0], indices[1]]] + np.array(
                [[distance], [0], [0], [0]]
            )

        sensor_world_matrix = self.get_matrix(self._get_sensor_position(cam))
        baseline = np.dot(sensor_world_matrix, cords)

        return baseline

    def _baseline_to_box(self, baseline, cam, height=1):
        """Transforms a baseline (in world coords) into a 2d box (in sensor coords)

        Args:
            baseline ([type]): [description]
            cam ([type]): [description]
            height (int, optional): Box height. Defaults to 1.

        Returns:
            [type]: Box in sensor coords
        """
        cords_x_y_z = np.array(
            self._world_to_sensor(baseline, self._get_sensor_position(cam))[:3, :]
        )

        cords = np.hstack(
            (cords_x_y_z, np.fliplr(cords_x_y_z + np.array([[0], [0], [height]])))
        )

        return self._coords_to_2d_bb(cords)

    def _coords_to_2d_bb(self, cords):
        """Returns coords of a 2d box given points in sensor coords

        Args:
            cords ([type]): [description]

        Returns:
            [type]: [description]
        """
        cords_y_minus_z_x = np.vstack((cords[1, :], -cords[2, :], cords[0, :]))

        bbox = (self._sensor_data["calibration"] @ cords_y_minus_z_x).T

        camera_bbox = np.vstack(
            [bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]]
        ).T

        if np.any(camera_bbox[:, 2] > 0):

            camera_bbox = np.array(camera_bbox)
            _positive_bb = camera_bbox[camera_bbox[:, 2] > 0]

            min_x = int(
                np.clip(np.min(_positive_bb[:, 0]), 0, self._sensor_data["width"])
            )
            min_y = int(
                np.clip(np.min(_positive_bb[:, 1]), 0, self._sensor_data["height"])
            )
            max_x = int(
                np.clip(np.max(_positive_bb[:, 0]), 0, self._sensor_data["width"])
            )
            max_y = int(
                np.clip(np.max(_positive_bb[:, 1]), 0, self._sensor_data["height"])
            )

            return [(min_x, min_y), (max_x, max_y)]
        else:
            return None

    def _change_seg_stop(self, seg_img, depth_img, stop_signs, cam, _region_size=6):
        """Adds a stop class to the segmentation image

        Args:
            seg_img ([type]): [description]
            depth_img ([type]): [description]
            stop_signs ([type]): [description]
            cam ([type]): [description]
            _region_size (int, optional): [description]. Defaults to 6.
        """
        for stop in stop_signs:

            _dist = self._get_distance(stop.get_transform().location)

            _region = np.abs(depth_img - _dist)

            seg_img[(_region < _region_size) & (seg_img == 12)] = 26

            # lane markings
            trigger = stop.trigger_volume

            _trig_loc_world = self._trig_to_world(
                np.array([[0], [0], [0], [1.0]]).T, stop, trigger
            )
            _x = self._world_to_sensor(_trig_loc_world, self._get_sensor_position(cam))[
                0, 0
            ]

            if _x > 0:  # stop is in front of camera

                bb = self._create_2d_bb_points(trigger, 4)
                trig_loc_world = self._trig_to_world(bb, stop, trigger)
                cords_x_y_z = self._world_to_sensor(
                    trig_loc_world, self._get_sensor_position(cam), True
                )

                # if cords_x_y_z.size:
                cords_x_y_z = cords_x_y_z[:3, :]
                cords_y_minus_z_x = np.concatenate(
                    [cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]]
                )
                bbox = (self._sensor_data["calibration"] @ cords_y_minus_z_x).T

                camera_bbox = np.concatenate(
                    [bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]],
                    axis=1,
                )

                if np.any(camera_bbox[:, 2] > 0):

                    camera_bbox = np.array(camera_bbox)

                    polygon = [
                        (camera_bbox[i, 0], camera_bbox[i, 1])
                        for i in range(len(camera_bbox))
                    ]

                    img = Image.new(
                        "L",
                        (self._sensor_data["width"], self._sensor_data["height"]),
                        0,
                    )
                    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
                    _region = np.array(img)

                    # seg_img[(_region == 1)] = 27
                    seg_img[(_region == 1) & (seg_img == 6)] = 27

    def _trig_to_world(self, bb, parent, trigger):
        """Transforms the trigger coordinates to world coordinates

        Args:
            bb ([type]): [description]
            parent ([type]): [description]
            trigger ([type]): [description]

        Returns:
            [type]: [description]
        """
        bb_transform = carla.Transform(trigger.location)
        bb_vehicle_matrix = self.get_matrix(bb_transform)
        vehicle_world_matrix = self.get_matrix(parent.get_transform())
        bb_world_matrix = vehicle_world_matrix @ bb_vehicle_matrix
        world_cords = bb_world_matrix @ bb.T
        return world_cords

    def _create_2d_bb_points(self, actor_bb, scale_factor=1):
        """
        Returns 2D floor bounding box for an actor.
        """

        cords = np.zeros((4, 4))
        extent = actor_bb.extent
        x = extent.x * scale_factor
        y = extent.y * scale_factor
        z = extent.z * scale_factor
        cords[0, :] = np.array([x, y, 0, 1])
        cords[1, :] = np.array([-x, y, 0, 1])
        cords[2, :] = np.array([-x, -y, 0, 1])
        cords[3, :] = np.array([x, -y, 0, 1])
        return cords

    def _get_sensor_position(self, cam):
        """returns the sensor position and rotation

        Args:
            cam ([type]): [description]

        Returns:
            [type]: [description]
        """
        sensor_transform = self._sensors[cam].get_transform()
        return sensor_transform

    def _world_to_sensor(self, cords, sensor, move_cords=False):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = self.get_matrix(sensor)
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)

        if move_cords:
            _num_cords = range(sensor_cords.shape[1])
            modified_cords = np.array([])
            for i in _num_cords:
                if sensor_cords[0, i] < 0:
                    for j in _num_cords:
                        if sensor_cords[0, j] > 0:
                            _direction = sensor_cords[:, i] - sensor_cords[:, j]
                            _distance = -sensor_cords[0, j] / _direction[0]
                            new_cord = (
                                sensor_cords[:, j]
                                + _distance[0, 0] * _direction * 0.9999
                            )
                            modified_cords = (
                                np.hstack([modified_cords, new_cord])
                                if modified_cords.size
                                else new_cord
                            )
                else:
                    modified_cords = (
                        np.hstack([modified_cords, sensor_cords[:, i]])
                        if modified_cords.size
                        else sensor_cords[:, i]
                    )

            return modified_cords
        else:
            return sensor_cords

    def get_matrix(self, transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

    def _change_seg_tl(self, seg_img, depth_img, traffic_lights, _region_size=4):
        """Adds 3 traffic light classes (green, yellow, red) to the segmentation image

        Args:
            seg_img ([type]): [description]
            depth_img ([type]): [description]
            traffic_lights ([type]): [description]
            _region_size (int, optional): [description]. Defaults to 4.
        """
        for tl in traffic_lights:
            _dist = self._get_distance(tl.get_transform().location)

            _region = np.abs(depth_img - _dist)

            if tl.get_state() == carla.TrafficLightState.Red:
                state = 23
            elif tl.get_state() == carla.TrafficLightState.Yellow:
                state = 24
            elif tl.get_state() == carla.TrafficLightState.Green:
                state = 25
            else:  # none of the states above, do not change class
                state = 18

            # seg_img[(_region >= _region_size)] = 0
            seg_img[(_region < _region_size) & (seg_img == 18)] = state

    def _get_dist(self, p1, p2):
        """Returns the distance between p1 and p2

        Args:
            target ([type]): [description]

        Returns:
            [type]: [description]
        """

        distance = np.sqrt(
            (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
        )

        return distance

    def _get_distance(self, target):
        """Returns the distance from the (rgb_front) camera to the target

        Args:
            target ([type]): [description]

        Returns:
            [type]: [description]
        """
        sensor_transform = self._sensors["rgb_front" + "_{}".format(self.vehicle_num)].get_transform()

        distance = np.sqrt(
            (sensor_transform.location.x - target.x) ** 2
            + (sensor_transform.location.y - target.y) ** 2
            + (sensor_transform.location.z - target.z) ** 2
        )

        return distance

    def _get_depth(self, data):
        """Transforms the depth image into meters

        Args:
            data ([type]): [description]

        Returns:
            [type]: [description]
        """

        data = data.astype(np.float32)

        normalized = np.dot(data, [65536.0, 256.0, 1.0])
        normalized /= 256 * 256 * 256 - 1
        in_meters = 1000 * normalized

        return in_meters

    def _find_obstacle(self, obstacle_type="*traffic_light*"):
        """Find all actors of a certain type that are close to the vehicle

        Args:
            obstacle_type (str, optional): [description]. Defaults to '*traffic_light*'.

        Returns:
            [type]: [description]
        """
        obst = list()

        _actors = self._world.get_actors()
        _obstacles = _actors.filter(obstacle_type)

        for _obstacle in _obstacles:
            trigger = _obstacle.trigger_volume

            _obstacle.get_transform().transform(trigger.location)
            distance_to_car = trigger.location.distance(self._vehicle.get_location())

            a = np.sqrt(
                trigger.extent.x**2 + trigger.extent.y**2 + trigger.extent.z**2
            )
            b = np.sqrt(
                self._vehicle.bounding_box.extent.x**2
                + self._vehicle.bounding_box.extent.y**2
                + self._vehicle.bounding_box.extent.z**2
            )

            s = a + b + 10

            if distance_to_car <= s:
                # the actor is affected by this obstacle.
                obst.append(_obstacle)

        return obst

    def _get_camera_to_car_calibration(self, sensor):
        """returns the calibration matrix for the given sensor

        Args:
            sensor ([type]): [description]

        Returns:
            [type]: [description]
        """
        calibration = np.identity(3)
        calibration[0, 2] = sensor["width"] / 2.0
        calibration[1, 2] = sensor["height"] / 2.0
        calibration[0, 0] = calibration[1, 1] = sensor["width"] / (
            2.0 * np.tan(sensor["fov"] * np.pi / 360.0)
        )
        return calibration
