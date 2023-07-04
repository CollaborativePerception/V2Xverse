import imp
import time
import torch
import math
import cv2
import carla
import numpy as np

import pdb

from interfuser.timm.models import create_model
from team_code.planner import RoutePlanner

from leaderboard.autoagents import autonomous_agent


from team_code.v2x_infer_action import V2X_infer
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from leaderboard.sensors.fixed_sensors import RoadSideUnit, get_rsu_point

def get_entry_point():
    return "V2X_Agent"

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
class V2X_Agent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file, ego_vehicles_num):

        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        self._rgb_sensor_data = {"width": 800, "height": 600, "fov": 100}
        self._sensor_data = self._rgb_sensor_data.copy()

        self.config = imp.load_source("MainModel", path_to_conf_file).GlobalConfig()
        self.skip_frames = self.config.skip_frames
        self.first_generate_rsu = True
        self.change_rsu_frame = 25
        self.ego_vehicles_num = ego_vehicles_num   

        ############
        ###### load the model parameters
        ############
        self.perception_model = create_model(self.config.perception_model['name'], fusion_mode=self.config.fusion_mode)
        path_to_model_file = self.config.perception_model['path']
        print('load perception model: %s' % path_to_model_file)
        self.perception_model.load_state_dict(torch.load(path_to_model_file)["state_dict"])
        self.perception_model.cuda()
        self.perception_model.eval()

        self.planning_model = create_model(self.config.planning_model['name'])
        path_to_model_file = self.config.planning_model['path']
        print('load planning model: %s' % path_to_model_file)
        self.planning_model.load_state_dict(torch.load(path_to_model_file)["state_dict"])
        self.planning_model.cuda()
        self.planning_model.eval()
    
        ############
        ###### core module, infer the action from data
        ############
        self.infer = V2X_infer(config=self.config,
                               ego_vehicles_num=self.ego_vehicles_num,
                               perception_model=self.perception_model,
                               planning_model=self.planning_model)


    def _init(self):
        self._route_planner = RoutePlanner(4.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        self.vehicle_num = 0

    def _get_position(self, tick_data):
        ##########
        ###### GPS coordinate!
        gps = tick_data["gps"]
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps

    def pose_def(self):
        self.lidar_pose = carla.Transform(carla.Location(x=1.3,y=0.0,z=1.85), # 2.5
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

    def tick(self, input_data, vehicle_num):
        """
        You shall not change the content of input_data in this function.
        """
        _vehicle = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
        # rgb
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
        lidar = input_data["lidar_{}".format(vehicle_num)][1]   ## HERE was a BUG, but I have fixed it now.
        gps = input_data["gps_{}".format(vehicle_num)][1][:2]
        speed = input_data["speed_{}".format(vehicle_num)][1]["speed"]
        compass = input_data["imu_{}".format(vehicle_num)][1][-1]
        if (
            math.isnan(compass) == True
        ):  # It can happen that the compass sends nan for a few frames
            compass = 0.0

        # set the current pose!
        cur_lidar_pose = carla.Location(x=self.lidar_pose.location.x,
                                            y=self.lidar_pose.location.y,
                                            z=self.lidar_pose.location.z)
        _vehicle.get_transform().transform(cur_lidar_pose)
        self.intrinsics = get_camera_intrinsic(self._rgb_sensor_data)
        self.lidar2front = get_camera_extrinsic(self.camera_front_pose, self.lidar_pose)
        self.lidar2left = get_camera_extrinsic(self.camera_left_pose, self.lidar_pose)
        self.lidar2right = get_camera_extrinsic(self.camera_right_pose, self.lidar_pose)
        self.lidar2rear = get_camera_extrinsic(self.camera_rear_pose, self.lidar_pose)

        pos = self._get_position({"gps": gps})
        next_wp, next_cmd = self._route_planner.run_step(pos, vehicle_num)
        # TODO: What's this?
        theta = compass + np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
        local_command_point = R.T.dot(local_command_point)
        local_command_point = np.clip(local_command_point, a_min=[-12, -36], a_max=[12, 12])
        # And what's that?

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
        result = {
            "rgb_front": rgb_front,
            "rgb_left": rgb_left,
            "rgb_right": rgb_right,
            "rgb_rear": rgb_rear,
            "lidar": lidar,
            "measurements": mes
        }
        return result

    def spawn_rsu(self):
        # delete rsu and spawn new ones
        if self.step % self.change_rsu_frame == 0:
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
                # spawn rsu
                self.rsu.append(RoadSideUnit(save_path="/GPFS/public/InterFuser/results_cop3/image/rsu",
                                            id=int((self.step/self.change_rsu_frame+1)*1000+vehicle_num),
                                            is_None=(vehicle is None)))
                if vehicle is not None:
                    spawn_loc = get_rsu_point(vehicle)
                    self.rsu[vehicle_num].setup_sensors(parent=None, spawn_point=spawn_loc)
        return
    
    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        self.step += 1

        # Try spawn the rsu.
        # print('Start to spawn the RSU.')
        self.spawn_rsu()
        rsu_data = []
        # print('Successfully spawn the RSU.')
        if self.step % self.change_rsu_frame != 0:
            for vehicle_num in range(self.ego_vehicles_num):
                # print("RSU: {}".format(vehicle_num))
                if CarlaDataProvider.get_hero_actor(hero_id=vehicle_num) is not None:
                    rsu_data.append(
                        self.rsu[vehicle_num].process(self.rsu[vehicle_num].tick(),is_train=True)
                    )
                else:
                    rsu_data.append(None)
                # provide None
        # print('Fetching the RSU data.')
        
        # If the frame is skipped.
        if self.step % self.skip_frames != 0 and self.step > 4:
            # return the previous control signal.   
            return self.infer.prev_control
        
        control_all = []
        
        ''' Detailed data structure, including N cars with M RSUs.
        path_to_the_data/
            - ego_vehicle_0/
                - rgb_(x)/          # x in [front, left, right]
                    0000.jpg        # note: only one frame! ALWAYS 0000.jpg,
                                    #       NO 0001.jpg
                - measurements/
                    0000.json:      # note: only one frame!
                        - command
                        - speed
                        - theta
                        - x_command
                        - y_command
                        - gps_x
                        - gps_y
                        - x
                        - y
                        - lidar_pose_x
                        - lidar_pose_y
                - lidar
                    0000.npy         # note: only one frame!
            - ego_vehicle_1/
            - ego_vehicle_2/
            - ...
            - ego_vehicle_(N-1)/
            - rsu_0/
            - rsu_1/
            - ...
            - rsu_(M-1)/
        '''
        # get the ego_data
        # print('Start fetching the car data.')
        ego_data = [
            self.tick(input_data, vehicle_num)
            if CarlaDataProvider.get_hero_actor(hero_id=vehicle_num) is not None
            else None
            for vehicle_num in range(self.ego_vehicles_num)
        ]
        # print('End Fetching the car data.')
        # pdb.set_trace()
        # control_all = self.infer.get_action_from_route(route_path="path_to_the_data", 
        #                                                    model=self.net, 
        #                                                    step=self.step,
        #                                                    timestamp=timestamp)
        
        if self.config.perception_fusion_mode == 'cheat':
            control_all = self.infer.get_action_from_list_for_interfuser_cheat( car_data_raw=ego_data,
                                                            rsu_data_raw=rsu_data,
                                                            step=self.step,
                                                            timestamp=timestamp)
        else:
            print('Undefined perception_fusion_mode.')
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