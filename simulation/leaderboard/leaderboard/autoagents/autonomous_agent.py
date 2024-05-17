#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the base class for all autonomous agents
"""

from __future__ import print_function

from enum import Enum

import carla
from srunner.scenariomanager.timer import GameTime

from leaderboard.utils.route_manipulation import downsample_route
from leaderboard.envs.sensor_interface import SensorInterface

import time


class Track(Enum):

    """
    This enum represents the different tracks of the CARLA AD leaderboard.
    """
    SENSORS = 'SENSORS'
    MAP = 'MAP'

class AutonomousAgent(object):

    """
    Autonomous agent base class. All user agents have to be derived from this class
    """

    def __init__(self, path_to_conf_file,ego_vehicles_num=1):
        self.agent_name='auto_agent'
        self.rsu = []
        self.track = Track.SENSORS
        #  current global plans to reach a destination
        self._global_plan = None
        self._global_plan_world_coord = None

        # this data structure will contain all sensor data
        self.sensor_interface = SensorInterface()

        # agent's initialization
        self.setup(path_to_conf_file, ego_vehicles_num)

        self.wallclock_t0 = None

    def setup(self, path_to_conf_file, ego_vehicles_num):
        """
        Initialize everything needed by your agent and set the track attribute to the right type:
            Track.SENSORS : CAMERAS, LIDAR, RADAR, GPS and IMU sensors are allowed
            Track.MAP : OpenDRIVE map is also allowed
        """
        pass

    def sensors(self):  # pylint: disable=no-self-use
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]

        """
        sensors = []

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        :return: control
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False

        return control

    def destroy(self):
        """
        Destroy (clean-up) the agent
        :return:
        """
        pass

    def __call__(self):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """

        # Obtaining data from sensors
        input_data = self.sensor_interface.get_data() # gps, lidar, imu, rgb, speed

        # 
        timestamp = GameTime.get_time()
        if not self.wallclock_t0:
            self.wallclock_t0 = GameTime.get_wallclocktime()

        # get control signal, function run_step should be rewritten in your customized agent
        control = self.run_step(input_data, timestamp)
        for i in range(len(control)):
            if control[i]:  
                control[i].manual_gear_shift = False

        return control

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """
        vehicle_num = len(global_plan_gps)
        self._global_plan_world_coord = []
        self._global_plan = []
        self._global_plan_world_coord_all = []
        self._global_plan_all = []

        distance = 50
        if isinstance(self.config, dict):
            if 'control' in self.config:
                if 'target_point_distance' in self.config['control']:
                    distance = self.config['control']['target_point_distance']
            if 'target_point_distance' in self.config:
                distance = self.config['target_point_distance']            
        elif self.config.__module__ == 'MainModel':
            if hasattr(self.config, 'target_point_distance'):
                distance = self.config.target_point_distance

        print('gps distance:', distance)

        for vehicle_id in range(vehicle_num):
            ds_ids = downsample_route(global_plan_world_coord[vehicle_id], distance)
            global_plan_world_coord_tmp = [(global_plan_world_coord[vehicle_id][x][0], global_plan_world_coord[vehicle_id][x][1]) for x in ds_ids]
            global_plan = [global_plan_gps[vehicle_id][x] for x in ds_ids]
            global_plan_world_coord_all = [(global_plan_world_coord[vehicle_id][x][0], global_plan_world_coord[vehicle_id][x][1]) for x in range(len(global_plan_world_coord[vehicle_id]))]
            global_plan_all = global_plan_gps[vehicle_id]
            self._global_plan_world_coord.append(global_plan_world_coord_tmp)
            self._global_plan.append(global_plan)
            self._global_plan_world_coord_all.append(global_plan_world_coord_all)
            self._global_plan_all.append(global_plan_all)


