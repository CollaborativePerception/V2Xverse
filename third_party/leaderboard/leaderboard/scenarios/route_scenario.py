#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides Challenge routes as standalone scenarios
"""

from __future__ import print_function

import math
import os
import xml.etree.ElementTree as ET
import numpy.random as random
import torch
import py_trees
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import carla

from agents.navigation.local_planner import RoadOption

# pylint: disable=line-too-long
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration, ActorConfigurationData
# pylint: enable=line-too-long
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import Idle, ScenarioTriggerer
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenarios.control_loss import ControlLoss
from srunner.scenarios.follow_leading_vehicle import FollowLeadingVehicle
from srunner.scenarios.object_crash_vehicle import DynamicObjectCrossing
from srunner.scenarios.object_crash_intersection import VehicleTurningRoute
from srunner.scenarios.other_leading_vehicle import OtherLeadingVehicle
from srunner.scenarios.maneuver_opposite_direction import ManeuverOppositeDirection
from srunner.scenarios.junction_crossing_route import SignalJunctionCrossingRoute, NoSignalJunctionCrossingRoute

from srunner.scenariomanager.scenarioatomics.atomic_criteria import (CollisionTest,
                                                                     InRouteTest,
                                                                     RouteCompletionTest,
                                                                     OutsideRouteLanesTest,
                                                                     RunningRedLightTest,
                                                                     RunningStopTest,
                                                                     ActorSpeedAboveThresholdTest)
from srunner.tools.scenario_helper import get_location_in_distance_from_wp

from leaderboard.utils.route_parser import RouteParser, TRIGGER_THRESHOLD, TRIGGER_ANGLE_THRESHOLD
from leaderboard.utils.route_manipulation import interpolate_trajectory
from leaderboard.sensors.fixed_sensors import TrafficLightSensor
from third_party.scenario_runner.srunner.scenarios import ScenarioClassRegistry


ROUTESCENARIO = ["RouteScenario"]

SECONDS_GIVEN_PER_METERS = 0.8 # for timeout
INITIAL_SECONDS_DELAY = 8.0

NUMBER_CLASS_TRANSLATION = {
    "Scenario1": ControlLoss,
    "Scenario2": FollowLeadingVehicle,
    "Scenario3": DynamicObjectCrossing,
    "Scenario4": VehicleTurningRoute,
    "Scenario5": OtherLeadingVehicle,
    "Scenario6": ManeuverOppositeDirection,
    "Scenario7": SignalJunctionCrossingRoute,
    "Scenario8": SignalJunctionCrossingRoute,
    "Scenario9": SignalJunctionCrossingRoute,
    "Scenario10": NoSignalJunctionCrossingRoute
}


def oneshot_behavior(name, variable_name, behaviour):
    """
    This is taken from py_trees.idiom.oneshot.
    """
    # Initialize the variables
    blackboard = py_trees.blackboard.Blackboard()
    _ = blackboard.set(variable_name, False)

    # Wait until the scenario has ended
    subtree_root = py_trees.composites.Selector(name=name)
    check_flag = py_trees.blackboard.CheckBlackboardVariable(
        name=variable_name + " Done?",
        variable_name=variable_name,
        expected_value=True,
        clearing_policy=py_trees.common.ClearingPolicy.ON_INITIALISE
    )
    set_flag = py_trees.blackboard.SetBlackboardVariable(
        name="Mark Done",
        variable_name=variable_name,
        variable_value=True
    )
    # If it's a sequence, don't double-nest it in a redundant manner
    if isinstance(behaviour, py_trees.composites.Sequence):
        behaviour.add_child(set_flag)
        sequence = behaviour
    else:
        sequence = py_trees.composites.Sequence(name="OneShot")
        sequence.add_children([behaviour, set_flag])

    subtree_root.add_children([check_flag, sequence])
    return subtree_root


def convert_json_to_transform(actor_dict):
    """
    Convert a JSON string to a CARLA transform
    """
    return carla.Transform(location=carla.Location(x=float(actor_dict['x']), y=float(actor_dict['y']),
                                                   z=float(actor_dict['z'])),
                           rotation=carla.Rotation(roll=0.0, pitch=0.0, yaw=float(actor_dict['yaw'])))

# NOTE（GJH): Select the scenario according to the proportion of each scenario
def selScenario(scenario_config: dict) -> str:
    """
    
    Select a scenario according to the proportion of each scenario

    Args:
        scenario_config: a dict of scenario definition config containing proportion in this Scenario

    Returns:
        selected_scenario: a string of selected scenario

    """
    try:
        scenarios = list(scenario_config.keys())
        scenario_proportion = []
        for scenario_name in scenarios:
            scenario_proportion.append(scenario_config[scenario_name]["proportion"])
        return np.random.choice(scenarios, 1, p=scenario_proportion).tolist()[0]
    except Exception as e:
        print("Select Scenario Error: ", e, "Using the first scenario in the config file.")
        scenarios = list(scenario_config.keys())
        print(scenarios)
        return scenarios[0]


def convert_json_to_actor(actor_dict):
    """
    Convert a JSON string to an ActorConfigurationData dictionary
    """
    node = ET.Element('waypoint')
    node.set('x', actor_dict['x'])
    node.set('y', actor_dict['y'])
    node.set('z', actor_dict['z'])
    node.set('yaw', actor_dict['yaw'])

    return ActorConfigurationData.parse_from_node(node, 'simulation')


def convert_transform_to_location(transform_vec):
    """
    Convert a vector of transforms to a vector of locations
    """
    location_vec = []
    for transform_tuple in transform_vec:
        location_vec.append((transform_tuple[0].location, transform_tuple[1]))

    return location_vec


def compare_scenarios(scenario_choice, existent_scenario):
    """
    Compare function for scenarios based on distance of the scenario start position
    """
    def transform_to_pos_vec(scenario):
        """
        Convert left/right/front to a meaningful CARLA position
        """
        position_vec = [scenario['trigger_position']]
        if scenario['other_actors'] is not None:
            if 'left' in scenario['other_actors']:
                position_vec += scenario['other_actors']['left']
            if 'front' in scenario['other_actors']:
                position_vec += scenario['other_actors']['front']
            if 'right' in scenario['other_actors']:
                position_vec += scenario['other_actors']['right']

        return position_vec

    # put the positions of the scenario choice into a vec of positions to be able to compare

    choice_vec = transform_to_pos_vec(scenario_choice)
    existent_vec = transform_to_pos_vec(existent_scenario)
    for pos_choice in choice_vec:
        for pos_existent in existent_vec:

            dx = float(pos_choice['x']) - float(pos_existent['x'])
            dy = float(pos_choice['y']) - float(pos_existent['y'])
            dz = float(pos_choice['z']) - float(pos_existent['z'])
            dist_position = math.sqrt(dx * dx + dy * dy + dz * dz)
            dyaw = float(pos_choice['yaw']) - float(pos_choice['yaw'])
            dist_angle = math.sqrt(dyaw * dyaw)
            if dist_position < TRIGGER_THRESHOLD and dist_angle < TRIGGER_ANGLE_THRESHOLD:
                return True

    return False


class RouteScenario(BasicScenario):

    """
    Implementation of a RouteScenario, i.e. a scenario that consists of driving along a pre-defined route,
    along which several smaller scenarios are triggered
    """

    category = "RouteScenario"

    def __init__(self, world, config, debug_mode=0, criteria_enable=True, ego_vehicles_num=1, crazy_level=0,crazy_proportion=0,log_dir=None, scenario_parameter=None,trigger_distance=10):
        """
        Setup all relevant parameters and create scenarios along route

        Args:
            world: carla.libcarla.World
            config: srunner.scenarioconfigs.route_scenario_configuration.RouteScenarioConfiguration,
                    route information(name, town, trajectory, weather)
            ego_vehicles_num: int, number of communicating vehicles
            log_dir: str, directory to save log
            scenario information:
                {crazy_level,
                crazy_propotion,
                trigger_distance
                }

        """

        # load or initialize params
        self.config = config
        self.route = None
        self.sampled_scenarios_definitions = None
        self.ego_vehicles_num=ego_vehicles_num
        self.new_config_trajectory=None
        self.crazy_level = crazy_level
        self.crazy_proportion = crazy_proportion
        self.trigger_distance = trigger_distance
        self.sensor_tf_num = 0
        self.sensor_tf_list = []
        self.log_dir = log_dir
        
        self.scenario_parameter = scenario_parameter
        self.route_scenario_dic = {}

        # update waypoints and scenarios along the routes
        self._update_route(world, config, debug_mode>0)

        # set traffic sensors
        for j in range(self.sensor_tf_num):
            tf_sensor=TrafficLightSensor(config.save_path_root,j)
            self.sensor_tf_list.append(tf_sensor)
        self._init_tf_sensors()

        # spawn ego vehicles
        ego_vehicles = self._update_ego_vehicle(world)
        # update ego_num, for some ego may fail to spawn
        # self.ego_vehicles_num = len(ego_vehicles)

        
        if self.scenario_parameter is not None:
            self.list_scenarios = self._build_scenario_parameter_instances(world,
                                                             ego_vehicles,
                                                             self.sampled_scenarios_definitions,
                                                             scenarios_per_tick=10,
                                                             timeout=self.timeout,
                                                             debug_mode=debug_mode>1,
                                                             scenario_parameter=self.scenario_parameter)
        else:
            self.list_scenarios = self._build_scenario_instances(world,
                                                                ego_vehicles,
                                                                self.sampled_scenarios_definitions,
                                                                scenarios_per_tick=10,
                                                                timeout=self.timeout,
                                                                debug_mode=debug_mode>1)
        super(RouteScenario, self).__init__(name=config.name,
                                            ego_vehicles=ego_vehicles,
                                            config=config,
                                            world=world,
                                            debug_mode=debug_mode>1,
                                            terminate_on_failure=False,
                                            criteria_enable=criteria_enable)
        print('route_scenarios:',self.route_scenario_dic)

    def _get_multi_tf(self, trajectory, tf_num=1) -> List:
        """
        calculate the closest tfs and return them
        """
        decay_factor=0.5
        tf_list = []
        world = CarlaDataProvider.get_world()
        # lights_list is a list of all tfs
        lights_list = world.get_actors().filter("*traffic_light*")
        tf_tensor = torch.tensor([[light.get_transform().location.x,
                       light.get_transform().location.y,
                       light.get_transform().location.z 
                    ] for light in lights_list],dtype=float)
        # Find top k closest tfs
        dist_tensor=torch.zeros(tf_tensor.shape[0],dtype=float)
        for waypoint in trajectory[0]:
            waypoint_tensor=torch.tensor([waypoint.x,waypoint.y,waypoint.z] 
                                ,dtype=float).reshape(1,3).repeat(tf_tensor.shape[0],1)
            dist_tensor += ((tf_tensor-waypoint_tensor)**2).sum(dim=1,keepdim=False)*decay_factor
            decay_factor = decay_factor ** 2
        # get the idx of top k closest tfs
        _,idx=dist_tensor.topk(tf_num,
                                    largest=False,
                                    sorted=False)
        [tf_list.append(lights_list[idx[j].item()]) for j in range(tf_num)]
        return tf_list

    def _init_tf_sensors(self) -> None:
        """
        This function is to find the proper tf and set up sensors on it.
        """
        tf_list = self._get_multi_tf(self.get_new_config_trajectory().copy(), 
                                          tf_num=self.sensor_tf_num)
        for j in range(self.sensor_tf_num):
            self.sensor_tf_list[j].setup_sensors(tf_list[j])
        return

    def get_sensor_tf(self) -> List:
        return self.sensor_tf_list

    def _cal_multi_routes(self, world: carla.libcarla.World, config) -> List:
        """
        Given the waypoints of one route as anchors, computes waypoints that those ego vehicles will pass by around those anchors.
        Args:
            world: Carla world
            config:
                config.trajectory: list of carla.libcarla.Location, sparse waypoints list
        Returns:
            trajectory: trajectory[i] represents waypoints of the ith ego vehicle, trajectory[i] is list of carla.libcarla.Location
        
        Given config.trajectory[A(start point), B, C, ..., L(end point)]
        ego 0, ego 1, ego 2 will reach B, C, ..., L individually, but start at different location around A, like
        ////////////////////////////
        //      |       ||[ego 1]|        //
        //      |       ||       |        //
        //      |       ||       |        //
        //      |       ||       |        //
        //      |       ||       |        //
        //      |[ego 3]||[ego 0]|        //
        //      |       ||       |        //
        //      |       ||       |        //
        //      |       ||       |        //
        //      |       ||[ego 2]|        //
        //      |       ||       |        //
            

        """
        trajectory=[]
        distance_gap_straight = 12
        distance_gap_left = 0
        distance_gap_right = 0
        distance_gap_rear = 0

        # trajectory's element is a list of waypoint, a carla.Location object
        trajectory.append(config.trajectory) 
        # initialize trajectory
        trajectory.extend([[] for _ in range(1,self.ego_vehicles_num)])

        # spawn_points is a list of carla.Transform
        spawn_points=world.get_map().get_spawn_points()
        spawn_tensor=torch.tensor([[spawn_point.location.x,
                       spawn_point.location.y,
                       spawn_point.location.z 
                    ] for spawn_point in spawn_points],dtype=float)

        # calculate waypoints via knn
        for point, waypoint in enumerate(trajectory[0]):
            waypoint_tensor=torch.tensor([waypoint.x,waypoint.y,waypoint.z] 
                                ,dtype=float).reshape(1,3).repeat(spawn_tensor.shape[0],1)
            dist_tensor=((spawn_tensor-waypoint_tensor)**2).sum(dim=1,keepdim=False)
            val,idx=dist_tensor.topk(7*self.ego_vehicles_num,
                                        largest=False,
                                        sorted=True
                                        )
            curk=0
            travel_distance_start = 12
            travel_distance_start_pre = 0
            travel_distance_start_rear = 12
            travel_distance_start_rear_pre = 0
            travel_distance_start_right = 10
            travel_distance_start_right_pre = 0
            travel_distance_start_left = 10
            travel_distance_start_left_pre = 0
            waypoint_start = None
            for k in range(1,self.ego_vehicles_num):
                if point == 0:
                    # curk+=1
                    for _ in range(curk, len(val)):
                        waypoint_carla = CarlaDataProvider.get_map().get_waypoint(waypoint)
                        location, travel_distance_start = get_location_in_distance_from_wp(waypoint_carla, travel_distance_start, direction='foward')
                        if abs(travel_distance_start_pre - travel_distance_start)<5.0:
                            waypoint_start_right = waypoint_carla.get_right_lane()
                            if waypoint_start_right:
                                location_right, travel_distance_start_right = get_location_in_distance_from_wp(waypoint_start_right, travel_distance_start_right, direction='foward')
                            
                            waypoint_start_left = waypoint_carla.get_left_lane()
                            if waypoint_start_left:
                                location_left, travel_distance_start_left = get_location_in_distance_from_wp(waypoint_start_left, travel_distance_start_left, direction='rear')
                            
                            location_rear, travel_distance_start_rear = get_location_in_distance_from_wp(waypoint_carla, travel_distance_start_rear, direction='rear')

                            if abs(travel_distance_start_right-travel_distance_start_right_pre)>7.0 \
                                and (not(waypoint_start_right is None or waypoint_start_right.lane_type == carla.LaneType.Sidewalk\
                                    or waypoint_start_right.lane_type == carla.LaneType.Shoulder)):
                                trajectory[k].append(location_right)
                                print("ego{} located at right".format(k))
                                travel_distance_start_right_pre = travel_distance_start_right
                                travel_distance_start_right += distance_gap_right
                                break

                            elif abs(travel_distance_start_rear-travel_distance_start_rear_pre)>8.0:
                                trajectory[k].append(location_rear)
                                print("ego{} located at rear".format(k))
                                travel_distance_start_rear_pre = travel_distance_start_rear
                                travel_distance_start_rear += distance_gap_rear
                                break                                    
                            elif abs(travel_distance_start_left-travel_distance_start_left_pre)>3.0 \
                                and (not(waypoint_start_left is None or waypoint_start_left.lane_type == carla.LaneType.Sidewalk\
                                    or waypoint_start_left.lane_type == carla.LaneType.Shoulder)):                         
                                trajectory[k].append(location_left)
                                print("ego{} located at left".format(k))
                                travel_distance_start_left_pre = travel_distance_start_left
                                travel_distance_start_left += distance_gap_left
                                break
                            else :
                                while min([spawn_points[idx[curk]].location.distance(trajectory[kk-1][0]) for kk in range(1,k+1)]) <20 and curk<len(idx)-2:
                                        curk += 1
                                trajectory[k].append(spawn_points[idx[curk]].location)
                                curk = (curk + 1)%len(idx)
                                print("ego{} located at the closest spawn point".format(k))
                                break
                        else:
                            trajectory[k].append(location)
                            print("ego{} located at forward".format(k))
                            travel_distance_start_pre = travel_distance_start 
                            travel_distance_start  += distance_gap_straight
                            break 
                else:
                    if val[curk]>5.0:
                        trajectory[k].append(waypoint)
                    else:
                        trajectory[k].append(spawn_points[idx[curk]].location)
                        curk = (curk + 1)%len(idx)
        self.new_config_trajectory=trajectory.copy()
        return trajectory

    def _cal_multi_routes_for_parallel_driving(self, world, config):
        """
        Make cars drive in parallel from the start
        """
        trajectory=[]
        # trajectory's element is a list of waypoint, a carla.Location object
        trajectory.append(config.trajectory) 
        # initialize trajectory
        trajectory.extend([[] for _ in range(1,self.ego_vehicles_num)])

        # spawn_points is a list of carla.Transform
        spawn_points=world.get_map().get_spawn_points()
        spawn_tensor=torch.tensor([[spawn_point.location.x,
                       spawn_point.location.y,
                       spawn_point.location.z 
                    ] for spawn_point in spawn_points],dtype=float)

        # calculate waypoints via knn
        for point, waypoint in enumerate(trajectory[0]):
            waypoint_tensor=torch.tensor([waypoint.x,waypoint.y,waypoint.z] 
                                ,dtype=float).reshape(1,3).repeat(spawn_tensor.shape[0],1)
            dist_tensor=((spawn_tensor-waypoint_tensor)**2).sum(dim=1,keepdim=False)
            val,idx=dist_tensor.topk(7*self.ego_vehicles_num,
                                        largest=False,
                                        sorted=True
                                        )
            curk=0

            waypoint_carla = CarlaDataProvider.get_map().get_waypoint(waypoint)
            waypoint_start_right = waypoint_carla.get_right_lane()

            # record lane in this row
            lane_list = []
            lane_list.append(waypoint)
            max_lane_num = 20
            # record lane on the right
            for lane in range(max_lane_num):
                waypoint_lane_changed = waypoint_carla.get_right_lane()
                if waypoint_lane_changed.lane_type == carla.LaneType.Driving:
                    lane_list.insert(0,waypoint_lane_changed)
                else:
                    break
            # record lane on the left
            for lane in range(max_lane_num):
                waypoint_lane_changed = waypoint_carla.get_left_lane()
                if waypoint_lane_changed.lane_type == carla.LaneType.Driving:
                    lane_list.append(waypoint_lane_changed)
                else:
                    break

            print('initial lane num:', len(lane_list))


            for k in range(1,self.ego_vehicles_num):
                if point == 0:
                    trajectory[k].append(lane_list[k].transform.location)

                else:
                    if val[curk]>5.0:
                        trajectory[k].append(waypoint)
                    else:
                        # _route_distance += travel_distance_route
                        # location, travel_distance_route = get_location_in_distance_from_wp(waypoint_carla, _start_distance)
                        trajectory[k].append(spawn_points[idx[curk]].location)
                        curk = (curk + 1)%len(idx)
        # route = open("route.txt",'w')
        # route.write(str(trajectory))
        self.new_config_trajectory=trajectory.copy()
        return trajectory

    def get_new_config_trajectory(self):
        return self.new_config_trajectory

    def draw_route(self):
        """
        draw waypoints coordinates from self.route
        """
        fig = plt.figure(dpi=400)
        colors = ['tab:red','tab:blue','tab:orange', 'tab:purple','tab:green']
        center_x = self.route[0][0][0].location.x
        center_y = self.route[0][0][0].location.y
        for i in range(len(self.route)):
            for j in range(len(self.route[i])):
                point_x = self.route[i][j][0].location.x - center_x + 1*i
                point_y = self.route[i][j][0].location.y - center_y + 1*i
                if j==0:
                    plt.scatter(point_x, point_y, s=50, c=colors[i], label='ego{}'.format(i))
                    plt.text(point_x+0.1, point_y+0.1, 'ego{} start'.format(i))
                elif j==(len(self.route[i])-1):
                    plt.scatter(point_x, point_y, s=50, c=colors[i])
                    plt.text(point_x+0.1, point_y+2*(i+1), 'ego{} end'.format(i))
                else:
                    plt.scatter(point_x, point_y, s=20, c=colors[i])
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(self.log_dir,'point_coordinates.png'))
        plt.close()

    def _update_route(self, world, config, debug_mode):
        """
        Update the input route, i.e. refine waypoint list, and extract possible scenario locations

        Parameters:
            world: CARLA world
            config: Scenario configuration (RouteConfiguration)
        Main target variable:
            self.route: trajectory plan to reach for each ego vehicle
            self.sampled_scenarios_definitions: scenarios will be triggered on each ego vehicle's route
        """

        # Transform the scenario file into a dictionary, defines possible trigger position for each type of scenario 
        world_annotations = RouteParser.parse_annotations_file(config.scenario_file)

        # generate trajectory for ego-vehicles
        # trajectory's element is a list of waypoint(carla.Location object)
        trajectory = self._cal_multi_routes(world, config)
        gps_route=[]
        route=[]
        potential_scenarios_definitions=[]

        # prepare route's trajectory (interpolate and add the GPS route)
        for i, tr in enumerate(trajectory):
            # tr is a list of waypoint, each a carla.Location object
            gps, r = interpolate_trajectory(world, tr)
            gps_route.append(gps)
            route.append(r)
            print('load scenarios for ego{}'.format(i))
            potential_scenarios_definition, _ = RouteParser.scan_route_for_scenarios(
                config.town, r, world_annotations)
            potential_scenarios_definitions.append(potential_scenarios_definition)
        # print(potential_scenarios_definitions)
        # self.route is a list of ego_vehicles' routes
        self.route = route
        if self.log_dir is not None:
            # plot waypoints coordinates
            self.draw_route()
        CarlaDataProvider.set_ego_vehicle_route([convert_transform_to_location(self.route[j]) for j in range(self.ego_vehicles_num)])
        config.agent.set_global_plan(gps_route, self.route)

        # Sample the scenarios to be used for this route instance. A list for ego_vehicles.
        self.sampled_scenarios_definitions = [self._scenario_sampling(potential_scenarios_definition) 
                                                for potential_scenarios_definition in potential_scenarios_definitions]

        # Timeout of each ego_vehicle in scenario in seconds
        self.timeout = self._estimate_route_timeout()

        # Print route in debug mode
        if debug_mode:
            [self._draw_waypoints(world, self.route[j], vertical_shift=1.0, persistency=50000.0) for j in range(self.ego_vehicles_num)]

    def _update_ego_vehicle(self, world) -> List:
        """
        Set/Update the start position of the ego_vehicles
        Returns:
            ego_vehicles (list): list of ego_vehicles.
        """
        # move ego vehicles to correct position
        ego_vehicles=[]
        for j in range(self.ego_vehicles_num):
            elevate_transform = self.route[j][0][0]
            # ego vehicle will float in the air at a height of 0.5m in the first frame
            elevate_transform.location.z += (0.5)
            print("ego id:{}".format(j))
            print("transform:{}".format(elevate_transform))
            ego_vehicle = CarlaDataProvider.request_new_actor('vehicle.lincoln.mkz2017',
                                                            elevate_transform,
                                                            rolename='hero_{}'.format(j))
            ego_vehicles.append(ego_vehicle)

            # set the spectator location above the first ego vehicle
            if j==0:
                spectator = CarlaDataProvider.get_world().get_spectator()
                ego_trans = ego_vehicle.get_transform()
                spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=50),
                                                            carla.Rotation(pitch=-90)))
        return ego_vehicles

    def _estimate_route_timeout(self):
        """
        Estimate the duration of the route
        """
        route_length = 0.0  # in meters

        prev_point = self.route[0][0][0]
        for current_point, _ in self.route[0][1:]:
            dist = current_point.location.distance(prev_point.location)
            route_length += dist
            prev_point = current_point

        return int(SECONDS_GIVEN_PER_METERS * route_length + INITIAL_SECONDS_DELAY)

    # pylint: disable=no-self-use
    def _draw_waypoints(self, world, waypoints, vertical_shift, persistency=-1):
        """
        Draw a list of waypoints at a certain height given in vertical_shift.
        """
        for w in waypoints:
            wp = w[0].location + carla.Location(z=vertical_shift)

            size = 0.2
            if w[1] == RoadOption.LEFT:  # Yellow
                color = carla.Color(255, 255, 0)
            elif w[1] == RoadOption.RIGHT:  # Cyan
                color = carla.Color(0, 255, 255)
            elif w[1] == RoadOption.CHANGELANELEFT:  # Orange
                color = carla.Color(255, 64, 0)
            elif w[1] == RoadOption.CHANGELANERIGHT:  # Dark Cyan
                color = carla.Color(0, 64, 255)
            elif w[1] == RoadOption.STRAIGHT:  # Gray
                color = carla.Color(128, 128, 128)
            else:  # LANEFOLLOW
                color = carla.Color(0, 255, 0) # Green
                size = 0.1

            world.debug.draw_point(wp, size=size, color=color, life_time=persistency)

        world.debug.draw_point(waypoints[0][0].location + carla.Location(z=vertical_shift), size=0.2,
                               color=carla.Color(0, 0, 255), life_time=persistency)
        world.debug.draw_point(waypoints[-1][0].location + carla.Location(z=vertical_shift), size=0.2,
                               color=carla.Color(255, 0, 0), life_time=persistency)

    def _scenario_sampling(self, potential_scenarios_definitions, random_seed=0):
        """
        The function used to sample the scenarios that are going to happen for this route.
        Args:
            potential_scenarios_definitions: OrderedDict, len(possible_scenarios) is the number of position that is possible to trigger scenarios along this route, 
                                and possible_scenarios[i] is a list of scenarios that is possible to be triggered at the ith position
        Returns:
            sampled_scenarios: list of Dict(), sampled from possible scenarios, sampled_scenarios[i] represents the ith scenario to be triggered along this route
        """

        # fix the random seed for reproducibility
        rgn = random.RandomState(random_seed)

        def position_sampled(scenario_choice, sampled_scenarios):
            """
            Check if a position was already sampled, i.e. used for another scenario
            """
            for existent_scenario in sampled_scenarios:
                # If the scenarios have equal positions then it is true.
                if compare_scenarios(scenario_choice, existent_scenario):
                    return True

            return False

        def select_scenario(list_scenarios):
            # priority to the scenarios with higher number: 10 has priority over 9, etc.
            higher_id = -1
            selected_scenario = None
            for scenario in list_scenarios:
                try:
                    scenario_number = int(scenario['name'].split('Scenario')[1])
                except:
                    scenario_number = -1

                if scenario_number >= higher_id:
                    higher_id = scenario_number
                    selected_scenario = scenario

            return selected_scenario

        def select_scenario_randomly(list_scenarios):
            # randomly select a scenario
            # if scenario3 in select list, select it with a probability, if not select randomly
            selected_scenario = None
            # for scenario in list_scenarios:
            #     if scenario['name'] == 'Scenario3':
            #         if rgn.random()>0.0:
            #             selected_scenario = rgn.choice(list_scenarios)
            #         selected_scenario = scenario
            selected_scenario = rgn.choice(list_scenarios)
            # if selected_scenario == None:
            #     selected_scenario = rgn.choice(list_scenarios)
            # record number of each type of scenario along this route
            if not selected_scenario['name'] in self.route_scenario_dic:
                self.route_scenario_dic[selected_scenario['name']] = 1
            else:
                self.route_scenario_dic[selected_scenario['name']] += 1

            # if selected_scenario['name'] != 'Scenario3':
            #     print(selected_scenario['name'])
            return selected_scenario

        # The idea is to randomly sample a scenario per trigger position.
        sampled_scenarios = []
        for trigger in potential_scenarios_definitions.keys():
            possible_scenarios = potential_scenarios_definitions[trigger]

            # scenario_choice = select_scenario(possible_scenarios) # original prioritized sampling
            scenario_choice = select_scenario_randomly(possible_scenarios) # random sampling
            if scenario_choice == None:
                continue
            print('load {} at (x={}, y={})'.format(scenario_choice['name'], scenario_choice['trigger_position']['x'], scenario_choice['trigger_position']['y']))
            del possible_scenarios[possible_scenarios.index(scenario_choice)]
            # Keep sampling and testing if this position is present on any of the scenarios.
            while position_sampled(scenario_choice, sampled_scenarios):
                if possible_scenarios is None or not possible_scenarios:
                    scenario_choice = None
                    break
                scenario_choice = rgn.choice(possible_scenarios)
                del possible_scenarios[possible_scenarios.index(scenario_choice)]

            if scenario_choice is not None:
                sampled_scenarios.append(scenario_choice)

        return sampled_scenarios

    def _build_scenario_instances(self, world, ego_vehicles, scenario_definitions,
                                  scenarios_per_tick=5, timeout=300, debug_mode=False) -> List:
        """
        Based on the parsed route and possible scenarios, build all the scenario classes.
        Args:
            world: Carla world
            ego_vehicles: list of Carla vehicle
            scenario_definitions: scenario_definitions[j] represents scenario to be triggered on the jth ego vehicle's route
        Returns:
            scenario_instance_vecs: scenario_instance_vecs[j] represents a list of scenario instance to meet with the jth ego vehicle
        """
        scenario_instance_vecs = []

        for j in range(len(scenario_definitions)):
            scenario_definition = scenario_definitions[j]
            scenario_instance_vec = []
            if debug_mode:
                for scenario in scenario_definition:
                    loc = carla.Location(scenario['trigger_position']['x'],
                                        scenario['trigger_position']['y'],
                                        scenario['trigger_position']['z']) + carla.Location(z=2.0)
                    world.debug.draw_point(loc, size=0.3, color=carla.Color(255, 0, 0), life_time=100000)
                    world.debug.draw_string(loc, str(scenario['name']), draw_shadow=False,
                                            color=carla.Color(0, 0, 255), life_time=100000, persistent_lines=True)

            for scenario_number, definition in enumerate(scenario_definition):
                # Get the class possibilities for this scenario number
                # TODO(gjh): USE REGISTRY TO DEFINE SCENARIOS
                scenario_class = NUMBER_CLASS_TRANSLATION[definition['name']]

                # Create the other actors that are going to appear
                if definition['other_actors'] is not None:
                    list_of_actor_conf_instances = self._get_actors_instances(definition['other_actors'])
                else:
                    list_of_actor_conf_instances = []
                # Create an actor configuration for the ego-vehicle trigger position

                egoactor_trigger_position = convert_json_to_transform(definition['trigger_position'])
                scenario_configuration = ScenarioConfiguration()
                scenario_configuration.other_actors = list_of_actor_conf_instances
                scenario_configuration.trigger_points = [egoactor_trigger_position]
                scenario_configuration.subtype = definition['scenario_type']
                scenario_configuration.ego_vehicle = ActorConfigurationData('vehicle.lincoln.mkz2017',
                                                                            ego_vehicles[j].get_transform(),
                                                                            'hero_{}'.format(j))
                route_var_name = "ScenarioRouteNumber{}".format(scenario_number)
                scenario_configuration.route_var_name = route_var_name
                try:
                    if definition['name']=='Scenario3':
                        scenario_instance = scenario_class(world, [ego_vehicles[j]], scenario_configuration,
                                                    criteria_enable=False, timeout=timeout, trigger_distance=self.trigger_distance)
                    else:
                        scenario_instance = scenario_class(world, [ego_vehicles[j]], scenario_configuration,
                                                    criteria_enable=False, timeout=timeout)
                    # Do a tick every once in a while to avoid spawning everything at the same time
                    if scenario_number % scenarios_per_tick == 0:
                        if CarlaDataProvider.is_sync_mode():
                            world.tick()
                        else:
                            world.wait_for_tick()
                except Exception as e:
                    print("Skipping scenario '{}' due to setup error: {}".format(definition['name'], e))
                    continue

                scenario_instance_vec.append(scenario_instance)
            scenario_instance_vecs.append(scenario_instance_vec)

        return scenario_instance_vecs

    def _build_scenario_parameter_instances(self, world:carla.libcarla.World, ego_vehicles: List, scenario_definitions: List,
                                  scenarios_per_tick:int=5, timeout:int=300, debug_mode:bool=False, scenario_parameter:dict=None)-> List:
        """
        Based on the parsed route and possible scenarios, build all the scenario classes.

        Args:
            world: carla world
            ego_vehicles: list of ego_vehicles
            scenario_definitions: list of scenario_definitions
            scenarios_per_tick: number of scenarios per tick
            timeout: number of timeout
            debug_mode: if open debug_mode
            scenario_parameter: a dict of predefined scenario_parameter in yaml file

        Returns:
            scenario_instance_vecs: list of scenario_instance_vecs
        """
        scenario_instance_vecs = []
        for j in range(len(scenario_definitions)):
            scenario_definition = scenario_definitions[j]
            scenario_instance_vec = []
            if debug_mode:
                for scenario in scenario_definition:
                    loc = carla.Location(scenario['trigger_position']['x'],
                                        scenario['trigger_position']['y'],
                                        scenario['trigger_position']['z']) + carla.Location(z=2.0)
                    world.debug.draw_point(loc, size=0.3, color=carla.Color(255, 0, 0), life_time=100000)
                    world.debug.draw_string(loc, str(scenario['name']), draw_shadow=False,
                                            color=carla.Color(0, 0, 255), life_time=100000, persistent_lines=True)

            for scenario_number, definition in enumerate(scenario_definition):
                # Get the class possibilities for this scenario number
                # NOTE(GJH): Use scenario_config.yaml to define scenarios
                scenario = scenario_parameter[definition['name']]
                scenario_class_name = selScenario(scenario)
                scenario_class = ScenarioClassRegistry[scenario_class_name]
                scenario_class_parameter = scenario[scenario_class_name]
                # Create the other actors that are going to appear
                if definition['other_actors'] is not None:
                    list_of_actor_conf_instances = self._get_actors_instances(definition['other_actors'])
                else:
                    list_of_actor_conf_instances = []
                # Create an actor configuration for the ego-vehicle trigger position

                egoactor_trigger_position = convert_json_to_transform(definition['trigger_position'])
                scenario_configuration = ScenarioConfiguration()
                scenario_configuration.other_actors = list_of_actor_conf_instances
                scenario_configuration.trigger_points = [egoactor_trigger_position]
                scenario_configuration.subtype = definition['scenario_type']
                scenario_configuration.ego_vehicle = ActorConfigurationData('vehicle.lincoln.mkz2017',
                                                                            ego_vehicles[j].get_transform(),
                                                                            'hero_{}'.format(j))
                route_var_name = "ScenarioRouteNumber{}".format(scenario_number)
                scenario_configuration.route_var_name = route_var_name
                scenario_instance = scenario_class(world, [ego_vehicles[j]], scenario_configuration,
                                                    criteria_enable=False, timeout=timeout, scenario_parameter=scenario_class_parameter)
                
                # Do a tick every once in a while to avoid spawning everything at the same time
                if scenario_number % scenarios_per_tick == 0:
                    if CarlaDataProvider.is_sync_mode():
                        world.tick()
                    else:
                        world.wait_for_tick()

                # except Exception as e:
                #     print("Skipping scenario '{}' due to setup error: {}".format(definition['name'], e))
                #     continue

                scenario_instance_vec.append(scenario_instance)
            scenario_instance_vecs.append(scenario_instance_vec)

        return scenario_instance_vecs


    def _get_actors_instances(self, list_of_antagonist_actors):
        """
        Get the full list of actor instances.
        """

        def get_actors_from_list(list_of_actor_def):
            """
                Receives a list of actor definitions and creates an actual list of ActorConfigurationObjects
            """
            sublist_of_actors = []
            for actor_def in list_of_actor_def:
                sublist_of_actors.append(convert_json_to_actor(actor_def))

            return sublist_of_actors

        list_of_actors = []
        # Parse vehicles to the left
        if 'front' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['front'])

        if 'left' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['left'])

        if 'right' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['right'])

        return list_of_actors

    # pylint: enable=no-self-use

    def _initialize_actors(self, config):
        """
        Set other_actors to the superset of all scenario actors
        """
        # Create the background activity of the route
        town_amount = {
            'Town01': 120,
            'Town02': 100,
            'Town03': 120,
            'Town04': 200,
            'Town05': 120, #120
            'Town06': 150,
            'Town07': 110,
            'Town08': 180,
            'Town09': 300,
            'Town10HD': 120, # town10 doesn't load properly for some reason
        }

        amount = town_amount[config.town] if config.town in town_amount else 0
        # amount_vehicle = amount
        amount_vehicle = int(amount)
        amount_pedestrain = int(amount)
        new_actors_vehicle = CarlaDataProvider.request_new_batch_actors('vehicle.*',
                                                                amount_vehicle,
                                                                carla.Transform(),
                                                                autopilot=True,
                                                                random_location=True,
                                                                rolename='background',
                                                                crazy_level = 0,
                                                                crazy_proportion = 0
                                                                )
        new_actors_pedestrian = CarlaDataProvider.request_new_batch_actors('walker.pedestrian.*',
                                                                amount_pedestrain,
                                                                carla.Transform(),
                                                                autopilot=True,
                                                                random_location=True,
                                                                rolename='background',
                                                                crazy_level = self.crazy_level,
                                                                crazy_proportion = self.crazy_proportion
                                                                )
        new_actors_bicycle = CarlaDataProvider.request_new_batch_actors('vehicle.diamondback.century',
                                                                amount_pedestrain,
                                                                carla.Transform(),
                                                                autopilot=True,
                                                                random_location=True,
                                                                rolename='background',
                                                                crazy_level = self.crazy_level,
                                                                crazy_proportion = self.crazy_proportion
                                                                )
                                                                
        # TODO: add other types of actors
        new_actors = new_actors_vehicle + new_actors_pedestrian + new_actors_bicycle

        if new_actors is None:
            raise Exception("Error: Unable to add the background activity, all spawn points were occupied")

        for _actor in new_actors:
            self.other_actors.append(_actor)

        # Add all the actors of the specific scenarios to self.other_actors
        for list_scenarios in self.list_scenarios:
            for scenario in list_scenarios:
                self.other_actors.extend(scenario.other_actors)

    def _create_behavior(self):
        """
        Basic behavior do nothing, i.e. Idle
        """
        scenario_trigger_distance = 1.5  # Max trigger distance between route and scenario
        behavior = []

        for ego_vehicle_id in range(len(self.list_scenarios)):
            behavior_tmp = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

            subbehavior = py_trees.composites.Parallel(name="Behavior",
                                                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)

            scenario_behaviors = []
            blackboard_list = []
            list_scenarios = self.list_scenarios[ego_vehicle_id]
            for i, scenario in enumerate(list_scenarios):
                if scenario.scenario.behavior is not None:
                    route_var_name = scenario.config.route_var_name

                    if route_var_name is not None:
                        scenario_behaviors.append(scenario.scenario.behavior)
                        blackboard_list.append([scenario.config.route_var_name,
                                                scenario.config.trigger_points[0].location])
                    else:
                        name = "{} - {}".format(i, scenario.scenario.behavior.name)
                        oneshot_idiom = oneshot_behavior(
                            name=name,
                            variable_name=name,
                            behaviour=scenario.scenario.behavior)
                        scenario_behaviors.append(oneshot_idiom)

            # Add behavior that manages the scenarios trigger conditions
            scenario_triggerer = ScenarioTriggerer(
                self.ego_vehicles[ego_vehicle_id],
                self.route[ego_vehicle_id],
                blackboard_list,
                scenario_trigger_distance,
                repeat_scenarios=False
            )

            subbehavior.add_child(scenario_triggerer)  # make ScenarioTriggerer the first thing to be checked
            subbehavior.add_children(scenario_behaviors)
            subbehavior.add_child(Idle())  # The behaviours cannot make the route scenario stop
            behavior_tmp.add_child(subbehavior)
            behavior.append(behavior_tmp)
        return behavior

    def _create_test_criteria(self):
        """
        """
        criteria_all = []
        for ego_vehicle_id in range(len(self.list_scenarios)):
            criteria = []
            route = convert_transform_to_location(self.route[ego_vehicle_id])
            collision_criterion = CollisionTest(self.ego_vehicles[ego_vehicle_id], terminate_on_failure=False)

            route_criterion = InRouteTest(self.ego_vehicles[ego_vehicle_id],
                                        route=route,
                                        offroad_max=30,
                                        terminate_on_failure=True)
                                        
            completion_criterion = RouteCompletionTest(self.ego_vehicles[ego_vehicle_id], route=route)

            outsidelane_criterion = OutsideRouteLanesTest(self.ego_vehicles[ego_vehicle_id], route=route)

            red_light_criterion = RunningRedLightTest(self.ego_vehicles[ego_vehicle_id])

            stop_criterion = RunningStopTest(self.ego_vehicles[ego_vehicle_id])

            blocked_criterion = ActorSpeedAboveThresholdTest(self.ego_vehicles[ego_vehicle_id],
                                                            speed_threshold=0.5,
                                                            below_threshold_max_time=30.0,
                                                            terminate_on_failure=True,
                                                            name="AgentBlockedTest")

            criteria.append(completion_criterion)
            criteria.append(outsidelane_criterion)
            criteria.append(collision_criterion)
            criteria.append(red_light_criterion)
            criteria.append(stop_criterion)
            criteria.append(route_criterion)
            criteria.append(blocked_criterion)
            criteria_all.append(criteria)
            
        return criteria_all

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
