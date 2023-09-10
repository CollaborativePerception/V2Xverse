#!/usr/bin/env python
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Object crash without prior vehicle action scenario:
The scenario realizes the user controlled ego vehicle
moving along the road and encountering a cyclist ahead.
"""

from __future__ import print_function

import math
import py_trees
import carla
import random

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (InitializeActor,
                                                                      ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      AccelerateToVelocity,
                                                                      HandBrakeVehicle,
                                                                      KeepVelocity,
                                                                      StopVehicle)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocationAlongRoute,
                                                                               InTimeToArrivalToVehicle,
                                                                               DriveDistance)
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import get_location_in_distance_from_wp
from . import ScenarioClassRegistry


@ScenarioClassRegistry.register
class StationaryObjectCrossing(BasicScenario):

    """
    This class holds everything required for a simple object crash
    without prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a stationary cyclist.

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60, scenario_parameter=None):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(
            config.trigger_points[0].location)
        self.timeout = timeout

        if scenario_parameter is None:
            # ego vehicle parameters
            self._ego_vehicle_distance_driven = 40
            # other vehicle parameters
            self._other_actor_target_velocity = 10 + 5
            # Timeout of scenario in seconds
            self._start_distance = 40
        else:
            # NOTE(GJH): Use scenario_parameter to assign.
            # ego vehicle parameters
            self._ego_vehicle_distance_driven = scenario_parameter['ego_vehicle_distance_driven']

            # other vehicle parameters
            self._other_actor_target_velocity = scenario_parameter['other_actor_target_velocity']
            # Timeout of scenario in seconds
            self._start_distance = scenario_parameter['start_distance']

        super(StationaryObjectCrossing, self).__init__("Stationaryobjectcrossing",
                                                       ego_vehicles,
                                                       config,
                                                       world,
                                                       debug_mode,
                                                       criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        # TODO(GJH):Add start distance as a parameter
        # _start_distance = self._start_distance
        lane_width = self._reference_waypoint.lane_width
        location, _ = get_location_in_distance_from_wp(
            self._reference_waypoint, self._start_distance)
        waypoint = self._wmap.get_waypoint(location)
        offset = {"orientation": 270, "position": 90, "z": 0.4, "k": 0.2}
        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + \
            offset['orientation']
        offset_location = carla.Location(
            offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
            offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z += offset['z']
        self.transform = carla.Transform(
            location, carla.Rotation(yaw=orientation_yaw))
        static = CarlaDataProvider.request_new_actor(
            'static.prop.container', self.transform)
        static.set_simulate_physics(True)
        self.other_actors.append(static)

    def _create_behavior(self):
        """
        Only behavior here is to wait
        """
        lane_width = self.ego_vehicles[0].get_world().get_map().get_waypoint(
            self.ego_vehicles[0].get_location()).lane_width
        # print(self.ego_vehicles[0])
        lane_width = lane_width + (1.25 * lane_width)

        # leaf nodes
        actor_stand = TimeOut(15)
        actor_removed = ActorDestroy(self.other_actors[0])
        end_condition = DriveDistance(
            self.ego_vehicles[0], self._ego_vehicle_distance_driven)

        # non leaf nodes
        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        scenario_sequence = py_trees.composites.Sequence()

        # building tree
        root.add_child(scenario_sequence)
        scenario_sequence.add_child(ActorTransformSetter(
            self.other_actors[0], self.transform))
        scenario_sequence.add_child(actor_stand)
        scenario_sequence.add_child(actor_removed)
        scenario_sequence.add_child(end_condition)

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicles[0])
        criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()


@ScenarioClassRegistry.register
class DynamicObjectCrossing(BasicScenario):

    """
    This class holds everything required for a simple object crash
    without prior vehicle action involving a vehicle and a cyclist/pedestrian,
    The ego vehicle is passing through a road,
    And encounters a cyclist/pedestrian crossing the road.

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, adversary_type=False, timeout=60, scenario_parameter=None):
        """
        Setup all relevant parameters and create scenario
        """
        self.world = world
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(
                config.trigger_points[0].location)
        self._trigger_location = config.trigger_points[0].location
        self.transform = None
        self.transform2 = None
        self.timeout = timeout
        self._spawn_attempted = 0
        self.vehilce_length = 0
        route_id = int(config.ego_vehicle.name.split("_")[1])
        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()[route_id]
        
        if scenario_parameter is None:
            self.trigger_distance = 10
            # ego vehicle parameters
            self._ego_vehicle_distance_driven = 40
            # other vehicle parameters
            self._other_actor_target_velocity = 3  # + 5
            self._other_actor_max_brake = 1.0
            self._time_to_reach = 10
            # flag to select either pedestrian (False) or cyclist (True)
            self._adversary_type = adversary_type
            self._walker_yaw = 0
            self._num_lane_changes = 1
            # Total Number of attempts to relocate a vehicle before spawning
            self._number_of_attempts = 500  # 20
        else:
            # NOTE(GJH): Use scenario_parameter to assign.
            self.trigger_distance = scenario_parameter['trigger_distance']
            # ego vehicle parameters
            self._ego_vehicle_distance_driven = scenario_parameter['ego_vehicle_distance_driven']
            # other vehicle parameters
            self._other_actor_target_velocity = scenario_parameter['other_actor_target_velocity']
            self._other_actor_max_brake = scenario_parameter['other_actor_max_brake']
            self._time_to_reach = scenario_parameter['time_to_reach']
            # flag to select either pedestrian (False) or cyclist (True)
            self._adversary_type = scenario_parameter['adversary_type']
            self._walker_yaw = scenario_parameter['walker_yaw']
            self._num_lane_changes = scenario_parameter['num_lane_changes']
            # Total Number of attempts to relocate a vehicle before spawning
            self._number_of_attempts = scenario_parameter['number_of_attempts']
    
        super(DynamicObjectCrossing, self).__init__("DynamicObjectCrossing",
                                                    ego_vehicles,
                                                    config,
                                                    world,
                                                    debug_mode,
                                                    criteria_enable=criteria_enable)

    def _calculate_base_transform(self, _start_distance, waypoint, k=1.0):

        lane_width = waypoint.lane_width

        # Patches false junctions
        if self._reference_waypoint.is_junction:
            stop_at_junction = False
        else:
            stop_at_junction = True

        location, _ = get_location_in_distance_from_wp(
            waypoint, _start_distance, stop_at_junction)
        waypoint = self._wmap.get_waypoint(location)
        offset = {"orientation": 270, "position": 90, "z": 0.6, "k": k}  # 1.0
        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + \
            offset['orientation']
        offset_location = carla.Location(
            offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
            offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z = self._trigger_location.z + offset['z']  # + 0.5
        return carla.Transform(location, carla.Rotation(yaw=orientation_yaw)), orientation_yaw

    def _calculate_base_vehicle_transform(self, _start_distance, waypoint):

        lane_width = waypoint.lane_width

        # Patches false junctions
        stop_at_junction = False

        location, _ = get_location_in_distance_from_wp(
            waypoint, _start_distance, stop_at_junction)
        waypoint = self._wmap.get_waypoint(location)
        offset = {"orientation": 270, "position": 90, "z": 0.6, "k": 1.0}
        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + \
            offset['orientation']
        offset_location = carla.Location(
            offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
            offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z = self._trigger_location.z + offset['z'] + 0.5
        return carla.Transform(location, carla.Rotation(yaw=orientation_yaw)), orientation_yaw

    def _spawn_adversary(self, transform, orientation_yaw):

        self._time_to_reach *= self._num_lane_changes

        if self._adversary_type is False:
            self._walker_yaw = orientation_yaw
            if self._other_actor_target_velocity < 0 or self._other_actor_target_velocity is None:
                self._other_actor_target_velocity = 3 + \
                    (0.4 * self._num_lane_changes)  # + 3
            else:
                pass
            walker = CarlaDataProvider.request_new_actor('walker.*', transform)
            adversary = walker
        else:
            self._other_actor_target_velocity = self._other_actor_target_velocity * \
                self._num_lane_changes
            first_vehicle = CarlaDataProvider.request_new_actor(
                'vehicle.diamondback.century', transform)
            first_vehicle.set_simulate_physics(enabled=False)
            adversary = first_vehicle

        return adversary

    def _spawn_blocker(self, transform, orientation_yaw):
        """
        Spawn the blocker prop that blocks the vision from the egovehicle of the jaywalker
        :return:
        """
        # static object transform
        shift = 1  # 0.9
        x_ego = self._reference_waypoint.transform.location.x
        y_ego = self._reference_waypoint.transform.location.y
        x_cycle = transform.location.x
        y_cycle = transform.location.y
        x_static = x_ego + shift * (x_cycle - x_ego)
        y_static = y_ego + shift * (y_cycle - y_ego)

        spawn_point_wp = self.ego_vehicles[0].get_world(
        ).get_map().get_waypoint(transform.location)

        self.transform2 = carla.Transform(carla.Location(x_static, y_static,
                                                         spawn_point_wp.transform.location.z),  # ori:  + 0.3
                                          carla.Rotation(yaw=orientation_yaw + 180))

        # 'static.prop.kiosk_01' 'static.prop.vendingmachine', 'static.prop.busstop'
        static = CarlaDataProvider.request_new_actor(
            'static.prop.kiosk_01', self.transform2)
        static.set_simulate_physics(enabled=False)

        return static

    def _spawn_blocker_vehicle(self, transform):
        """
        Spawn the blocker prop that blocks the vision from the egovehicle of the jaywalker
        :return:
        """
        # static object transform
        # shift = 0.9
        # x_ego = self._reference_waypoint.transform.location.x
        # y_ego = self._reference_waypoint.transform.location.y
        # x_cycle = transform.location.x
        # y_cycle = transform.location.y
        # x_static = x_ego + shift * (x_cycle - x_ego)
        # y_static = y_ego + shift * (y_cycle - y_ego)

        spawn_point_wp = self.ego_vehicles[0].get_world(
        ).get_map().get_waypoint(transform.location)

        self.transform2 = carla.Transform(carla.Location(transform.location.x, transform.location.y,
                                                         spawn_point_wp.transform.location.z),
                                          carla.Rotation(transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll))
        if self._reference_waypoint.is_junction:
            actor_type = "vehicle.*"
        else:
            blocker_vehicle_list = [
                "vehicle.volkswagen.t2", "vehicle.carlamotors.carlacola", "vehicle.volkswagen.t2"]
            actor_type = random.choice(blocker_vehicle_list)
        static = CarlaDataProvider.request_new_vehicle_actor(
            actor_type, self.transform2)
        self.vehilce_length = max(
            static.bounding_box.extent.x, static.bounding_box.extent.y)
        # print(static)

        # print(self.ego_vehicles[0])
        static.set_simulate_physics(enabled=False)

        return static

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        # cyclist transform
        _start_distance = self.trigger_distance  # 12, 9
        # We start by getting and waypoint in the closest sidewalk.
        waypoint = self._reference_waypoint
        vehicle_waypoint = None
        while True:
            wp_next = waypoint.get_right_lane()
            self._num_lane_changes += 1

            if wp_next is not None:
                if wp_next.lane_type == carla.LaneType.Driving:
                    vehicle_waypoint = wp_next

            if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
                break
            elif wp_next.lane_type == carla.LaneType.Shoulder:
                # Filter Parkings considered as Shoulders
                if wp_next.lane_width > 2:
                    _start_distance += 1.5
                    waypoint = wp_next
                break
            elif wp_next.lane_type == carla.LaneType.Driving:
                waypoint = wp_next
            else:
                _start_distance += 1.5
                waypoint = wp_next

        bias = 0
        while True:  # We keep trying to spawn avoiding props

            try:
                if vehicle_waypoint:  # vehicle_waypoint:  False
                    self.trig_distance = self.trigger_distance  # 10
                    # if vehicle_waypoint.is_junction:
                    #     stop_at_junction = False
                    # else:
                    #     stop_at_junction = True
                    # transform_for_vehicle = CarlaDataProvider.get_map().get_waypoint(self.transform.location).get_left_lane()
                    vehicle_location, _ = get_location_in_distance_from_wp(
                        vehicle_waypoint, _start_distance, stop_at_junction=True)
                    # print("spawn special vehicle")
                    waypoint_for_vehicle = CarlaDataProvider.get_map().get_waypoint(vehicle_location)
                    # transform_for_vehicle = self.transform.get_left_lane()
                    blocker = self._spawn_blocker_vehicle(
                        waypoint_for_vehicle.transform)
                    self.transform, orientation_yaw = self._calculate_base_vehicle_transform(
                        _start_distance + self.vehilce_length + 1, waypoint)
                    first_vehicle = self._spawn_adversary(
                        self.transform, orientation_yaw)
                else:
                    # print("spawn walker road blocker")
                    self.trig_distance = self.trigger_distance  # 10
                    self.transform, orientation_yaw = self._calculate_base_transform(
                        _start_distance+2, waypoint, k=1.4+bias)
                    first_vehicle = self._spawn_adversary(
                        self.transform, orientation_yaw)
                    self.transform, orientation_yaw = self._calculate_base_transform(
                        _start_distance, waypoint)
                    blocker = self._spawn_blocker(
                        self.transform, orientation_yaw)
                    self.transform, orientation_yaw = self._calculate_base_transform(
                        _start_distance+2, waypoint, k=1.4+bias)

                    # self.transform, orientation_yaw = self._calculate_base_transform(_start_distance, waypoint, k=1.4)
                break
            except RuntimeError as r:
                # We keep retrying until we spawn
                # print("Base transform is blocking objects ", self.transform)
                _start_distance += 0.4
                self._spawn_attempted += 1
                # print(_start_distance)
                if self._spawn_attempted >= self._number_of_attempts/2:
                    bias = 20*random.random()
                if self._spawn_attempted >= self._number_of_attempts:
                    raise r

        # Now that we found a possible position we just put the vehicle to the underground
        disp_transform = carla.Transform(
            carla.Location(self.transform.location.x,
                           self.transform.location.y,
                           self.transform.location.z - 500),
            self.transform.rotation)

        prop_disp_transform = carla.Transform(
            carla.Location(self.transform2.location.x,
                           self.transform2.location.y,
                           self.transform2.location.z - 500),
            self.transform2.rotation)

        first_vehicle.set_transform(disp_transform)
        blocker.set_transform(prop_disp_transform)
        first_vehicle.set_simulate_physics(enabled=False)
        blocker.set_simulate_physics(enabled=False)
        self.other_actors.append(first_vehicle)
        self.other_actors.append(blocker)

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        then after 60 seconds, a timeout stops the scenario
        """

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="OccludedObjectCrossing")
        lane_width = self._reference_waypoint.lane_width
        lane_width = lane_width + (lane_width * self._num_lane_changes)

        # + self._num_lane_changes  ## 12 + self._num_lane_changes  -> 12
        dist_to_trigger = self.trig_distance
        # leaf nodes
        start_behaviour = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="start")
        if self._ego_route is not None:
            start_condition = InTriggerDistanceToLocationAlongRoute(self.ego_vehicles[0],
                                                                    self._ego_route,
                                                                    self.transform.location,
                                                                    dist_to_trigger)
        else:
            start_condition = InTimeToArrivalToVehicle(self.ego_vehicles[0],
                                                       self.other_actors[0],
                                                       self._time_to_reach)

        actor_velocity = KeepVelocity(self.other_actors[0],
                                      self._other_actor_target_velocity,  # self._other_actor_target_velocity  8
                                      name="walker velocity")
        actor_drive = DriveDistance(self.other_actors[0],
                                    0.5 * lane_width,  # 0.5 * lane_width
                                    name="walker drive distance")
        actor_start_cross_lane = AccelerateToVelocity(self.other_actors[0],
                                                      1.0*40,  # *40
                                                      self._other_actor_target_velocity,  # self._other_actor_target_velocity  -> 8
                                                      name="walker crossing lane accelerate velocity")
        actor_cross_lane = DriveDistance(self.other_actors[0],
                                         lane_width,  # lane_width
                                         name="walker drive distance for lane crossing ")
        actor_stop_crossed_lane = StopVehicle(self.other_actors[0],
                                              self._other_actor_max_brake,
                                              name="walker stop")
        ego_pass_machine = DriveDistance(self.ego_vehicles[0],
                                         5,
                                         name="ego vehicle passed prop")
        actor_remove = ActorDestroy(self.other_actors[0],
                                    name="Destroying walker")
        static_remove = ActorDestroy(self.other_actors[1],
                                     name="Destroying Prop")
        end_condition = DriveDistance(self.ego_vehicles[0],
                                      self._ego_vehicle_distance_driven,
                                      name="End condition ego drive distance")

        # non leaf nodes

        scenario_sequence = py_trees.composites.Sequence()
        keep_velocity_other = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="keep velocity other")
        keep_velocity = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="keep velocity")

        # building tree

        root.add_child(scenario_sequence)

        # scenario_sequence.add_child(InitializeActor(1,2,3))

        scenario_sequence.add_child(ActorTransformSetter(self.other_actors[0], self.transform,
                                                         name='TransformSetterTS3walker'))
        scenario_sequence.add_child(ActorTransformSetter(self.other_actors[1], self.transform2,
                                                         name='TransformSetterTS3coca', physics=False))
        scenario_sequence.add_child(
            HandBrakeVehicle(self.other_actors[0], True))
        # scenario_sequence.add_child(start_condition)
        scenario_sequence.add_child(start_behaviour)
        scenario_sequence.add_child(
            HandBrakeVehicle(self.other_actors[0], False))
        scenario_sequence.add_child(keep_velocity)
        scenario_sequence.add_child(keep_velocity_other)
        scenario_sequence.add_child(actor_stop_crossed_lane)
        scenario_sequence.add_child(actor_remove)
        scenario_sequence.add_child(static_remove)
        scenario_sequence.add_child(end_condition)

        start_behaviour.add_child(start_condition)
        start_behaviour.add_child(TimeOut(10))
        keep_velocity.add_child(actor_velocity)
        keep_velocity.add_child(actor_drive)
        keep_velocity.add_child(TimeOut(20))
        keep_velocity_other.add_child(actor_start_cross_lane)
        keep_velocity_other.add_child(actor_cross_lane)
        keep_velocity_other.add_child(ego_pass_machine)
        keep_velocity_other.add_child(TimeOut(20))

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicles[0])
        criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
