#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
Object crash with prior vehicle action scenario:
The scenario realizes the user controlled ego vehicle
moving along the road and encounters a cyclist ahead after taking a right or left turn.
"""

from __future__ import print_function

import math
import py_trees
import random
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      KeepVelocity,
                                                                      HandBrakeVehicle)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocationAlongRoute,
                                                                               InTriggerDistanceToVehicle,
                                                                               InTriggerDistanceToVehicle_Away,
                                                                               DriveDistance)
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import generate_target_waypoint, generate_target_waypoint_in_route
from srunner.tools.scenario_helper import get_location_in_distance_from_wp,get_location_previous_from_wp,generate_target_waypoint_and_roadoption_in_route
from agents.navigation.local_planner import RoadOption
from . import ScenarioClassRegistry

def get_opponent_transform(added_dist, waypoint, trigger_location):
    """
    Calculate the transform of the adversary
    """
    lane_width = waypoint.lane_width

    offset = {"orientation": 270, "position": 90, "k": 1.0}
    _wp = waypoint.next(added_dist)
    if _wp:
        _wp = _wp[-1]
    else:
        raise RuntimeError("Cannot get next waypoint !")

    location = _wp.transform.location
    orientation_yaw = _wp.transform.rotation.yaw + offset["orientation"]
    position_yaw = _wp.transform.rotation.yaw + offset["position"]

    offset_location = carla.Location(
        offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
        offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
    location += offset_location
    location.z = trigger_location.z
    transform = carla.Transform(location, carla.Rotation(yaw=orientation_yaw))

    return transform

def get_right_transform(waypoint, trigger_location):
    """
    Calculate the transform of the adversary
    """
    lane_width = waypoint.lane_width

    offset = {"orientation": 270, "position": 90, "k": 1.0}
    _wp = waypoint.next(1)
    if _wp:
        _wp = _wp[-1]
    else:
        raise RuntimeError("Cannot get next waypoint !")

    location = _wp.transform.location
    orientation_yaw = _wp.transform.rotation.yaw + offset["orientation"]
    position_yaw = _wp.transform.rotation.yaw + offset["position"]

    offset_location = carla.Location(
        offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
        offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
    location += offset_location
    location.z = trigger_location.z
    transform = carla.Transform(location, carla.Rotation(yaw=orientation_yaw))

    return transform


def get_right_driving_lane(waypoint):
    """
    Gets the driving / parking lane that is most to the right of the waypoint
    as well as the number of lane changes done
    """
    lane_changes = 0

    while True:
        wp_next = waypoint.get_right_lane()
        lane_changes += 1

        if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
            break
        elif wp_next.lane_type == carla.LaneType.Shoulder:
            # Filter Parkings considered as Shoulders
            if is_lane_a_parking(wp_next):
                lane_changes += 1
                waypoint = wp_next
            break
        else:
            waypoint = wp_next

    return waypoint, lane_changes


def is_lane_a_parking(waypoint):
    """
    This function filters false negative Shoulder which are in reality Parking lanes.
    These are differentiated from the others because, similar to the driving lanes,
    they have, on the right, a small Shoulder followed by a Sidewalk.
    """

    # Parking are wide lanes
    if waypoint.lane_width > 2:
        wp_next = waypoint.get_right_lane()

        # That are next to a mini-Shoulder
        if wp_next is not None and wp_next.lane_type == carla.LaneType.Shoulder:
            wp_next_next = wp_next.get_right_lane()

            # Followed by a Sidewalk
            if wp_next_next is not None and wp_next_next.lane_type == carla.LaneType.Sidewalk:
                return True

    return False


@ScenarioClassRegistry.register
class VehicleTurningRight(BasicScenario):

    """
    This class holds everything required for a simple object crash
    with prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a cyclist after taking a right turn. (Traffic Scenario 4)

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario
        """

        self._other_actor_target_velocity = 3
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self._trigger_location = config.trigger_points[0].location
        self._other_actor_transform = None
        self._num_lane_changes = 0
        # Timeout of scenario in seconds
        self.timeout = timeout
        # Total Number of attempts to relocate a vehicle before spawning
        self._number_of_attempts = 100
        # Number of attempts made so far
        self._spawn_attempted = 0
        self.sidewalk_flag = False
        self.accum_distance = 0
        route_id = int(config.ego_vehicle.name.split("_")[1])
        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()[route_id]

        super(VehicleTurningRight, self).__init__("VehicleTurningRight",
                                                  ego_vehicles,
                                                  config,
                                                  world,
                                                  debug_mode,
                                                  criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        waypoint = self._reference_waypoint 

        # allow the turning right scenario to be complicated   
        if random.random()>=0.0:
            self.sidewalk_flag = True
            other_actor_loacatin, _ = get_location_previous_from_wp(waypoint,10, stop_at_junction=True)
            waypoint_spe = CarlaDataProvider.get_map().get_waypoint(other_actor_loacatin)

        # Get the waypoint of the right lane
        if self.sidewalk_flag:
            while True:
                wp_next = waypoint_spe.get_right_lane()
                if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
                    break
                elif wp_next.lane_type == carla.LaneType.Shoulder:
                    # Filter Parkings considered as Shoulders
                    if wp_next.lane_width > 2:
                        waypoint_spe = wp_next
                    break
                else:
                    waypoint_spe = wp_next

        # Original Code
        if not self.sidewalk_flag:
        # Get the waypoint right after the junction
            waypoint = generate_target_waypoint(self._reference_waypoint, 1)

            # Move a certain distance to the front
            start_distance = 8
            waypoint = waypoint.next(start_distance)[0]

            # Get the last driving lane to the right
            waypoint, self._num_lane_changes = get_right_driving_lane(waypoint)
            # And for synchrony purposes, move to the front a bit
            added_dist = self._num_lane_changes

        # Flag to identify if it is original (False is original)
        # self.sidewalk_flag = True
        while True:
            # Try to spawn the actor
            try:    
                if self.sidewalk_flag:
                    added_dist = 0
                    self._other_actor_transform = waypoint_spe.transform
                    world = CarlaDataProvider.get_world()
                    actors = world.get_actors()
                    bikes = actors.filter("walker.*")
                    loaction_to_do = self._other_actor_transform.location
                    loaction_to_do.z = loaction_to_do.z - 500
                    for actor in bikes:
                        if loaction_to_do.distance(actor.get_transform().location)<=2:
                            self.accum_distance += 2
                            loaction_to_do, _ = get_location_previous_from_wp(waypoint_spe, self.accum_distance, stop_at_junction=True)
                            waypoint_tp = CarlaDataProvider.get_map().get_waypoint(loaction_to_do)
                            while True:
                                wp_next = waypoint_tp.get_right_lane()
                                if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
                                    waypoint_spe = wp_next
                                    self.sidewalk_flag = True
                                    break
                                elif wp_next.lane_type == carla.LaneType.Shoulder:
                                    # Filter Parkings considered as Shoulders
                                    if wp_next.lane_width > 2:
                                        waypoint_tp = wp_next
                                        self.sidewalk_flag = True
                                    break
                                else:
                                    waypoint_tp = wp_next
                                    self.sidewalk_flag = False
                            # loaction_to_do, _ = get_location_previous_from_wp(waypoint_spe, start_distance + self.accum_distance, stop_at_junction=True)
                            print("Warning:The Scenario Actor is Placed in another place!")
                            # other_actor_location = CarlaDataProvider.get_map().get_waypoint(loaction_to_do).transform.location
                    
                    while True:
                        if loaction_to_do.z < 0.0:
                            loaction_to_do.z = loaction_to_do.z + 500.0
                        else:
                            break
                    # print(CarlaDataProvider.get_map().get_waypoint(loaction_to_do).lane_type)
                    # self._other_actor_transform = CarlaDataProvider.get_map().get_waypoint(loaction_to_do).transform
                    loaction_to_do = waypoint_spe.transform.location
                    position_yaw = waypoint_spe.transform.rotation.yaw
                    lane_width = waypoint_spe.lane_width
                    offset_location = carla.Location(
                        lane_width * math.cos(math.radians(position_yaw)),
                        lane_width * math.sin(math.radians(position_yaw)))
                    loaction_to_do += offset_location
                    loaction_to_do.z = self._trigger_location.z
                    self._other_actor_transform = carla.Transform(
                                                carla.Location(loaction_to_do.x,
                                                            loaction_to_do.y,
                                                            loaction_to_do.z + 2),
                                                waypoint_spe.transform.rotation)
                    # print(self._other_actor_transform)
                   
                    first_vehicle = CarlaDataProvider.request_new_actor(
                        "walker.*", self._other_actor_transform)
                    first_vehicle.set_simulate_physics(enabled=False)
                    # print("spawn sidewalk")
                else:
                    self._other_actor_transform = get_opponent_transform(added_dist, waypoint, self._trigger_location)
                    first_vehicle = CarlaDataProvider.request_new_actor(
                        'vehicle.diamondback.century', self._other_actor_transform)
                    first_vehicle.set_simulate_physics(enabled=False)
                break

            # Move the spawning point a bit and try again
            except RuntimeError as r:
                # In the case there is an object just move a little bit and retry
                print(" Base transform is blocking objects ", self._other_actor_transform)
                added_dist += 0.5
                self._spawn_attempted += 1
                if self._spawn_attempted >= self._number_of_attempts:
                    raise r

        # Set the transform to -500 z after we are able to spawn it
        actor_transform = carla.Transform(
            carla.Location(self._other_actor_transform.location.x,
                           self._other_actor_transform.location.y,
                           self._other_actor_transform.location.z - 500),
            self._other_actor_transform.rotation)
        first_vehicle.set_transform(actor_transform)
        first_vehicle.set_simulate_physics(enabled=False)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter the in the trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        ego vehicle has to avoid the crash after a right turn, but
        continue driving after the road is clear.If this does not happen
        within 90 seconds, a timeout stops the scenario.
        """

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="IntersectionRightTurn")

        lane_width = self._reference_waypoint.lane_width
        if not self.sidewalk_flag:
            dist_to_travel = lane_width + (1.10 * lane_width * self._num_lane_changes)
        else:
            dist_to_travel = 4 * lane_width + 30 + (1.10 * lane_width * self._num_lane_changes) + self.accum_distance

        # print(dist_to_travel)
        if not self.sidewalk_flag:
            bycicle_start_dist = 13 + dist_to_travel
        else:
            bycicle_start_dist = 13 + dist_to_travel

        if self.sidewalk_flag:
            trigger_distance = InTriggerDistanceToVehicle_Away(self.other_actors[0],
                                                                self.ego_vehicles[0],
                                                                2.0)
        else:
            if self._ego_route is not None:
                trigger_distance = InTriggerDistanceToLocationAlongRoute(self.ego_vehicles[0],
                                                                        self._ego_route,
                                                                        self._other_actor_transform.location,
                                                                        bycicle_start_dist)
            else:
                trigger_distance = InTriggerDistanceToVehicle(self.other_actors[0],
                                                          self.ego_vehicles[0],
                                                          bycicle_start_dist)

        actor_velocity = KeepVelocity(self.other_actors[0], self._other_actor_target_velocity)
        actor_traverse = DriveDistance(self.other_actors[0], 0.30 * dist_to_travel)
        post_timer_velocity_actor = KeepVelocity(self.other_actors[0], self._other_actor_target_velocity)
        post_timer_traverse_actor = DriveDistance(self.other_actors[0], 0.70 * dist_to_travel)
        end_condition = TimeOut(5)

        # non leaf nodes
        scenario_sequence = py_trees.composites.Sequence()

        actor_ego_sync = py_trees.composites.Parallel(
            "Synchronization of actor and ego vehicle",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        after_timer_actor = py_trees.composites.Parallel(
            "After timeout actor will cross the remaining lane_width",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # building the tree
        root.add_child(scenario_sequence)
        scenario_sequence.add_child(ActorTransformSetter(self.other_actors[0], self._other_actor_transform,
                                                         name='TransformSetterTS4'))
        scenario_sequence.add_child(HandBrakeVehicle(self.other_actors[0], True))
        scenario_sequence.add_child(trigger_distance)
        scenario_sequence.add_child(HandBrakeVehicle(self.other_actors[0], False))
        scenario_sequence.add_child(actor_ego_sync)
        scenario_sequence.add_child(after_timer_actor)
        scenario_sequence.add_child(end_condition)
        scenario_sequence.add_child(ActorDestroy(self.other_actors[0]))

        actor_ego_sync.add_child(actor_velocity)
        actor_ego_sync.add_child(actor_traverse)

        after_timer_actor.add_child(post_timer_velocity_actor)
        after_timer_actor.add_child(post_timer_traverse_actor)

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
class VehicleTurningLeft(BasicScenario):

    """
    This class holds everything required for a simple object crash
    with prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a cyclist after taking a left turn. (Traffic Scenario 4)

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario
        """

        self._other_actor_target_velocity = 10
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self._trigger_location = config.trigger_points[0].location
        self._other_actor_transform = None
        self._num_lane_changes = 0
        # Timeout of scenario in seconds
        self.timeout = timeout
        # Total Number of attempts to relocate a vehicle before spawning
        self._number_of_attempts = 100
        # Number of attempts made so far
        self._spawn_attempted = 0

        route_id = int(config.ego_vehicle.name.split("_")[1])
        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()[route_id]

        super(VehicleTurningLeft, self).__init__("VehicleTurningLeft",
                                                 ego_vehicles,
                                                 config,
                                                 world,
                                                 debug_mode,
                                                 criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """

        # Get the waypoint right after the junction
        waypoint = generate_target_waypoint(self._reference_waypoint, -1)

        # Move a certain distance to the front
        start_distance = 8
        waypoint = waypoint.next(start_distance)[0]

        # Get the last driving lane to the right
        waypoint, self._num_lane_changes = get_right_driving_lane(waypoint)
        # And for synchrony purposes, move to the front a bit
        added_dist = self._num_lane_changes

        while True:

            # Try to spawn the actor
            try:
                self._other_actor_transform = get_opponent_transform(added_dist, waypoint, self._trigger_location)
                first_vehicle = CarlaDataProvider.request_new_actor(
                    'vehicle.diamondback.century', self._other_actor_transform)
                first_vehicle.set_simulate_physics(enabled=False)
                break

            # Move the spawning point a bit and try again
            except RuntimeError as r:
                # In the case there is an object just move a little bit and retry
                print(" Base transform is blocking objects ", self._other_actor_transform)
                added_dist += 0.5
                self._spawn_attempted += 1
                if self._spawn_attempted >= self._number_of_attempts:
                    raise r

        # Set the transform to -500 z after we are able to spawn it
        actor_transform = carla.Transform(
            carla.Location(self._other_actor_transform.location.x,
                           self._other_actor_transform.location.y,
                           self._other_actor_transform.location.z - 500),
            self._other_actor_transform.rotation)
        first_vehicle.set_transform(actor_transform)
        first_vehicle.set_simulate_physics(enabled=False)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter the in the trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        ego vehicle has to avoid the crash after a left turn, but
        continue driving after the road is clear.If this does not happen
        within 90 seconds, a timeout stops the scenario.
        """

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="IntersectionLeftTurn")

        lane_width = self._reference_waypoint.lane_width
        dist_to_travel = lane_width + (1.10 * lane_width * self._num_lane_changes)

        bycicle_start_dist = 13 + dist_to_travel

        if self._ego_route is not None:
            trigger_distance = InTriggerDistanceToLocationAlongRoute(self.ego_vehicles[0],
                                                                     self._ego_route,
                                                                     self._other_actor_transform.location,
                                                                     bycicle_start_dist)
        else:
            trigger_distance = InTriggerDistanceToVehicle(self.other_actors[0],
                                                          self.ego_vehicles[0],
                                                          bycicle_start_dist)

        actor_velocity = KeepVelocity(self.other_actors[0], self._other_actor_target_velocity)
        actor_traverse = DriveDistance(self.other_actors[0], 0.30 * dist_to_travel)
        post_timer_velocity_actor = KeepVelocity(self.other_actors[0], self._other_actor_target_velocity)
        post_timer_traverse_actor = DriveDistance(self.other_actors[0], 0.70 * dist_to_travel)
        end_condition = TimeOut(5)

        # non leaf nodes
        scenario_sequence = py_trees.composites.Sequence()

        actor_ego_sync = py_trees.composites.Parallel(
            "Synchronization of actor and ego vehicle",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        after_timer_actor = py_trees.composites.Parallel(
            "After timeout actor will cross the remaining lane_width",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # building the tree
        root.add_child(scenario_sequence)
        scenario_sequence.add_child(ActorTransformSetter(self.other_actors[0], self._other_actor_transform,
                                                         name='TransformSetterTS4'))
        scenario_sequence.add_child(HandBrakeVehicle(self.other_actors[0], True))
        scenario_sequence.add_child(trigger_distance)
        scenario_sequence.add_child(HandBrakeVehicle(self.other_actors[0], False))
        scenario_sequence.add_child(actor_ego_sync)
        scenario_sequence.add_child(after_timer_actor)
        scenario_sequence.add_child(end_condition)
        scenario_sequence.add_child(ActorDestroy(self.other_actors[0]))

        actor_ego_sync.add_child(actor_velocity)
        actor_ego_sync.add_child(actor_traverse)

        after_timer_actor.add_child(post_timer_velocity_actor)
        after_timer_actor.add_child(post_timer_traverse_actor)

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
class VehicleTurningRoute(BasicScenario):

    """
    This class holds everything required for a simple object crash
    with prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a cyclist after taking a turn. This is the version used when the ego vehicle
    is following a given route. (Traffic Scenario 4)

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario
        """

        self._other_actor_target_velocity = 10
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self._trigger_location = config.trigger_points[0].location
        self._other_actor_transform = None
        self._num_lane_changes = 0
        # Timeout of scenario in seconds
        self.timeout = timeout
        # Total Number of attempts to relocate a vehicle before spawning
        self._number_of_attempts = 100
        # Number of attempts made so far
        self._spawn_attempted = 0
        self.sidewalk_flag = False
        self.accum_distance = 0
        route_id = int(config.ego_vehicle.name.split("_")[1])
        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()[route_id]

        super(VehicleTurningRoute, self).__init__("VehicleTurningRoute",
                                                  ego_vehicles,
                                                  config,
                                                  world,
                                                  debug_mode,
                                                  criteria_enable=criteria_enable)


    def _initialize_actors(self, config):
        """
        Custom initialization
        """

        # Get the waypoint right after the junction
        waypoint, road_option = generate_target_waypoint_and_roadoption_in_route(self._reference_waypoint, self._ego_route)
        if road_option == RoadOption.RIGHT or road_option == RoadOption.LANEFOLLOW:
            self.sidewalk_flag = True
        # Move a certain distance to the front
                # allow the turning right scenario to be complicated   
        if self.sidewalk_flag:
            other_actor_loacation, _ = get_location_previous_from_wp(waypoint,6, stop_at_junction=True)
            waypoint_spe = CarlaDataProvider.get_map().get_waypoint(other_actor_loacation)

            # Get the waypoint of the right lane
            while True:
                wp_next = waypoint_spe.get_right_lane()
                if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
                    break
                elif wp_next.lane_type == carla.LaneType.Shoulder:
                    # Filter Parkings considered as Shoulders
                    if wp_next.lane_width > 2:
                        waypoint_spe = wp_next
                    break
                else:
                    waypoint_spe = wp_next
        else:
            start_distance = 8
            waypoint = waypoint.next(start_distance)[0]

            # Get the last driving lane to the right
            waypoint, self._num_lane_changes = get_right_driving_lane(waypoint)
            # And for synchrony purposes, move to the front a bit
            added_dist = self._num_lane_changes

        while True:

            # Try to spawn the actor
            try:
                if self.sidewalk_flag:
                    added_dist = 0
                    self._other_actor_transform = waypoint_spe.transform
                    world = CarlaDataProvider.get_world()
                    actors = world.get_actors()
                    bikes = actors.filter("walker.*")
                    loaction_to_do = self._other_actor_transform.location
                    loaction_to_do.z = loaction_to_do.z - 500
                    for actor in bikes:
                        if loaction_to_do.distance(actor.get_transform().location)<=2:
                            self.accum_distance += 2
                            loaction_to_do, _ = get_location_previous_from_wp(waypoint_spe, self.accum_distance, stop_at_junction=True)
                            waypoint_spe = CarlaDataProvider.get_map().get_waypoint(loaction_to_do)
                            while True:
                                wp_next = waypoint_spe.get_right_lane()
                                if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
                                    break
                                elif wp_next.lane_type == carla.LaneType.Shoulder:
                                    # Filter Parkings considered as Shoulders
                                    if wp_next.lane_width > 2:
                                        waypoint_spe = wp_next
                                    break
                                else:
                                    waypoint_spe = wp_next
                            
                            # loaction_to_do, _ = get_location_previous_from_wp(waypoint_spe, start_distance + self.accum_distance, stop_at_junction=True)
                            print("Warning:The Scenario Actor is Placed in another place!")
                            # other_actor_location = CarlaDataProvider.get_map().get_waypoint(loaction_to_do).transform.location
                    
                    while True:
                        if loaction_to_do.z < 0.0:
                            loaction_to_do.z = loaction_to_do.z + 500.0
                        else:
                            break
                    # print(CarlaDataProvider.get_map().get_waypoint(loaction_to_do).lane_type)
                    # self._other_actor_transform = CarlaDataProvider.get_map().get_waypoint(loaction_to_do).transform
                    loaction_to_do = waypoint_spe.transform.location
                    position_yaw = waypoint_spe.transform.rotation.yaw
                    lane_width = waypoint_spe.lane_width
                    offset_location = carla.Location(
                        lane_width * math.cos(math.radians(position_yaw)),
                        lane_width * math.sin(math.radians(position_yaw)))
                    loaction_to_do += offset_location
                    loaction_to_do.z = self._trigger_location.z
                    self._other_actor_transform = carla.Transform(
                                                carla.Location(loaction_to_do.x,
                                                            loaction_to_do.y,
                                                            loaction_to_do.z + 2),
                                                waypoint_spe.transform.rotation)
                    # print(self._other_actor_transform)
                   
                    first_vehicle = CarlaDataProvider.request_new_actor(
                        "walker.*", self._other_actor_transform)
                    first_vehicle.set_simulate_physics(enabled=False)
                    # print("spawn sidewalk")
                else:
                    self._other_actor_transform = get_opponent_transform(added_dist, waypoint, self._trigger_location)
                    first_vehicle = CarlaDataProvider.request_new_actor(
                        'vehicle.diamondback.century', self._other_actor_transform)
                    first_vehicle.set_simulate_physics(enabled=False)
                break

            # Move the spawning point a bit and try again
            except RuntimeError as r:
                # In the case there is an object just move a little bit and retry
                # print(" Base transform is blocking objects ", self._other_actor_transform)
                added_dist += 0.5
                self._spawn_attempted += 1
                if self._spawn_attempted >= self._number_of_attempts:
                    raise r

        # Set the transform to -500 z after we are able to spawn it
        actor_transform = carla.Transform(
            carla.Location(self._other_actor_transform.location.x,
                           self._other_actor_transform.location.y,
                           self._other_actor_transform.location.z - 500),
            self._other_actor_transform.rotation)
        first_vehicle.set_transform(actor_transform)
        first_vehicle.set_simulate_physics(enabled=False)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter the in the trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        ego vehicle has to avoid the crash after a turn, but
        continue driving after the road is clear.If this does not happen
        within 90 seconds, a timeout stops the scenario.
        """

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="IntersectionRouteTurn")

        lane_width = self._reference_waypoint.lane_width
        dist_to_travel = lane_width + (1.10 * lane_width * self._num_lane_changes)

        bycicle_start_dist = 13 + dist_to_travel

        if not self.sidewalk_flag:
            dist_to_travel = lane_width + (1.10 * lane_width * self._num_lane_changes)
        else:
            dist_to_travel = 4 * lane_width + 30 + (1.10 * lane_width * self._num_lane_changes) + self.accum_distance

        if self.sidewalk_flag:
            trigger_distance = InTriggerDistanceToVehicle_Away(self.other_actors[0],
                                                                self.ego_vehicles[0],
                                                                2.0)
        else:
            if self._ego_route is not None:
                trigger_distance = InTriggerDistanceToLocationAlongRoute(self.ego_vehicles[0],
                                                                        self._ego_route,
                                                                        self._other_actor_transform.location,
                                                                        bycicle_start_dist)
            else:
                trigger_distance = InTriggerDistanceToVehicle(self.other_actors[0],
                                                            self.ego_vehicles[0],
                                                            bycicle_start_dist)

        actor_velocity = KeepVelocity(self.other_actors[0], self._other_actor_target_velocity)
        actor_traverse = DriveDistance(self.other_actors[0], 0.30 * dist_to_travel)
        post_timer_velocity_actor = KeepVelocity(self.other_actors[0], self._other_actor_target_velocity)
        post_timer_traverse_actor = DriveDistance(self.other_actors[0], 0.70 * dist_to_travel)
        end_condition = TimeOut(5)

        # non leaf nodes
        scenario_sequence = py_trees.composites.Sequence()

        actor_ego_sync = py_trees.composites.Parallel(
            "Synchronization of actor and ego vehicle",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        after_timer_actor = py_trees.composites.Parallel(
            "After timeout actor will cross the remaining lane_width",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # building the tree
        root.add_child(scenario_sequence)
        scenario_sequence.add_child(ActorTransformSetter(self.other_actors[0], self._other_actor_transform,
                                                         name='TransformSetterTS4'))
        scenario_sequence.add_child(HandBrakeVehicle(self.other_actors[0], True))
        scenario_sequence.add_child(trigger_distance)
        scenario_sequence.add_child(HandBrakeVehicle(self.other_actors[0], False))
        scenario_sequence.add_child(actor_ego_sync)
        scenario_sequence.add_child(after_timer_actor)
        scenario_sequence.add_child(end_condition)
        scenario_sequence.add_child(ActorDestroy(self.other_actors[0]))

        actor_ego_sync.add_child(actor_velocity)
        actor_ego_sync.add_child(actor_traverse)

        after_timer_actor.add_child(post_timer_velocity_actor)
        after_timer_actor.add_child(post_timer_traverse_actor)

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
