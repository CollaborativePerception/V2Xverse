#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the ScenarioManager implementations.
It must not be modified and is for reference only!
"""

from __future__ import print_function
import signal
import sys
import time

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.autoagents.agent_wrapper import AgentWrapper, AgentError
from leaderboard.envs.sensor_interface import SensorReceivedNoData
from leaderboard.utils.result_writer import ResultOutputProvider


class ScenarioManager(object):

    """
    Basic scenario manager class. This class holds all functionality
    required to start, run and stop a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.run_scenario()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. If needed, cleanup with manager.stop_scenario()
    """


    def __init__(self, timeout, debug_mode=False):
        """
        Setups up the parameters, which will be filled at load_scenario()
        """
        self.scenario = []
        self.scenario_tree = []
        self.scenario_class = None
        self.ego_vehicles = None
        self.other_actors = None
        
        self._debug_mode = debug_mode
        self._agent = None
        self._running = False
        self._timestamp_last_run = 0.0
        self._timeout = float(timeout)

        # Used to detect if the simulation is down
        watchdog_timeout = max(5, self._timeout - 2)
        self._watchdog = Watchdog(watchdog_timeout)

        # Avoid the agent from freezing the simulation
        agent_timeout = watchdog_timeout - 1
        self._agent_watchdog = Watchdog(agent_timeout)

        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        self.end_game_time = None

        # Register the scenario tick as callback for the CARLA world
        # Use the callback_id inside the signal handler to allow external interrupts
        signal.signal(signal.SIGINT, self.signal_handler)

        self.prev_ego_trans = None
        self.first_entry = []
        self.set_flag = True

        # record simulation time cost
        self.time_record = []
        self.c_time_record = []
        self.a_time_record = []
        self.sc_time_record = []
        
    def signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        self._running = False

    def cleanup(self):
        """
        Reset all parameters
        """
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        self.end_game_time = None

    def load_scenario(self, scenario, agent, rep_number, ego_vehicles_num, save_root=None, sensor_tf_list=None, is_crazy=False):
        """
        Load scenario instance and agent instance into manager
        Args:
            scenario: RouteScenario, scenario instance
            agent: agent instance
            rep_number: number of repetition
            ego_vehicles_num: number of ego vehicles
            save_root: root directory to save sensor data
        """

        GameTime.restart()
        self.is_crazy=is_crazy
        self._agent = AgentWrapper(agent)
        self.sensor_tf_list = sensor_tf_list
        self.scenario_class = scenario
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors
        self.repetition_number = rep_number
        self.scenario=[] # important!!!!
        self.scenario_tree=[] # important!!!!

        self.ego_vehicles_num = ego_vehicles_num
        if self.ego_vehicles_num != 1 :
            for ego_vehicle_id in range(ego_vehicles_num):
                self.scenario.append(scenario.scenario[ego_vehicle_id])
            for ego_vehicle_id in range(ego_vehicles_num):
                self.scenario_tree.append(self.scenario[ego_vehicle_id].scenario_tree)
        else:
            self.scenario.append(scenario.scenario)
            self.scenario_tree.append(self.scenario[0].scenario_tree)

        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)

        for vehicle_num in range(self.ego_vehicles_num):
            print("set ip sensor for ego vehicle {}".format(vehicle_num))
            self._agent.setup_sensors(self.ego_vehicles[vehicle_num], vehicle_num, save_root, self._debug_mode)
            self.first_entry.append(True)

    def run_scenario(self):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        self._watchdog.start()
        self._running = True
        
        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                if self.is_crazy:
                    # turn off traffic light at every frame to prevent from turning on in accident
                    [tf.set_state(carla.libcarla.TrafficLightState.Green) for tf in world.get_actors().filter("*traffic_light*") if hasattr(tf,"set_state")]
                    [tf.freeze(True) for tf in world.get_actors().filter("*traffic_light*") if hasattr(tf,"freeze")]
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                self._tick_scenario(timestamp)

    def _tick_scenario(self, timestamp):
        """
        Run next tick of scenario and the agent and tick the world.
        """

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()
            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()

            # Agent take action (eg. save data/produce control signal)
            try:
                ego_action = self._agent()

            # Special exception inside the agent that isn't caused by the agent
            except SensorReceivedNoData as e:
                raise RuntimeError(e)

            except Exception as e:
                raise AgentError(e)

            # destroy ego if it is not alive
            for vehicle_num in range(self.ego_vehicles_num):
                if  CarlaDataProvider.get_hero_actor(hero_id=vehicle_num) and not CarlaDataProvider.get_hero_actor(hero_id=vehicle_num).is_alive:
                    self._agent.del_ego_sensor(vehicle_num)
                    self._agent.cleanup_single(vehicle_num)
                    self._agent.cleanup_rsu(vehicle_num)
                    print("destroy ego type 1 : {}".format(vehicle_num))
                    CarlaDataProvider.remove_actor_by_id(CarlaDataProvider.get_hero_actor(hero_id=vehicle_num).id)

            # Execute driving control signal
            for vehicle_num in range(self.ego_vehicles_num):
                try:
                    ego = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
                    if ego:
                        if ego.is_alive:
                            self.ego_vehicles[vehicle_num].apply_control(ego_action[vehicle_num])
                except:
                    pass

            # Tick scenario
            for vehicle_num in range(self.ego_vehicles_num):
                try:
                    ego = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
                    if ego and ego.is_alive:
                        self.scenario_tree[vehicle_num].tick_once()
                except:
                    pass

            if self._debug_mode:
                print("\n")
                for vehicle_num in range(self.ego_vehicles_num):
                    # if self.scenario_tree[vehicle_num].status == py_trees.common.Status.RUNNING \
                    #    or self.scenario_tree[vehicle_num].status == py_trees.common.Status.INVALID:
                    try:
                        ego = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
                        if ego and ego.is_alive:
                        # if CarlaDataProvider.get_hero_actor(hero_id=vehicle_num).is_alive:
                            py_trees.display.print_ascii_tree(
                                self.scenario_tree[vehicle_num], show_status=True)
                            sys.stdout.flush()
                    except:
                        pass

            # destroy ego if it is not in RUNNING status or not alive
            stop_flag = 0
            for vehicle_num in range(self.ego_vehicles_num):
                if CarlaDataProvider.get_hero_actor(hero_id=vehicle_num) is None:
                    stop_flag += 1
                    if CarlaDataProvider.get_hero_actor(hero_id=vehicle_num):
                        self._agent.del_ego_sensor(vehicle_num)
                        self._agent.cleanup_single(vehicle_num)
                        self._agent.cleanup_rsu(vehicle_num)
                        print("destroy ego type 2 {}".format(vehicle_num))
                        CarlaDataProvider.remove_actor_by_id(CarlaDataProvider.get_hero_actor(hero_id=vehicle_num).id)
                    if stop_flag == self.ego_vehicles_num:
                        self._running = False
                
                elif self.scenario_tree[vehicle_num].status != py_trees.common.Status.RUNNING or not CarlaDataProvider.get_hero_actor(hero_id=vehicle_num).is_alive:
                    stop_flag += 1
                    if CarlaDataProvider.get_hero_actor(hero_id=vehicle_num):
                        self._agent.del_ego_sensor(vehicle_num)
                        self._agent.cleanup_single(vehicle_num)
                        self._agent.cleanup_rsu(vehicle_num)
                        print("destroy ego type 3 {}".format(vehicle_num))
                        print('flag1:', self.scenario_tree[vehicle_num].status != py_trees.common.Status.RUNNING)
                        print('flag2:', not CarlaDataProvider.get_hero_actor(hero_id=vehicle_num).is_alive)
                        CarlaDataProvider.remove_actor_by_id(CarlaDataProvider.get_hero_actor(hero_id=vehicle_num).id)
                    if stop_flag == self.ego_vehicles_num:
                        self._running = False

            # set spectator
            spectator = CarlaDataProvider.get_world().get_spectator()
            if CarlaDataProvider.get_hero_actor(hero_id=0):
                ego_trans = CarlaDataProvider.get_hero_actor(hero_id=0).get_transform()
                self.prev_ego_trans = ego_trans
            else:
                for vehicle_num in range(1, self.ego_vehicles_num): 
                    if CarlaDataProvider.get_hero_actor(hero_id=vehicle_num):
                        ego_trans = self.ego_vehicles[vehicle_num].get_transform()
                        self.prev_ego_trans = ego_trans
                        break
                # if none of the ego vehicle is alive
                ego_trans = self.prev_ego_trans
            spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=50),
                                                        carla.Rotation(pitch=-90)))

        if self._running and self.get_running_status():
            CarlaDataProvider.get_world().tick(self._timeout)

    def get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        return self._watchdog.get_status()

    def stop_scenario(self):
        """
        This function triggers a proper termination of a scenario
        """
        self._watchdog.stop()

        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time

        if self.get_running_status():
            # print("terminate ego vehicle in the first step {}".format(ego_vehicle_id))
            for ego_vehicle_id in range(len(self.ego_vehicles)):
                if self.scenario[ego_vehicle_id] is not None:
                    # print("terminate ego vehicle {}".format(ego_vehicle_id))
                    self.scenario[ego_vehicle_id].terminate()

            if self._agent is not None:
                self._agent.cleanup()
                self._agent = None

            if self.sensor_tf_list is not None:
                [_sensor.cleanup() for _sensor in self.sensor_tf_list]
                self.sensor_tf_list = None

            self.analyze_scenario()

    def analyze_scenario(self):
        """
        Analyzes and prints the results of the route
        """
        global_result = '\033[92m'+'SUCCESS'+'\033[0m'
        for ego_vehicle_id in range(len(self.ego_vehicles)):
            for criterion in self.scenario[ego_vehicle_id].get_criteria():
                if criterion.test_status != "SUCCESS":
                    global_result = '\033[91m'+'FAILURE'+'\033[0m'

            if self.scenario[ego_vehicle_id].timeout_node.timeout:
                global_result = '\033[91m'+'FAILURE'+'\033[0m'

            ResultOutputProvider(self, global_result, ego_vehicle_id)
