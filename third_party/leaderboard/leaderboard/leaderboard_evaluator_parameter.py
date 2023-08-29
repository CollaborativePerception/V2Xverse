#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Corporation.
# authors: German Ros (german.ros@intel.com), Felipe Codevilla (felipe.alcm@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Challenge Evaluator Routes

Provisional code to evaluate Autonomous Agents for the CARLA Autonomous Driving challenge
"""
from __future__ import print_function

import traceback
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
from distutils.version import LooseVersion
import importlib
import os
import sys
import gc
import pkg_resources
import sys
import carla
import copy
import signal
import torch
import time
import json
import yaml
import numpy as np
import random

from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from leaderboard.scenarios.scenario_manager import ScenarioManager
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.envs.sensor_interface import SensorInterface, SensorConfigurationInvalid
from leaderboard.autoagents.agent_wrapper import  AgentWrapper, AgentError
from leaderboard.utils.statistics_manager import StatisticsManager
from leaderboard.utils.route_indexer import RouteIndexer

class Logger(object):
    def __init__(self, file_name = 'temp.log', stream = sys.stdout) -> None:
        self.terminal = stream
        self.file_name = file_name

    def write(self, message):
        local_time = time.localtime(time.time())
        if message=='\n':
            time_str = ''
        else:
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
        try:
            self.terminal.write(message)
            self.log = open(self.file_name, "a")
            self.log.write(time_str+'  '+message)
            # self.log.close()
        except Exception as e:
            pass

    def flush(self):
        pass

sensors_to_icons = {
    'sensor.camera.rgb':        'carla_camera',
    'sensor.camera.semantic_segmentation': 'carla_camera',
    'sensor.camera.depth':      'carla_camera',
    'sensor.lidar.ray_cast':    'carla_lidar',
    'sensor.lidar.ray_cast_semantic':    'carla_lidar',
    'sensor.other.radar':       'carla_radar',
    'sensor.other.gnss':        'carla_gnss',
    'sensor.other.imu':         'carla_imu',
    'sensor.opendrive_map':     'carla_opendrive_map',
    'sensor.speedometer':       'carla_speedometer'
}


class LeaderboardEvaluator(object):

    """
    TODO: document me!
    """

    ego_vehicles = []

    # Tunable parameters
    client_timeout = 10.0  # in seconds
    wait_for_world = 20.0  # in seconds
    frame_rate = 20.0      # in Hz

    def __init__(self, args, statistics_manager):
        """
        Setup CARLA client and world
        Setup ScenarioManager
        """
        self.statistics_manager = statistics_manager
        self.sensors = None
        self.sensor_icons = []
        self._vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam

        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        self.client = carla.Client(args.host, int(args.port))
        if args.timeout:
            self.client_timeout = float(args.timeout)
        self.client.set_timeout(self.client_timeout)

        self.traffic_manager = self.client.get_trafficmanager(int(args.trafficManagerPort))

        dist = pkg_resources.get_distribution("carla")
        # if dist.version != 'leaderboard':
        #     if LooseVersion(dist.version) < LooseVersion('0.9.10'):
        #         raise ImportError("CARLA version 0.9.10.1 or newer required. CARLA version found: {}".format(dist))

        # Load agent
        module_name = os.path.basename(args.agent).split('.')[0]
        sys.path.insert(0, os.path.dirname(args.agent))
        self.module_agent = importlib.import_module(module_name)

        # Create the ScenarioManager
        self.manager = ScenarioManager(args.timeout, args.debug > 1)

        # Time control for summary purposes
        self._start_time = GameTime.get_time()
        self._end_time = None

        # Create the agent timer
        self._agent_watchdog = Watchdog(int(float(args.timeout)))
        signal.signal(signal.SIGINT, self._signal_handler)

        self.ego_vehicles_num = args.ego_num

    def _signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        if self._agent_watchdog and not self._agent_watchdog.get_status():
            raise RuntimeError("Timeout: Agent took too long to setup")
        elif self.manager:
            self.manager.signal_handler(signum, frame)

    def __del__(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """

        self._cleanup()
        if hasattr(self, 'manager') and self.manager:
            del self.manager
        if hasattr(self, 'world') and self.world:
            del self.world

    def _cleanup(self):
        """
        Remove and destroy all actors
        """

        # Simulation still running and in synchronous mode?
        if self.manager and self.manager.get_running_status() \
                and hasattr(self, 'world') and self.world:
            # Reset to asynchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(False)

        if self.manager:
            self.manager.cleanup()

        CarlaDataProvider.cleanup()

        for i, _ in enumerate(self.ego_vehicles):
            if self.ego_vehicles[i]:
                self.ego_vehicles[i].destroy()
                self.ego_vehicles[i] = None
        self.ego_vehicles = []

        if self._agent_watchdog._timer:
            self._agent_watchdog.stop()

        if hasattr(self, 'agent_instance') and self.agent_instance:
            self.agent_instance.destroy()
            self.agent_instance = None

        if hasattr(self, 'statistics_manager') and self.statistics_manager:
            for j in range(self.ego_vehicles_num):
                self.statistics_manager[j].scenario = None

    def _prepare_ego_vehicles(self, ego_vehicles, wait_for_ego_vehicles=False):
        """
        Spawn or update the ego vehicles
        """

        if not wait_for_ego_vehicles:
            for vehicle in ego_vehicles:
                self.ego_vehicles.append(CarlaDataProvider.request_new_actor(vehicle.model,
                                                                             vehicle.transform,
                                                                             vehicle.rolename,
                                                                             color=vehicle.color,
                                                                             vehicle_category=vehicle.category))

        else:
            ego_vehicle_missing = True
            while ego_vehicle_missing:
                self.ego_vehicles = []
                ego_vehicle_missing = False
                for ego_vehicle in ego_vehicles:
                    ego_vehicle_found = False
                    carla_vehicles = CarlaDataProvider.get_world().get_actors().filter('vehicle.*')
                    for carla_vehicle in carla_vehicles:
                        if carla_vehicle.attributes['role_name'] == ego_vehicle.rolename:
                            ego_vehicle_found = True
                            self.ego_vehicles.append(carla_vehicle)
                            break
                    if not ego_vehicle_found:
                        ego_vehicle_missing = True
                        break

            for i, _ in enumerate(self.ego_vehicles):
                self.ego_vehicles[i].set_transform(ego_vehicles[i].transform)

        # sync state
        CarlaDataProvider.get_world().tick()

    def _load_and_wait_for_world(self, args, town, ego_vehicles=None):
        """
        Load a new CARLA world and provide data to CarlaDataProvider
        """

        self.world = self.client.load_world(town)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / self.frame_rate
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        self.world.reset_all_traffic_lights()

        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(int(args.trafficManagerPort))
        CarlaDataProvider.set_random_seed(int(args.carlaProviderSeed))

        np.random.seed(int(args.carlaProviderSeed))
        random.seed(int(args.carlaProviderSeed))
        torch.manual_seed(int(args.carlaProviderSeed))
        torch.cuda.manual_seed_all(int(args.carlaProviderSeed))

        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(int(args.trafficManagerSeed))

        # Wait for the world to be ready
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()

        if CarlaDataProvider.get_map().name != town:
            raise Exception("The CARLA server uses the wrong map!"
                            "This scenario requires to use map {}".format(town))

    def _register_statistics(self, config, ego_car_num, checkpoint, entry_status, crash_message="",):
        """
        Computes and saved the simulation statistics
        """
        # register statistics
        
        current_stats_record = []
        current_stats_record.extend([[] for _ in range(0,ego_car_num)])
        for i in range(ego_car_num):
            
            current_stats_record[i] = self.statistics_manager[i].compute_route_statistics(
                config,
                self.manager.scenario_duration_system,
                self.manager.scenario_duration_game,
                crash_message
            )

            print("\033[1m> Registering the route statistics\033[0m")
            path_tmp = os.path.join(os.path.dirname(checkpoint), "ego_vehicle_{}".format(i), os.path.basename(checkpoint))
            folder_path = os.path.join(os.path.dirname(checkpoint), "ego_vehicle_{}".format(i))
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            self.statistics_manager[i].save_record(current_stats_record[i], config.index, path_tmp)
            self.statistics_manager[i].save_entry_status(entry_status, False, path_tmp)

    def _load_and_run_scenario(self, args, config):
        """
        Load and run the scenario given by config.

        Depending on what code fails, the simulation will either stop the route and
        continue from the next one, or report a crash and stop.

        Args:
            args: argparse.Namespace, global config
            config: srunner.scenarioconfigs.route_scenario_configuration.RouteScenarioConfiguration, config for route scenarios

        """
        crash_message = ""
        entry_status = "Started"

        print("\n\033[1m========= Preparing {} (repetition {}) =========".format(config.name, config.repetition_index))
        print("> Setting up the agent\033[0m")

        # Prepare the statistics of the route
        for j in range(self.ego_vehicles_num):
            self.statistics_manager[j].set_route(config.name, config.index)

        # Set up the user's agent, and the timer self._agent_watchdog to avoid freezing the simulation
        try:
            self._agent_watchdog.start()
            # agent_class_name for example 'AutoPilot', 'PnP_Agent' .etc
            agent_class_name = getattr(self.module_agent, 'get_entry_point')()
            self.agent_instance = getattr(self.module_agent, agent_class_name)(args.agent_config, 
                                                                               self.ego_vehicles_num)
            config.agent = self.agent_instance
            if hasattr(self.agent_instance, "get_save_path"):
                print("Data Generation Confirmed!")
                config.save_path_root = self.agent_instance.get_save_path()
                log_root_dir = config.save_path_root
            else:
                print("Evaluation Process!")
                config.save_path_root = None
                log_root_dir = os.path.dirname(os.environ["CHECKPOINT_ENDPOINT"])
                try:
                    log_root_dir = self.agent_instance.get_save_path()
                except:
                    print('load save path failed')

            # Log simulation information(or error), every printed string will be logged in file log.log
            log_dir = None
            log_dir = os.path.join(log_root_dir,'log')
            self.log_dir = log_dir
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            log_file_dir = os.path.join(log_dir,'log.log')
            self.log_file_dir = log_file_dir
            sys.stdout = Logger(log_file_dir)
            args_file_dir = os.path.join(log_dir,'args.json')
            args_dict = vars(args)
            json_str = json.dumps(args_dict, indent=2)
            with open(args_file_dir, 'w') as json_file:
                json_file.write(json_str)
            print("{} (repetition {}) Log file initialized!".format(config.name, config.repetition_index))

            # Check and store the sensors
            if not self.sensors:
                self.sensors = self.agent_instance.sensors()
                track = self.agent_instance.track
                AgentWrapper.validate_sensor_configuration(self.sensors, track, args.track)
                self.sensor_icons = [sensors_to_icons[sensor['type']] for sensor in self.sensors]
                for j in range(self.ego_vehicles_num):
                    self.statistics_manager[j].save_sensors(self.sensor_icons, args.checkpoint)
            self._agent_watchdog.stop()

        except SensorConfigurationInvalid as e:
            # The sensors are invalid -> set the ejecution to rejected and stop
            print("\n\033[91mThe sensor's configuration used is invalid:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()
            traceback.print_exc(file=open(log_file_dir,'a'))
            crash_message = "Agent's sensors were invalid"
            entry_status = "Rejected"
            # self._register_statistics(config, args.ego_num, args.checkpoint, entry_status, crash_message)
            self._cleanup()
            sys.exit(-1)

        except Exception as e:
            # The agent setup has failed -> start the next route
            print("\n\033[91mCould not set up the required agent:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()
            traceback.print_exc(file=open(log_file_dir,'a'))
            crash_message = "Agent couldn't be set up"
            # self._register_statistics(config,  args.ego_num, args.checkpoint, entry_status, crash_message)
            self._cleanup()
            return

        # Load the world and the scenario
        print("\033[1m> Loading the world\033[0m")    
        try:
            # create world
            self._load_and_wait_for_world(args, config.town, config.ego_vehicles)
            if args.crazy_level != 0:
                # crazy traffic light
                print("crazy traffic lights")
                [tf.set_state(carla.libcarla.TrafficLightState.Off) for tf in self.world.get_actors().filter("*traffic_light*") if hasattr(tf,"set_state")]
                [tf.freeze(True) for tf in self.world.get_actors().filter("*traffic_light*") if hasattr(tf,"freeze")]
            self._prepare_ego_vehicles(config.ego_vehicles, False)
            with open(args.scenario_parameter, 'r', encoding='utf-8') as f:
                scenario_parameter = yaml.load(f.read(), Loader=yaml.FullLoader)
            # NOTE(GJH)：change the arg of RouteScenario - add the arg of scenario_config
            # print(args.scenario_parameter)
            scenario = RouteScenario(world=self.world, config=config, debug_mode=args.debug, \
                ego_vehicles_num=self.ego_vehicles_num, crazy_level=args.crazy_level, \
                crazy_propotion=args.crazy_propotion, log_dir=log_dir,scenario_parameter=scenario_parameter)
            config.trajectory=scenario.get_new_config_trajectory()
            if self.ego_vehicles_num != 1 :
                for j in range(self.ego_vehicles_num):
                    self.statistics_manager[j].set_scenario(scenario.scenario[j])
            else:
                self.statistics_manager[0].set_scenario(scenario.scenario)


            # self.agent_instance._init()
            # self.agent_instance.sensor_interface = SensorInterface()

            # Night mode
            if config.weather.sun_altitude_angle < 0.0:
                for vehicle in scenario.ego_vehicles:
                    vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))

            # Load scenario and agent into manager then prepare to run it
            if args.record:
                self.client.start_recorder("{}/{}_rep{}.log".format(args.record, config.name, config.repetition_index))
            self.manager.load_scenario(scenario, self.agent_instance, config.repetition_index, self.ego_vehicles_num, save_root=config.save_path_root, sensor_tf_list=scenario.get_sensor_tf(), is_crazy=(args.crazy_level != 0))

        except Exception as e:
            # The scenario is wrong -> set the ejecution to crashed and stop
            print("\n\033[91mThe scenario could not be loaded:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()
            traceback.print_exc(file=open(log_file_dir,'a'))

            crash_message = "Simulation crashed"
            entry_status = "Crashed"

            self._register_statistics(config,  args.ego_num, args.checkpoint, entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()

            self._cleanup()
            sys.exit(-1)

        print("\033[1m> Running the route\033[0m")

        # Run the scenario
        try:
            self.manager.run_scenario()
        except AgentError as e:
            # The agent has failed -> stop the route
            print("\n\033[91mStopping the route, the agent has crashed:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()
            traceback.print_exc(file=open(log_file_dir,'a'))

            crash_message = "Agent crashed"
        except Exception as e:
            print("\n\033[91mError during the simulation:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()
            traceback.print_exc(file=open(log_file_dir,'a'))

            crash_message = "Run scenario crashed"
            entry_status = "Crashed"

        # Stop the scenario
        try:
            print("\033[1m> Stopping the route\033[0m")
            self.manager.stop_scenario()
            #GXK111
            self._register_statistics(config,  args.ego_num, args.checkpoint, entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()

            # Remove all actors
            scenario.remove_all_actors()

            self._cleanup()
            # Final clean up!
            # print(self.world.get_actors())
            # print(self.world.get_actors().filter("*sensor*"))
            for zombie in self.world.get_actors().filter("*sensor*"):
                zombie.stop()
                zombie.destroy()
                zombie = None            

        except Exception as e:
            print("\n\033[91mFailed to stop the scenario, the statistics might be empty:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()
            traceback.print_exc(file=open(log_file_dir,'a'))

            crash_message = "Simulation crashed"

        # if crash_message == "Simulation crashed":
        #     sys.exit(-1)

    def run(self, args: argparse.Namespace):
        """
        Run the challenge mode
        """

        route_indexer = RouteIndexer(args.routes, args.scenarios, args.repetitions)

        if args.resume:
            route_indexer.resume(args.checkpoint)
            for i in range(self.ego_vehicles_num):
                self.statistics_manager[i].resume(args.checkpoint)
        else:
            for i in range(self.ego_vehicles_num):              
                self.statistics_manager[i].clear_record(args.checkpoint)
            route_indexer.save_state(args.checkpoint)

        print("Start Running!")
        while route_indexer.peek():
            try:
                # setup, load config of the next route
                config = route_indexer.next()

                # run
                self._load_and_run_scenario(args, config)

                for i in range(args.ego_num):
                    folder_path = os.path.join(os.path.dirname(args.checkpoint), "ego_vehicle_{}".format(i))
                    if not os.path.exists(folder_path):
                        os.mkdir(folder_path)
                    path_tmp = os.path.join(os.path.dirname(args.checkpoint), "ego_vehicle_{}".format(i), os.path.basename(args.checkpoint))
                    route_indexer.save_state(path_tmp)

            except Exception as e:
                print('route error:',e)

        # save global statistics
        print("\033[1m> Registering the global statistics\033[0m")
        # TODO: save global records for every statistics manager.
        try:
            for i in range(self.ego_vehicles_num):
                global_stats_record = self.statistics_manager[i].compute_global_statistics(route_indexer.total)
                # print("------------ego_{}---------".format(i))
                # print(global_stats_record)
                path_tmp = os.path.join(os.path.dirname(args.checkpoint), "ego_vehicle_{}".format(i), os.path.basename(args.checkpoint)) 
                self.statistics_manager[i].save_global_record(global_stats_record, self.sensor_icons, route_indexer.total, path_tmp)
        except Exception as e:
            print('route error:',e)
            traceback.print_exc()
            traceback.print_exc(file=open(self.log_file_dir,'a'))      

def main():
    description = "CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n"

    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default='40000', help='TCP port to listen to (default: 2000)')
    parser.add_argument('--trafficManagerPort', default='2702',
                        help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--trafficManagerSeed', default='1',
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--carlaProviderSeed', default='2000',
                        help='Seed used by the CarlaProvider (default: 2000)')
    parser.add_argument('--debug', type=int, help='Run with debug output', default=0)
    parser.add_argument('--record', type=str, default='',
                        help='Use CARLA recording feature to create a recording of the scenario')
    parser.add_argument('--timeout', default="600.0",
                        help='Set the CARLA client timeout value in seconds')

    # simulation setup
    parser.add_argument('--routes',
                        default='third_party/leaderboard/data/evaluation_routes/final/town05_short_r0.xml',
                        help='Name of the route to be executed. Point to the route_xml_file to be executed.')
    parser.add_argument('--scenarios',
                        default='third_party/leaderboard/data/scenarios/town05_all_scenarios.json',
                        help='Name of the scenario annotation file to be mixed with the route.')
    parser.add_argument('--scenario_parameter', 
                        default='third_party/leaderboard/scenarios/scenario_para.yaml',
                        help='Defination of the scenario parameters.')
    parser.add_argument('--repetitions',
                        type=int,
                        default=1,
                        help='Number of repetitions per route.')
    # crazy level: 0-5, the probability of ignoring front car.
    # crazy propotion: the probability of a car is crazy 
    parser.add_argument('--crazy-level',type=int,  default=3, help='Level background vehicles driving obey rule')
    parser.add_argument('--crazy-propotion', type=int, default=70, help='The number of background vehicles driving obey rule')


    # agent-related options
    parser.add_argument("-a", "--agent", type=str, default='third_party/leaderboard/team_code/pnp_agent.py', help="Path to Agent's py file to evaluate")
    parser.add_argument("--agent-config", type=str, default='third_party/leaderboard/team_code/pnp_config.py', help="Path to Agent's configuration file")

    parser.add_argument("--track", type=str, default='SENSORS', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=int, default=0, help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str,
                        default='/GPFS/data/gjliu/Auto-driving/Cop3/results/eval/test/results.json',
                        help="Path to checkpoint used for saving statistics and resuming")
    parser.add_argument('--ego-num', type=int, default=1, help='The number of ego vehicles')

    arguments = parser.parse_args()

    if not os.path.exists(os.environ["SAVE_PATH"]):
        os.makedirs(os.environ["SAVE_PATH"])
    if not os.path.exists(os.path.dirname(os.environ["CHECKPOINT_ENDPOINT"])):
        os.makedirs(os.path.dirname(os.environ["CHECKPOINT_ENDPOINT"]))
        
    statistics_manager_all = []
    for i in range(arguments.ego_num):
        statistics_manager = StatisticsManager(ego_car_id=i)
        statistics_manager_all.append(statistics_manager)

    try:
        leaderboard_evaluator = LeaderboardEvaluator(arguments, statistics_manager_all)
        leaderboard_evaluator.run(arguments)

    except Exception as e:
        traceback.print_exc()
    finally:
        del leaderboard_evaluator


if __name__ == '__main__':
    main()
