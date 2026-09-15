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
import shutil

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
from leaderboard.utils.checkpoint_tools import fetch_dict, create_default_json_msg, save_dict

def check_log_file(file_path, target_string):
    try:
        # 打开文件
        with open(file_path, 'r') as file:
            # 读取文件内容
            file_content = file.read()

            # 判断文件内容中是否包含特定字符
            if target_string in file_content:
                # print(f"文件中包含目标字符串 '{target_string}'")
                return True
            else:
                # print(f"文件中不包含目标字符串 '{target_string}'")
                return False

    except FileNotFoundError:
        print(f"文件 '{file_path}' 不存在")

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

def backup_script(full_path, folders_to_save=["simulation/leaderboard/leaderboard","simulation/leaderboard/team_code", "simulation/scenario_runner"]):
    target_folder = os.path.join(full_path, 'scripts')
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    else:
        return
    
    current_path = os.path.dirname(__file__)  # __file__ refer to this file, then the dirname is "?/tools"

    for folder_name in folders_to_save:
        ttarget_folder = os.path.join(target_folder, folder_name)
        source_folder = os.path.join(current_path, f'../../../{folder_name}')
        shutil.copytree(source_folder, ttarget_folder)

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
        self.reuse_agent = bool(getattr(args, "reuse_agent", False))
        self._best_metrics = None
        self._received_sigint = False
        self._teardown_profile = CarlaDataProvider.is_teardown_profile_enabled()

    def _get_rl_trainer(self):
        if not getattr(self, "agent_instance", None):
            return None
        infer = getattr(self.agent_instance, "infer", None)
        if infer is None:
            return None
        return getattr(infer, "rl_trainer", None)

    def _is_rl_eval_only(self):
        if not getattr(self, "agent_instance", None):
            return False
        infer = getattr(self.agent_instance, "infer", None)
        if infer is None:
            return False
        return bool(getattr(infer, "rl_eval_only", False))

    def _maybe_save_best(self, found_routes, args):
        best_every = int(getattr(args, "best_every", 0) or 0)
        if best_every <= 0:
            return
        if found_routes <= 0 or (found_routes % best_every) != 0:
            return

        global_record = self.statistics_manager[0].compute_global_statistics(found_routes)
        score = float(global_record.scores.get("score_composed", 0.0))
        completion = float(global_record.scores.get("score_route", 0.0))
        infractions = global_record.infractions
        collisions_total = (
            float(infractions.get("collisions_pedestrian", 0.0))
            + float(infractions.get("collisions_vehicle", 0.0))
            + float(infractions.get("collisions_layout", 0.0))
        )
        collisions_per_route = collisions_total / max(1, found_routes)

        candidate = {
            "score": score,
            "collisions_per_route": collisions_per_route,
            "completion": completion,
            "routes": int(found_routes),
            "timestamp": datetime.utcnow().isoformat(),
        }

        def _is_better(a, b, eps=1e-6):
            if a["score"] > b["score"] + eps:
                return True
            if abs(a["score"] - b["score"]) <= eps:
                if a["collisions_per_route"] < b["collisions_per_route"] - eps:
                    return True
                if abs(a["collisions_per_route"] - b["collisions_per_route"]) <= eps:
                    if a["completion"] > b["completion"] + eps:
                        return True
            return False

        if self._best_metrics is None or _is_better(candidate, self._best_metrics):
            self._best_metrics = candidate
            if self._is_rl_eval_only():
                return
            rl_trainer = self._get_rl_trainer()
            if rl_trainer is None or not rl_trainer.log_dir:
                return
            ckpt_dir = os.path.join(rl_trainer.log_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "ckpt_best.pt")
            rl_trainer.save_checkpoint(ckpt_path)
            meta_path = os.path.join(ckpt_dir, "ckpt_best_meta.json")
            meta = dict(candidate)
            meta["update_count"] = int(getattr(rl_trainer, "update_count", 0))
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

    def _flush_rl_pending(self):
        if self._is_rl_eval_only():
            return
        rl_trainer = self._get_rl_trainer()
        if rl_trainer is None:
            return
        if hasattr(rl_trainer, "flush_pending"):
            try:
                rl_trainer.flush_pending()
            except Exception:
                pass

    def _signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        self._received_sigint = True
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

    @staticmethod
    def _safe_destroy_actor_handle(actor, stop_first=False):
        """
        Best-effort actor destroy for idempotent cleanup.
        """
        if actor is None:
            return False

        if stop_first and hasattr(actor, "stop"):
            try:
                actor.stop()
            except Exception:
                pass

        try:
            if hasattr(actor, "is_alive") and not actor.is_alive:
                return False
        except Exception:
            return False

        try:
            actor.destroy()
            return True
        except RuntimeError:
            return False
        except Exception:
            return False

    def _cleanup(self):
        """
        Remove and destroy all actors
        """
        cleanup_start = time.time() if self._teardown_profile else None
        manager_cleanup_ms = 0.0
        provider_cleanup_ms = 0.0
        ego_destroy_ms = 0.0
        agent_destroy_ms = 0.0

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
            manager_cleanup_start = time.time() if self._teardown_profile else None
            self.manager.cleanup()
            if self._teardown_profile:
                manager_cleanup_ms = (time.time() - manager_cleanup_start) * 1000.0

        provider_cleanup_start = time.time() if self._teardown_profile else None
        CarlaDataProvider.cleanup()
        if self._teardown_profile:
            provider_cleanup_ms = (time.time() - provider_cleanup_start) * 1000.0

        ego_destroy_start = time.time() if self._teardown_profile else None
        for i, _ in enumerate(self.ego_vehicles):
            if self.ego_vehicles[i]:
                self._safe_destroy_actor_handle(self.ego_vehicles[i])
                self.ego_vehicles[i] = None
        self.ego_vehicles = []
        if self._teardown_profile:
            ego_destroy_ms = (time.time() - ego_destroy_start) * 1000.0

        if self._agent_watchdog._timer:
            self._agent_watchdog.stop()

        if hasattr(self, 'agent_instance') and self.agent_instance and not self.reuse_agent:
            agent_destroy_start = time.time() if self._teardown_profile else None
            self.agent_instance.destroy()
            self.agent_instance = None
            if self._teardown_profile:
                agent_destroy_ms = (time.time() - agent_destroy_start) * 1000.0

        if hasattr(self, 'statistics_manager') and self.statistics_manager:
            for j in range(self.ego_vehicles_num):
                self.statistics_manager[j].scenario = None

        if self._teardown_profile and cleanup_start is not None:
            total_cleanup_ms = (time.time() - cleanup_start) * 1000.0
            print(
                "[EVAL_CLEANUP] manager_ms={:.1f} provider_ms={:.1f} ego_destroy_ms={:.1f} "
                "agent_destroy_ms={:.1f} total_ms={:.1f}".format(
                    manager_cleanup_ms,
                    provider_cleanup_ms,
                    ego_destroy_ms,
                    agent_destroy_ms,
                    total_cleanup_ms,
                )
            )

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
        settings.deterministic_ragdolls = True
        self.world.apply_settings(settings)

        self.world.reset_all_traffic_lights()

        # self.world.set_pedestrians_seed(int(args.trafficManagerSeed))

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

        # Notify agent about route end for RL updates
        if hasattr(self, "agent_instance") and hasattr(self.agent_instance, "on_route_end"):
            try:
                record = current_stats_record[0]
                completed = bool(record.scores.get("score_route", 0) >= 100.0) or record.status.startswith("Completed")
                collisions = (
                    len(record.infractions.get("collisions_pedestrian", [])) > 0
                    or len(record.infractions.get("collisions_vehicle", [])) > 0
                    or len(record.infractions.get("collisions_layout", [])) > 0
                )
                self.agent_instance.on_route_end(completed=completed, collided=collisions)
            except Exception:
                pass

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
            if not self.reuse_agent or not getattr(self, "agent_instance", None):
                self._agent_watchdog.start()
                # agent_class_name for example 'AutoPilot', 'PnP_Agent' .etc
                agent_class_name = getattr(self.module_agent, 'get_entry_point')()
                self.agent_instance = getattr(self.module_agent, agent_class_name)(args.agent_config, 
                                                                                   self.ego_vehicles_num)
                self._agent_watchdog.stop()
            config.agent = self.agent_instance

            # Route-specific reset for persistent agents
            if hasattr(self.agent_instance, "on_route_start"):
                try:
                    self.agent_instance.on_route_start(args.routes)
                except Exception:
                    pass

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
            return False

        # Load the world and the scenario
        print("\033[1m> Loading the world\033[0m")    
        try:
            # create world
            self._load_and_wait_for_world(args, config.town, config.ego_vehicles)
            self._prepare_ego_vehicles(config.ego_vehicles, False)
            with open(args.scenario_parameter, 'r', encoding='utf-8') as f:
                scenario_parameter = yaml.load(f.read(), Loader=yaml.FullLoader)
            # NOTE(GJH)：change the arg of RouteScenario - add the arg of scenario_config
            # print(args.scenario_parameter)
            if scenario_parameter['Background']['turn_off_light']:
                # crazy traffic light
                print("crazy traffic lights")
                [tf.set_state(carla.libcarla.TrafficLightState.Green) for tf in self.world.get_actors().filter("*traffic_light*") if hasattr(tf,"set_state")]
                [tf.freeze(True) for tf in self.world.get_actors().filter("*traffic_light*") if hasattr(tf,"freeze")]

            scenario = RouteScenario(world=self.world, config=config, debug_mode=args.debug, \
                    ego_vehicles_num=self.ego_vehicles_num, log_dir=log_dir,scenario_parameter=scenario_parameter)
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
            self.manager.load_scenario(scenario, self.agent_instance, config.repetition_index, self.ego_vehicles_num, save_root=config.save_path_root, sensor_tf_list=scenario.get_sensor_tf(), is_crazy=scenario_parameter['Background']['turn_off_light'])

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

            # If user interrupted with Ctrl+C, keep checkpoint cursor at this route
            # by not committing the interrupted route statistics/progress.
            if self._received_sigint:
                print("\033[93m> Interrupted by user; route will be resumed from this route on next run.\033[0m")
            else:
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
                self._safe_destroy_actor_handle(zombie, stop_first=True)
                zombie = None            

            if self._received_sigint:
                return False

        except Exception as e:
            print("\n\033[91mFailed to stop the scenario, the statistics might be empty:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()
            traceback.print_exc(file=open(log_file_dir,'a'))

            crash_message = "Simulation crashed"

        return True

        # if crash_message == "Simulation crashed":
        #     sys.exit(-1)

    def run(self, args: argparse.Namespace):
        """
        Run the challenge mode
        """

        if args.num_routes and int(args.num_routes) > 0:
            self._run_route_id_loop(args)
            return

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
        found_routes = 0
        while route_indexer.peek():
            try:
                # setup, load config of the next route
                config = route_indexer.next()

                # reinitialize random seed after each route
                np.random.seed(int(args.carlaProviderSeed))
                random.seed(int(args.carlaProviderSeed))
                torch.manual_seed(int(args.carlaProviderSeed))
                torch.cuda.manual_seed_all(int(args.carlaProviderSeed))
                # run
                route_committed = self._load_and_run_scenario(args, config)
                if self._received_sigint:
                    break
                if route_committed is False:
                    continue
                found_routes += 1
                self._maybe_save_best(found_routes, args)

                for i in range(args.ego_num):
                    folder_path = os.path.join(os.path.dirname(args.checkpoint), "ego_vehicle_{}".format(i))
                    if not os.path.exists(folder_path):
                        os.mkdir(folder_path)
                    path_tmp = os.path.join(os.path.dirname(args.checkpoint), "ego_vehicle_{}".format(i), os.path.basename(args.checkpoint))
                    route_indexer.save_state(path_tmp)

            except Exception as e:
                print('route error:',e)

        if self._received_sigint:
            self._flush_rl_pending()
            print("\033[93m> Interrupted by user. Checkpoint preserved for resume.\033[0m")
            return

        self._flush_rl_pending()

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

    def _run_route_id_loop(self, args: argparse.Namespace):
        """
        Run sequential routes by iterating route IDs and skipping missing files.
        """
        def _stats_path(ego_idx, ensure_parent=False):
            folder_path = os.path.join(
                os.path.dirname(args.checkpoint), "ego_vehicle_{}".format(ego_idx)
            )
            if ensure_parent and not os.path.exists(folder_path):
                os.mkdir(folder_path)
            return os.path.join(folder_path, os.path.basename(args.checkpoint))

        def _scan_next_route_id(completed_in_pass, route_id_start, route_id_max):
            route_id = route_id_start
            found = 0
            while found < completed_in_pass and route_id <= route_id_max:
                route_path = os.path.join(
                    args.routes_dir, args.routes_pattern.format(route_id)
                )
                route_id += 1
                if not os.path.exists(route_path):
                    continue
                found += 1
            return route_id, found

        def _infer_cursor_from_found(found_routes_total):
            total_target = int(num_routes_target * passes)
            found_local = max(0, min(int(found_routes_total), total_target))
            pass_idx_local = min(found_local // max(1, num_routes_target), passes)
            if pass_idx_local < passes:
                pass_found_target = found_local - (pass_idx_local * num_routes_target)
                next_route_id_local, matched = _scan_next_route_id(
                    pass_found_target, route_id_start, route_id_max
                )
                # If route-id layout changed between runs, clamp to what is actually matchable.
                if matched != pass_found_target:
                    found_local = pass_idx_local * num_routes_target + matched
                return pass_idx_local, matched, next_route_id_local, found_local
            return pass_idx_local, 0, route_id_start, found_local

        def _save_route_id_loop_state(
            found_routes_total, pass_idx, pass_found, next_route_id, passes_total, num_routes_target, completed=False
        ):
            loop_state = {
                "version": 1,
                "pass_idx": int(pass_idx),
                "pass_found": int(pass_found),
                "next_route_id": int(next_route_id),
                "found_routes": int(found_routes_total),
                "passes_total": int(passes_total),
                "num_routes_target": int(num_routes_target),
                "route_id_start": int(args.route_id_start),
                "route_id_max": int(args.route_id_max),
                "completed": bool(completed),
            }
            progress_total = int(num_routes_target * passes_total)

            for ego_idx in range(self.ego_vehicles_num):
                path_tmp = _stats_path(ego_idx, ensure_parent=True)
                data = fetch_dict(path_tmp)
                if not data:
                    data = create_default_json_msg()
                data["_checkpoint"]["progress"] = [int(found_routes_total), progress_total]
                data["_checkpoint"]["route_id_loop_state"] = loop_state
                save_dict(path_tmp, data)

        found_routes = 0
        passes = max(1, int(args.route_passes))
        num_routes_target = int(args.num_routes)
        route_id_start = int(args.route_id_start)
        route_id_max = int(args.route_id_max)

        start_pass_idx = 0
        start_pass_found = 0
        start_route_id = route_id_start

        # initialize stats
        if args.resume:
            resume_data = fetch_dict(_stats_path(0, ensure_parent=False))
            record_count = 0
            if resume_data and resume_data.get("_checkpoint", {}).get("records"):
                record_count = len(resume_data["_checkpoint"]["records"])
                found_routes = record_count
            for i in range(self.ego_vehicles_num):
                self.statistics_manager[i].resume(_stats_path(i, ensure_parent=False))

            loop_state = None
            if resume_data:
                loop_state = resume_data.get("_checkpoint", {}).get("route_id_loop_state")

            if isinstance(loop_state, dict):
                state_found = int(loop_state.get("found_routes", record_count))
                state_passes = int(loop_state.get("passes_total", passes))
                state_num_routes = int(loop_state.get("num_routes_target", num_routes_target))
                state_route_id_start = int(loop_state.get("route_id_start", route_id_start))
                state_route_id_max = int(loop_state.get("route_id_max", route_id_max))

                state_compatible = (
                    state_passes == passes
                    and state_num_routes == num_routes_target
                    and state_route_id_start == route_id_start
                    and state_route_id_max == route_id_max
                )
                state_consistent = (state_found == record_count)

                if state_compatible and state_consistent:
                    start_pass_idx = max(0, min(int(loop_state.get("pass_idx", 0)), passes))
                    start_pass_found = max(0, int(loop_state.get("pass_found", 0)))
                    start_route_id = int(loop_state.get("next_route_id", route_id_start))
                    found_routes = record_count
                else:
                    # Rebuild cursor from recorded routes to avoid replay/duplication from stale loop_state.
                    start_pass_idx, start_pass_found, start_route_id, found_routes = _infer_cursor_from_found(record_count)
            elif record_count > 0:
                # Backward-compatible inference when old checkpoints do not have route_id_loop_state.
                start_pass_idx, start_pass_found, start_route_id, found_routes = _infer_cursor_from_found(record_count)
        else:
            self.statistics_manager[0].clear_record(args.checkpoint)
            for i in range(self.ego_vehicles_num):
                self.statistics_manager[i].clear_record(_stats_path(i, ensure_parent=True))

        print("Start Running (route-id loop)!")
        interrupted = False
        for pass_idx in range(start_pass_idx, passes):
            if pass_idx == start_pass_idx:
                pass_found = start_pass_found
                route_id = start_route_id
            else:
                pass_found = 0
                route_id = route_id_start
            os.environ["ROUTE_PASS"] = "pass{:02d}".format(pass_idx + 1)
            while pass_found < num_routes_target and route_id <= route_id_max:
                route_path = os.path.join(
                    args.routes_dir, args.routes_pattern.format(route_id)
                )
                route_id += 1
                if not os.path.exists(route_path):
                    continue

                args.routes = route_path
                os.environ["ROUTES"] = route_path
                # Force per-route repetitions to 1; full-pass repeats handled externally.
                route_indexer = RouteIndexer(args.routes, args.scenarios, 1)
                route_committed = True

                while route_indexer.peek():
                    config = route_indexer.next()
                    config.index = found_routes

                    # reinitialize random seed after each route
                    np.random.seed(int(args.carlaProviderSeed))
                    random.seed(int(args.carlaProviderSeed))
                    torch.manual_seed(int(args.carlaProviderSeed))
                    torch.cuda.manual_seed_all(int(args.carlaProviderSeed))

                    route_committed = self._load_and_run_scenario(args, config)
                    if self._received_sigint:
                        interrupted = True
                        break

                if interrupted:
                    break
                if route_committed is False:
                    continue

                pass_found += 1
                found_routes += 1
                self._maybe_save_best(found_routes, args)
                _save_route_id_loop_state(
                    found_routes_total=found_routes,
                    pass_idx=pass_idx,
                    pass_found=pass_found,
                    next_route_id=route_id,
                    passes_total=passes,
                    num_routes_target=num_routes_target,
                    completed=False,
                )

            if interrupted:
                break

            if pass_found >= num_routes_target:
                _save_route_id_loop_state(
                    found_routes_total=found_routes,
                    pass_idx=pass_idx + 1,
                    pass_found=0,
                    next_route_id=route_id_start,
                    passes_total=passes,
                    num_routes_target=num_routes_target,
                    completed=False,
                )

            if pass_found < num_routes_target:
                print(
                    "Route-id loop stopped at max id {} (pass {}/{}): ran {} of {} routes.".format(
                        args.route_id_max, pass_idx + 1, passes, pass_found, args.num_routes
                    )
                )

        if interrupted or self._received_sigint:
            self._flush_rl_pending()
            print("\033[93m> Interrupted by user. Checkpoint preserved for resume.\033[0m")
            return

        self._flush_rl_pending()

        # save global statistics
        print("\033[1m> Registering the global statistics\033[0m")
        try:
            # Ensure checkpoint progress matches total routes executed
            for i in range(self.ego_vehicles_num):
                path_tmp = os.path.join(
                    os.path.dirname(args.checkpoint),
                    "ego_vehicle_{}".format(i),
                    os.path.basename(args.checkpoint),
                )
                data = fetch_dict(path_tmp)
                if not data:
                    data = create_default_json_msg()
                data["_checkpoint"]["progress"] = [found_routes, found_routes]
                data["_checkpoint"]["route_id_loop_state"] = {
                    "version": 1,
                    "pass_idx": passes,
                    "pass_found": 0,
                    "next_route_id": route_id_start,
                    "found_routes": found_routes,
                    "passes_total": passes,
                    "num_routes_target": num_routes_target,
                    "route_id_start": route_id_start,
                    "route_id_max": route_id_max,
                    "completed": True,
                }
                save_dict(path_tmp, data)

            for i in range(self.ego_vehicles_num):
                global_stats_record = self.statistics_manager[i].compute_global_statistics(found_routes)
                path_tmp = os.path.join(
                    os.path.dirname(args.checkpoint),
                    "ego_vehicle_{}".format(i),
                    os.path.basename(args.checkpoint),
                )
                self.statistics_manager[i].save_global_record(global_stats_record, self.sensor_icons, found_routes, path_tmp)
        except Exception:
            traceback.print_exc()

def main():
    description = "CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n"

    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default='50010', help='TCP port to listen to (default: 2000)')
    parser.add_argument('--trafficManagerPort', default='50050',
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
                        default='simulation/leaderboard/data/evaluation_routes/demo/town05_short_r2.xml',
                        help='Name of the route to be executed. Point to the route_xml_file to be executed.')
    parser.add_argument('--route-id-start',
                        type=int,
                        default=0,
                        help='Start route id for sequential training (skip missing ids).')
    parser.add_argument('--num-routes',
                        type=int,
                        default=0,
                        help='Number of routes to run sequentially (0 = use --routes file).')
    parser.add_argument('--route-passes',
                        type=int,
                        default=0,
                        help='Full-pass repeats over the route set (0 or 1 = single pass).')
    parser.add_argument('--routes-dir',
                        default='simulation/leaderboard/data/evaluation_routes',
                        help='Directory containing town05_short_r{ID}.xml files.')
    parser.add_argument('--routes-pattern',
                        default='town05_short_r{}.xml',
                        help='Filename pattern for route files (format with route id).')
    parser.add_argument('--reuse-agent',
                        type=int,
                        default=0,
                        help='Reuse a single agent instance across routes (1 = reuse).')
    parser.add_argument('--route-id-max',
                        type=int,
                        default=331,
                        help='Maximum route id to try before stopping a pass.')
    parser.add_argument('--best-every',
                        type=int,
                        default=10,
                        help='Save a best checkpoint every N completed routes (0 = disabled).')
    parser.add_argument('--scenarios',
                        default='simulation/leaderboard/data/scenarios/town05_all_scenarios_2.json',
                        help='Name of the scenario annotation file to be mixed with the route.')
    parser.add_argument('--scenario_parameter', 
                        default='simulation/leaderboard/leaderboard/scenarios/scenario_parameter_demo_0.yaml',
                        help='Defination of the scenario parameters.')
    parser.add_argument('--repetitions',
                        type=int,
                        default=1,
                        help='Number of repetitions per route.')

    # agent-related options
    parser.add_argument("-a", "--agent", type=str, default='simulation/leaderboard/team_code/pnp_agent_e2e_demo.py', help="Path to Agent's py file to evaluate")
    parser.add_argument("--agent-config", type=str, default='simulation/leaderboard/team_code/agent_config_e2e/pnp_config_codriving_5_10.yaml', help="Path to Agent's configuration file")

    parser.add_argument("--track", type=str, default='SENSORS', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=int, default=0, help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str,
                        default='/GPFS/data/gjliu-1/Auto-driving/V2Xverse/out_dir/demo/results.json',
                        help="Path to checkpoint used for saving statistics and resuming")
    parser.add_argument('--ego-num', type=int, default=1, help='The number of ego vehicles')
    parser.add_argument('--skip_existed', type=int, default=0, help='If the result exist, return')
    # crazy level: 0-5, the probability of ignoring front car.
    # crazy proportion: the probability of a car is crazy 
    
    arguments = parser.parse_args()

    check_result = False

    if arguments.skip_existed:
        if os.path.exists(os.path.join(os.path.dirname(arguments.checkpoint), "ego_vehicle_{}".format(0), os.path.basename(arguments.checkpoint))):
            image_save_path = os.environ['SAVE_PATH']
            route_list = os.listdir(image_save_path)
            route_list.sort()
            log_dir = route_list[-1]
            log_file_path = "{}/{}/log/log.log".format(image_save_path, log_dir)
            
            for invalid_str in ['Traceback','No space']:
                if check_log_file(log_file_path, invalid_str):
                    check_result = True
                    break

            if not check_log_file(log_file_path, 'RouteCompletionTest'):
                check_result = True

            if check_result:
                print('Invalid results, rerun!')
            else:
                print('Valid results, skip!')
                return
        else:
            print('{} do not exists, continue!'.format(os.path.join(os.path.dirname(arguments.checkpoint), "ego_vehicle_{}".format(0), os.path.basename(arguments.checkpoint))))

    if not os.path.exists(os.environ["SAVE_PATH"]):
        os.makedirs(os.environ["SAVE_PATH"])
    if not os.path.exists(os.path.dirname(os.environ["CHECKPOINT_ENDPOINT"])):
        os.makedirs(os.path.dirname(os.environ["CHECKPOINT_ENDPOINT"]))
        
    if not "RESULT_ROOT" in os.environ:
        os.environ["RESULT_ROOT"] = os.environ["DATA_ROOT"]

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
