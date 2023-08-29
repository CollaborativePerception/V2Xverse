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

sys.path.append("/GPFS/data/gjliu/Auto-driving/V2Xverse/third_party/scenario_runner")
sys.path.append("/GPFS/data/gjliu/Auto-driving/V2Xverse/third_party/leaderboard")
sys.path.append("/GPFS/data/gjliu/Auto-driving/V2Xverse")
sys.path.append('/GPFS/data/gjliu/Auto-driving/Cop3/carla/PythonAPI/carla')

os.environ["CARLA_ROOT"] = '/GPFS/data/gjliu/Auto-driving/Cop3/carla'
os.environ["DATA_ROOT"] = 'dataset_cop3'
os.environ["YAML_ROOT"] = 'simulation/hypes_yaml'
os.environ["ROUTES"]='/GPFS/data/gjliu/Auto-driving/Cop3/leaderboard/data/evaluation_routes/final/town05_short_r0.xml'
os.environ["LEADERBOARD_ROOT"]='third_party/leaderboard'
os.environ["CHALLENGE_TRACK_CODENAME"]='SENSORS'

# os.environ["PORT"]=2000 # same as the carla server port

os.environ["TEAM_AGENT"]='third_party/leaderboard/team_code/pnp_agent.py'   #'third_party/leaderboard/team_code/auto_pilot.py' , 'third_party/leaderboard/team_code/pnp_agent.py' 
os.environ["TEAM_CONFIG"]='third_party/leaderboard/team_code/pnp_config.py' # 'simulation/hypes_yaml/weather-0.yaml' , 'third_party/leaderboard/team_code/pnp_config.py'
os.environ["CHECKPOINT_ENDPOINT"]='/GPFS/data/gjliu/Auto-driving/Cop3/results/eval/test/results.json'
os.environ["SCENARIOS"]='third_party/leaderboard/data/scenarios/town05_all_scenarios.json'
os.environ["SAVE_PATH"]='/GPFS/data/gjliu/Auto-driving/Cop3/results/eval/test/image'

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
from leaderboard_evaluator_parameter import main


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__ == '__main__':
    main()
