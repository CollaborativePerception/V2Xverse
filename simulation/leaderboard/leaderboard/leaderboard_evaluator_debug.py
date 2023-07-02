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

sys.path.append("/GPFS/data/gjliu/Auto-driving/Cop3/scenario_runner")
sys.path.append("/GPFS/data/gjliu/Auto-driving/Cop3/leaderboard")
sys.path.append('/GPFS/data/gjliu/Auto-driving/Cop3/carla/PythonAPI/carla')

os.environ["CARLA_ROOT"] = '/GPFS/data/gjliu/Auto-driving/Cop3/carla'
os.environ["DATA_ROOT"] = 'dataset_cop3'
os.environ["YAML_ROOT"] = 'data_collection/yamls'
os.environ["ROUTES"]='leaderboard/data/evaluation_routes/routes_town05_debug.xml'
os.environ["LEADERBOARD_ROOT"]='leaderboard'
os.environ["CHALLENGE_TRACK_CODENAME"]='SENSORS'
# os.environ["PORT"]=2000 # same as the carla server port

os.environ["TEAM_AGENT"]='leaderboard/team_code/auto_pilot.py'
os.environ["TEAM_CONFIG"]='data_collection/yamls/debug.yaml'
os.environ["CHECKPOINT_ENDPOINT"]='/GPFS/data/gjliu/Auto-driving/Cop3/results/eval/test/results.json'
os.environ["SCENARIOS"]='leaderboard/data/scenarios/town05_all_scenarios.json'
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
from leaderboard_evaluator import main


# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

if __name__ == '__main__':
    main()
