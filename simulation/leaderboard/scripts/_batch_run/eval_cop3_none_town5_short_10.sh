#!/bin/bash

export CARLA_ROOT=/GPFS/data/gjliu/Auto-driving/Cop3/carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner

export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=40020 # IMPORTANT: same as the carla server port
export TM_PORT=47020 # port for traffic manager, required when spawning multiple servers/clients
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export ROUTES=leaderboard/data/evaluation_routes/routes_town05_short_10.xml
# verify the evaluation route, including start point and end point.
export CHECKPOINT_ENDPOINT=/GPFS/data/gjliu/Auto-driving/Cop3/results/eval/none/cop3_none_town5_short_10.json 
# path to save the result json file
export SAVE_PATH=/GPFS/data/gjliu/Auto-driving/Cop3/results/eval/none/image
# path to save the images.

export TEAM_AGENT=leaderboard/team_code/cop3_baseline_agent.py 
# agent
export TEAM_CONFIG=leaderboard/team_code/cop3_baseline_config.py 
# model config file, we have to change the param `fusion_mode`!
export SCENARIOS=leaderboard/data/scenarios/town05_all_scenarios.json
export RESUME=0
export EGO_NUM=1
export CRAZY_LEVEL=3
export CRAZY_PROPROTION=70

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT} \
--ego-num=${EGO_NUM} \
--crazy-level=${CRAZY_LEVEL} \
--crazy-propotion=${CRAZY_PROPROTION}


