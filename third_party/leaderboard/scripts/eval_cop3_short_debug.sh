#!/bin/bash

export CARLA_ROOT=/DB/rhome/weibomao/GPFS/InterFuser/carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner

export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=7000 # same as the carla server port
export TM_PORT=7015 # port for traffic manager, required when spawning multiple servers/clients
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export ROUTES=leaderboard/data/evaluation_routes/routes_town05_short.xml
# export ROUTES=leaderboard/data/evaluation_routes/routes_town05_debug.xml
export TEAM_AGENT=leaderboard/team_code/cop3_baseline_agent.py # agent
export TEAM_CONFIG=leaderboard/team_code/cop3_baseline_config.py # model checkpoint, not required for expert
export CHECKPOINT_ENDPOINT=/GPFS/public/InterFuser/results_cop3/debug/cop3_baseline_N1_short.json # results file
export SCENARIOS=leaderboard/data/scenarios/town05_all_scenarios.json
export SAVE_PATH=/GPFS/public/InterFuser/results_cop3/image/debug # TODO, update the path for saving episodes while evaluating
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


