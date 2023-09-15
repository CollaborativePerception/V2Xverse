#!/bin/bash

export CARLA_ROOT=/DB/rhome/weibomao/GPFS/InterFuser/carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner

#######################
######## HYPER-PARAM
#######################
export AGENT_MAX_SPEED=$1
export TRIGGER_DISTANCE=$2

export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=$3 # IMPORTANT: same as the carla server port
export TM_PORT=`expr $PORT + 5` # port for traffic manager, required when spawning multiple servers/clients
export DEBUG_CHALLENGE=0

export REPETITIONS=1 # multiple evaluation runs
export ROUTES=leaderboard/data/evaluation_routes/final/town05_short_r$4.xml
# verify the evaluation route, including start point and end point.
export SCENARIOS=leaderboard/data/scenarios/town05_all_scenarios.json

export RESULT_ROOT=/GPFS/public/InterFuser/results_close_$7
export EVAL_SETTING=v2x_final/town05_short_collab_v${AGENT_MAX_SPEED}d${TRIGGER_DISTANCE}/r$4_repeat$5
export CHECKPOINT_ENDPOINT=${RESULT_ROOT}/${EVAL_SETTING}/results.json 
# path to save the result json file
export SAVE_PATH=${RESULT_ROOT}/image/${EVAL_SETTING}
# path to save the images.

export TEAM_AGENT=leaderboard/team_code/v2xverse_agent.py 
# V2X agent with BEV input to indicate the drivable area.
export TEAM_CONFIG=leaderboard/team_code/v2xverse_config$8.py 
# model config file, we have to change the param `fusion_mode`!

export RESUME=0
export EGO_NUM=$6
export CRAZY_LEVEL=3
export CRAZY_PROPROTION=50

mkdir -p $SAVE_PATH
mkdir -p ${RESULT_ROOT}/${EVAL_SETTING}


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
--crazy-propotion=${CRAZY_PROPROTION} \
--max-speed=${AGENT_MAX_SPEED} \
--trigger-distance=${TRIGGER_DISTANCE}


