#!/bin/bash

# $1, route id
# $2, Carla port
# $3, exp_name
# $4, repeat
# $5, agent config
# $6, scenario config

export CARLA_ROOT=external_paths/carla_root
export LEADERBOARD_ROOT=simulation/leaderboard
export SCENARIO_RUNNER_ROOT=simulation/scenario_runner
export DATA_ROOT=external_paths/data_root
export SAVE_DIR=results

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${LEADERBOARD_ROOT}
export PYTHONPATH=$PYTHONPATH:${LEADERBOARD_ROOT}/team_code
export PYTHONPATH=$PYTHONPATH:${SCENARIO_RUNNER_ROOT}

export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=${2:-40000} # IMPORTANT: same as the carla server port
export TM_PORT=`expr $PORT + 5` # port for traffic manager, required when spawning multiple servers/clients
export DEBUG_CHALLENGE=0
export TRAFFIC_SEED=2000
export CARLA_SEED=2000
export REPETITIONS=1 # multiple evaluation runs
export ROUTES=${LEADERBOARD_ROOT}/data/evaluation_routes/town05_short_r${1:-0}.xml
# verify the evaluation route, including start point and end point.
export SCENARIOS=${LEADERBOARD_ROOT}/data/scenarios/town05_all_scenarios_2.json
export SCENARIOS_PARAMETER=${LEADERBOARD_ROOT}/leaderboard/scenarios/scenario_parameter$6.yaml
export RESULT_ROOT=${SAVE_DIR}/results_driving_${3:-debug}
export EVAL_SETTING=v2x_final/town05_short_collab/r${1:-0}_repeat${4:-0}
export CHECKPOINT_ENDPOINT=${RESULT_ROOT}/${EVAL_SETTING}/results.json
# path to save the result json file
export SAVE_PATH=${RESULT_ROOT}/image/${EVAL_SETTING}
# path to save the images.

export TEAM_AGENT=simulation/leaderboard/team_code/pnp_agent_e2e.py
# V2X agent with BEV input to indicate the drivable area.
export TEAM_CONFIG=simulation/leaderboard/team_code/agent_config/pnp_config_$5.yaml
# model config file!

export RESUME=0
export EGO_NUM=1
export SKIP_EXISTED=1

mkdir -p $SAVE_PATH
mkdir -p ${RESULT_ROOT}/${EVAL_SETTING}


python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_parameter.py \
--scenarios=${SCENARIOS}  \
--scenario_parameter=${SCENARIOS_PARAMETER}  \
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
--carlaProviderSeed=${CARLA_SEED} \
--trafficManagerSeed=${TRAFFIC_SEED} \
--ego-num=${EGO_NUM} \
--timeout 600 \
--skip_existed=${SKIP_EXISTED}
