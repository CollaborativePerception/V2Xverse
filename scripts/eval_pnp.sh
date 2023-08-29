#!/bin/bash
export CARLA_ROOT=external_paths/carla_root
export LEADERBOARD_ROOT=third_party/leaderboard
export SCENARIO_RUNNER_ROOT=third_party/scenario_runner
export DATA_ROOT=external_paths/data_root

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${LEADERBOARD_ROOT}
export PYTHONPATH=$PYTHONPATH:${LEADERBOARD_ROOT}/team_code
export PYTHONPATH=$PYTHONPATH:${SCENARIO_RUNNER_ROOT}

export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=40000 #${1:-40000} # IMPORTANT: same as the carla server port
export TM_PORT=`expr $PORT + 5` # port for traffic manager, required when spawning multiple servers/clients
export DEBUG_CHALLENGE=0
export TRAFFIC_SEED=2000
export CARLA_SEED=2000
export REPETITIONS=1 # multiple evaluation runs
export ROUTE_ID=0 #${2:-0}
export ROUTES=${LEADERBOARD_ROOT}/data/evaluation_routes/final/town05_short_r0.xml
# verify the evaluation route, including start point and end point.
export SCENARIOS=${LEADERBOARD_ROOT}/data/scenarios/town05_all_scenarios.json
export SCENARIOS_PARAMETER=${LEADERBOARD_ROOT}/leaderboard/scenarios/scenario_parameter.yaml
export RESULT_ROOT=/GPFS/public/InterFuser/results_close_ #${4:-debug}
export EVAL_SETTING=v2x_final/town05_short_collab/r0_repeat #$3
export CHECKPOINT_ENDPOINT=${RESULT_ROOT}/${EVAL_SETTING}/results.json 
# path to save the result json file
export SAVE_PATH=${RESULT_ROOT}/image/${EVAL_SETTING}
# path to save the images.

export TEAM_AGENT=third_party/leaderboard/team_code/pnp_agent.py  # pnp_agent.py
# V2X agent with BEV input to indicate the drivable area.
export TEAM_CONFIG=third_party/leaderboard/team_code/pnp_config.py # third_party/leaderboard/team_code/pnp_config.py simulation/data_collection/yamls/weather-0.yaml
# model config file, we have to change the param `fusion_mode`!

export RESUME=0
export EGO_NUM=1 #${5:-1}
export CRAZY_LEVEL=3
export CRAZY_PROPROTION=50

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
--crazy-level=${CRAZY_LEVEL} \
--crazy-propotion=${CRAZY_PROPROTION} \
--timeout 600
