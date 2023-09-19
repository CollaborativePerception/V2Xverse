#!/bin/bash
export CARLA_ROOT=external_paths/carla_root
export LEADERBOARD_ROOT=third_party/leaderboard
export SCENARIO_RUNNER_ROOT=third_party/scenario_runner
export DATA_ROOT=external_paths/data_root

export YAML_ROOT=simulation/data_collection/yamls
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${LEADERBOARD_ROOT}
export PYTHONPATH=$PYTHONPATH:${LEADERBOARD_ROOT}/team_code
export PYTHONPATH=$PYTHONPATH:${SCENARIO_RUNNER_ROOT}

export CHECKPOINT_ENDPOINT=${DATA_ROOT}/weather-0/results/routes_town01_0.json
export SAVE_PATH=${DATA_ROOT}/weather-0/data
export TEAM_CONFIG=${YAML_ROOT}/weather-0.yaml
export TRAFFIC_SEED=2000
export CARLA_SEED=2000
export SCENARIOS=${LEADERBOARD_ROOT}/data/scenarios/town01_all_scenarios_complete.json
export SCENARIOS_PARAMETER=${LEADERBOARD_ROOT}/leaderboard/scenarios/scenario_parameter.yaml
export ROUTES=${LEADERBOARD_ROOT}/data/training_routes/original_routes/routes_town01_long.xml
export TM_PORT=40500
export PORT=40000
export CHALLENGE_TRACK_CODENAME=SENSORS
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export TEAM_AGENT=${LEADERBOARD_ROOT}/team_code/auto_pilot.py # agent
export RESUME=0
export EGO_NUM=1
export CRAZY_LEVEL=3
export CRAZY_PROPORTION=70

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
--crazy-proportion=${CRAZY_PROPORTION} \
--timeout 600
