#!/bin/bash
export CARLA_ROOT=/GPFS/data/gjliu/Auto-driving/Cop3/carla
export DATA_ROOT=/GPFS/public/InterFuser/dataset_coslam_example_0701
export YAML_ROOT=data_collection/yamls
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner

export LEADERBOARD_ROOT=leaderboard

export CHECKPOINT_ENDPOINT=${DATA_ROOT}/weather-0/results/routes_town03_31.json
export SAVE_PATH=${DATA_ROOT}/weather-0/data
export TEAM_CONFIG=${YAML_ROOT}/weather-0.yaml
export TRAFFIC_SEED=2000
export CARLA_SEED=2000
export SCENARIOS=${LEADERBOARD_ROOT}/data/scenarios/town03_all_scenarios.json
export ROUTES=${LEADERBOARD_ROOT}/data/training_routes/splitted_routes/routes_town03_31.xml
export TM_PORT=40510
export PORT=40010
export CHALLENGE_TRACK_CODENAME=SENSORS
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export TEAM_AGENT=${LEADERBOARD_ROOT}/team_code/auto_pilot.py # agent
export RESUME=0
export EGO_NUM=2
export CRAZY_LEVEL=3
export CRAZY_PROPROTION=70

python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
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
--carlaProviderSeed=${CARLA_SEED} \
--trafficManagerSeed=${TRAFFIC_SEED} \
--ego-num=${EGO_NUM} \
--crazy-level=${CRAZY_LEVEL} \
--crazy-propotion=${CRAZY_PROPROTION} \
--timeout 600
