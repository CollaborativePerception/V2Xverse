import os
import random
import shutil
from tqdm import tqdm
import re

def generate_script(
    port, tm_port, route, scenario, carla_seed, traffic_seed, config_path
):
    """
    generate script based on a base_script.sh
    
    """
    lines = []
    lines.append("export PORT=%d\n" % port)
    lines.append("export TM_PORT=%d\n" % tm_port)
    lines.append("export ROUTES=${LEADERBOARD_ROOT}/data/%s\n" % route)
    lines.append("export SCENARIOS=${LEADERBOARD_ROOT}/data/%s\n" % scenario)
    lines.append("export CARLA_SEED=%d\n" % carla_seed)
    lines.append("export TRAFFIC_SEED=%d\n" % traffic_seed)
    lines.append("export TEAM_CONFIG=${YAML_ROOT}/%s\n" % config_path)
    lines.append("export SAVE_PATH=${DATA_ROOT}/%s/data\n" % config_path.split(".")[0])
    lines.append(
        "export CHECKPOINT_ENDPOINT=${DATA_ROOT}/%s/results/%s.json\n"
        % (config_path.split(".")[0], route.split("/")[2].split(".")[0])
    )
    lines.append("\n")
    base = open("base_script.sh").readlines()

    for line in lines:
        base.insert(13, line)

    return base

# route: dict, contain every route file path and its corresponding scenario json file
# route[town_(x)_route_file_path] = town(x)_all_scenario.json
routes = {}
routes_dir = '/GPFS/data/gjliu/Auto-driving/V2Xverse/third_party/leaderboard/data/training_routes/splitted_routes'
pattern = re.compile('.*town(\d\d)')
for route in os.listdir(routes_dir):
    res = pattern.findall(route)
    town = res[0]
    routes["training_routes/splitted_routes/{}".format(route)] = "scenarios/town{}_all_scenarios.json".format(town)

# port and traffic manager port to be used
# make sure each port and traffic manager port is not already in used
ip_ports = []
for port in range(40000, 40028, 2):
    ip_ports.append(("localhost", port, port + 500))

# set seed
carla_seed = 2000
traffic_seed = 2000

# configs list for different weather (0-13)
configs = []
for i in range(14):
    configs.append("weather-%d.yaml" % i)

# reset the file folder bashs/
if os.path.exists("scripts"):
    shutil.rmtree("scripts")
    print("remove succeed")
if not os.path.exists("scripts"):
    os.mkdir("scripts")
    print("create scripts folder")

town_list = ['town01', 'town02', 'town03', 'town04', 'town05', 'town06','town07', 'town10']

town_dict = {'town01':{'id':0, 'port_split':[16,17], 'total_routes': 33, 'routes':{}},
            'town02':{'id':1, 'port_split':[10,11], 'total_routes': 21, 'routes':{}},
            'town03':{'id':2, 'port_split':[21,21], 'total_routes': 42, 'routes':{}},
            'town04':{'id':3, 'port_split':[22,22], 'total_routes': 44, 'routes':{}},
            'town05':{'id':4, 'port_split':[21,21], 'total_routes': 42, 'routes':{}},
            'town06':{'id':5, 'port_split':[14,14], 'total_routes': 28, 'routes':{}},
            'town07':{'id':6, 'port_split':[7,7], 'total_routes': 14, 'routes':{}},
            'town10':{'id':8, 'port_split':[9], 'total_routes': 9, 'routes':{}}}

port = 40000
tm_port = 40500

# compute port for every single route generation
for town in town_list:
    route_order = 0
    for port_len in town_dict[town]['port_split']:
        for p in range(port_len):
            if not route_order in town_dict[town]['routes']:
                town_dict[town]['routes'][route_order] = {}
            town_dict[town]['routes'][route_order]['port'] = port
            town_dict[town]['routes'][route_order]['tm_port'] = tm_port
            route_order += 1
        port += 2
        tm_port += 2

# here only scripts for weather-0 will be generated
# generate seperate scripts for every single route
weathers_to_generate = [0]

for i, weather in enumerate(weathers_to_generate):
    os.mkdir("scripts/weather-%d" % weather)
    for route in routes:
        _, port, tm_port = ip_ports[weather]
        route_id = route.split("/")[2].split(".")[0].split('_')[2]
        town_str = route.split("/")[2].split(".")[0].split('_')[1][-6:]

        port = town_dict[town_str]['routes'][int(route_id)]['port'] + i*1000
        tm_port = town_dict[town_str]['routes'][int(route_id)]['tm_port'] + i*1000

        script = generate_script(
            port,
            tm_port,
            route,
            routes[route],
            carla_seed,
            traffic_seed,
            configs[weather],
        )
        fw = open(
            "scripts/weather-%d/%s.sh" % (weather, route.split("/")[2].split(".")[0]), "w"
        )
        for line in script:
            fw.write(line)

# create a file generate_v2xverse_all.sh to collect data in parallel
if os.path.exists("batch_run"):
    shutil.rmtree("batch_run")
    print("remove batch_run/ succeed")
if not os.path.exists("batch_run"):
    os.mkdir("batch_run")
    print("create batch_run/ folder")

fw_all = open("generate_v2xverse_all.sh", "w")
for _, weather in enumerate(weathers_to_generate):
    os.mkdir("batch_run/weather-%d" % weather)
    for town in town_list:
        route_order = 0
        for i, port_len in enumerate(town_dict[town]['port_split']):
            fw = open("batch_run/weather-%d/town_%s_%s.sh" % (weather,town,i), "w")

            for _ in range(port_len):
                route = 'routes_{}_{}'.format(town, route_order)
                fw.write("bash data_collection/bashs/weather-%d/%s.sh \n" % (weather, route))
                route_order += 1

            fw_all.write("bash data_collection/batch_run/weather-%d/town_%s_%s.sh & \n" % (weather,town,i))

