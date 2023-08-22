import os
import random
import shutil
from tqdm import tqdm
import re

def generate_script(
    ip, port, tm_port, route, scenario, carla_seed, traffic_seed, config_path
):
    """
    generate script based on a base_script.sh
    
    """
    lines = []
    # lines.append("export HOST=%s\n" % ip)
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

routes = {}
routes_dir = '../../third_party/leaderboard/data/training_routes/splitted_routes'
pattern = re.compile('.*town(\d\d)')
for route in os.listdir(routes_dir):
    res = pattern.findall(route)
    town = res[0]
    routes["training_routes/splitted_routes/{}".format(route)] = "scenarios/town{}_all_scenarios.json".format(town)

ip_ports = []

for port in range(40000, 40028, 2):
    ip_ports.append(("localhost", port, port + 500))

carla_seed = 2000
traffic_seed = 2000

configs = []
for i in range(14):
    configs.append("weather-%d.yaml" % i)

if os.path.exists("bashs"):
    shutil.rmtree("bashs")
    print("remove succeed")

split_len = [0,10,10,10,10,10,10,8,4]

for i in tqdm(range(1)):
    if not os.path.exists("bashs"):
        os.mkdir("bashs")
    os.mkdir("bashs/weather-%d" % i)
    for route in routes:
        ip, port, tm_port = ip_ports[i]
        route_id = route.split("/")[2].split(".")[0].split('_')[2]
        town_id = route.split("/")[2].split(".")[0].split('_')[1][-2:]
        town_id = int(town_id)
        if town_id ==10:
            town_id = 8
        port += (town_id-1)*4
        tm_port += (town_id-1)*4
        if int(route_id) > split_len[town_id]:
            port += 2
            tm_port += 2

        script = generate_script(
            ip,
            port,
            tm_port,
            route,
            routes[route],
            carla_seed,
            traffic_seed,
            configs[i],
        )
        fw = open(
            "bashs/weather-%d/%s.sh" % (i, route.split("/")[2].split(".")[0]), "w"
        )
        for line in script:
            fw.write(line)
