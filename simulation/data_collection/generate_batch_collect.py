import os
import re
import numpy as np

routes = {}
routes_num = np.zeros(20)
routes_dir = '/GPFS/data/gjliu/Auto-driving/Cop3/leaderboard/data/training_routes/cop3_split_routes'
pattern = re.compile('.*town(\d\d)')
for route in os.listdir(routes_dir):
    res = pattern.findall(route)
    town = res[0]
    routes["training_routes/cop3_split_routes/{}".format(route)] = "scenarios/town{}_all_scenarios.json".format(town)
    routes_num[int(town)] += 1

routes_list = []
for route in routes:
    routes_list.append(route.split("/")[2].split(".")[0])

if os.path.exists("batch_run"):
    shutil.rmtree("batch_run")
    print("remove succeed")
if not os.path.exists("batch_run"):
    os.mkdir("batch_run")

fw_all = open("generate_cop3_all.sh", "w")

split_len = [0,10,10,10,10,10,10,8,0,0,4]

for i in range(1):
    for town in [1,2,3,4,5,6,7,10]:
        fw = open("batch_run/town_%s_%s.sh" % (town,0), "w")
        fw1 = open("batch_run/town_%s_%s.sh" % (town,1), "w")
        for route_id in range(int(routes_num[town])):
            route = 'routes_town{}_{}'.format(str(town).zfill(2),route_id)
            if int(route_id) > split_len[town]:
                fw1.write("bash data_collection/bashs/weather-%d/%s.sh \n" % (i, route))
            else:
                fw.write("bash data_collection/bashs/weather-%d/%s.sh \n" % (i, route))
        fw_all.write("bash data_collection/batch_run/town_%s_%s.sh & \n" % (town,0))
        fw_all.write("bash data_collection/batch_run/town_%s_%s.sh & \n" % (town,1))