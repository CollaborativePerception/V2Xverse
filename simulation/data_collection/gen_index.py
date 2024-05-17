import os
import sys
from tqdm import tqdm
import time
data_index = ""
frame_all = 0
sample_all = 0
route_num = {}
town_num = {}

dataset_directory='dataset'

for i in range(1):
    subs = os.listdir(os.path.join(dataset_directory,"weather-%d/data" % i))
    for sub in tqdm(subs):
        seq_len = 1000000
        sub_path = os.path.join(dataset_directory,"weather-{}/data/{}/".format(i, sub))
        try:
            agent_list = os.listdir(sub_path)
        except:
            continue
        ego_list = [ego for ego in agent_list if ego.startswith('ego')]
        rsu_list = [ego for ego in agent_list if ego.startswith('rsu')]
        for ego in ego_list:
            ego_path = os.path.join(sub_path, ego)
            seq_len_ego = len(os.listdir(os.path.join(ego_path,'rgb_front')))
            if seq_len > seq_len_ego:
                seq_len = seq_len_ego
        if seq_len>50:
            if len(ego_list)==1 and len(rsu_list)==0:
                continue
            # data_index += "{} {} {}\n".format(sub_path, seq_len, len(ego_list))
            # frame_all += seq_len
            # sample_all += seq_len*len(ego_list)

            town_route_id = sub.split('_')[1]+'_'+sub.split('_')[2]
            if not town_route_id in route_num:
                town = int(sub.split('_')[1][-2:])
                # if town not in [7,10]:
                #     continue
                route_num[town_route_id] = {'seq_len':seq_len, 'sub_path':sub_path, 'len(ego_list)':len(ego_list)}
                if not town in town_num:
                    town_num[town] = 1
                else:
                    town_num[town] += 1
            elif route_num[town_route_id]['seq_len'] < seq_len:
                route_num[town_route_id] = {'seq_len':seq_len, 'sub_path':sub_path, 'len(ego_list)':len(ego_list)}


exist_path = []
print(len(exist_path))
a=0

with open(os.path.join(dataset_directory,"dataset_index.txt"), 'w') as f:
    for town_route_id in route_num:

        if route_num[town_route_id]['sub_path'] in exist_path:
            continue
            # time.sleep(1000)
        route_num[town_route_id]['seq_len']-=25
        data_index += "{} {} {}\n".format(route_num[town_route_id]['sub_path'], route_num[town_route_id]['seq_len'], route_num[town_route_id]['len(ego_list)'])
        seq_len = route_num[town_route_id]['seq_len']
        frame_all += seq_len
        sample_all += seq_len*len(ego_list)
    f.write(data_index)
print('frames:',frame_all)
print('samples:',sample_all)
print(town_num)