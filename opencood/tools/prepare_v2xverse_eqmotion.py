from opencood.hypes_yaml.yaml_utils import load_yaml, save_yaml
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import os
import re
import json
import math

def _load_json(path, root_dir):
    try:
        json_value = json.load(open(os.path.join(root_dir,path)))
    except Exception as e:
        n = path[-9:-5]
        new_path = path[:-9] + "%04d.json" % (int(n) - 1)
        json_value = json.load(open(os.path.join(root_dir,new_path)))
    return json_value

if __name__ == '__main__':
    root_dir = '/GPFS/public/InterFuser/dataset_cop3_lidarmini'
    save_path = '/GPFS/data/gjliu/Auto-driving/Eqmotion/v2xverse/dataset5'
    splits = ['train', 'validate', 'test']
    weathers = [0,1,2,3,4,5,6,7,8,9,10]
    dataset_indexs = open(os.path.join(root_dir,'dataset_index.txt'), 'r').read().split('\n')
    traj_len = 20
    max_agent_num = 20

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for split in splits:

        print('{} dataset'.format(split))
        if split == 'train':
            towns = [1,2,3,4]
        elif split == 'validate':
            towns = [6,7,10] # [6,7,8,9,10]
        elif split == 'test':
            towns = [5]

        route_frames = []
        num_agent = []
        traj_data = []

        pattern = re.compile('weather-(\d+).*town(\d\d)')
        for line in tqdm(dataset_indexs):
            if len(line.split()) != 3:
                continue
            path, frames, egos = line.split()
            route_path = os.path.join(root_dir, path)
            frames = int(frames)
            res = pattern.findall(path)
            if len(res) != 1:
                continue
            weather = int(res[0][0])
            town = int(res[0][1])            
            if weather not in weathers or town not in towns:
                continue

            files = os.listdir(route_path)
            ego_files = [file for file in files if file.startswith('ego')]
            rsu_files = [file for file in files if file.startswith('rsu')]
            for j, file in enumerate(ego_files):
                ego_path = os.path.join(route_path, file)
                seq_len = len(os.listdir(os.path.join(ego_path, "actors_data")))
                if seq_len <= traj_len:
                    continue
                for k in range(0, seq_len-traj_len, traj_len//4):
                    traj_dict = {}
                    one_sample = np.zeros((1, max_agent_num, traj_len, 2))
                    
                    for t in range(traj_len):
                        frame_id = k + t
                        actors_data = _load_json(os.path.join(ego_path, "actors_data", "%04d.json" % frame_id), root_dir)

                        measurements_file_path = "{}/measurements/{:0>4d}.json".format(ego_path, frame_id)
                        measurements = json.load(open(measurements_file_path))
                        ego_x = measurements["lidar_pose_x"]
                        ego_y = measurements["lidar_pose_y"]
                        ego_z = measurements["lidar_pose_z"]
                        ego_theta = measurements["theta"] + np.pi # !note, plus pi in extra.
                        # rotate counterclockwise by ego_theta
                        R = np.array(
                            [
                                [np.cos(ego_theta), -np.sin(ego_theta)],
                                [np.sin(ego_theta), np.cos(ego_theta)],
                            ]
                        )

                        for _id in actors_data.keys():
                            if actors_data[_id]["tpe"] == 2:
                                continue
                            if not _id in traj_dict.keys():
                                traj_dict[_id] = {}
                            traj_dict[_id][t] = actors_data[_id]['loc'][:2]
                    agent_order = 0
                    for _id in traj_dict.keys():
                        if not len(traj_dict[_id].keys()) == traj_len:
                            continue
                        for t in range(traj_len):
                            one_sample[0, agent_order, t, :] = traj_dict[_id][t]
                        agent_order += 1
                        if agent_order == max_agent_num:
                            break
                    if not math.isnan(one_sample.max()):
                        num_agent.append(agent_order)
                        traj_data.append(one_sample)
        num_agent = np.array(num_agent)
        traj_data = np.concatenate(traj_data, axis=0)
        np.save(os.path.join(save_path, 'v2xverse_data_{}_{}.npy'.format(split, traj_len)), traj_data)
        np.save(os.path.join(save_path, 'v2xverse_num_{}_{}.npy'.format(split, traj_len)), num_agent)