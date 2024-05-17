
import os
from collections import OrderedDict
import cv2
import h5py
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import json
import random
import re
import math

import logging
_logger = logging.getLogger(__name__)

import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.camera_utils import load_camera_data
from opencood.utils.transformation_utils import x1_to_x2
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor


class V2XVERSEBaseDataset(Dataset):
    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train

        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)
        self.data_augmentor = DataAugmentor(params['data_augment'],
                                            train)

        self.frame_gap = params.get('frame_gap',200)
        self.time_delay = params.get('time_delay',0)

        if 'target_assigner_config' in self.params['loss']['args']:
            self.det_range = self.params['loss']['args']['target_assigner_config']['cav_lidar_range'] # [-36, -36, -22, 36, 36, 14]
        else:
            self.det_range = [-36, -36, -22, 36, 36, 14]

        if self.time_delay % self.frame_gap != 0:
            print("Time delay of v2xverse dataset should be a multiple of frame_gap !")
        self.frame_delay = int(self.time_delay / self.frame_gap)
        print(f'*** time_delay = {self.time_delay} ***')

        self.test_flag = False
        if self.train:
            root_dir = params['root_dir']
            towns = [1,2,3,4,6]
        elif not visualize:
            root_dir = params['validate_dir']
            towns = [7,10] # [6,7,8,9,10]
        else:
            root_dir = params['test_dir']
            towns = [5]
            self.test_flag = True
        self.root_dir = root_dir 
        self.clock = 0

        print("Dataset dir:", root_dir)

        if 'train_params' not in params or \
                'max_cav' not in params['train_params']:
            self.max_cav = 5
        else:
            self.max_cav = params['train_params']['max_cav']

        self.load_lidar_file = True if 'lidar' in params['input_source'] or self.visualize else False
        self.load_camera_file = True if 'camera' in params['input_source'] else False
        self.load_depth_file = True if 'depth' in params['input_source'] else False

        self.label_type = params['label_type'] # 'lidar' or 'camera'
        self.generate_object_center = self.generate_object_center_lidar if self.label_type == "lidar" \
                                            else self.generate_object_center_camera
        self.generate_object_center_single = self.generate_object_center # will it follows 'self.generate_object_center' when 'self.generate_object_center' change?

        if self.load_camera_file:
            self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

        # by default, we load lidar, camera and metadata. But users may
        # define additional inputs/tasks
        self.add_data_extension = \
            params['add_data_extension'] if 'add_data_extension' \
                                            in params else []

        if "noise_setting" not in self.params:
            self.params['noise_setting'] = OrderedDict()
            self.params['noise_setting']['add_noise'] = False

        if root_dir is None:
            print('Not loading from an existing dataset!')
            return
        if not os.path.exists(root_dir):
            print('Dataset path do not exists!')
            return

        # first load all paths of different scenarios
        scenario_folders = sorted([os.path.join(root_dir, x)
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x))])
        self.scenario_folders = scenario_folders

        #################################
        ## v2xverse data load
        #################################

        self.rsu_change_frame = 25
        self.route_frames = []

        data_index_name = 'dataset_index.txt'
        if 'index_file' in self.params:
            data_index_name = self.params['index_file'] + '.txt'
        print('data_index_name:', data_index_name)
        dataset_indexs = self._load_text(data_index_name).split('\n')

        filter_file = None
        if 'filte_danger' in self.params:
            if os.path.exists(os.path.join(self.root_dir,self.params['filte_danger'])):
                filter_file = self._load_json(self.params['filte_danger'])

        weathers = [0,1,2,3,4,5,6,7,8,9,10]
        pattern = re.compile('weather-(\d+).*town(\d\d)')
        for line in dataset_indexs:
            if len(line.split()) != 3:
                continue
            path, frames, egos = line.split()
            route_path = os.path.join(self.root_dir, path)
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

            # recompute rsu change frames
            file_len_list = []
            if len(rsu_files) > 0:
                for rsu_file in ['rsu_1000', 'rsu_1001']:
                    if rsu_file in rsu_files:
                        rsu_frame_len = len(os.listdir(os.path.join(route_path,rsu_file,'measurements')))
                        file_len_list.append(rsu_frame_len)
            self.rsu_change_frame = max(file_len_list) + 1

            for j, file in enumerate(ego_files):
                ego_path = os.path.join(path, file)
                others_list = ego_files[:j]+ego_files[j+1:]
                others_path_list = []
                for others in others_list:
                    others_path_list.append(os.path.join(path, others))

                for i in range(frames):
                    # reduce the ratio of frames not at junction
                    if filter_file is not None:
                        danger_frame_flag = False
                        for route_id in filter_file:
                            if route_path.endswith(filter_file[route_id]['sub_path']):
                                for junction_range in filter_file[route_id]['selected_frames'][file]:
                                    if i > junction_range[0] and i < junction_range[1]+15:
                                        danger_frame_flag = True
                        if (not danger_frame_flag):
                            continue
                    scene_dict = {}
                    scene_dict['ego'] = ego_path
                    scene_dict['other_egos'] = others_path_list
                    scene_dict['num_car'] = len(ego_files)
                    scene_dict['rsu'] = []
                    # order of rsu
                    if i%self.rsu_change_frame != 0  and len(rsu_files)>0:
                        order = int(i/self.rsu_change_frame)+1 #  int(i/10)+1 
                        rsu_path = 'rsu_{}00{}'.format(order, ego_path[-1])
                        if True: # os.path.exists(os.path.join(route_path, rsu_path,'measurements','{}.json'.format(str(i).zfill(4)))):
                            scene_dict['rsu'].append(os.path.join(path, rsu_path))

                    self.route_frames.append((scene_dict, i)) # (scene_dict, i)
        self.label_mode = self.params.get('label_mode', 'v2xverse')
        self.first_det = False
        print("Sub route dir nums: %d" % len(self.route_frames))

    def _load_text(self, path):
        text = open(os.path.join(self.root_dir,path), 'r').read()
        return text

    def _load_image(self, path):
        trans_totensor = torchvision.transforms.ToTensor()
        trans_toPIL = torchvision.transforms.ToPILImage()
        try:
            img = Image.open(os.path.join(self.root_dir,path))
            img_tensor = trans_totensor(img)
            img_PIL = trans_toPIL(img_tensor)
        except Exception as e:
            _logger.info(path)
            n = path[-8:-4]
            new_path = path[:-8] + "%04d.jpg" % (int(n) - 1)
            img = Image.open(os.path.join(self.root_dir,new_path))
            img_tensor = trans_totensor(img)
            img_PIL = trans_toPIL(img_tensor)
        return img_PIL

    def _load_json(self, path):
        try:
            json_value = json.load(open(os.path.join(self.root_dir,path)))
        except Exception as e:
            _logger.info(path)
            n = path[-9:-5]
            new_path = path[:-9] + "%04d.json" % (int(n) - 1)
            json_value = json.load(open(os.path.join(self.root_dir,new_path)))
        return json_value

    def _load_npy(self, path):
        try:
            array = np.load(os.path.join(self.root_dir,path), allow_pickle=True)
        except Exception as e:
            _logger.info(path)
            n = path[-8:-4]
            new_path = path[:-8] + "%04d.npy" % (int(n) - 1)
            array = np.load(os.path.join(self.root_dir,new_path), allow_pickle=True)
        return array

    def get_one_record(self, route_dir, frame_id, agent='ego', visible_actors=None, tpe='all', extra_source=None):
        '''
        Parameters
        ----------
        scene_dict: str, index given by dataloader.
        frame_id: int, frame id.

        Returns
        -------
        data:  
            structure: dict{
                ####################
                # input to the model
                ####################
                'agent': 'ego' or 'other_ego', # whether it is the ego car
                'rgb_[direction]': torch.Tenser, # direction in [left, right, center], shape (3, 128, 128)
                'rgb': torch.Tensor, front rgb image , # shape (3, 224, 224) 
                'measurements': torch.Tensor, size [7]: the first 6 dims is the onehot vector of command, and the last dim is car speed
                'command': int, 0-5, discrete command signal 0:left, 1:right, 2:straight, 
                                                    # 3: lane follow, 4:lane change left, 5: lane change right
                'pose': np.array, shape(3,), lidar pose[gps_x, gps_y, theta]
                'detmap_pose': pose for density map
                'target_point': torch.Tensor, size[2], (x,y) coordinate in the left hand coordinate system,
                                                                 where X-axis towards right side of the car
                'lidar': np.ndarray, # shape (3, 224, 224), 2D projection of lidar, range x:[-28m, 28m], y:[-28m,28m]
                                        in the right hand coordinate system with X-axis towards left of car
                ####################
                # target of model
                ####################
                'img_traffic': not yet used in model
                'command_waypoints': torch.Tensor, size[10,2], 10 (x,y) coordinates in the same coordinate system with target point
                'is_junction': int, 0 or 1, 1 means the car is at junction
                'traffic_light_state': int, 0 or 1
                'det_data': np.array, (400,7), flattened density map, 7 feature dims corresponds to 
                                                [prob_obj, box bias_X, box bias_Y, box_orientation, l, w, speed]
                'img_traj': not yet used in model
                'stop_sign': int, 0 or 1, exist of stop sign
        },
        '''

        output_record = OrderedDict()

        if agent == 'ego':
            output_record['ego'] = True
        else:
            output_record['ego'] = False

        BEV = None

        if route_dir is not None:
            measurements = self._load_json(os.path.join(route_dir, "measurements", "%04d.json" % frame_id))
            actors_data = self._load_json(os.path.join(route_dir, "actors_data", "%04d.json" % frame_id))
        elif extra_source is not None:
            if 'actors_data' in extra_source:
                actors_data = extra_source['actors_data']
            else:
                actors_data = {}
            measurements = extra_source['measurements']

        ego_loc = np.array([measurements['x'], measurements['y']])
        output_record['params'] = {}
        
        cam_list = ['front','right','left','rear']
        cam_angle_list = [0, 60, -60, 180]
        for cam_id in range(4):
            output_record['params']['camera{}'.format(cam_id)] = {}
            output_record['params']['camera{}'.format(cam_id)]['cords'] = [measurements['x'], measurements['y'], 1.0,\
	 						                                                0,measurements['theta']/np.pi*180+cam_angle_list[cam_id],0]
            output_record['params']['camera{}'.format(cam_id)]['extrinsic'] = measurements['camera_{}_extrinsics'.format(cam_list[cam_id])]
            output_record['params']['camera{}'.format(cam_id)]['intrinsic'] = measurements['camera_{}_intrinsics'.format(cam_list[cam_id])]

        if 'speed' in measurements:
            output_record['params']['ego_speed'] = measurements['speed']*3.6
        else:
            output_record['params']['ego_speed'] = 0

        output_record['params']['lidar_pose'] = \
                        [measurements['lidar_pose_x'], measurements['lidar_pose_y'], 0, \
                        0,measurements['theta']/np.pi*180-90,0]
        self.distance_to_map_center = (self.det_range[3]-self.det_range[0])/2+self.det_range[0]
        output_record['params']['map_pose'] = \
                        [measurements['lidar_pose_x'] + self.distance_to_map_center*np.cos(measurements["theta"]-np.pi/2),
                         measurements['lidar_pose_y'] + self.distance_to_map_center*np.sin(measurements["theta"]-np.pi/2), 0, \
                        0,measurements['theta']/np.pi*180-90,0]
        detmap_pose_x = measurements['lidar_pose_x'] + self.distance_to_map_center*np.cos(measurements["theta"]-np.pi/2)
        detmap_pose_y = measurements['lidar_pose_y'] + self.distance_to_map_center*np.sin(measurements["theta"]-np.pi/2)
        detmap_theta = measurements["theta"] + np.pi/2
        output_record['detmap_pose'] = np.array([-detmap_pose_y, detmap_pose_x, detmap_theta])
        output_record['params']['lidar_pose_clean'] = output_record['params']['lidar_pose']
        output_record['params']['plan_trajectory'] = []
        output_record['params']['true_ego_pos'] = \
                        [measurements['lidar_pose_x'], measurements['lidar_pose_y'], 0, \
                         0,measurements['theta']/np.pi*180,0]
        output_record['params']['predicted_ego_pos'] = \
                        [measurements['lidar_pose_x'], measurements['lidar_pose_y'], 0, \
                        0,measurements['theta']/np.pi*180,0]
        
        if tpe == 'all':
            if route_dir is not None:
                lidar = self._load_npy(os.path.join(route_dir, "lidar", "%04d.npy" % frame_id))
                output_record['rgb_front'] = self._load_image(os.path.join(route_dir, "rgb_front", "%04d.jpg" % frame_id))
                output_record['rgb_left'] = self._load_image(os.path.join(route_dir, "rgb_left", "%04d.jpg" % frame_id))
                output_record['rgb_right'] = self._load_image(os.path.join(route_dir, "rgb_right", "%04d.jpg" % frame_id))
                output_record['rgb_rear'] = self._load_image(os.path.join(route_dir, "rgb_rear", "%04d.jpg" % frame_id))
                if agent != 'rsu':
                    BEV = self._load_image(os.path.join(route_dir, "birdview", "%04d.jpg" % frame_id))
            elif extra_source is not None:
                lidar = extra_source['lidar']
                if 'rgb_front' in extra_source:
                    output_record['rgb_front'] = extra_source['rgb_front']
                    output_record['rgb_left'] = extra_source['rgb_left']
                    output_record['rgb_right'] = extra_source['rgb_right']
                    output_record['rgb_rear'] = extra_source['rgb_rear']
                else:
                    output_record['rgb_front'] = None
                    output_record['rgb_left'] = None
                    output_record['rgb_right'] = None
                    output_record['rgb_rear'] = None
                BEV = None

            output_record['lidar_np'] = lidar
            lidar_transformed = np.zeros((output_record['lidar_np'].shape))
            lidar_transformed[:,0] = output_record['lidar_np'][:,1]
            lidar_transformed[:,1] = -output_record['lidar_np'][:,0]
            lidar_transformed[:,2:] = output_record['lidar_np'][:,2:]
            output_record['lidar_np'] = lidar_transformed.astype(np.float32)
            output_record['lidar_np'][:, 2] += measurements['lidar_pose_z']

        if visible_actors is not None:
            actors_data = self.filter_actors_data_according_to_visible(actors_data, visible_actors)

        ################ LSS debug TODO: clean up this function #####################
        if not self.first_det:
            import copy
            if True: # agent=='rsu':
                measurements["affected_light_id"] = -1
                measurements["is_vehicle_present"] = []
                measurements["is_bike_present"] = []
                measurements["is_junction_vehicle_present"] = []
                measurements["is_pedestrian_present"] = []
                measurements["future_waypoints"] = []
            cop3_range = [36,12,12,12, 0.25]
            heatmap = generate_heatmap_multiclass(
                copy.deepcopy(measurements), copy.deepcopy(actors_data), max_distance=36
            )
            self.det_data = (
                generate_det_data_multiclass(
                    heatmap, copy.deepcopy(measurements), copy.deepcopy(actors_data), cop3_range
                )
                .reshape(3, int((cop3_range[0]+cop3_range[1])/cop3_range[4]
                            *(cop3_range[2]+cop3_range[3])/cop3_range[4]), -1) #(2, H*W,7)
                .astype(np.float32)
            )
            self.first_det = True
            if self.label_mode == 'cop3':
                self.first_det = False
        output_record['det_data'] = self.det_data
        ##############################################################
        if agent == 'rsu' :
            for actor_id in actors_data.keys():
                if actors_data[actor_id]['tpe'] == 0:
                    box = actors_data[actor_id]['box']
                    if abs(box[0]-0.8214) < 0.01 and abs(box[1]-0.18625) < 0.01 :
                        actors_data[actor_id]['tpe'] = 3

        output_record['params']['vehicles'] = {}
        for actor_id in actors_data.keys():

            ######################
            ## debug
            ######################
            # if agent == 'ego':
            #     continue

            if tpe in [0, 1, 3]:
                if actors_data[actor_id]['tpe'] != tpe:
                    continue

            # exclude ego car
            loc_actor = np.array(actors_data[actor_id]['loc'][0:2])
            dis = np.linalg.norm(ego_loc - loc_actor)
            if dis < 0.1:
                continue

            if not ('box' in actors_data[actor_id].keys() and 'ori' in actors_data[actor_id].keys() and 'loc' in actors_data[actor_id].keys()):
                continue
            output_record['params']['vehicles'][actor_id] = {}
            output_record['params']['vehicles'][actor_id]['tpe'] = actors_data[actor_id]['tpe']
            yaw = math.degrees(math.atan(actors_data[actor_id]['ori'][1]/actors_data[actor_id]['ori'][0]))
            pitch = math.degrees(math.asin(actors_data[actor_id]['ori'][2]))
            output_record['params']['vehicles'][actor_id]['angle'] = [0,yaw,pitch]
            output_record['params']['vehicles'][actor_id]['center'] = [0,0,actors_data[actor_id]['box'][2]]
            output_record['params']['vehicles'][actor_id]['extent'] = actors_data[actor_id]['box']
            output_record['params']['vehicles'][actor_id]['location'] = [actors_data[actor_id]['loc'][0],actors_data[actor_id]['loc'][1],0]
            output_record['params']['vehicles'][actor_id]['speed'] = 3.6 * math.sqrt(actors_data[actor_id]['vel'][0]**2+actors_data[actor_id]['vel'][1]**2 )

        direction_list = ['front','left','right','rear']
        theta_list = [0,-60,60,180]
        dis_list = [0,0,0,-2.6]
        camera_data_list = []
        for i, direction in enumerate(direction_list):
            if 'rgb_{}'.format(direction) in output_record:
                camera_data_list.append(output_record['rgb_{}'.format(direction)])
            dis_to_lidar = dis_list[i]
            output_record['params']['camera{}'.format(i)]['cords'] = \
                                                                    [measurements['x'] + dis_to_lidar*np.sin(measurements['theta']), measurements['y'] - dis_to_lidar*np.cos(measurements['theta']), 2.3,\
                                                                    0,measurements['theta']/np.pi*180 - 90  + theta_list[i],0]
            output_record['params']['camera{}'.format(i)]['extrinsic'] = measurements['camera_{}_extrinsics'.format(direction_list[i])]
            output_record['params']['camera{}'.format(i)]['intrinsic'] = measurements['camera_{}_intrinsics'.format(direction_list[i])]
        output_record['camera_data'] = camera_data_list
        bev_visibility_np = 255*np.ones((256,256,3), dtype=np.uint8)
        output_record['bev_visibility.png'] = bev_visibility_np

        if agent != 'rsu':
            output_record['BEV'] = BEV
        else:
            output_record['BEV'] = None
        return output_record

    def filter_actors_data_according_to_visible(self, actors_data, visible_actors):
        to_del_id = []
        for actors_id in actors_data.keys():
            if actors_id in visible_actors:
                continue
            to_del_id.append(actors_id)
        for actors_id in to_del_id:
            del actors_data[actors_id]
        return actors_data

    def get_visible_actors_one_term(self, route_dir, frame_id):
        cur_visible_actors = []
        actors_data = self._load_json(os.path.join(route_dir, "actors_data", "%04d.json" % frame_id))

        for actors_id in actors_data:
            if actors_data[actors_id]['tpe']==2:
                continue
            if not 'lidar_visible' in actors_data[actors_id]:
                cur_visible_actors.append(actors_id)
                print('Lose of lidar_visible!')
                continue
            if actors_data[actors_id]['lidar_visible']==1:
                cur_visible_actors.append(actors_id)
        return cur_visible_actors

    def get_visible_actors(self, scene_dict, frame_id):
        visible_actors = {} # id only
        if self.test_flag:
            visible_actors['car_0'] = None
            for i, route_dir in enumerate(scene_dict['other_egos']):
                visible_actors['car_{}'.format(i+1)] = None
            for i, rsu_dir in enumerate(scene_dict['rsu']):
                visible_actors['rsu_{}'.format(i)] = None
        else:
            visible_actors['car_0'] = self.get_visible_actors_one_term(scene_dict['ego'], frame_id)
            if self.params['train_params']['max_cav'] > 1:
                for i, route_dir in enumerate(scene_dict['other_egos']):
                    visible_actors['car_{}'.format(i+1)] = self.get_visible_actors_one_term(route_dir, frame_id)
                for i, rsu_dir in enumerate(scene_dict['rsu']):
                    visible_actors['rsu_{}'.format(i)] = self.get_visible_actors_one_term(rsu_dir, frame_id)
            for keys in visible_actors:
                visible_actors[keys] = list(set(visible_actors[keys]))
        return visible_actors

    def retrieve_base_data(self, idx, tpe='all', extra_source=None, data_dir=None):
        if extra_source is None:
            if data_dir is not None:
                scene_dict, frame_id = data_dir
            else:
                scene_dict, frame_id = self.route_frames[idx]
            frame_id_latency = frame_id - self.frame_delay
            visible_actors = None
            visible_actors = self.get_visible_actors(scene_dict, frame_id)
            data = OrderedDict()
            data['car_0'] = self.get_one_record(scene_dict['ego'], frame_id , agent='ego', visible_actors=visible_actors['car_0'], tpe=tpe)
            if self.params['train_params']['max_cav'] > 1:
                for i, route_dir in enumerate(scene_dict['other_egos']):
                    try:
                        data['car_{}'.format(i+1)] = self.get_one_record(route_dir, frame_id_latency , agent='other_ego', visible_actors=visible_actors['car_{}'.format(i+1)], tpe=tpe)
                    except:
                        print('load other ego failed')
                        continue
            if self.params['train_params']['max_cav'] > 2:
                for i, rsu_dir in enumerate(scene_dict['rsu']):
                    try:
                        data['rsu_{}'.format(i)] = self.get_one_record(rsu_dir, frame_id_latency, agent='rsu', visible_actors=visible_actors['rsu_{}'.format(i)], tpe=tpe)
                    except:
                        print('load rsu failed')
                        continue
        else:
            data = OrderedDict()
            scene_dict = None
            frame_id = None
            data['car_0'] = self.get_one_record(route_dir=None, frame_id=None , agent='ego', visible_actors=None, tpe=tpe, extra_source=extra_source['car_data'][0])
            if self.params['train_params']['max_cav'] > 1:
                if len(extra_source['car_data']) > 1:
                    for i in range(len(extra_source['car_data'])-1):
                        data['car_{}'.format(i+1)] = self.get_one_record(route_dir=None, frame_id=None , agent='other_ego', visible_actors=None, tpe=tpe, extra_source=extra_source['car_data'][i+1])
                for i in range(len(extra_source['rsu_data'])):
                    data['rsu_{}'.format(i)] = self.get_one_record(route_dir=None, frame_id=None , agent='rsu', visible_actors=None, tpe=tpe, extra_source=extra_source['rsu_data'][i])            
        data['car_0']['scene_dict'] = scene_dict
        data['car_0']['frame_id'] = frame_id
        return data


    def __len__(self):
        return len(self.route_frames)

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        pass

    @staticmethod
    def extract_timestamps(yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split('/')[-1]

            timestamp = res.replace('.yaml', '')
            timestamps.append(timestamp)

        return timestamps

    @staticmethod
    def return_timestamp_key(scenario_database, timestamp_index):
        """
        Given the timestamp index, return the correct timestamp key, e.g.
        2 --> '000078'.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all contents in the current scenario.

        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        # get all timestamp keys
        timestamp_keys = list(scenario_database.items())[0][1]
        # retrieve the correct index
        timestamp_key = list(timestamp_keys.items())[timestamp_index][0]

        return timestamp_key

    @staticmethod
    def find_camera_files(cav_path, timestamp, sensor="camera"):
        """
        Retrieve the paths to all camera files.

        Parameters
        ----------
        cav_path : str
            The full file path of current cav.

        timestamp : str
            Current timestamp

        sensor : str
            "camera" or "depth" 

        Returns
        -------
        camera_files : list
            The list containing all camera png file paths.
        """
        camera0_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}0.png')
        camera1_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}1.png')
        camera2_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}2.png')
        camera3_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}3.png')
        return [camera0_file, camera1_file, camera2_file, camera3_file]


    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        Given the raw point cloud, augment by flipping and rotation.

        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape

        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw

        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask


    def generate_object_center_lidar(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.
        The object_bbx_center is in ego coordinate.

        Notice: it is a wrap of postprocessor

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        return self.post_processor.generate_object_center(cav_contents,
                                                        reference_lidar_pose)

    def generate_object_center_camera(self, 
                                cav_contents, 
                                reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.
        The object_bbx_center is in ego coordinate.

        Notice: it is a wrap of postprocessor

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.
        
        visibility_map : np.ndarray
            for OPV2V, its 256*256 resolution. 0.39m per pixel. heading up.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        return self.post_processor.generate_visible_object_center(
            cav_contents, reference_lidar_pose
        )

    def get_ext_int(self, params, camera_id):
        if self.params['extrinsic'] == 1:
            return self.get_ext_int_1(params, camera_id)
        elif self.params['extrinsic'] == 2:
            return self.get_ext_int_2(params, camera_id)
    def get_ext_int_1(self, params, camera_id):
        camera_coords = np.array(params["camera%d" % camera_id]["cords"]).astype(
            np.float32)
        camera_to_lidar = x1_to_x2(
            camera_coords, params["lidar_pose_clean"]
        ).astype(np.float32)  # T_LiDAR_camera
        camera_to_lidar = camera_to_lidar @ np.array(
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
            dtype=np.float32)  # UE4 coord to opencv coord
        camera_intrinsic = np.array(params["camera%d" % camera_id]["intrinsic"]).astype(
            np.float32
        )
        return camera_to_lidar, camera_intrinsic
    def get_ext_int_2(self, params, camera_id):
        camera_extrinsic = np.array(params["camera%d" % camera_id]["extrinsic"]).astype(
            np.float32)
        camera_extrinsic = camera_extrinsic @ np.array(
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
            dtype=np.float32)  # UE4 coord to opencv coord
        camera_intrinsic = np.array(params["camera%d" % camera_id]["intrinsic"]).astype(
            np.float32
        )
        return camera_extrinsic, camera_intrinsic
VALUES = [255]
EXTENT = [0]
def generate_heatmap_multiclass(measurements, actors_data, max_distance=30, pixels_per_meter=8):
    actors_data_multiclass = {
        0: {}, 1: {}, 2:{}, 3:{}
    }
    for _id in actors_data.keys():
        actors_data_multiclass[actors_data[_id]['tpe']][_id] = actors_data[_id]
    heatmap_0 = generate_heatmap(measurements, actors_data_multiclass[0], max_distance, pixels_per_meter)
    heatmap_1 = generate_heatmap(measurements, actors_data_multiclass[1], max_distance, pixels_per_meter)
    # heatmap_2 = generate_heatmap(measurements, actors_data_multiclass[2], max_distance, pixels_per_meter) # traffic light, not used
    heatmap_3 = generate_heatmap(measurements, actors_data_multiclass[3], max_distance, pixels_per_meter)
    return {0: heatmap_0, 1: heatmap_1, 3: heatmap_3}

def get_yaw_angle(forward_vector):
    forward_vector = forward_vector / np.linalg.norm(forward_vector)
    yaw = math.acos(forward_vector[0])
    if forward_vector[1] < 0:
        yaw = 2 * np.pi - yaw
    return yaw

def generate_heatmap(measurements, actors_data, max_distance=30, pixels_per_meter=8):
    img_size = max_distance * pixels_per_meter * 2
    img = np.zeros((img_size, img_size, 3), np.int)
    ego_x = measurements["lidar_pose_x"]
    ego_y = measurements["lidar_pose_y"]
    ego_theta = measurements["theta"]
    R = np.array(
        [
            [np.cos(ego_theta), -np.sin(ego_theta)],
            [np.sin(ego_theta), np.cos(ego_theta)],
        ]
    )
    ego_id = None
    for _id in actors_data:
        color = np.array([1, 1, 1])
        if actors_data[_id]["tpe"] == 2:
            if int(_id) == int(measurements["affected_light_id"]):
                if actors_data[_id]["sta"] == 0:
                    color = np.array([1, 1, 1])
                else:
                    color = np.array([0, 0, 0])
                yaw = get_yaw_angle(actors_data[_id]["ori"])
                TR = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
                actors_data[_id]["loc"] = np.array(
                    actors_data[_id]["loc"][:2]
                ) + TR.T.dot(np.array(actors_data[_id]["taigger_loc"])[:2])
                actors_data[_id]["ori"] = np.array(actors_data[_id]["ori"])
                actors_data[_id]["box"] = np.array(actors_data[_id]["trigger_box"]) * 2
            else:
                continue
        raw_loc = actors_data[_id]["loc"]
        if (raw_loc[0] - ego_x) ** 2 + (raw_loc[1] - ego_y) ** 2 <= 2:
            ego_id = _id
            color = np.array([0, 1, 1])
        new_loc = R.T.dot(np.array([raw_loc[0] - ego_x, raw_loc[1] - ego_y]))
        actors_data[_id]["loc"] = np.array(new_loc)
        raw_ori = actors_data[_id]["ori"]
        new_ori = R.T.dot(np.array([raw_ori[0], raw_ori[1]]))
        actors_data[_id]["ori"] = np.array(new_ori)
        actors_data[_id]["box"] = np.array(actors_data[_id]["box"])
        if int(_id) in measurements["is_vehicle_present"]:
            color = np.array([1, 1, 1])
        elif int(_id) in measurements["is_bike_present"]:
            color = np.array([1, 1, 1])
        elif int(_id) in measurements["is_junction_vehicle_present"]:
            color = np.array([1, 1, 1])
        elif int(_id) in measurements["is_pedestrian_present"]:
            color = np.array([1, 1, 1])
        actors_data[_id]["color"] = color

    if ego_id is not None and ego_id in actors_data:
        del actors_data[ego_id]  # Do not show ego car
    for _id in actors_data:
        if actors_data[_id]["tpe"] == 2:
            continue  # FIXME donot add traffix light
            if int(_id) != int(measurements["affected_light_id"]):
                continue
            if actors_data[_id]["sta"] != 0:
                continue
        act_img = np.zeros((img_size, img_size, 3), np.uint8)
        loc = actors_data[_id]["loc"][:2]
        ori = actors_data[_id]["ori"][:2]
        box = actors_data[_id]["box"]
        if box[0] < 1.5:
            box = box * 1.5  # FIXME enlarge the size of pedstrian and bike
        color = actors_data[_id]["color"]
        for i in range(len(VALUES)):
            act_img = add_rect(
                act_img,
                loc,
                ori,
                box + EXTENT[i],
                VALUES[i],
                pixels_per_meter,
                max_distance,
                color,
            )
        act_img = np.clip(act_img, 0, 255)
        img = img + act_img
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    img = img[:, :, 0]
    return img

def add_rect(img, loc, ori, box, value, pixels_per_meter, max_distance, color):
    img_size = max_distance * pixels_per_meter * 2
    vet_ori = np.array([-ori[1], ori[0]])
    hor_offset = box[0] * ori
    vet_offset = box[1] * vet_ori
    left_up = (loc + hor_offset + vet_offset + max_distance) * pixels_per_meter
    left_down = (loc + hor_offset - vet_offset + max_distance) * pixels_per_meter
    right_up = (loc - hor_offset + vet_offset + max_distance) * pixels_per_meter
    right_down = (loc - hor_offset - vet_offset + max_distance) * pixels_per_meter
    left_up = np.around(left_up).astype(np.int)
    left_down = np.around(left_down).astype(np.int)
    right_down = np.around(right_down).astype(np.int)
    right_up = np.around(right_up).astype(np.int)
    left_up = list(left_up)
    left_down = list(left_down)
    right_up = list(right_up)
    right_down = list(right_down)
    color = [int(x) for x in value * color]
    cv2.fillConvexPoly(img, np.array([left_up, left_down, right_down, right_up]), color)
    return img

def generate_det_data_multiclass(
    heatmap, measurements, actors_data, det_range=[30,10,10,10, 0.8]
):  
    actors_data_multiclass = {
        0: {}, 1: {}, 2: {}, 3:{}
    }
    for _id in actors_data.keys():
        actors_data_multiclass[actors_data[_id]['tpe']][_id] = actors_data[_id]
    det_data = []
    for _class in range(4):
        if _class != 2:
            det_data.append(generate_det_data(heatmap[_class], measurements, actors_data_multiclass[_class], det_range))

    return np.array(det_data)

from skimage.measure import block_reduce

def generate_det_data(
    heatmap, measurements, actors_data, det_range=[30,10,10,10, 0.8]
):
    res = det_range[4]
    max_distance = max(det_range)
    traffic_heatmap = block_reduce(heatmap, block_size=(int(8*res), int(8*res)), func=np.mean)
    traffic_heatmap = np.clip(traffic_heatmap, 0.0, 255.0)
    traffic_heatmap = traffic_heatmap[:int((det_range[0]+det_range[1])/res), int((max_distance-det_range[2])/res):int((max_distance+det_range[3])/res)]
    det_data = np.zeros((int((det_range[0]+det_range[1])/res), int((det_range[2]+det_range[3])/res), 7)) # (50,25,7)
    vertical, horizontal = det_data.shape[:2]

    ego_x = measurements["lidar_pose_x"]
    ego_y = measurements["lidar_pose_y"]
    ego_theta = measurements["theta"]
    R = np.array(
        [
            [np.cos(ego_theta), -np.sin(ego_theta)],
            [np.sin(ego_theta), np.cos(ego_theta)],
        ]
    )
    need_deleted_ids = []
    for _id in actors_data:
        raw_loc = actors_data[_id]["loc"]
        new_loc = R.T.dot(np.array([raw_loc[0] - ego_x, raw_loc[1] - ego_y]))
        new_loc[1] = -new_loc[1]
        actors_data[_id]["loc"] = np.array(new_loc)
        raw_ori = actors_data[_id]["ori"]
        new_ori = R.T.dot(np.array([raw_ori[0], raw_ori[1]]))
        dis = new_loc[0] ** 2 + new_loc[1] ** 2
        if (
            dis <= 2
            or dis >= (max_distance) ** 2 * 2
            or "box" not in actors_data[_id]
            or actors_data[_id]['tpe'] == 2
        ):
            need_deleted_ids.append(_id)
            continue
        actors_data[_id]["ori"] = np.array(new_ori)
        actors_data[_id]["box"] = np.array(actors_data[_id]["box"])

    for _id in need_deleted_ids:
        del actors_data[_id]

    for i in range(vertical):  # 50
        for j in range(horizontal):  # 25
            if traffic_heatmap[i][j] < 0.05 * 255.0:
                continue
            center_x, center_y = convert_grid_to_xy(i, j, det_range)
            min_dis = 1000
            min_id = None
            for _id in actors_data:
                loc = actors_data[_id]["loc"][:2]
                ori = actors_data[_id]["ori"][:2]
                box = actors_data[_id]["box"]
                dis = (loc[0] - center_x) ** 2 + (loc[1] - center_y) ** 2
                if dis < min_dis:
                    min_dis = dis
                    min_id = _id

            if min_id is None:
                continue

            loc = actors_data[min_id]["loc"][:2]
            ori = actors_data[min_id]["ori"][:2]
            box = actors_data[min_id]["box"]
            theta = (get_yaw_angle(ori) / np.pi + 2) % 2
            speed = np.linalg.norm(actors_data[min_id]["vel"])

            # prob = np.power(0.5 / max(0.5, np.sqrt(min_dis)), 0.5)

            det_data[i][j] = np.array(
                [
                    0,
                    (loc[0] - center_x) * 3.0,
                    (loc[1] - center_y) * 3.0,
                    theta / 2.0,
                    box[0] / 7.0,
                    box[1] / 4.0,
                    0,
                ]
            )

    heatmap = np.zeros((int((det_range[0]+det_range[1])/res), int((det_range[2]+det_range[3])/res))) # (50,25)
    for _id in actors_data:
        loc = actors_data[_id]["loc"][:2]
        ori = actors_data[_id]["ori"][:2]
        box = actors_data[_id]["box"]
        try:
            x,y = loc
            i,j = convert_xy_to_grid(x,y,det_range)
            i = int(np.around(i))
            j = int(np.around(j))

            if i < vertical and i > 0 and j > 0 and j < horizontal:
                det_data[i][j][-1] = 1.0

            ################## Gaussian Heatmap #####################
            w, h = box[:2]/det_range[4]
            heatmap = draw_heatmap(heatmap, h, w, j, i)
            #########################################################

            # theta = (get_yaw_angle(ori) / np.pi + 2) % 2
            # center_x, center_y = convert_grid_to_xy(i, j, det_range)

            # det_data[i][j] = np.array(
            #     [
            #         0,
            #         (loc[0] - center_x) * 3.0,
            #         (loc[1] - center_y) * 3.0,
            #         theta / 2.0,
            #         box[0] / 7.0,
            #         box[1] / 4.0,
            #         0,
            #     ]
            # )

        except:
            print('actor data error, skip!')
    det_data[:,:,0] = heatmap
    return det_data

def convert_grid_to_xy(i, j, det_range):
    x = det_range[4]*(j + 0.5) - det_range[2]
    y = det_range[0] - det_range[4]*(i+0.5)
    return x, y

def convert_xy_to_grid(x, y, det_range):
    j = (x + det_range[2]) / det_range[4] - 0.5
    i = (det_range[0] - y) / det_range[4] - 0.5
    return i, j

def draw_heatmap(heatmap, h, w, x, y):
    feature_map_size = heatmap.shape
    radius = gaussian_radius(
                    (h, w),
                    min_overlap=0.1)
    radius = max(2, int(radius))

    # throw out not in range objects to avoid out of array
    # area when creating the heatmap
    if not (0 <= y < feature_map_size[0]
            and 0 <= x < feature_map_size[1]):
        return heatmap

    heatmap = draw_gaussian(heatmap, (x,y), radius) 
    return heatmap

def draw_gaussian(heatmap, center, radius, k=1):
    """Get gaussian masked heatmap.

    Args:
        heatmap (torch.Tensor): Heatmap to be masked.
        center (torch.Tensor): Center coord of the heatmap.
        radius (int): Radius of gausian.
        K (int): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        torch.Tensor: Masked heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                                radius - left:radius + right]
    
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        # torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    #     masked_heatmap = np.max([masked_heatmap[None,], (masked_gaussian * k)[None,]], axis=0)[0]
    # heatmap[y - top:y + bottom, x - left:x + right] = masked_heatmap
    return heatmap

def gaussian_2d(shape, sigma=1):
    """Generate gaussian map.

    Args:
        shape (list[int]): Shape of the map.
        sigma (float): Sigma to generate gaussian map.
            Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian_radius(det_size, min_overlap=0.5):
    """Get radius of gaussian.

    Args:
        det_size (tuple[torch.Tensor]): Size of the detection result.
        min_overlap (float): Gaussian_overlap. Defaults to 0.5.

    Returns:
        torch.Tensor: Computed radius.
    """
    height, width = det_size
    
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)