import os
import copy
import re
import io
import logging
import json
import numpy as np
import torch
import carla
import cv2
import math
import datetime
import pathlib
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
from skimage.measure import block_reduce

import matplotlib.pyplot as plt
from team_code.planner import RoutePlanner
import torch.nn.functional as F
import pygame

import pdb

from team_code.v2x_controller import V2X_Controller
from team_code.eval_utils import turn_traffic_into_bbox_fast
from team_code.render_mwb import render, render_self_car, render_waypoints
from team_code.v2x_utils import (generate_relative_heatmap, 
				 generate_heatmap, generate_det_data,
				 get_yaw_angle, boxes_to_corners_3d, get_points_in_rotated_box_3d  # visibility related functions
				 )

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

####### Input: raw_data, N(actor)+M(RSU)
####### Output: actors action, N(actor)
####### Generate the action with the trained model.

SAVE_PATH = os.environ.get("SAVE_PATH", 'eval')
os.environ["SDL_VIDEODRIVER"] = "dummy"

class DisplayInterface(object):
	def __init__(self):
		self._width = 2300
		self._height = 600
		self._surface = None

		pygame.init()
		pygame.font.init()
		self._clock = pygame.time.Clock()
		self._display = pygame.display.set_mode(
			(self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF
		)
		pygame.display.set_caption("V2X Agent")

	def run_interface(self, input_data):
		rgb = input_data['rgb']
		map = input_data['map']
		lidar = input_data['lidar']
		surface = np.zeros((600, 2300, 3),np.uint8)
		surface[:, :800] = rgb
		surface[:,800:1400] = lidar
		surface[:,1400:2000] = input_data['lidar_rsu']
		surface[:,2000:2300] = input_data['map']
		surface[:150,:200] = input_data['rgb_left']
		surface[:150, 600:800] = input_data['rgb_right']
		surface[:150, 325:475] = input_data['rgb_focus']
		surface = cv2.putText(surface, input_data['control'], (20,580), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
		surface = cv2.putText(surface, input_data['meta_infos'][1], (20,560), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
		surface = cv2.putText(surface, input_data['meta_infos'][2], (20,540), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
		surface = cv2.putText(surface, input_data['time'], (20,520), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)

		surface = cv2.putText(surface, 'Left  View', (40,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)
		surface = cv2.putText(surface, 'Focus View', (335,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)
		surface = cv2.putText(surface, 'Right View', (640,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)

		surface = cv2.putText(surface, 'Single GT', (2180,45), cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255), 2)

		# surface = cv2.putText(surface, 'Future Prediction', (940,420), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
		# surface = cv2.putText(surface, 't', (1160,385), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0), 2)
		# surface = cv2.putText(surface, '0', (1170,385), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
		# surface = cv2.putText(surface, 't', (960,585), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0), 2)
		# surface = cv2.putText(surface, '1', (970,585), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
		# surface = cv2.putText(surface, 't', (1160,585), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0), 2)
		# surface = cv2.putText(surface, '2', (1170,585), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)

		# surface[:150,198:202]=0
		# surface[:150,323:327]=0
		# surface[:150,473:477]=0
		# surface[:150,598:602]=0
		# surface[148:152, :200] = 0
		# surface[148:152, 325:475] = 0
		# surface[148:152, 600:800] = 0
		# surface[430:600, 998:1000] = 255
		# surface[0:600, 798:800] = 255
		# surface[0:600, 1198:1200] = 255
		# surface[0:2, 800:1200] = 255
		# surface[598:600, 800:1200] = 255
		# surface[398:400, 800:1200] = 255
		surface[:, 798:802] = 255
		surface[:, 1398:1402] = 255
		surface[:, 1998:2002] = 255


		# display image
		self._surface = pygame.surfarray.make_surface(surface.swapaxes(0, 1))
		if self._surface is not None:
			self._display.blit(self._surface, (0, 0))

		pygame.display.flip()
		pygame.event.get()
		return surface

	def _quit(self):
		pygame.quit()



class BasePreprocessor(object):
    """
    Basic Lidar pre-processor.
    Parameters
    ----------
    preprocess_params : dict
        The dictionary containing all parameters of the preprocessing.
    train : bool
        Train or test mode.
    """

    def __init__(self, preprocess_params, train):
        self.params = preprocess_params
        self.train = train


class SpVoxelPreprocessor(BasePreprocessor):
    def __init__(self, preprocess_params, train):
        super(SpVoxelPreprocessor, self).__init__(preprocess_params,
                                                  train)
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
        except:
            from spconv.utils import VoxelGenerator

        self.lidar_range = self.params['cav_lidar_range']
        self.voxel_size = self.params['args']['voxel_size']
        self.max_points_per_voxel = self.params['args']['max_points_per_voxel']

        if train:
            self.max_voxels = self.params['args']['max_voxel_train']
        else:
            self.max_voxels = self.params['args']['max_voxel_test']

        grid_size = (np.array(self.lidar_range[3:6]) -
                     np.array(self.lidar_range[0:3])) / np.array(self.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)

        # use sparse conv library to generate voxel
        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.lidar_range,
            max_num_points=self.max_points_per_voxel,
            max_voxels=self.max_voxels
        )

    def preprocess(self, pcd_np):
        data_dict = {}
        voxel_output = self.voxel_generator.generate(pcd_np)
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = \
                voxel_output['voxels'], voxel_output['coordinates'], \
                voxel_output['num_points_per_voxel']
        else:
            voxels, coordinates, num_points = voxel_output

        data_dict['voxel_features'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points


        return data_dict


def turn_back_into_theta(input):
    B,K,_,H,W = input.shape
    output = torch.cat([input[:,:,:2],torch.atan2(input[:,:,2:3], input[:,:,-1:]),input[:,:,3:]],dim=2)
    assert output.shape[2] == input.shape[2]
    return output

def turn_traffic_into_map(pred_traffic, det_range):
    data_total = []
    for idx in range(pred_traffic.shape[0]):
        all_bbox = []
        cls_num = pred_traffic.shape[1]
        for i in range(cls_num):
            map_cur = pred_traffic[idx, i]
            object_bbox, _ = turn_traffic_into_bbox_fast(map_cur,det_range)
            if len(object_bbox) > 0:
                all_bbox.append(object_bbox)

        if len(all_bbox) > 0:
            all_bbox = np.concatenate(all_bbox,axis=0)
        else:
            all_bbox = np.zeros((1,4,2))
        # plt.cla()

        fig = plt.figure(figsize=(6, 12), dpi=16)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        ax = plt.gca()
        ax.set_facecolor("black")

        plt.xlim((-det_range[2], det_range[3]))
        plt.ylim((-det_range[1], det_range[0]))

        for i in range(len(all_bbox)):
            plt.fill(all_bbox[i,:,0], all_bbox[i,:,1], color = 'white')

        # plt.axis('off')
        # If we haven't already shown or saved the plot, then we need to
        # draw the figure first...
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # H=192, W=96, 3
        data_total.append(data[:, :, 0])
        # plt.savefig('/GPFS/public/InterFuser/results/cop3/pnp/multiclass_finetune_fusion_none/test.png')
        plt.close()

    occ_map = np.stack(data_total, axis=0) # B * T_p, H, W
    return occ_map



def x_to_world(pose):
    """
    The transformation matrix from x-coordinate system to carla world system
    Also is the pose in world coordinate: T_world_x

    Parameters
    ----------
    pose : list
        [x, y, z, roll, yaw, pitch], degree
        [x, y, roll], radians
    Returns
    -------
    matrix : np.ndarray
        The transformation matrix.
    """
    x, y, roll= pose[:]
    z = 0
    yaw = 0
    pitch = 0

    # used for rotation matrix
    c_r = np.cos(roll)
    s_r = np.sin(roll)

    matrix = np.identity(4)

    # translation matrix
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # rotation matrix
    matrix[0,0] = c_r
    matrix[0,1] = -s_r
    matrix[1,0] = s_r
    matrix[1,1] = c_r

    return matrix

def get_pairwise_transformation(pose, max_cav):
    """
    Get pair-wise transformation matrix accross different agents.

    Parameters
    ----------
    base_data_dict : dict
        Key : cav id, item: transformation matrix to ego, lidar points.

    max_cav : int
        The maximum number of cav, default 5

    Return
    ------
    pairwise_t_matrix : np.array
        The pairwise transformation matrix across each cav.
        shape: (L, L, 4, 4), L is the max cav number in a scene
        pairwise_t_matrix[i, j] is Tji, i_to_j
    """
    pairwise_t_matrix = np.tile(np.eye(4), (max_cav, max_cav, 1, 1)) # (L, L, 4, 4)


    t_list = []

    # save all transformation matrix in a list in order first.
    for i in range(max_cav):
        lidar_pose = pose[i]
        t_list.append(x_to_world(lidar_pose))  # Twx

    for i in range(len(t_list)):
        for j in range(len(t_list)):
            # identity matrix to self
            if i != j:
                # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                # t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                t_matrix = np.linalg.solve(t_list[j], t_list[i])  # Tjw*Twi = Tji
                pairwise_t_matrix[i, j] = t_matrix

    return pairwise_t_matrix

def warp_affine_simple(src, M, dsize,
        align_corners=False):

    B, C, H, W = src.size()
    grid = F.affine_grid(M,
                        [B, C, dsize[0], dsize[1]],
                        align_corners=align_corners).to(src)
    return F.grid_sample(src, grid, align_corners=align_corners)

def warp_image(det_pose, occ_map):
    '''
    det_pose: B, T_p, 3, torch.Tensor
    occ_map: B, T_p, C, H, W, torch.Tensor
    '''
    B, T, C, H, W = occ_map.shape
    occ_fused = []
    for b in range(B):
        pairwise_t_matrix = \
            get_pairwise_transformation(det_pose[b].cpu(), T)
        # t_matrix[i, j]-> from i to j
        pairwise_t_matrix = pairwise_t_matrix[:,:,[0, 1],:][:,:,:,[0, 1, 3]] # [N, N, 2, 3]
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (12)  #(downsample_rate * discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (24)

        t_matrix = torch.from_numpy(pairwise_t_matrix[:T, :T, :, :])
        
        neighbor_feature = warp_affine_simple(occ_map[b],
                                        t_matrix[-1, :, :, :],
                                        (H, W))                               
        # print(neighbor_feature.shape)
        occ_fused.append(neighbor_feature)
    
    return torch.stack(occ_fused)


class PnP_infer():
	def __init__(self, config=None, ego_vehicles_num=1, perception_model=None, planning_model=None) -> None:
		self.config = config
		self._hic = DisplayInterface()
		self.ego_vehicles_num = ego_vehicles_num

		self.memory_measurements = [[], [], [], [], []]
		self.memory_actors_data = [[], [], [], [], []]
		self.det_range = [36, 12, 12, 12, 0.25]
		self.max_distance = 36
		self.distance_to_map_center = (self.det_range[0]+self.det_range[1])/2-self.det_range[1]

		#### Voxelization Process
		voxel_args = {
			'args': {
				'voxel_size': [0.125, 0.125, 4], # 
				'max_points_per_voxel': 32,
				'max_voxel_train': 70000,
				'max_voxel_test': 40000
			},
			'cav_lidar_range': [-12, -36, -22, 12, 12, 14]   # x_min, y_min, z_min, x_max, y_max, z_max
		}
		self.voxel_preprocess = SpVoxelPreprocessor(voxel_args, train=False)
	

		self.perception_model = perception_model
		self.planning_model = planning_model

		self.perception_memory_bank = [{}]

		self.controller = [V2X_Controller(self.config) for _ in range(self.ego_vehicles_num)]

		self.input_lidar_size = 224
		self.lidar_range = [36, 36, 36, 36]

		self.softmax = torch.nn.Softmax(dim=0)
		self.traffic_meta_moving_avg = np.zeros((ego_vehicles_num, 400, 7))
		self.momentum = self.config.momentum
		self.prev_lidar = []
		self.prev_control = {}
		self.prev_surround_map = {}

		############
		###### multi-agent related components
		############

		### generate the save files for images
		self.skip_frames = self.config.skip_frames
		self.save_path = None
		if SAVE_PATH is not None:
			now = datetime.datetime.now()
			string = pathlib.Path(os.environ["ROUTES"]).stem + "_"
			string += "_".join(
				map(
					lambda x: "%02d" % x,
					(now.month, now.day, now.hour, now.minute, now.second),
				)
			)

			print(string)

			self.save_path = pathlib.Path(SAVE_PATH) / string
			self.save_path.mkdir(parents=True, exist_ok=False)
			(self.save_path / "meta").mkdir(parents=True, exist_ok=False)


	def get_action_from_list_inter(self, car_data_raw, rsu_data_raw, step, timestamp):
		'''
		generate the action for N cars from the record data.

		Parameters
		----------
		car_data : list[ dict{}, dict{}, None, ...],
		rsu_data : list[ dict{}, dict{}, None, ...],
		model : trained model, probably we can store it in the initialization.
		step : int, frame in the game, 20hz.
		timestamp : float, time in the game.
	
		Returns
		-------
		controll_all: list, detailed actions for N cars.
		'''
		
		### Check the data is None or not.
		### If the data is not None, return the filtered data with new keys
		### 	for batch collection.
		car_data, car_mask = self.check_data(car_data_raw)
		rsu_data, _ = self.check_data(rsu_data_raw, car=False)
		batch_data = self.collate_batch_infer_perception(car_data, rsu_data)  # batch_size: N*(N+M)
		# batch_data for perception
		
		### Get peception results!
		### model inferece: N*(N+M) -> N, generate perception for
		### cars only. 
		with torch.no_grad():
			# for key in batch_data.keys():
			# 	print(key, batch_data[key].shape)
			# pdb.set_trace()
			perception_output = self.perception_model(batch_data)
		### batch_size: N
		perception_output['bbox_preds'] = turn_back_into_theta(perception_output['bbox_preds'])
		
		pred_traffic = torch.cat((perception_output['cls_preds'].sigmoid(), perception_output['bbox_preds']), dim=2).permute(0, 1, 3, 4, 2).cpu().numpy()
		# N, K, H, W, C=7
		occ_map = turn_traffic_into_map(pred_traffic, self.det_range)
		occ_map_shape = occ_map.shape
		occ_map = torch.from_numpy(occ_map).cuda().contiguous().view((-1, 1) + occ_map_shape[1:]) 
		# N, 1, H, W
		
		
		da = []
		for i in range(len(car_data_raw)):
			da.append(torch.from_numpy(car_data_raw[i]['drivable_area']).cuda().float().unsqueeze(0))
		

		self.perception_memory_bank.pop(0)
		if len(self.perception_memory_bank)<5:
			for _ in range(5 - len(self.perception_memory_bank)):
				self.perception_memory_bank.append({
					'occ_map': occ_map, # N, 1, H, W
					'drivable_area': torch.stack(da), # N, 1, H, W
					'detmap_pose': batch_data['detmap_pose'][:len(car_data_raw)], # N, 3
					'target': batch_data['target'][:len(car_data_raw)], # N, 2
				})
		

		#### Turn the memoried perception output into planning input
		planning_input = self.generate_planning_input()

		predicted_waypoints = self.planning_model(planning_input)
		# predicted_waypoints: N, T_f=10, 2

		### output postprocess to generate the action, list of actions for N agents
		control_all = self.generate_action_from_model_output(predicted_waypoints, car_data_raw, rsu_data_raw, car_data, rsu_data, batch_data, planning_input, car_mask, step, timestamp)
		return control_all
	
	def generate_planning_input(self):
		
		occ_final = torch.zeros(self.ego_vehicles_num, 5, 6, 192, 96).cuda().float()
		# N, T, C, H, W

		self_car_map = render_self_car( 
			loc=np.array([0, 0]),
			ori=np.array([0, -1]),
			box=np.array([2.45, 1.0]),
			color=[1, 1, 0], 
			pixels_per_meter=8,
			max_distance=self.max_distance
		)[:, :, 0]
		self_car_map = block_reduce(self_car_map, block_size=(2, 2), func=np.mean)
		self_car_map = np.clip(self_car_map, 0.0, 255.0)
		self_car_map = self_car_map[:48*4, 48*2:48*4]  # H, W
		occ_ego_temp = torch.from_numpy(self_car_map).cuda().float()[None, None, None, :, :].repeat(self.ego_vehicles_num, 5, 1, 1, 1)


		coordinate_map = torch.ones((5, 2, 192, 96))
		for h in range(192):
			coordinate_map[:, 0, h, :] *= h*self.det_range[-1]-self.det_range[0]
		for w in range(96):
			coordinate_map[:, 1, :, w] *= w*self.det_range[-1]-self.det_range[2]
		coordinate_map = coordinate_map.cuda().float()

		occ_to_warp = torch.zeros(self.ego_vehicles_num, 5, 2, 192, 96).cuda().float()
		# B, T, 2, H, W
		occ_to_warp[:, :, 1:2] = occ_ego_temp
		det_map_pose = torch.zeros(self.ego_vehicles_num, 5, 3).cuda().float()

		for agent_i in range(self.ego_vehicles_num):
			for t in range(5):
				occ_others = self.perception_memory_bank[t]['occ_map'][agent_i]  # 1, H, W
				occ_to_warp[agent_i, t, 0:1] = occ_others
				det_map_pose[:, t] = self.perception_memory_bank[t]['detmap_pose'] # N, 3
			
			local_command_map = render_self_car( 
				loc=self.perception_memory_bank[-1]['target'][agent_i].cpu().numpy(),
				ori=np.array([0, 1]),
				box=np.array([1.0, 1.0]),
				color=[1, 1, 0], 
				pixels_per_meter=8,
				max_distance=self.max_distance
			)[:, :, 0]	
			local_command_map = block_reduce(local_command_map, block_size=(2, 2), func=np.mean)
			local_command_map = np.clip(local_command_map, 0.0, 255.0)
			local_command_map = torch.from_numpy(local_command_map[:48*4, 48*2:48*4]).cuda().float()[None, None, :, :].repeat(5, 1, 1, 1)

			da = self.perception_memory_bank[-1]['drivable_area'][agent_i][None, :, :, :].repeat(5, 1, 1, 1) # 5, 1, H, W

			occ_final[agent_i, :, 2:3] = local_command_map
			occ_final[agent_i, :, 3:5] = coordinate_map
			occ_final[agent_i, :, 5:6] = da
		
		occ_warped = warp_image(det_map_pose, occ_to_warp)
		occ_final[:, :, :2] = occ_warped

		return {
			"occupancy": occ_final, # N, T=5, C=6, H=192, W=96
			"target": self.perception_memory_bank[-1]['target']  # N, 2
		}

	def generate_action_from_model_output(self, pred_waypoints_total, car_data_raw, rsu_data_raw, car_data, rsu_data, batch_data, planning_input, car_mask, step, timestamp):
		control_all = []
		tick_data = []
		ego_i = -1
		for count_i in range(self.ego_vehicles_num):
			if not car_mask[count_i]:
				control_all.append(None)
				tick_data.append(None)
				continue

			# store the data for visualization
			tick_data.append({})
			ego_i += 1
			# get the data for current vehicle
			pred_waypoints = pred_waypoints_total[ego_i].detach().cpu().numpy()

			route_info = {
				'speed': car_data_raw[ego_i]['measurements']["speed"],
				'waypoints': pred_waypoints,
				'target': car_data_raw[ego_i]['measurements']["target_point"],
				'route_length': 0,
				'route_time': 0,
				'drive_length': 0,
				'drive_time': 0
			}

			steer, throttle, brake, meta_infos = self.controller[ego_i].run_step(
				route_info
			)

			if brake < 0.05:
				brake = 0.0
			if brake > 0.1:
				throttle = 0.0

			control = carla.VehicleControl()
			control.steer = float(steer)
			control.throttle = float(throttle)
			control.brake = float(brake)

			if step % 2 != 0 and step > 4:
				control = self.prev_control[ego_i]
			else:
				self.prev_control[ego_i] = control


			control_all.append(control)

			cur_actors = planning_input["occupancy"][ego_i][-1][:3].cpu().permute(1, 2, 0).contiguous().numpy()
			cur_bev = (planning_input["occupancy"][ego_i][-1][-1:].cpu().permute(1, 2, 0).repeat(1, 1, 3)*120).contiguous().numpy()
			tick_data[ego_i]["map"] = np.where(cur_actors.sum(axis=2, keepdims=True)>5, cur_actors, cur_bev)
			# pdb.set_trace()
			tick_data[ego_i]["map"] = (tick_data[ego_i]["map"]/tick_data[ego_i]["map"].max()*255).astype(np.uint8)
			# 192, 96, 3
			# planning_input["occupancy"][ego_i][-1][0] = perception_total_total[ego_i][-1]
			cur_actors = planning_input["occupancy"][ego_i][-1][:3].cpu().permute(1, 2, 0).contiguous().numpy()
			cur_bev = (planning_input["occupancy"][ego_i][-1][-1:].cpu().permute(1, 2, 0).repeat(1, 1, 3)*120).contiguous().numpy()
			tick_data[ego_i]["map_gt"] = np.where(cur_actors.sum(axis=2, keepdims=True)>5, cur_actors, cur_bev)
			# pdb.set_trace()
			tick_data[ego_i]["map_gt"] = (tick_data[ego_i]["map_gt"]/tick_data[ego_i]["map_gt"].max()*255).astype(np.uint8)
			# 192, 96, 3
			tick_data[ego_i]["map_t1"] = planning_input["occupancy"][ego_i][-2][:3].cpu().permute(1, 2, 0).numpy()

			# tick_data[ego_i]["map_gt"] = perception_total[ego_i][-1][:3].cpu().permute(1, 2, 0).numpy()
			tick_data[ego_i]["rgb_raw"] = car_data_raw[ego_i]["rgb_front"]
			# print(car_data_raw[ego_i]["rgb_front"].shape)
			# print(batch_data[ego_i]["lidar"].shape)
			tick_data[ego_i]["lidar"] = np.rot90((np.transpose(car_data[ego_i]["lidar_original"], (1, 2, 0))*127).astype(np.uint8), k=1, axes=(1,0))
			try:
				tick_data[ego_i]["lidar_rsu"] = np.rot90((np.transpose(rsu_data[ego_i]["lidar_original"], (1, 2, 0))*127).astype(np.uint8), k=1, axes=(1,0))
			except:
				tick_data[ego_i]["lidar_rsu"] = np.ones_like(tick_data[ego_i]["lidar"])
			tick_data[ego_i]["rgb_left_raw"] = car_data_raw[ego_i]["rgb_left"]
			tick_data[ego_i]["rgb_right_raw"] = car_data_raw[ego_i]["rgb_right"]
			# print(tick_data[ego_i]["rgb_raw"].shape)
			# print(tick_data[ego_i]["map"].shape)
			# raise ValueError
			# pdb.set_trace()
			for t_i in range(10):
				tick_data[ego_i]["map"][int(pred_waypoints[t_i][1]*4+144), int(pred_waypoints[t_i][0]*4+48)] = np.array([255, 0, 0])
				# tick_data[ego_i]["map"] = cv2.circle(tick_data[ego_i]["map"], (int(pred_waypoints[t_i][1]*4+144), int(pred_waypoints[t_i][0]*4+48)), radius=2, color=(255, 255, 255))
			tick_data[ego_i]["map"] = cv2.resize(tick_data[ego_i]["map"], (300, 600))
			# print(tick_data[ego_i]["map"].shape)
			tick_data[ego_i]["map_t1"] = cv2.resize(tick_data[ego_i]["map_t1"], (300, 600))
			tick_data[ego_i]["map_gt"] = cv2.resize(tick_data[ego_i]["map_gt"], (300, 600))
			tick_data[ego_i]["rgb"] = cv2.resize(tick_data[ego_i]["rgb_raw"], (800, 600))
			tick_data[ego_i]["lidar"] = cv2.resize(tick_data[ego_i]["lidar"], (600, 600))
			tick_data[ego_i]["lidar_rsu"] = cv2.resize(tick_data[ego_i]["lidar_rsu"], (600, 600))
			tick_data[ego_i]["rgb_left"] = cv2.resize(tick_data[ego_i]["rgb_left_raw"], (200, 150))
			tick_data[ego_i]["rgb_right"] = cv2.resize(tick_data[ego_i]["rgb_right_raw"], (200, 150))
			tick_data[ego_i]["rgb_focus"] = cv2.resize(tick_data[ego_i]["rgb_raw"][244:356, 344:456], (150, 150))
			if len(rsu_data_raw)>0:
				tick_data[ego_i]["control"] = "throttle: %.2f, steer: %.2f, brake: %.2f, ego: %.2f, %.2f/rsu: %.2f, %.2f" % (
					control.throttle,
					control.steer,
					control.brake,
					car_data_raw[ego_i]['measurements']["lidar_pose_x"],
					car_data_raw[ego_i]['measurements']["lidar_pose_y"],
					rsu_data_raw[ego_i]['measurements']["lidar_pose_x"],
					rsu_data_raw[ego_i]['measurements']["lidar_pose_y"],
				)
			else:
				tick_data[ego_i]["control"] = "throttle: %.2f, steer: %.2f, brake: %.2f, ego: %.2f, %.2f/rsu: None" % (
					control.throttle,
					control.steer,
					control.brake,
					car_data_raw[ego_i]['measurements']["lidar_pose_x"],
					car_data_raw[ego_i]['measurements']["lidar_pose_y"]
				)
			meta_infos[2] += ", target point: %.2f, %.2f" % (batch_data['target'][ego_i][0], batch_data['target'][ego_i][1])
			tick_data[ego_i]["meta_infos"] = meta_infos
			tick_data[ego_i]["mes"] = "speed: %.2f" % car_data_raw[ego_i]['measurements']["speed"]
			tick_data[ego_i]["time"] = "time: %.3f" % timestamp


			# NOTE: to-be check
			surface = self._hic.run_interface(tick_data[ego_i])
			tick_data[ego_i]["surface"] = surface
		
		if SAVE_PATH is not None:
			self.save(tick_data, step)
		
		return control_all


	def save(self, tick_data, frame):
		if frame % self.skip_frames != 0:
			return
		for ego_i in range(self.ego_vehicles_num):
			folder_path = self.save_path / pathlib.Path("ego_vehicle_{}".format(ego_i))
			if not os.path.exists(folder_path):
				os.mkdir(folder_path)
			Image.fromarray(tick_data[ego_i]["surface"]).save(
				folder_path / ("%04d.jpg" % frame)
			)
		return


	def generate_last_info(self, measurements_last):
		# print(measurements_last.keys())
		ego_theta = measurements_last["theta"]
		ego_x = measurements_last["gps_x"]
		ego_y = measurements_last["gps_y"]

		egp_pos_last = {
			"x": ego_y,
			"y": -ego_x,
			"theta": ego_theta
		}

		R = np.array(
			[
				[np.cos(ego_theta), -np.sin(ego_theta)],
				[np.sin(ego_theta), np.cos(ego_theta)],
			]
		)

		ego_last = {
			'egp_pos_last': egp_pos_last,
			'ego_x': ego_x,
			'ego_y': ego_y,
			'ego_theta': ego_theta,
			'R': R,
			'local_command_point': measurements_last['target_point']
		}
		return ego_last
	

	def reduce_image(self, img, pixel_per_meter=1):
		img_after = block_reduce(img, block_size=(pixel_per_meter, pixel_per_meter), func=np.mean)
		# occ_map: 75, 75
		
		img_after = np.clip(img_after, 0.0, 255.0)
		# TODO: change it into auto calculation
		img_after = torch.from_numpy(img_after[:48*8, 48*4:48*8])

		return img_after
	

	def generate_occupancy_map(self, measurements, actors_data, ego_last):
		pixel_per_meter = 1
		heatmap_visible = generate_relative_heatmap(
				copy.deepcopy(measurements), copy.deepcopy(actors_data), ego_last['egp_pos_last'],
				pixels_per_meter=8,
				max_distance=self.max_distance,
				judge_visibility=True
			)
		heatmap_total = generate_relative_heatmap(
				copy.deepcopy(measurements), copy.deepcopy(actors_data), ego_last['egp_pos_last'],
				pixels_per_meter=8,
				max_distance=self.max_distance,
				judge_visibility=False
			)

		# max_distance(32) * pixel_per_meter(8) * 2, max_distance*pixel_per_meter*2
		occ_map = self.reduce_image(heatmap_visible, pixel_per_meter)
		occ_map_total = self.reduce_image(heatmap_total, pixel_per_meter)

		# calculate the position of past frames according to current pos.
		self_car_map = render_self_car( 
			loc=np.array(ego_last['R'].T.dot(np.array(
					[measurements["gps_y"]-ego_last['ego_y'], -1*(measurements["gps_x"]-ego_last['ego_x'])]
					))),
			ori=np.array([np.sin(measurements["theta"]-ego_last['ego_theta']), -np.cos(measurements["theta"]-ego_last['ego_theta'])]),
			box=np.array([2.45, 1.0]),
			color=[1, 1, 0], 
			pixels_per_meter=8,
			max_distance=self.max_distance
		)[:, :, 0]
		self_car_map = self.reduce_image(self_car_map, pixel_per_meter)


		# ego_last['local_command_point'] = np.array(ego_last['local_command_point'])
		local_command_map = render_self_car( 
			loc=np.array(ego_last['local_command_point']),
			ori=np.array([0, 1]),
			box=np.array([1.0, 1.0]),
			color=[1, 1, 0], 
			pixels_per_meter=8,
			max_distance=self.max_distance
		)[:, :, 0]
		local_command_map = self.reduce_image(local_command_map, pixel_per_meter)


		coordinate_map = np.ones((2,)+local_command_map.shape)
		for h in range(local_command_map.shape[0]):
			coordinate_map[0, h, :] *= h*self.det_range[-1]-self.det_range[0]
		for w in range(local_command_map.shape[1]):
			coordinate_map[1, :, w] *= w*self.det_range[-1]-self.det_range[2]

		occupancy_map = torch.cat((occ_map.unsqueeze(0), self_car_map.unsqueeze(0), local_command_map.unsqueeze(0), torch.from_numpy(coordinate_map)), dim=0)[:, ::2, ::2] # C=1+1+1+2, H, W
		
		return occupancy_map.cuda().float(), occ_map_total.cuda().float()


	def check_data(self, raw_data, car=True):
		mask = []
		data = [] # without None
		for i in raw_data:
			if i is not None:
				mask.append(1) # filter the data!
				data.append(self.preprocess_data(copy.deepcopy(i), car=car))
			else:
				mask.append(0)
				data.append(0)
		return data, mask
	

	def preprocess_data(self, data, car=True):
		output_record = {
		}
		
		##########
		## load and pre-process images
		##########
		
		##########
		## load environment data and control signal
		##########    
		measurements = data['measurements']
		cmd_one_hot = [0, 0, 0, 0, 0, 0]
		if not car:
			measurements['command'] = -1
			measurements["speed"] = 0
			measurements['target_point'] = np.array([0, 0])
		cmd = measurements['command'] - 1
		if cmd < 0:
			cmd = 3
		cmd_one_hot[cmd] = 1
		cmd_one_hot.append(measurements["speed"])
		mes = np.array(cmd_one_hot)
		mes = torch.from_numpy(mes).cuda().float()

		output_record["measurements"] = mes
		output_record['command'] = cmd

		lidar_pose_x = measurements["lidar_pose_x"]
		lidar_pose_y = measurements["lidar_pose_y"]
		lidar_theta = measurements["theta"] + np.pi
		
		output_record['lidar_pose'] = np.array([-lidar_pose_y, lidar_pose_x, lidar_theta])

		## 计算density map中心点的世界坐标，目前density map预测范围为左右10m前18m后2m
		detmap_pose_x = measurements['lidar_pose_x'] + self.distance_to_map_center*np.cos(measurements["theta"]-np.pi/2)
		detmap_pose_y = measurements['lidar_pose_y'] + self.distance_to_map_center*np.sin(measurements["theta"]-np.pi/2)
		detmap_theta = measurements["theta"] + np.pi/2
		output_record['detmap_pose'] = np.array([-detmap_pose_y, detmap_pose_x, detmap_theta])
		output_record["target_point"] = torch.from_numpy(measurements['target_point']).cuda().float()
		
		##########
		## load and pre-process LiDAR from 3D point cloud to 2D map
		##########
		lidar_unprocessed = data['lidar'][:, :3]
		# print(lidar_unprocessed.shape)
		lidar_unprocessed[:, 1] *= -1
		if not car:
			lidar_unprocessed[:, 2] = lidar_unprocessed[:, 2] + np.array([measurements["lidar_pose_z"]])[np.newaxis, :] - np.array([2.1])[np.newaxis, :] 
		
		lidar_processed = self.lidar_to_histogram_features(
			lidar_unprocessed, crop=self.input_lidar_size, lidar_range=self.lidar_range
		)        
		# if self.lidar_transform is not None:
		# 	lidar_processed = self.lidar_transform(lidar_processed)
		output_record["lidar_original"] = lidar_processed

		lidar_unprocessed[:, 0] *= -1

		voxel_dict = self.voxel_preprocess.preprocess(lidar_unprocessed)
		output_record["lidar"] = voxel_dict
		return output_record


	def collect_actor_data_with_visibility(self, measurements, lidar_data):
		lidar_data = lidar_data[:, :3]
		lidar_data[:, 1] *= -1
		actors_data = self.collect_actor_data()
		original_actors_data = copy.deepcopy(actors_data)

		
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
			raw_loc = actors_data[_id]['loc'][:2]
			new_loc = R.T.dot(np.array([raw_loc[0] - ego_x , raw_loc[1] - ego_y]))
			new_loc[1] = -new_loc[1]
			actors_data[_id]['loc'][:2] = np.array(new_loc)
			actors_data[_id]['loc'][2] -= (actors_data[_id]['box'][2] + ego_z)
			raw_ori = actors_data[_id]['ori'][:2]
			new_ori = R.T.dot(np.array([raw_ori[0], raw_ori[1]]))
			actors_data[_id]['ori'][:2] = np.array(new_ori)
		
		boxes_corner = [] # pose and orientation of the box,
				# (x, y, z, scale_x, scale_y, scale_z, yaw)
		id_map = {}
		count = 0
		for _id in actors_data.keys():
			cur_data = actors_data[_id]
			yaw = get_yaw_angle(cur_data['ori'][:2])
			cur_data['loc'][2] += cur_data['box'][2]
			boxes_corner.append(cur_data['loc']+ [i*2 for i in cur_data['box']] + [yaw])
			id_map[count] = _id
			count += 1
		boxes_corner = np.array(boxes_corner)   

		corners = boxes_to_corners_3d(boxes_corner, order='lwh')

		lidar_visible = []
		# print(lidar_unprocessed[:20])
		for N in range(boxes_corner.shape[0]):
			if actors_data[id_map[N]]['tpe']==2:
				original_actors_data[id_map[N]]['lidar_visible'] = 0
				original_actors_data[id_map[N]]['camera_visible'] = 0
				continue
			num_lidar_points = get_points_in_rotated_box_3d(lidar_data, corners[N])
			# print(len(num_lidar_points))
			if len(num_lidar_points)>8:
				original_actors_data[id_map[N]]['lidar_visible'] = 1
				original_actors_data[id_map[N]]['camera_visible'] = 0
				lidar_visible += [1]
			else:
				original_actors_data[id_map[N]]['lidar_visible'] = 0
				original_actors_data[id_map[N]]['camera_visible'] = 0
				lidar_visible += [0]
		# print(lidar_visible)
		return original_actors_data



	def collect_actor_data(self):
		data = {}
		vehicles = CarlaDataProvider.get_world().get_actors().filter("*vehicle*")
		for actor in vehicles:
			loc = actor.get_location()
			if loc.z<-1:
				continue
			_id = actor.id
			data[_id] = {}
			data[_id]["loc"] = [loc.x, loc.y, loc.z]
			ori = actor.get_transform().rotation.get_forward_vector()
			data[_id]["ori"] = [ori.x, ori.y, ori.z]
			box = actor.bounding_box.extent
			data[_id]["box"] = [box.x, box.y, box.z]
			vel = actor.get_velocity()
			data[_id]["vel"] = [vel.x, vel.y, vel.z]
			data[_id]["tpe"] = 0

		walkers = CarlaDataProvider.get_world().get_actors().filter("*walker*")
		for actor in walkers:
			loc = actor.get_location()
			if loc.z<-1:
				continue
			_id = actor.id
			data[_id] = {}
			data[_id]["loc"] = [loc.x, loc.y, loc.z]
			ori = actor.get_transform().rotation.get_forward_vector()
			data[_id]["ori"] = [ori.x, ori.y, ori.z]
			try:
				box = actor.bounding_box.extent
				data[_id]["box"] = [box.x, box.y, box.z]
			except:
				data[_id]["box"] = [1, 1, 1]
			try:
				vel = actor.get_velocity()
				data[_id]["vel"] = [vel.x, vel.y, vel.z]
			except:
				data[_id]["vel"] = [0, 0, 0]
			data[_id]["tpe"] = 1

		
		return data


	# load data
	def _load_image(self, path):
		try:
			img = Image.open(self.root_path + path)
		except Exception as e:
			print('[Error] Can not find the IMAGE path.')
			n = path[-8:-4]
			new_path = path[:-8] + "%04d.jpg" % (int(n) - 1)
			img = Image.open(self.root_path + new_path)
		return img
	
	def _load_json(self, path):
		try:
			json_value = json.load(open(self.root_path + path))
		except Exception as e:
			print('[Error] Can not find the JSON path.')
			n = path[-9:-5]
			new_path = path[:-9] + "%04d.json" % (int(n) - 1)
			json_value = json.load(open(self.root_path + new_path))
		return json_value

	def _load_npy(self, path):
		try:
			array = np.load(self.root_path + path, allow_pickle=True)
		except Exception as e:
			print('[Error] Can not find the NPY path.')
			n = path[-8:-4]
			new_path = path[:-8] + "%04d.npy" % (int(n) - 1)
			array = np.load(self.root_path + new_path, allow_pickle=True)
		return array
	

	def lidar_to_histogram_features(self, lidar, crop=256, lidar_range=[28,28,28,28]):
		"""
		Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
		"""

		def splat_points(point_cloud):
			# 256 x 256 grid
			pixels_per_meter = 4
			hist_max_per_pixel = 5
			# x_meters_max = 28
			# y_meters_max = 28
			xbins = np.linspace(
				- lidar_range[3],
				lidar_range[2],
				(lidar_range[2]+lidar_range[3])* pixels_per_meter + 1,
			)
			ybins = np.linspace(-lidar_range[0], lidar_range[1], (lidar_range[0]+lidar_range[1]) * pixels_per_meter + 1)
			hist = np.histogramdd(point_cloud[..., :2], bins=(xbins, ybins))[0]
			hist[hist > hist_max_per_pixel] = hist_max_per_pixel
			overhead_splat = hist / hist_max_per_pixel
			return overhead_splat

		below = lidar[lidar[..., 2] <= -1.45]
		above = lidar[lidar[..., 2] > -1.45]
		below_features = splat_points(below)
		above_features = splat_points(above)
		total_features = below_features + above_features
		features = np.stack([below_features, above_features, total_features], axis=-1)
		features = np.transpose(features, (2, 0, 1)).astype(np.float32)
		return features

	def collate_batch_infer_perception(self, car_data: list, rsu_data: list) -> dict:
		'''
		Re-collate a batch
		'''

		output_dict = {
            "lidar_pose": [],
            "voxel_features": [],
            "voxel_num_points": [],
            "voxel_coords": [],

            "lidar_original": [],

            "detmap_pose": [],

            "record_len": [],
	    
			"target": [],
		}
		
		count = 0
		for j in range(len(car_data)):
			output_dict["record_len"].append(len(car_data)+len(rsu_data))
			output_dict["target"].append(car_data[j]['target_point'].unsqueeze(0).float())

			# Set j-th car as the ego-car.
			output_dict["lidar_original"].append(torch.from_numpy(car_data[j]['lidar_original']).unsqueeze(0))
			output_dict["voxel_features"].append(car_data[j]['lidar']['voxel_features'])
			output_dict["voxel_num_points"].append(car_data[j]['lidar']['voxel_num_points'])
			coords =car_data[j]['lidar']["voxel_coords"]
			output_dict["voxel_coords"].append(
				np.pad(coords, ((0, 0), (1, 0)),
					mode='constant', constant_values=count)) 
                    
			output_dict["lidar_pose"].append(torch.from_numpy(car_data[j]['lidar_pose']).unsqueeze(0).cuda().float())
			output_dict["detmap_pose"].append(torch.from_numpy(car_data[j]['detmap_pose']).unsqueeze(0).cuda().float())
			count += 1
			for i in range(len(car_data)):
				if i==j:
					continue
				output_dict["lidar_original"].append(torch.from_numpy(car_data[i]['lidar_original']).unsqueeze(0))
				output_dict["voxel_features"].append(car_data[i]['lidar']['voxel_features'])
				output_dict["voxel_num_points"].append(car_data[i]['lidar']['voxel_num_points'])
				coords =car_data[i]['lidar']["voxel_coords"]
				output_dict["voxel_coords"].append(
					np.pad(coords, ((0, 0), (1, 0)),
						mode='constant', constant_values=count)) 
						
				output_dict["lidar_pose"].append(torch.from_numpy(car_data[i]['lidar_pose']).unsqueeze(0).cuda().float())
				output_dict["detmap_pose"].append(torch.from_numpy(car_data[i]['detmap_pose']).unsqueeze(0).cuda().float())
				count += 1
			for i in range(len(rsu_data)):
				output_dict["lidar_original"].append(torch.from_numpy(rsu_data[i]['lidar_original']).unsqueeze(0))
				output_dict["voxel_features"].append(rsu_data[i]['lidar']['voxel_features'])
				output_dict["voxel_num_points"].append(rsu_data[i]['lidar']['voxel_num_points'])
				coords =rsu_data[i]['lidar']["voxel_coords"]
				output_dict["voxel_coords"].append(
					np.pad(coords, ((0, 0), (1, 0)),
						mode='constant', constant_values=count)) 
						
				output_dict["lidar_pose"].append(torch.from_numpy(rsu_data[i]['lidar_pose']).unsqueeze(0).cuda().float())
				output_dict["detmap_pose"].append(torch.from_numpy(rsu_data[i]['detmap_pose']).unsqueeze(0).cuda().float())
				count += 1
		for key in ["target", "lidar_pose", "detmap_pose", "lidar_original"]:
			output_dict[key] = torch.cat(output_dict[key], dim=0)
		
		for key in ["voxel_features", "voxel_coords", "voxel_num_points"]:
			output_dict[key] = torch.from_numpy(np.concatenate(output_dict[key], axis=0)).cuda().float()
	    
		output_dict["record_len"] = torch.from_numpy(np.array(output_dict["record_len"]))

		return output_dict
