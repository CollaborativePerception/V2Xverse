from typing import List, Dict, Any, Iterable
import os
import copy
import re
import io
import logging
import json
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

from .basedataset import BaseIODataset
from common.heatmap import generate_relative_heatmap, render_self_car
from common.detection import generate_det_data

from skimage.measure import block_reduce

from codriving import CODRIVING_REGISTRY


# CarlaMVDetDataset_planner
@CODRIVING_REGISTRY.register
class CarlaMVDatasetWithGTInput(BaseIODataset):
	"""Carla multi-vehicle dataset with ground-truth as input data

	TODO: change to a more informative name.
    """

	route_frames : List
	"""records file paths for the ego vehicle.

	.. code-block::

		structure:
			list[
				tuple(
					str, # verify the path
					int  # verify the start frame
				), ...
			]
		example:
			[
				(/dataset_example/weather-0/data/routes_town05_long_w0_02_15_02_37_28/ego_vehicle_0, 0),
				(/dataset_example/weather-0/data/routes_town05_long_w0_02_15_02_37_28/ego_vehicle_0, 5)
			]
	"""

	def __init__(
		self,
		root,
		towns,
		weathers,
		input_frame : int=5,
		output_points : int=10,
		skip_frames : int=1,
        det_range : Iterable[int]=(36, 12, 12, 12, 0.125),
	):
		"""
		Initialize Carla dataset. [KEY] contribution: self.route_frames.

		Args:
			towns: TODO (weibo)
			weathers: TODO (weibo)
			input_frame: TODO (weibo)
			output_points: TODO (weibo)
			det_range: TODO (weibo)
		"""
		super().__init__()
		
		self.input_frame = input_frame # number of input frames
		self.output_points = output_points # number of output points
		self.skip_frames = skip_frames  # 1 control the nummber of frames to skip
		self.det_range = det_range # [front, back, left, right, resolution]
		self.distance_to_map_center = (self.det_range[0]+self.det_range[1])/2-self.det_range[1]
		self.with_drivable_area = True


		##########
		## Load file paths and build the route structure.
		##########
		self.route_frames = []

		self.pixel_per_meter = int(det_range[-1]*8)
		self.max_distance = max(det_range)
		self.root = root

		dataset_indexs = self._load_text(os.path.join(root, 'dataset_index.txt')).split('\n')
		pattern = re.compile('weather-(\d+).*town(\d\d)')
		for line in dataset_indexs:
			if len(line.split()) != 3:
				continue
			path, frames, egos = line.split()
			route_path = os.path.join(root, path)
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
			for j, file in enumerate(ego_files):
				ego_path = os.path.join(route_path, file)
				for i in range(0, frames-input_frame*self.skip_frames-output_points*self.skip_frames, 1):
					self.route_frames.append((ego_path, i))
		logging.info("Sub route dir nums: %d" % len(self.route_frames))


	def __len__(self):
		return len(self.route_frames)


	def get_one_record(self,
			route_dir : str,
			frame_id : int,
		) -> Dict[str, Any]:
		"""Read data of one record

		Args:
			scene_dict: index given by dataloader
			frame_id: start frame id

		Return:
			batch data, including model input and training target
		"""
		output_record = {
			'occupancy_map': [],   # T=5, H=20, W=40
			'occupancy_ego': [],   # T=5, H=20, W=40
			'occupancy_local_command': [],
			'command_waypoints': 0, # T=10, 2,
			'target': [],
			'drivable_area': None,
		}
		
		if self.with_drivable_area:
			bev_image = self._load_image(os.path.join(route_dir, "birdview", "%04d.jpg" % (frame_id+(self.input_frame-1)*self.skip_frames)))
			img_c = bev_image.crop([140, 20, 260, 260])
			img_r = np.array(img_c.resize((96, 192))) # 96, 192, 3
			# print('Image r: ', img_r.shape)
			driavble_area = np.where(img_r.sum(axis=2)>200, 1, 0) # 192, 96, only 0/1.
			# print('Drivable area: ', driavble_area.shape)
			output_record['drivable_area'] = driavble_area


		##########
		## load environment data and generate temporal input sequence.
		##########        
		measurements_last = self._load_json(os.path.join(route_dir, "measurements", "%04d.json" % (frame_id+(self.input_frame-1)*self.skip_frames)))

		ego_theta = measurements_last["theta"]
		ego_x = measurements_last["gps_x"]
		ego_y = measurements_last["gps_y"]

		x_command = measurements_last["x_command"]
		y_command = measurements_last["y_command"]

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

		local_command_point = np.array([y_command - ego_y, -1*(x_command - ego_x)])
		local_command_point = R.T.dot(local_command_point)
		local_command_point = np.clip(local_command_point, a_min=[-12, -36], a_max=[12, 12])
		local_command_point[np.isnan(local_command_point)] = 0
		# clip the point out of detection range
		
		# NOTE: we should take the ground-truth future position as the supervision!
		# instead of future waypopints which might not be reached.
		command_waypoints = []
		
		for i in range(0, self.output_points*self.skip_frames, self.skip_frames):
			measurements_i = self._load_json(os.path.join(route_dir, "measurements", "%04d.json" % (frame_id+self.input_frame*self.skip_frames+i)))
			command_waypoints.append(R.T.dot(np.array([measurements_i["gps_y"]-ego_y, -1*(measurements_i["gps_x"]-ego_x)])))
		output_record["command_waypoints"] = torch.from_numpy(np.array(command_waypoints))
		
		detmap_pose = []
		for frame_cur in range(0, self.input_frame*self.skip_frames, self.skip_frames):
			measurements = self._load_json(os.path.join(route_dir, "measurements", "%04d.json" % (frame_id+frame_cur)))
			detmap_pose_x = measurements['lidar_pose_x'] + self.distance_to_map_center*np.cos(measurements["theta"]-np.pi/2)
			detmap_pose_y = measurements['lidar_pose_y'] + self.distance_to_map_center*np.sin(measurements["theta"]-np.pi/2)
			detmap_theta = measurements["theta"] + np.pi/2
			detmap_pose.append(np.array([-detmap_pose_y, detmap_pose_x, detmap_theta]))

			actors_data = self._load_json(os.path.join(route_dir, "actors_data", "%04d.json" % (frame_id+frame_cur)))

			heatmap = generate_relative_heatmap(
				copy.deepcopy(measurements), copy.deepcopy(actors_data), egp_pos_last,
				pixels_per_meter=8,
				max_distance=self.max_distance
			)
			# max_distance(32) * pixel_per_meter(8) * 2, max_distance*pixel_per_meter*2
			occ_map = block_reduce(heatmap, block_size=(self.pixel_per_meter, self.pixel_per_meter), func=np.mean)
			# occ_map: 75, 75
			occ_map = np.clip(occ_map, 0.0, 255.0)
			# TODO: change it into auto calculation
			occ_map = occ_map[:48*8, 48*4:48*8]
			# Verify the detection range.
			output_record["occupancy_map"].append(occ_map)

			# calculate the position of past frames according to current pos.
			self_car_map = render_self_car( 
				loc=np.array(R.T.dot(np.array(
						[measurements["gps_y"]-ego_y, -1*(measurements["gps_x"]-ego_x)]
						))),
				ori=np.array([np.sin(measurements["theta"]-ego_theta), -np.cos(measurements["theta"]-ego_theta)]),
				box=np.array([2.45, 1.0]),
				color=[1, 1, 0], 
				pixels_per_meter=8,
				max_distance=self.max_distance
			)[:, :, 0]
			
			self_car_map = block_reduce(self_car_map, block_size=(self.pixel_per_meter, self.pixel_per_meter), func=np.mean)
			self_car_map = np.clip(self_car_map, 0.0, 255.0)
			self_car_map = self_car_map[:48*8, 48*4:48*8]
			output_record["occupancy_ego"].append(self_car_map)	


			local_command_map = render_self_car( 
				loc=local_command_point,
				ori=np.array([0, 1]),
				box=np.array([1.0, 1.0]),
				color=[1, 1, 0], 
				pixels_per_meter=8,
				max_distance=self.max_distance
			)[:, :, 0]
			
			local_command_map = block_reduce(local_command_map, block_size=(self.pixel_per_meter, self.pixel_per_meter), func=np.mean)
			local_command_map = np.clip(local_command_map, 0.0, 255.0)
			local_command_map = local_command_map[:48*8, 48*4:48*8]
			output_record["occupancy_local_command"].append(local_command_map)	

		coordinate_map = np.ones((5, 2,)+local_command_map.shape)
		for h in range(local_command_map.shape[0]):
			coordinate_map[:, 0, h, :] *= h*self.det_range[-1]-self.det_range[0]
		for w in range(local_command_map.shape[1]):
			coordinate_map[:, 1, :, w] *= w*self.det_range[-1]-self.det_range[2]
		output_record["coordinate_map"] = coordinate_map
		output_record["target"] = local_command_point

		output_record['detmap_pose'] = detmap_pose

		return output_record


	# For validation only, verify the performance with different targets.
	def get_one_record_given_target(self, route_dir, frame_id, target):
		
		output_record = {
			'occupancy_map': [],   # T=5, H=20, W=40
			'occupancy_ego': [],   # T=5, H=20, W=40
			'occupancy_local_command': [],
			'command_waypoints': 0 # T=10, 2
		}
		
		##########
		## load environment data and generate temporal input sequence.
		##########        
		measurements_last = self._load_json(os.path.join(route_dir, "measurements", "%04d.json" % (frame_id+self.input_frame-1)))

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

		local_command_point = np.array(target)
		# clip the point out of detection range
		
		# NOTE: we should take the ground-truth future position as the supervision!
		# instead of future waypopints which might not be reached.
		command_waypoints = []
		for i in range(self.output_points):
			measurements_i = self._load_json(os.path.join(route_dir, "measurements", "%04d.json" % (frame_id+self.input_frame+i)))
			command_waypoints.append(R.T.dot(np.array([measurements_i["gps_y"]-ego_y, -1*(measurements_i["gps_x"]-ego_x)])))
		output_record["command_waypoints"] = torch.from_numpy(np.array(command_waypoints))
		

		for frame_cur in range(self.input_frame):
			measurements = self._load_json(os.path.join(route_dir, "measurements", "%04d.json" % (frame_id+frame_cur)))
			actors_data = self._load_json(os.path.join(route_dir, "actors_data", "%04d.json" % (frame_id+frame_cur)))

			heatmap = generate_relative_heatmap(
				copy.deepcopy(measurements), copy.deepcopy(actors_data), egp_pos_last,
				pixels_per_meter=8,
				max_distance=self.max_distance
			)
			# max_distance(32) * pixel_per_meter(8) * 2, max_distance*pixel_per_meter*2
			occ_map = block_reduce(heatmap, block_size=(self.pixel_per_meter, self.pixel_per_meter), func=np.mean)
			# occ_map: 75, 75
			occ_map = np.clip(occ_map, 0.0, 255.0)
			# TODO: change it into auto calculation
			occ_map = occ_map[:48*8, 48*4:48*8]
			# Verify the detection range.
			output_record["occupancy_map"].append(occ_map)

			# calculate the position of past frames according to current pos.
			self_car_map = render_self_car( 
				loc=np.array(R.T.dot(np.array(
						[measurements["gps_y"]-ego_y, -1*(measurements["gps_x"]-ego_x)]
						))),
				ori=np.array([np.sin(measurements["theta"]-ego_theta), -np.cos(measurements["theta"]-ego_theta)]),
				box=np.array([2.45, 1.0]),
				color=[1, 1, 0], 
				pixels_per_meter=8,
				max_distance=self.max_distance
			)[:, :, 0]
			
			self_car_map = block_reduce(self_car_map, block_size=(self.pixel_per_meter, self.pixel_per_meter), func=np.mean)
			self_car_map = np.clip(self_car_map, 0.0, 255.0)
			self_car_map = self_car_map[:48*8, 48*4:48*8]
			output_record["occupancy_ego"].append(self_car_map)	


			local_command_map = render_self_car( 
				loc=np.array(local_command_point),
				ori=np.array([0, 1]),
				box=np.array([1.0, 1.0]),
				color=[1, 1, 0], 
				pixels_per_meter=8,
				max_distance=self.max_distance
			)[:, :, 0]
			
			local_command_map = block_reduce(local_command_map, block_size=(self.pixel_per_meter, self.pixel_per_meter), func=np.mean)
			local_command_map = np.clip(local_command_map, 0.0, 255.0)
			local_command_map = local_command_map[:48*8, 48*4:48*8]
			output_record["occupancy_local_command"].append(local_command_map)	

		coordinate_map = np.ones((5, 2,)+local_command_map.shape)
		for h in range(local_command_map.shape[0]):
			coordinate_map[:, 0, h, :] *= h*self.det_range[-1]-self.det_range[0]
		for w in range(local_command_map.shape[1]):
			coordinate_map[:, 1, :, w] *= w*self.det_range[-1]-self.det_range[2]
		output_record["coordinate_map"] = coordinate_map

		return output_record



	def __getitem__(self, idx : int, data_dir=None) -> Dict:
		"""
		Given the index, return the corresponding data. 

		Args:
			idx: index given by dataloader.

		Return:
			output_dict: batched data in the following format:

				.. code-block::

					structure: list[
						dict{
						}, ... # see details for this dict in `get_one_record`
					] # len = 'num_cars'
		"""
		
		if idx is not None:
			scene, frame_id = self.route_frames[idx]
		elif data_dir is not None:
			scene, frame_id = os.path.join(self.root, data_dir[0]['ego']), data_dir[1]

		output_dict = self.get_one_record(scene, frame_id)

		# output_dict = []
		# for target in [[-5, -5], [-5, 0], [-5, 5], [0, -5], [0, 5], [5, -5], [5, 0], [5, 5]]:
		# 	output_dict.append(self.get_one_record_given_target(scene, frame_id, target=target))
	   
		return output_dict

	@classmethod
	def collate_fn(cls, batch : List, extra_source=None) -> Dict:
		"""Re-collate a batch.

		Args:
			batch: a batch of data len(batch)=batch_size

				- batch[i]: the ith data in the batch
				- batch[i][j]: the jth car in batch[i], batch[i][0] always center ego

				NOTE: BN = Σ_{i=0}^{B-1}(N_i)  # N_i is the num_car of the ith data in the batch

		Returns:
			batch data

				- model input: dict{'occupancy': torch.Tensor, size [B, 5, 2, 40, 20]},
				- model target: tuple(command_waypoints: torch.Tensor, size [B, 10])

		"""
		output_dict = {}
		occupancy = []
		future_waypoints = []
		target = []
		
		for i in range(len(batch)):
			occ_map = torch.from_numpy(np.array(batch[i]["occupancy_map"]))
			occ_ego = torch.from_numpy(np.array(batch[i]["occupancy_ego"]))
			occ_tar = torch.from_numpy(np.array(batch[i]["occupancy_local_command"]))
			occ_cor = torch.from_numpy(np.array(batch[i]["coordinate_map"]))
			occ_da  = torch.from_numpy(np.array(batch[i]["drivable_area"]))
			# print(occ_map.shape, occ_ego.shape, occ_tar.shape, occ_cor.shape)
			occ_img = torch.cat((occ_map.unsqueeze(1), occ_ego.unsqueeze(1), occ_tar.unsqueeze(1), occ_cor), dim=1)[:, :, ::2, ::2]

			if extra_source is not None:
				occ_img[:,0] = extra_source['occ'][0,:,0,:,:]
			# # T=5, C=1+1+1+2, H, W
			# print(occ_img.shape, occ_da.shape)
			# raise ValueError
			occ_final = torch.cat((occ_img, occ_da.unsqueeze(0).unsqueeze(0).repeat(5, 1, 1, 1)), dim=1)

			occupancy.append(occ_final) # T=5, C=1+1+1+2+1, H, W
			future_waypoints.append(batch[i]["command_waypoints"])
			target.append(torch.from_numpy(np.array(batch[i]["target"])))
		# print(len(target), target[0].shape)
		output_dict["occupancy"] = torch.stack(occupancy, dim=0).float()
		output_dict["target"] = torch.stack(target, dim=0).float()
		future_waypoints = torch.stack(future_waypoints, dim=0).float()
		output_dict["future_waypoints"] = future_waypoints

		return output_dict
