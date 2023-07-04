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

try:
    import pygame
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

import pdb

from team_code.tracker import Tracker
from team_code.interfuser_controller import InterfuserController

from team_code.render import render, render_self_car, render_waypoints

# from team_code.heatmap_utils import generate_heatmap, generate_future_waypoints
# from team_code.det_utils import generate_det_data

####### Input: raw_data, N(actor)+M(RSU)
####### Output: actors action, N(actor)
####### Generate the action with the trained model.

SAVE_PATH = os.environ.get("SAVE_PATH", 'eval')
os.environ["SDL_VIDEODRIVER"] = "dummy"


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class Resize2FixedSize:
	def __init__(self, size):
		self.size = size

	def __call__(self, pil_img):
		pil_img = pil_img.resize(self.size)
		return pil_img


def create_carla_rgb_transform(
		input_size, need_scale=True, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
	):

		if isinstance(input_size, (tuple, list)):
			img_size = input_size[-2:]
		else:
			img_size = input_size
		tfl = []

		if isinstance(input_size, (tuple, list)):
			input_size_num = input_size[-1]
		else:
			input_size_num = input_size

		if need_scale:
			if input_size_num == 112:
				tfl.append(Resize2FixedSize((170, 128)))
			elif input_size_num == 128:
				tfl.append(Resize2FixedSize((195, 146)))
			elif input_size_num == 224:
				tfl.append(Resize2FixedSize((341, 256)))
			elif input_size_num == 256:
				tfl.append(Resize2FixedSize((288, 288)))
			else:
				raise ValueError("Can't find proper crop size")
		tfl.append(transforms.CenterCrop(img_size))
		tfl.append(transforms.ToTensor())
		tfl.append(transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))

		return transforms.Compose(tfl)


class DisplayInterface(object):
    def __init__(self):
        self._width = 1200
        self._height = 600
        self._surface = None

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode(
            (self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("Human Agent")

    def run_interface(self, input_data):
        rgb = input_data['rgb']
        rgb_left = input_data['rgb_left']
        rgb_right = input_data['rgb_right']
        rgb_focus = input_data['rgb_focus']
        map = input_data['map']
        surface = np.zeros((600, 1200, 3),np.uint8)
        surface[:, :800] = rgb
        surface[:400,800:1200] = map
        surface[400:600,800:1000] = input_data['map_t1']
        surface[400:600,1000:1200] = input_data['map_t2']
        surface[:150,:200] = input_data['rgb_left']
        surface[:150, 600:800] = input_data['rgb_right']
        surface[:150, 325:475] = input_data['rgb_focus']
        surface = cv2.putText(surface, input_data['control'], (20,580), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
        surface = cv2.putText(surface, input_data['meta_infos'][0], (20,560), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
        surface = cv2.putText(surface, input_data['meta_infos'][1], (20,540), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
        surface = cv2.putText(surface, input_data['time'], (20,520), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)

        surface = cv2.putText(surface, 'Left  View', (40,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)
        surface = cv2.putText(surface, 'Focus View', (335,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)
        surface = cv2.putText(surface, 'Right View', (640,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)

        surface = cv2.putText(surface, 'Future Prediction', (940,420), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
        surface = cv2.putText(surface, 't', (1160,385), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0), 2)
        surface = cv2.putText(surface, '0', (1170,385), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
        surface = cv2.putText(surface, 't', (960,585), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0), 2)
        surface = cv2.putText(surface, '1', (970,585), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
        surface = cv2.putText(surface, 't', (1160,585), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0), 2)
        surface = cv2.putText(surface, '2', (1170,585), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)

        surface[:150,198:202]=0
        surface[:150,323:327]=0
        surface[:150,473:477]=0
        surface[:150,598:602]=0
        surface[148:152, :200] = 0
        surface[148:152, 325:475] = 0
        surface[148:152, 600:800] = 0
        surface[430:600, 998:1000] = 255
        surface[0:600, 798:800] = 255
        surface[0:600, 1198:1200] = 255
        surface[0:2, 800:1200] = 255
        surface[598:600, 800:1200] = 255
        surface[398:400, 800:1200] = 255


        # display image
        self._surface = pygame.surfarray.make_surface(surface.swapaxes(0, 1))
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))

        pygame.display.flip()
        pygame.event.get()
        return surface

    def _quit(self):
        pygame.quit()



class CoP3_infer():
	def __init__(self, config=None, ego_vehicles_num=1) -> None:
		self.config = config
		self._hic = DisplayInterface()


		self.controller = InterfuserController(self.config)

		self.input_lidar_size = 224
		self.lidar_range = [28,28,28,28]



		self.rgb_front_transform = create_carla_rgb_transform(224)
		self.rgb_left_transform = create_carla_rgb_transform(128)
		self.rgb_right_transform = create_carla_rgb_transform(128)
		self.rgb_center_transform = create_carla_rgb_transform(128, need_scale=False)

		self.softmax = torch.nn.Softmax(dim=0)
		self.traffic_meta_moving_avg = np.zeros((ego_vehicles_num, 400, 7))
		self.momentum = self.config.momentum
		self.prev_lidar = []
		self.prev_control = {}
		self.prev_surround_map = {}

		############
        ###### multi-agent related components
        ############
		self.ego_vehicles_num = ego_vehicles_num  
		self.trackers = [Tracker() for _ in range(self.ego_vehicles_num)]
		# list of tracker for each vehicle

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



	def get_action_from_route(self, route_path, model, step, timestamp):
		'''
        generate the action for N cars from the record data.

        Parameters
        ----------
        route_path : str, indicate the file path for the sensor/carla data.
		model : trained model, probably we can store it in the initialization.
		step : int, frame in the game, 20hz.
		timestamp : float, time in the game.
	
        Returns
        -------
        controll_all: list, detailed actions for N cars.
		'''
		
		### fetch the record data
		files = os.listdir(route_path)
		output_dict = []
		# record the inter-medium results
		tick_data = []

		####### NOTE: file order!
		for j, file in enumerate(files):
			ego_path = os.path.join(route_path, file)
			### data preprocess like dataloader
			if file[:3]=="rsu":
				output_dict.append(self.get_one_record(ego_path, frame_id=0, car=False))
			else:
				output_dict.append(self.get_one_record(ego_path, frame_id=0, car=True))
		
		batch_data = self.collate_batch_train(output_dict)  # batch_size: (N+M)*N
		
		### model inferece: (N+M)*N -> N, reduce the RSU data and generate the actions for
		### cars only.
		with torch.no_grad():
			(
				traffic_meta_total,
				pred_waypoints_total,
				is_junction_total,
				traffic_light_state_total,
				stop_sign_total,
				bev_feature_total,
			) = model(batch_data)
		### batch_size: N
		
		late_fusion = False
		if late_fusion:
			# update traffic_meta_total HERE
			pass

		### output postprocess to generate the action, list of actions for N agents
		control_all = []
		for ego_i in range(self.ego_vehicles_num):
			# get the data for current vehicle
			traffic_meta = traffic_meta_total[ego_i].detach().cpu().numpy()
			pred_waypoints = pred_waypoints_total[ego_i].detach().cpu().numpy()
			is_junction = self.softmax(is_junction_total[ego_i]).detach().cpu().numpy().reshape(-1)
			traffic_light_state = (
				self.softmax(traffic_light_state_total[ego_i]).detach().cpu().numpy().reshape(-1)
			)
			stop_sign = self.softmax(stop_sign_total[ego_i]).detach().cpu().numpy().reshape(-1)
			bev_feature = bev_feature_total[ego_i].detach().cpu().numpy()

			if step % 2 == 0 or step < 4:
				# NOTE: update the output_dict
				traffic_meta = self.trackers[ego_i].update_and_predict(traffic_meta.reshape(20, 20, -1), output_dict[ego_i]['gps'],  output_dict[ego_i]['compass'], step // 2)
				traffic_meta = traffic_meta.reshape(400, -1)
				self.traffic_meta_moving_avg[ego_i] = (
					self.momentum * self.traffic_meta_moving_avg[ego_i]
					+ (1 - self.momentum) * traffic_meta
				)

			traffic_meta = self.traffic_meta_moving_avg[ego_i]

			tick_data[ego_i]["raw"] = traffic_meta
			tick_data[ego_i]["bev_feature"] = bev_feature

			steer, throttle, brake, meta_infos = self.controller.run_step(
				tick_data[ego_i]["speed"],
				pred_waypoints,
				is_junction,
				traffic_light_state,
				stop_sign,
				self.traffic_meta_moving_avg,
			)

			if brake < 0.05:
				brake = 0.0
			if brake > 0.1:
				throttle = 0.0

			control = carla.VehicleControl()
			control.steer = float(steer)
			control.throttle = float(throttle)
			control.brake = float(brake)


			surround_map, box_info = render(traffic_meta.reshape(20, 20, 7), pixels_per_meter=20)
			surround_map = surround_map[:400, 160:560]
			surround_map = np.stack([surround_map, surround_map, surround_map], 2)

			self_car_map = render_self_car(
				loc=np.array([0, 0]),
				ori=np.array([0, -1]),
				box=np.array([2.45, 1.0]),
				color=[1, 1, 0], pixels_per_meter=20
			)[:400, 160:560]

			pred_waypoints = pred_waypoints.reshape(-1, 2)
			safe_index = 10
			for i in range(10):
				if pred_waypoints[i, 0] ** 2 + pred_waypoints[i, 1] ** 2> (meta_infos[3]+0.5) ** 2:
					safe_index = i
					break
			wp1 = render_waypoints(pred_waypoints[:safe_index], pixels_per_meter=20, color=(0, 255, 0))[:400, 160:560]
			wp2 = render_waypoints(pred_waypoints[safe_index:], pixels_per_meter=20, color=(255, 0, 0))[:400, 160:560]
			wp = wp1 + wp2

			surround_map = np.clip(
				(
					surround_map.astype(np.float32)
					+ self_car_map.astype(np.float32)
					+ wp.astype(np.float32)
				),
				0,
				255,
			).astype(np.uint8)

			map_t1, box_info = render(traffic_meta.reshape(20, 20, 7), pixels_per_meter=20, t=1)
			map_t1 = map_t1[:400, 160:560]
			map_t1 = np.stack([map_t1, map_t1, map_t1], 2)
			map_t1 = np.clip(map_t1.astype(np.float32) + self_car_map.astype(np.float32), 0, 255).astype(np.uint8)
			map_t1 = cv2.resize(map_t1, (200, 200))
			map_t2, box_info = render(traffic_meta.reshape(20, 20, 7), pixels_per_meter=20, t=2)
			map_t2 = map_t2[:400, 160:560]
			map_t2 = np.stack([map_t2, map_t2, map_t2], 2)
			map_t2 = np.clip(map_t2.astype(np.float32) + self_car_map.astype(np.float32), 0, 255).astype(np.uint8)
			map_t2 = cv2.resize(map_t2, (200, 200))

			if step % 2 != 0 and step > 4:
				control = self.prev_control[ego_i]
			else:
				self.prev_control[ego_i] = control
				self.prev_surround_map[ego_i] = surround_map


			control_all.append(control)

			tick_data[ego_i]["map"] = self.prev_surround_map
			tick_data[ego_i]["map_t1"] = map_t1
			tick_data[ego_i]["map_t2"] = map_t2
			tick_data[ego_i]["rgb_raw"] = tick_data["rgb"]
			tick_data[ego_i]["rgb_left_raw"] = tick_data["rgb_left"]
			tick_data[ego_i]["rgb_right_raw"] = tick_data["rgb_right"]

			tick_data[ego_i]["rgb"] = cv2.resize(tick_data["rgb"], (800, 600))
			tick_data[ego_i]["rgb_left"] = cv2.resize(tick_data["rgb_left"], (200, 150))
			tick_data[ego_i]["rgb_right"] = cv2.resize(tick_data["rgb_right"], (200, 150))
			tick_data[ego_i]["rgb_focus"] = cv2.resize(tick_data["rgb_raw"][244:356, 344:456], (150, 150))
			tick_data[ego_i]["control"] = "throttle: %.2f, steer: %.2f, brake: %.2f" % (
				control.throttle,
				control.steer,
				control.brake,
			)
			tick_data[ego_i]["meta_infos"] = meta_infos
			tick_data[ego_i]["box_info"] = "car: %d, bike: %d, pedestrian: %d" % (
				box_info["car"],
				box_info["bike"],
				box_info["pedestrian"],
			)
			tick_data[ego_i]["mes"] = "speed: %.2f" % tick_data[ego_i]["speed"]
			tick_data[ego_i]["time"] = "time: %.3f" % timestamp


			# NOTE: to-be check
			surface = self._hic.run_interface(tick_data)
			tick_data[ego_i]["surface"] = surface
		
		if SAVE_PATH is not None:
			self.save(tick_data)
			
		### delete/clean the route_path content
		os.removedirs(route_path)
		os.mkdir(route_path)
		return control_all
	

	def get_action_from_list(self, car_data_raw, rsu_data_raw, model, step, timestamp):
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
		# pdb.set_trace()
		car_data, car_mask = self.check_data(car_data_raw)
		rsu_data, rsu_mask = self.check_data(rsu_data_raw, car=False)
		tick_data = []
		batch_data = self.collate_batch_train_list(car_data, rsu_data)  # batch_size: (N+M)*N
		
		### model inferece: (N+M)*N -> N, reduce the RSU data and generate the actions for
		### cars only.
		with torch.no_grad():
			(
				traffic_meta_total,
				pred_waypoints_total,
				is_junction_total,
				traffic_light_state_total,
				stop_sign_total,
				bev_feature_total,
			) = model(batch_data)
		### batch_size: N
		
		late_fusion = False
		if late_fusion:
			# update traffic_meta_total HERE
			pass

		### output postprocess to generate the action, list of actions for N agents
		control_all = []
		ego_i = -1
		for count_i in range(self.ego_vehicles_num):
			if not car_mask[count_i]:
				control_all.append(None)
				tick_data.append(None)
				continue
			tick_data.append({})
			ego_i += 1
			# get the data for current vehicle
			traffic_meta = traffic_meta_total[ego_i].detach().cpu().numpy()
			pred_waypoints = pred_waypoints_total[ego_i].detach().cpu().numpy()
			# print(is_junction_total.shape)
			# pdb.set_trace()
			is_junction = self.softmax(is_junction_total[ego_i]).detach().cpu().numpy().reshape(-1)[0]
			traffic_light_state = (
				self.softmax(traffic_light_state_total[ego_i]).detach().cpu().numpy().reshape(-1)[0]
			)
			stop_sign = self.softmax(stop_sign_total[ego_i]).detach().cpu().numpy().reshape(-1)[0]
			bev_feature = bev_feature_total[ego_i].detach().cpu().numpy()

			if step % 2 == 0 or step < 4:
				# NOTE: update the output_dict
				traffic_meta = self.trackers[ego_i].update_and_predict(traffic_meta.reshape(20, 20, -1), 
							   [car_data_raw[ego_i]['measurements']['gps_x'], car_data_raw[ego_i]['measurements']['gps_y']],  
							   car_data_raw[ego_i]['measurements']['compass'], 
							   step // 2)
				traffic_meta = traffic_meta.reshape(400, -1)
				self.traffic_meta_moving_avg[ego_i] = (
					self.momentum * self.traffic_meta_moving_avg[ego_i]
					+ (1 - self.momentum) * traffic_meta
				)

			traffic_meta = self.traffic_meta_moving_avg[ego_i]

			tick_data[ego_i]["raw"] = traffic_meta
			tick_data[ego_i]["bev_feature"] = bev_feature

			steer, throttle, brake, meta_infos = self.controller.run_step(
				car_data_raw[ego_i]['measurements']["speed"],
				pred_waypoints,
				is_junction,
				traffic_light_state,
				stop_sign,
				self.traffic_meta_moving_avg[ego_i],
			)

			if brake < 0.05:
				brake = 0.0
			if brake > 0.1:
				throttle = 0.0

			control = carla.VehicleControl()
			control.steer = float(steer)
			control.throttle = float(throttle)
			control.brake = float(brake)


			surround_map, box_info = render(traffic_meta.reshape(20, 20, 7), pixels_per_meter=20)
			surround_map = surround_map[:400, 160:560]
			surround_map = np.stack([surround_map, surround_map, surround_map], 2)

			self_car_map = render_self_car(
				loc=np.array([0, 0]),
				ori=np.array([0, -1]),
				box=np.array([2.45, 1.0]),
				color=[1, 1, 0], pixels_per_meter=20
			)[:400, 160:560]

			pred_waypoints = pred_waypoints.reshape(-1, 2)
			safe_index = 10
			for i in range(10):
				if pred_waypoints[i, 0] ** 2 + pred_waypoints[i, 1] ** 2> (meta_infos[3]+0.5) ** 2:
					safe_index = i
					break
			wp1 = render_waypoints(pred_waypoints[:safe_index], pixels_per_meter=20, color=(0, 255, 0))[:400, 160:560]
			wp2 = render_waypoints(pred_waypoints[safe_index:], pixels_per_meter=20, color=(255, 0, 0))[:400, 160:560]
			wp = wp1 + wp2

			surround_map = np.clip(
				(
					surround_map.astype(np.float32)
					+ self_car_map.astype(np.float32)
					+ wp.astype(np.float32)
				),
				0,
				255,
			).astype(np.uint8)

			map_t1, box_info = render(traffic_meta.reshape(20, 20, 7), pixels_per_meter=20, t=1)
			map_t1 = map_t1[:400, 160:560]
			map_t1 = np.stack([map_t1, map_t1, map_t1], 2)
			map_t1 = np.clip(map_t1.astype(np.float32) + self_car_map.astype(np.float32), 0, 255).astype(np.uint8)
			map_t1 = cv2.resize(map_t1, (200, 200))
			map_t2, box_info = render(traffic_meta.reshape(20, 20, 7), pixels_per_meter=20, t=2)
			map_t2 = map_t2[:400, 160:560]
			map_t2 = np.stack([map_t2, map_t2, map_t2], 2)
			map_t2 = np.clip(map_t2.astype(np.float32) + self_car_map.astype(np.float32), 0, 255).astype(np.uint8)
			map_t2 = cv2.resize(map_t2, (200, 200))

			if step % 2 != 0 and step > 4:
				control = self.prev_control[ego_i]
			else:
				self.prev_control[ego_i] = control
				self.prev_surround_map[ego_i] = surround_map


			control_all.append(control)

			tick_data[ego_i]["map"] = self.prev_surround_map[ego_i]
			tick_data[ego_i]["map_t1"] = map_t1
			tick_data[ego_i]["map_t2"] = map_t2
			tick_data[ego_i]["rgb_raw"] = car_data_raw[ego_i]["rgb_front"]
			tick_data[ego_i]["rgb_left_raw"] = car_data_raw[ego_i]["rgb_left"]
			tick_data[ego_i]["rgb_right_raw"] = car_data_raw[ego_i]["rgb_right"]

			tick_data[ego_i]["rgb"] = cv2.resize(tick_data[ego_i]["rgb_raw"], (800, 600))
			tick_data[ego_i]["rgb_left"] = cv2.resize(tick_data[ego_i]["rgb_left_raw"], (200, 150))
			tick_data[ego_i]["rgb_right"] = cv2.resize(tick_data[ego_i]["rgb_right_raw"], (200, 150))
			tick_data[ego_i]["rgb_focus"] = cv2.resize(tick_data[ego_i]["rgb_raw"][244:356, 344:456], (150, 150))
			tick_data[ego_i]["control"] = "throttle: %.2f, steer: %.2f, brake: %.2f" % (
				control.throttle,
				control.steer,
				control.brake,
			)
			tick_data[ego_i]["meta_infos"] = meta_infos
			tick_data[ego_i]["box_info"] = "car: %d, bike: %d, pedestrian: %d" % (
				box_info["car"],
				box_info["bike"],
				box_info["pedestrian"],
			)
			tick_data[ego_i]["mes"] = "speed: %.2f" % car_data_raw[ego_i]['measurements']["speed"]
			tick_data[ego_i]["time"] = "time: %.3f" % timestamp


			# NOTE: to-be check
			surface = self._hic.run_interface(tick_data[ego_i])
			tick_data[ego_i]["surface"] = surface
		
		if SAVE_PATH is not None:
			self.save(tick_data, step)
		
		return control_all
	

	def get_action_from_list(self, car_data_raw, rsu_data_raw, model, step, timestamp):
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
		# pdb.set_trace()
		car_data, car_mask = self.check_data(car_data_raw)
		rsu_data, rsu_mask = self.check_data(rsu_data_raw, car=False)
		tick_data = []
		batch_data = self.collate_batch_train_list(car_data, rsu_data)  # batch_size: (N+M)*N
		
		### model inferece: (N+M)*N -> N, reduce the RSU data and generate the actions for
		### cars only.
		with torch.no_grad():
			(
				traffic_meta_total,
				pred_waypoints_total,
				is_junction_total,
				traffic_light_state_total,
				stop_sign_total,
				bev_feature_total,
			) = model(batch_data)
		### batch_size: N
		
		late_fusion = False
		if late_fusion:
			# update traffic_meta_total HERE
			pass

		### output postprocess to generate the action, list of actions for N agents
		control_all = []
		ego_i = -1
		for count_i in range(self.ego_vehicles_num):
			if not car_mask[count_i]:
				control_all.append(None)
				tick_data.append(None)
				continue
			tick_data.append({})
			ego_i += 1
			# get the data for current vehicle
			traffic_meta = traffic_meta_total[ego_i].detach().cpu().numpy()
			pred_waypoints = pred_waypoints_total[ego_i].detach().cpu().numpy()
			# print(is_junction_total.shape)
			# pdb.set_trace()
			is_junction = self.softmax(is_junction_total[ego_i]).detach().cpu().numpy().reshape(-1)[0]
			traffic_light_state = (
				self.softmax(traffic_light_state_total[ego_i]).detach().cpu().numpy().reshape(-1)[0]
			)
			stop_sign = self.softmax(stop_sign_total[ego_i]).detach().cpu().numpy().reshape(-1)[0]
			bev_feature = bev_feature_total[ego_i].detach().cpu().numpy()

			if step % 2 == 0 or step < 4:
				# NOTE: update the output_dict
				traffic_meta = self.trackers[ego_i].update_and_predict(traffic_meta.reshape(20, 20, -1), 
							   [car_data_raw[ego_i]['measurements']['gps_x'], car_data_raw[ego_i]['measurements']['gps_y']],  
							   car_data_raw[ego_i]['measurements']['compass'], 
							   step // 2)
				traffic_meta = traffic_meta.reshape(400, -1)
				self.traffic_meta_moving_avg[ego_i] = (
					self.momentum * self.traffic_meta_moving_avg[ego_i]
					+ (1 - self.momentum) * traffic_meta
				)

			traffic_meta = self.traffic_meta_moving_avg[ego_i]

			tick_data[ego_i]["raw"] = traffic_meta
			tick_data[ego_i]["bev_feature"] = bev_feature

			steer, throttle, brake, meta_infos = self.controller.run_step(
				car_data_raw[ego_i]['measurements']["speed"],
				pred_waypoints,
				is_junction,
				traffic_light_state,
				stop_sign,
				self.traffic_meta_moving_avg[ego_i],
			)

			if brake < 0.05:
				brake = 0.0
			if brake > 0.1:
				throttle = 0.0

			control = carla.VehicleControl()
			control.steer = float(steer)
			control.throttle = float(throttle)
			control.brake = float(brake)


			surround_map, box_info = render(traffic_meta.reshape(20, 20, 7), pixels_per_meter=20)
			surround_map = surround_map[:400, 160:560]
			surround_map = np.stack([surround_map, surround_map, surround_map], 2)

			self_car_map = render_self_car(
				loc=np.array([0, 0]),
				ori=np.array([0, -1]),
				box=np.array([2.45, 1.0]),
				color=[1, 1, 0], pixels_per_meter=20
			)[:400, 160:560]

			pred_waypoints = pred_waypoints.reshape(-1, 2)
			safe_index = 10
			for i in range(10):
				if pred_waypoints[i, 0] ** 2 + pred_waypoints[i, 1] ** 2> (meta_infos[3]+0.5) ** 2:
					safe_index = i
					break
			wp1 = render_waypoints(pred_waypoints[:safe_index], pixels_per_meter=20, color=(0, 255, 0))[:400, 160:560]
			wp2 = render_waypoints(pred_waypoints[safe_index:], pixels_per_meter=20, color=(255, 0, 0))[:400, 160:560]
			wp = wp1 + wp2

			surround_map = np.clip(
				(
					surround_map.astype(np.float32)
					+ self_car_map.astype(np.float32)
					+ wp.astype(np.float32)
				),
				0,
				255,
			).astype(np.uint8)

			map_t1, box_info = render(traffic_meta.reshape(20, 20, 7), pixels_per_meter=20, t=1)
			map_t1 = map_t1[:400, 160:560]
			map_t1 = np.stack([map_t1, map_t1, map_t1], 2)
			map_t1 = np.clip(map_t1.astype(np.float32) + self_car_map.astype(np.float32), 0, 255).astype(np.uint8)
			map_t1 = cv2.resize(map_t1, (200, 200))
			map_t2, box_info = render(traffic_meta.reshape(20, 20, 7), pixels_per_meter=20, t=2)
			map_t2 = map_t2[:400, 160:560]
			map_t2 = np.stack([map_t2, map_t2, map_t2], 2)
			map_t2 = np.clip(map_t2.astype(np.float32) + self_car_map.astype(np.float32), 0, 255).astype(np.uint8)
			map_t2 = cv2.resize(map_t2, (200, 200))

			if step % 2 != 0 and step > 4:
				control = self.prev_control[ego_i]
			else:
				self.prev_control[ego_i] = control
				self.prev_surround_map[ego_i] = surround_map


			control_all.append(control)

			tick_data[ego_i]["map"] = self.prev_surround_map[ego_i]
			tick_data[ego_i]["map_t1"] = map_t1
			tick_data[ego_i]["map_t2"] = map_t2
			tick_data[ego_i]["rgb_raw"] = car_data_raw[ego_i]["rgb_front"]
			tick_data[ego_i]["rgb_left_raw"] = car_data_raw[ego_i]["rgb_left"]
			tick_data[ego_i]["rgb_right_raw"] = car_data_raw[ego_i]["rgb_right"]

			tick_data[ego_i]["rgb"] = cv2.resize(tick_data[ego_i]["rgb_raw"], (800, 600))
			tick_data[ego_i]["rgb_left"] = cv2.resize(tick_data[ego_i]["rgb_left_raw"], (200, 150))
			tick_data[ego_i]["rgb_right"] = cv2.resize(tick_data[ego_i]["rgb_right_raw"], (200, 150))
			tick_data[ego_i]["rgb_focus"] = cv2.resize(tick_data[ego_i]["rgb_raw"][244:356, 344:456], (150, 150))
			tick_data[ego_i]["control"] = "throttle: %.2f, steer: %.2f, brake: %.2f" % (
				control.throttle,
				control.steer,
				control.brake,
			)
			tick_data[ego_i]["meta_infos"] = meta_infos
			tick_data[ego_i]["box_info"] = "car: %d, bike: %d, pedestrian: %d" % (
				box_info["car"],
				box_info["bike"],
				box_info["pedestrian"],
			)
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
			save_path_tmp = self.save_path / pathlib.Path("ego_vehicle_{}".format(ego_i))
			folder_path = save_path_tmp / pathlib.Path("meta")
			if not os.path.exists(save_path_tmp):
				os.mkdir(save_path_tmp)
			if not os.path.exists(folder_path):
				os.mkdir(folder_path)
			Image.fromarray(tick_data[ego_i]["surface"]).save(
				save_path_tmp / "meta" / ("%04d.jpg" % frame)
			)
		return




	def check_data(self, raw_data, car=True):
		mask = []
		data = [] # without None
		for i in raw_data:
			if i is not None:
				mask.append(1) # filter the data!
				data.append(self.preprocess_data(copy.deepcopy(i), car=car))
			else:
				mask.append(0)
		return data, mask
	

	def preprocess_data(self, data, car=True):
		output_record = {
		}
		
		##########
		## load and pre-process images
		##########
		rgb_main_image = self.rgb_front_transform(Image.fromarray(data['rgb_front'])).cuda().float()
		output_record['rgb'] = rgb_main_image

		rgb_center_image = self.rgb_center_transform(Image.fromarray(data['rgb_front'])).cuda().float()
		output_record['rgb_center'] = rgb_center_image
		
		output_record['rgb_left'] = self.rgb_left_transform(Image.fromarray(data['rgb_left'])).cuda().float()
		output_record['rgb_right'] = self.rgb_right_transform(Image.fromarray(data['rgb_right'])).cuda().float()

		
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
		detmap_pose_x = measurements['x'] + 8*np.cos(measurements["theta"]-np.pi/2)
		detmap_pose_y = measurements['y'] + 8*np.sin(measurements["theta"]-np.pi/2)
		detmap_theta = measurements["theta"] + np.pi/2
		output_record['detmap_pose'] = np.array([-detmap_pose_y, detmap_pose_x, detmap_theta])
		output_record["target_point"] = torch.from_numpy(measurements['target_point']).cuda().float()
		

		##########
		## load and pre-process LiDAR from 3D point cloud to 2D map
		##########
		lidar_unprocessed = data['lidar']
		lidar_unprocessed[:, 1] *= -1
		
		lidar_processed = self.lidar_to_histogram_features(
			lidar_unprocessed, crop=self.input_lidar_size, lidar_range=self.lidar_range
		)        
		# if self.lidar_transform is not None:
		# 	lidar_processed = self.lidar_transform(lidar_processed)
		output_record["lidar"] = lidar_processed
		return output_record



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

		below = lidar[lidar[..., 2] <= -2.0]
		above = lidar[lidar[..., 2] > -2.0]
		below_features = splat_points(below)
		above_features = splat_points(above)
		total_features = below_features + above_features
		features = np.stack([below_features, above_features, total_features], axis=-1)
		features = np.transpose(features, (2, 0, 1)).astype(np.float32)
		return features


	def get_one_record(self, route_dir, frame_id=0, car=True):
		'''
		Parameters
		----------
		scene_dict: str, index given by dataloader.
		frame_id: int, frame id.
		ego: bool, indicate whether it is ego car.

		Returns
		-------
		data:  
			structure: dict{
				####################
				# input to the model
				####################
				'ego': bool, # whether it is the ego car, True for ego
				'rgb_[direction]': torch.Tenser, # direction in [left, right, center], shape (3, 128, 128)
				'rgb': torch.Tensor, front rgb image , # shape (3, 224, 224) 
				'measurements': torch.Tensor, size [7]: the first 6 dims is the onehot vector of command, and the last dim is car speed
				'command': int, 0-5, discrete command signal 0:left, 1:right, 2:straight, 
													# 3: lane follow, 4:lane change left, 5: lane change right
				'pose': np.array, shape(3,), lidar pose[gps_x, gps_y, theta]
				'target_point': torch.Tensor, size[2], (x,y) coordinate in the left hand coordinate system,
																	where X-axis towards right side of the car
				'lidar': np.ndarray, # shape (3, 224, 224), 2D projection of lidar, range x:[-28m, 28m], y:[-28m,28m]
										in the right hand coordinate system with X-axis towards left of car
				
		},
		'''
		output_record = {
			'car': car,
		}
		
		##########
		## load and pre-process images
		##########
		output_record['rgb_front'] = self._load_image(os.path.join(route_dir, "rgb_front", "%04d.jpg" % frame_id))
		output_record['rgb_left'] = self._load_image(os.path.join(route_dir, "rgb_left", "%04d.jpg" % frame_id))
		output_record['rgb_right'] = self._load_image(os.path.join(route_dir, "rgb_right", "%04d.jpg" % frame_id))
		
		rgb_main_image = self.rgb_front_transform(output_record['rgb_front'])
		output_record["rgb"] = rgb_main_image

		rgb_center_image = self.rgb_center_transform(output_record['rgb_front'])
		output_record["rgb_center"] = rgb_center_image
		
		output_record['rgb_left'] = self.rgb_left_transform(output_record['rgb_left'])
		output_record["rgb_right"] = self.rgb_right_transform(output_record["rgb_right"])

		
		##########
		## load environment data and control signal
		##########        
		measurements = self._load_json(os.path.join(route_dir, "measurements", "%04d.json" % frame_id))
		cmd_one_hot = [0, 0, 0, 0, 0, 0]
		cmd = measurements["command"] - 1
		if cmd < 0:
			cmd = 3
		cmd_one_hot[cmd] = 1
		cmd_one_hot.append(measurements["speed"])
		mes = np.array(cmd_one_hot)
		mes = torch.from_numpy(mes).float()

		output_record["measurements"] = mes
		output_record['command'] = cmd

		if np.isnan(measurements["theta"]):
			measurements["theta"] = 0
		ego_theta = measurements["theta"]
		x_command = measurements["x_command"]
		y_command = measurements["y_command"]
		if "gps_x" in measurements:
			ego_x = measurements["gps_x"]
		else:
			ego_x = measurements["x"]
		if "gps_y" in measurements:
			ego_y = measurements["gps_y"]
		else:
			ego_y = measurements["y"]

		lidar_pose_x = measurements["lidar_pose_x"]
		lidar_pose_y = measurements["lidar_pose_y"]
		lidar_theta = measurements["theta"] + np.pi
		
		output_record['lidar_pose'] = np.array([-lidar_pose_y, lidar_pose_x, lidar_theta])

		## 计算density map中心点的世界坐标，目前density map预测范围为左右10m前18m后2m
		detmap_pose_x = measurements['x'] + 8*np.cos(measurements["theta"]-np.pi/2)
		detmap_pose_y = measurements['y'] + 8*np.sin(measurements["theta"]-np.pi/2)
		detmap_theta = measurements["theta"] + np.pi/2
		output_record['detmap_pose'] = np.array([-detmap_pose_y, detmap_pose_x, detmap_theta])
		
		R = np.array(
			[
				[np.cos(np.pi / 2 + ego_theta), -np.sin(np.pi / 2 + ego_theta)],
				[np.sin(np.pi / 2 + ego_theta), np.cos(np.pi / 2 + ego_theta)],
			]
		)
		local_command_point = np.array([x_command - ego_x, y_command - ego_y])
		local_command_point = R.T.dot(local_command_point)
		if any(np.isnan(local_command_point)):
			local_command_point[np.isnan(local_command_point)] = np.mean(
				local_command_point
			)
		local_command_point = torch.from_numpy(local_command_point).float()
		output_record["target_point"] = local_command_point
		

		##########
		## load and pre-process LiDAR from 3D point cloud to 2D map
		##########
		lidar_unprocessed = self._load_npy(
			os.path.join(route_dir, "lidar", "%04d.npy" % frame_id)
		)[..., :3]
		lidar_unprocessed[:, 1] *= -1
		
		lidar_processed = self.lidar_to_histogram_features(
			lidar_unprocessed, crop=self.input_lidar_size, lidar_range=self.lidar_range
		)        
		if self.lidar_transform is not None:
			lidar_processed = self.lidar_transform(lidar_processed)
		output_record["lidar"] = lidar_processed
		return output_record


	def collate_batch_train(self, origin_data):
		'''
		Re-collate a batch

		Parameters
		----------
		output_dict

		* Note
		BN = Σ_{i=0}^{B-1}(N_i)  # N_i is the num_car of the ith data in the batch

		Returns
		-------
		output_dict:  # input to the model
			dict{
				'rgb': torch.Tensor, size [N, 3, 224, 224]
				'rgb_[direction]': torch.Tensor, size [N, 3, 128, 128]
				'measurements': torch.Tensor, size [N, 7]
				'target_point': torch.Tensor, size [N, 2]
				'lidar': torch.Tensor, size [N, 3, 224, 224] # 2D projection of 3D point cloud
				'record_len': torch.Tensor, size [], record_len[i] is the num_car in batch[i]
				'lidar_pose': torch.Tensor, size [N, 3]
			}
			 
			 ... # see more details for this dict in `get_one_record`
		'''

		output_dict = {}
		output_dict["rgb"]=[]
		output_dict["rgb_left"]=[]
		output_dict["rgb_right"]=[]
		output_dict["rgb_center"]=[]
		output_dict["measurements"]=[]
		output_dict["target_point"]=[]
		output_dict["lidar"]=[]
		output_dict["record_len"]=[]
		output_dict["lidar_pose"]=[]
		output_dict["detmap_pose"]=[]

		
		for _ in range(self.ego_vehicles_num):
			output_dict["record_len"].append(self.ego_vehicles_num)
			for i in range(len(origin_data)):
				output_dict["record_len"].append(len(origin_data))
				output_dict["rgb"].append(origin_data[i]['rgb'].unsqueeze(0))
				output_dict["rgb_left"].append(origin_data[i]['rgb_left'].unsqueeze(0))
				output_dict["rgb_right"].append(origin_data[i]['rgb_right'].unsqueeze(0))
				output_dict["rgb_center"].append(origin_data[i]['rgb_center'].unsqueeze(0))
				output_dict["measurements"].append(origin_data[i]['measurements'].unsqueeze(0))
				output_dict["target_point"].append(origin_data[i]['target_point'].unsqueeze(0))
				output_dict["lidar"].append(torch.from_numpy(origin_data[i]['lidar']).unsqueeze(0))
				output_dict["lidar_pose"].append(torch.from_numpy(origin_data[i]['lidar_pose']).unsqueeze(0))
				output_dict["detmap_pose"].append(torch.from_numpy(origin_data[i]['detmap_pose']).unsqueeze(0))
			
		output_dict["record_len"] = torch.from_numpy(np.array(output_dict["record_len"]))
		output_dict["rgb"] = torch.cat(output_dict["rgb"], dim=0)
		output_dict["rgb_left"] = torch.cat(output_dict["rgb_left"], dim=0)
		output_dict["rgb_right"] = torch.cat(output_dict["rgb_right"], dim=0)
		output_dict["rgb_center"] = torch.cat(output_dict["rgb_center"], dim=0)
		output_dict["measurements"] = torch.cat(output_dict["measurements"], dim=0)
		output_dict["target_point"] = torch.cat(output_dict["target_point"], dim=0)
		output_dict["lidar"] = torch.cat(output_dict["lidar"], dim=0)
		output_dict["lidar_pose"] = torch.cat(output_dict["lidar_pose"], dim=0)
		output_dict["detmap_pose"] = torch.cat(output_dict["detmap_pose"], dim=0)

		return output_dict


	def collate_batch_train_list(self, car_data, rsu_data):
		'''
		Re-collate a batch

		Parameters
		----------
		output_dict

		* Note
		BN = Σ_{i=0}^{B-1}(N_i)  # N_i is the num_car of the ith data in the batch

		Returns
		-------
		output_dict:  # input to the model
			dict{
				'rgb': torch.Tensor, size [N, 3, 224, 224]
				'rgb_[direction]': torch.Tensor, size [N, 3, 128, 128]
				'measurements': torch.Tensor, size [N, 7]
				'target_point': torch.Tensor, size [N, 2]
				'lidar': torch.Tensor, size [N, 3, 224, 224] # 2D projection of 3D point cloud
				'record_len': torch.Tensor, size [], record_len[i] is the num_car in batch[i]
				'lidar_pose': torch.Tensor, size [N, 3]
			}
			 
			 ... # see more details for this dict in `get_one_record`
		'''

		output_dict = {}
		output_dict["rgb"]=[]
		output_dict["rgb_left"]=[]
		output_dict["rgb_right"]=[]
		output_dict["rgb_center"]=[]
		output_dict["measurements"]=[]
		output_dict["target_point"]=[]
		output_dict["lidar"]=[]
		output_dict["record_len"]=[]
		output_dict["lidar_pose"]=[]
		output_dict["detmap_pose"]=[]

		
		for j in range(len(car_data)):
			output_dict["record_len"].append(len(car_data)+len(rsu_data))
			output_dict["rgb"].append(car_data[j]['rgb'].unsqueeze(0))
			output_dict["rgb_left"].append(car_data[j]['rgb_left'].unsqueeze(0))
			output_dict["rgb_right"].append(car_data[j]['rgb_right'].unsqueeze(0))
			output_dict["rgb_center"].append(car_data[j]['rgb_center'].unsqueeze(0))
			output_dict["measurements"].append(car_data[j]['measurements'].unsqueeze(0))
			output_dict["target_point"].append(car_data[j]['target_point'].unsqueeze(0))
			output_dict["lidar"].append(torch.from_numpy(car_data[j]['lidar']).unsqueeze(0).cuda().float())
			output_dict["lidar_pose"].append(torch.from_numpy(car_data[j]['lidar_pose']).unsqueeze(0).cuda().float())
			output_dict["detmap_pose"].append(torch.from_numpy(car_data[j]['detmap_pose']).unsqueeze(0).cuda().float())
			for i in range(len(car_data)):
				if i==j:
					continue
				output_dict["rgb"].append(car_data[i]['rgb'].unsqueeze(0))
				output_dict["rgb_left"].append(car_data[i]['rgb_left'].unsqueeze(0))
				output_dict["rgb_right"].append(car_data[i]['rgb_right'].unsqueeze(0))
				output_dict["rgb_center"].append(car_data[i]['rgb_center'].unsqueeze(0))
				output_dict["measurements"].append(car_data[i]['measurements'].unsqueeze(0))
				output_dict["target_point"].append(car_data[i]['target_point'].unsqueeze(0))
				output_dict["lidar"].append(torch.from_numpy(car_data[i]['lidar']).unsqueeze(0).cuda().float())
				output_dict["lidar_pose"].append(torch.from_numpy(car_data[i]['lidar_pose']).unsqueeze(0).cuda().float())
				output_dict["detmap_pose"].append(torch.from_numpy(car_data[i]['detmap_pose']).unsqueeze(0).cuda().float())
			for i in range(len(rsu_data)):
				output_dict["rgb"].append(rsu_data[i]['rgb'].unsqueeze(0))
				output_dict["rgb_left"].append(rsu_data[i]['rgb_left'].unsqueeze(0))
				output_dict["rgb_right"].append(rsu_data[i]['rgb_right'].unsqueeze(0))
				output_dict["rgb_center"].append(rsu_data[i]['rgb_center'].unsqueeze(0))
				output_dict["measurements"].append(rsu_data[i]['measurements'].unsqueeze(0))
				output_dict["target_point"].append(rsu_data[i]['target_point'].unsqueeze(0))
				output_dict["lidar"].append(torch.from_numpy(rsu_data[i]['lidar']).unsqueeze(0).cuda().float())
				output_dict["lidar_pose"].append(torch.from_numpy(rsu_data[i]['lidar_pose']).unsqueeze(0).cuda().float())
				output_dict["detmap_pose"].append(torch.from_numpy(rsu_data[i]['detmap_pose']).unsqueeze(0).cuda().float())
			
		output_dict["record_len"] = torch.from_numpy(np.array(output_dict["record_len"]))
		output_dict["rgb"] = torch.cat(output_dict["rgb"], dim=0)
		output_dict["rgb_left"] = torch.cat(output_dict["rgb_left"], dim=0)
		output_dict["rgb_right"] = torch.cat(output_dict["rgb_right"], dim=0)
		output_dict["rgb_center"] = torch.cat(output_dict["rgb_center"], dim=0)
		output_dict["measurements"] = torch.cat(output_dict["measurements"], dim=0)
		output_dict["target_point"] = torch.cat(output_dict["target_point"], dim=0)
		output_dict["lidar"] = torch.cat(output_dict["lidar"], dim=0)
		output_dict["lidar_pose"] = torch.cat(output_dict["lidar_pose"], dim=0)
		output_dict["detmap_pose"] = torch.cat(output_dict["detmap_pose"], dim=0)

		return output_dict
