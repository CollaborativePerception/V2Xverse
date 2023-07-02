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

try:
    import pygame
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

import pdb

from team_code.tracker import Tracker
from team_code.interfuser_controller import InterfuserController

from team_code.render import render, render_self_car, render_waypoints


from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

####### Input: raw_data, N(actor)+M(RSU)
####### Output: actors action, N(actor)
####### Generate the action with the trained model.

SAVE_PATH = os.environ.get("SAVE_PATH", 'eval')
os.environ["SDL_VIDEODRIVER"] = "dummy"
VALUES = [255]
EXTENT = [0]

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


def convert_grid_to_xy(i, j):
    x = j - 9.5
    y = 17.5 - i
    return x, y


def generate_det_data(
    heatmap, measurements, actors_data, pixels_per_meter=5, max_distance=18
):
    traffic_heatmap = block_reduce(heatmap, block_size=(5, 5), func=np.mean)
    traffic_heatmap = np.clip(traffic_heatmap, 0.0, 255.0)
    traffic_heatmap = traffic_heatmap[:20, 8:28]
    det_data = np.zeros((20, 20, 7))

    ego_x = measurements["x"]
    ego_y = measurements["y"]
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
            dis <= 1
            or dis >= (max_distance + 3) ** 2 * 2
            or "box" not in actors_data[_id]
        ):
            need_deleted_ids.append(_id)
            continue
        actors_data[_id]["ori"] = np.array(new_ori)
        actors_data[_id]["box"] = np.array(actors_data[_id]["box"])

    for _id in need_deleted_ids:
        del actors_data[_id]

    for i in range(20):  # Vertical
        for j in range(20):  # horizontal
            if traffic_heatmap[i][j] < 0.05 * 255.0:
                continue
            center_x, center_y = convert_grid_to_xy(i, j)
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
            if not min_id:
                continue
            loc = actors_data[min_id]["loc"][:2]
            ori = actors_data[min_id]["ori"][:2]
            box = actors_data[min_id]["box"]
            theta = (get_yaw_angle(ori) / np.pi + 2) % 2
            speed = np.linalg.norm(actors_data[min_id]["vel"])
            prob = np.power(0.5 / max(0.5, np.sqrt(min_dis)), 0.5)
            det_data[i][j] = np.array(
                [
                    prob,
                    (loc[0] - center_x) / 3.5,
                    (loc[1] - center_y) / 3.5,
                    theta / 2.0,
                    box[0] / 3.5,
                    box[1] / 2.0,
                    speed / 8.0,
                ]
            )
    return det_data

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


def get_yaw_angle(forward_vector):
    forward_vector = forward_vector / np.linalg.norm(forward_vector)
    yaw = math.acos(forward_vector[0])
    if forward_vector[1] < 0:
        yaw = 2 * np.pi - yaw
    return yaw

def generate_heatmap(measurements, actors_data, pixels_per_meter=5, max_distance=18):
    img_size = max_distance * pixels_per_meter * 2
    img = np.zeros((img_size, img_size, 3), np.int)
    ego_x = measurements["x"]
    ego_y = measurements["y"]
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
        raw_loc = actors_data[_id]["loc"]
        if (raw_loc[0] - ego_x) ** 2 + (raw_loc[1] - ego_y) ** 2 <= 1:
            ego_id = _id
            color = np.array([0, 1, 1])
        new_loc = R.T.dot(np.array([raw_loc[0] - ego_x, raw_loc[1] - ego_y]))
        actors_data[_id]["loc"] = np.array(new_loc)
        raw_ori = actors_data[_id]["ori"]
        new_ori = R.T.dot(np.array([raw_ori[0], raw_ori[1]]))
        actors_data[_id]["ori"] = np.array(new_ori)
        actors_data[_id]["box"] = np.array(actors_data[_id]["box"])
        actors_data[_id]["color"] = color

    if ego_id is not None and ego_id in actors_data:
        del actors_data[ego_id]  # Do not show ego car
    for _id in actors_data:
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
		batch_data = self.collate_batch_train_list(car_data, rsu_data)  # batch_size: (N+M)*N
		
		### model inferece: (N+M)*N -> N, reduce the RSU data and generate the actions for
		### cars only.
		with torch.no_grad():
			model_output = model(batch_data)
			# traffic_meta_total,
			# pred_waypoints_total,
			# is_junction_total,
			# traffic_light_state_total,
			# stop_sign_total,
			# bev_feature_total,
		### batch_size: N
		
		late_fusion = False
		if late_fusion:
			# update traffic_meta_total HERE
			pass

		### output postprocess to generate the action, list of actions for N agents
		control_all = self.generate_action_from_model_output(model_output, car_data_raw, car_mask, step, timestamp)
		return control_all


	def get_action_from_list_for_interfuser(self, car_data_raw, rsu_data_raw, model, step, timestamp):
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
		batch_data = self.collate_batch_train_list(car_data, [])  # batch_size: N*N
		
		### model inferece: N*N -> N, reduce the RSU data and generate the actions for
		### cars only.
		with torch.no_grad():
			model_output = model(batch_data)
		### batch_size: N
		
		### output postprocess to generate the action, list of actions for N agents
		control_all = self.generate_action_from_model_output(model_output, car_data_raw, car_mask, step, timestamp)
		return control_all
	

	def get_action_from_list_for_interfuser_cheat(self, car_data_raw, rsu_data_raw, model, step, timestamp):
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
		batch_data = self.collate_batch_train_list(car_data, [])  # batch_size: N*N
		
		### model inferece: N*N -> N, reduce the RSU data and generate the actions for
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
		cheat_actor_data = self.collect_actor_data()
		for i in range(len(car_data_raw)):
			traffic_meta = generate_heatmap(copy.deepcopy(car_data_raw[i]['measurements']), copy.deepcopy(cheat_actor_data))
			det_data = (
				generate_det_data(
					traffic_meta, copy.deepcopy(car_data_raw[i]['measurements']), copy.deepcopy(cheat_actor_data)
				)
				.reshape(400, -1)
				.astype(np.float32)
			)
			traffic_meta_total[i] = torch.from_numpy(det_data).cuda().float()
		model_output = (
				traffic_meta_total,
				pred_waypoints_total,
				is_junction_total,
				traffic_light_state_total,
				stop_sign_total,
				bev_feature_total,
			)
		### output postprocess to generate the action, list of actions for N agents
		control_all = self.generate_action_from_model_output(model_output, car_data_raw, car_mask, step, timestamp)
		return control_all
	
	

	def generate_action_from_model_output(self, model_output, car_data_raw, car_mask, step, timestamp):
		(
			traffic_meta_total,
			pred_waypoints_total,
			is_junction_total,
			traffic_light_state_total,
			stop_sign_total,
			bev_feature_total,
		) = model_output
		control_all = []
		tick_data = []
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
			traffic_light_state *= 0 # unsensitive to traffic light
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
			data[_id]["box"] = [box.x, box.y]
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
				data[_id]["box"] = [box.x, box.y]
			except:
				data[_id]["box"] = [1, 1]
			try:
				vel = actor.get_velocity()
				data[_id]["vel"] = [vel.x, vel.y, vel.z]
			except:
				data[_id]["vel"] = [0, 0, 0]
			data[_id]["tpe"] = 1

		
		return data



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

	def collate_batch_train_list(self, car_data: list, rsu_data: list) -> dict:
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
			# Set j-th car as the ego-car.
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
