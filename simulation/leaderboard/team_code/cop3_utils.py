import os
import cv2
import math
import torch
import numpy as np
from torchvision import transforms
from skimage.measure import block_reduce

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

