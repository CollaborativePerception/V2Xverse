"""
common function related to detection
TODO (yinda): remove replicated implementation in other packages
              and resolve all import issues
"""
import math
import json
import os
import copy

from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np

from skimage.measure import block_reduce
from .heatmap_utils import generate_heatmap, get_yaw_angle


def convert_grid_to_xy(i : int, j : int, det_range):
    """
    Conversion from feature map (perception result) to Cartesian coordinate in Vehicle Systsm

    det_range: forward, backward, left, right, resolution
    """
    x = det_range[4]*(j + 0.5) - det_range[2]
    y = det_range[0] - det_range[4]*(i+0.5)
    return x, y

def convert_xy_to_grid(x, y, det_range):
    """
    TODO
    """
    j = (x + det_range[2]) / det_range[4] - 0.5
    i = (det_range[0] - y) / det_range[4] - 0.5
    return i, j


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
    TODO

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

def generate_det_data(
    heatmap, measurements, actors_data, det_range=[30,10,10,10, 0.8]
) -> np.array:
    """
    Broad cast perception result label (pose, shape, class) to occupancy map

    Return:
        np.array: shape=(H, W, 7), 7 =  + label
    """
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

            # det_data[i][j] = np.array(
            #     [
            #         0,
            #         (loc[0] - center_x) / 3.5,
            #         (loc[1] - center_y) / 3.5,
            #         theta / 2.0,
            #         box[0] / 3.5,
            #         box[1] / 2.0,
            #         0,
            #     ]
            # )
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

        x,y = loc
        i,j = convert_xy_to_grid(x,y,det_range)
        i = int(np.around(i))
        j = int(np.around(j))

        if i < vertical and i > 0 and j > 0 and j < horizontal:
            det_data[i][j][-1] = 1.0

        ################## Gaussian Heatmap #####################
        w, h = box[:2]/det_range[4]
        heatmap = draw_heatmap(heatmap, h, w, j, i)
        ################## Gaussian Heatmap #####################
    det_data[:,:,0] = heatmap
    return det_data


def generate_det_data_backup(
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
            prob = np.power(0.5 / max(0.5, np.sqrt(min_dis)), 0.5)
            det_data[i][j] = np.array(
                [
                    prob,
                    (loc[0] - center_x) / 3.5,
                    (loc[1] - center_y) / 3.5,
                    theta / 2.0,
                    box[0] / 3.5,
                    box[1] / 2.0,
                    0,
                ]
            )

    for _id in actors_data:
        loc = actors_data[_id]["loc"][:2]
        ori = actors_data[_id]["ori"][:2]
        box = actors_data[_id]["box"]

        x,y = loc
        i,j = convert_xy_to_grid(x,y,det_range)
        i = int(np.around(i))
        j = int(np.around(j))

        if i < vertical and i > 0 and j > 0 and j < horizontal:
            det_data[i][j][-1] = 1.0

    return det_data


def generate_det_data_multiclass(
    heatmap, measurements, actors_data, det_range=[30,10,10,10, 0.8]
):  
    """
    Extend `generate_det_data` to a multi-class version
    """
    # 0: vehicle, 1: pedestrian, 2: traffic light, 3: bicycle
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
