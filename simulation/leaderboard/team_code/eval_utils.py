import os
import numpy as np
import torch
from shapely.geometry import Polygon
import json
import pickle
from collections import OrderedDict
import re
import yaml
import math
import copy
from typing import List

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def get_yaw_angle(forward_vector):
    forward_vector = forward_vector / np.linalg.norm(forward_vector)
    yaw = math.acos(forward_vector[0])
    if forward_vector[1] < 0:
        yaw = 2 * np.pi - yaw
    return yaw

def boxes_to_corners_3d(boxes3d, order):
    """
        4 -------- 5
       /|         /|
      7 -------- 6 .
      | |        | |
      . 0 -------- 1
      |/         |/
      3 -------- 2
    Parameters
    __________
    boxes3d: np.ndarray or torch.Tensor
        (N, 7) [x, y, z, l, w, h, heading], or [x, y, z, h, w, l, heading]
               (x, y, z) is the box center.
    order : str
        'lwh' or 'hwl'
    Returns:
        corners3d: np.ndarray or torch.Tensor
        (N, 8, 3), the 8 corners of the bounding box.
    opv2v's left hand coord 
    
    ^ z
    |
    |
    | . x
    |/
    +-------> y
    """

    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)
    boxes3d_ = boxes3d

    if order == 'hwl':
        boxes3d_ = boxes3d[:, [0, 1, 2, 5, 4, 3, 6]]

    template = boxes3d_.new_tensor((
        [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1],
        [1, -1, 1], [1, 1, 1], [-1, 1, 1], [-1, -1, 1],
    )) / 2

    corners3d = boxes3d_[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3),
                                                   boxes3d_[:, 6]).view(-1, 8,
                                                                        3)
    corners3d += boxes3d_[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d

def get_points_in_rotated_box_3d(p, box_corner):
    """
    Get points within a rotated bounding box (3D version).

    Parameters
    ----------
    p : numpy.array
        Points to be tested with shape (N, 3).
    box_corner : numpy.array
        Corners of bounding box with shape (8, 3).

            0 --------------------- 1         
          ,"|                     ,"|       
         3 --------------------- 2  |     
         |  |                    |  |   
         |  |                    |  |  
         |  4  _ _ _ _ _ _ _ _ _ |  5 
         |,"                     |," 
         7 --------------------- 6

    Returns
    -------
    p_in_box : numpy.array
        Points within the box.

    """
    edge1 = box_corner[1, :] - box_corner[0, :]
    edge2 = box_corner[3, :] - box_corner[0, :]
    edge3 = box_corner[4, :] - box_corner[0, :]

    p_rel = p - box_corner[0, :].reshape(1, -1)

    l1 = get_projection_length_for_vector_projection(p_rel, edge1)
    l2 = get_projection_length_for_vector_projection(p_rel, edge2)
    l3 = get_projection_length_for_vector_projection(p_rel, edge3)
    # A point is within the box, if and only after projecting the
    # point onto the two edges s.t. p_rel = [edge1, edge2] @ [l1, l2]^T,
    # we have 0<=l1<=1 and 0<=l2<=1.
    mask1 = np.logical_and(l1 >= 0, l1 <= 1)
    mask2 = np.logical_and(l2 >= 0, l2 <= 1)
    mask3 = np.logical_and(l3 >= 0, l3 <= 1)

    mask = np.logical_and(mask1, mask2)
    mask = np.logical_and(mask, mask3)
    p_in_box = p[mask, :]

    return p_in_box

def get_projection_length_for_vector_projection(a, b):
    """
    Get projection length for the Vector projection of a onto b s.t.
    a_projected = length * b. (2D version) See
    https://en.wikipedia.org/wiki/Vector_projection#Vector_projection_2
    for more details.

    Parameters
    ----------
    a : numpy.array
        The vectors to be projected with shape (N, 2).

    b : numpy.array
        The vector that is projected onto with shape (2).

    Returns
    -------
    length : numpy.array
        The length of projected a with respect to b.
    """
    assert np.sum(b ** 2, axis=-1) > 1e-6
    length = a.dot(b) / np.sum(b ** 2, axis=-1)
    return length

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), radians, angle along z-axis, angle increases x ==> y
    Returns:
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3].float(), rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def process_lidar_visibility(actors_data: dict, lidar_np: np.array, lidar_pose: dict, change_actor_file: bool = False, mode: str = 'lidar', thresh: int = 5) -> (dict, List) : 
    """
    tag actors data with label of lidar visibility.
    If there are enough lidar points within the detection box range of an actor, this actor is visible for lidar, vice versa
    Args:
        actors_data: bounding box of actors in world coordinate
        lidar_np: point clouds in lidar coordinate
        lidar_pose: lidar pose in world coordinate
        change_actor_file: whether to rewrite the actor data
        mode: 'lidar' or 'camera'
        thresh: minimum number of lidar points to tell a box is visible
    Returns:
        actors_data: added with information fo lidar_visibility
        visible_actor_ids: id of visible actors
    """

    original_actors_data = copy.deepcopy(actors_data)    
    lidar_unprocessed = copy.deepcopy(lidar_np)
    full_lidar = lidar_unprocessed[..., :3]
    full_lidar[:, 1] *= -1

    ego_x = lidar_pose["lidar_pose_x"]
    ego_y = lidar_pose["lidar_pose_y"]
    ego_z = lidar_pose["lidar_pose_z"]
    ego_theta = lidar_pose["theta"] + np.pi

    # transform matrix from lidar to world
    R = np.array(
        [
            [np.cos(ego_theta), -np.sin(ego_theta)],
            [np.sin(ego_theta), np.cos(ego_theta)],
        ]
    )

    # transform actor bounding box from world coordinate to lidar coordinate
    for _id in original_actors_data.keys():
        if original_actors_data[_id]["tpe"] == 2:
            continue  # FIXME donot add traffix light
        raw_loc = original_actors_data[_id]['loc'][:2]
        new_loc = R.T.dot(np.array([raw_loc[0] - ego_x , raw_loc[1] - ego_y]))
        new_loc[1] = -new_loc[1]
        original_actors_data[_id]['loc'][:2] = np.array(new_loc)
        # if original_actors_data[_id]["tpe"] == 1:
        #     original_actors_data[_id]['loc'][2] -= original_actors_data[_id]['box'][2]
        original_actors_data[_id]['loc'][2] -= (ego_z)
        raw_ori = original_actors_data[_id]['ori'][:2]
        new_ori = R.T.dot(np.array([raw_ori[0], raw_ori[1]]))
        original_actors_data[_id]['ori'][:2] = np.array(new_ori)
    

    boxes_corner = [] # pose and orientation of the box,
            # (x, y, z, scale_x, scale_y, scale_z, yaw)
    id_map = {}
    count = 0
    for _id in original_actors_data.keys():
        if original_actors_data[_id]["tpe"] == 2:
            continue  # FIXME donot add traffix light
        cur_data = original_actors_data[_id]
        yaw = get_yaw_angle(cur_data['ori'][:2])
        cur_data['loc'][2] += cur_data['box'][2]
        boxes_corner.append(cur_data['loc']+ [i*2 for i in cur_data['box']] + [yaw])
        id_map[count] = _id
        count += 1
    boxes_corner = np.array(boxes_corner)   

    # transform boxes to form of corner
    corners = boxes_to_corners_3d(boxes_corner, order='lwh')

    # check the number of lidar points in each box and tag the box
    visible_actor_ids = []
    for N in range(boxes_corner.shape[0]):
        if actors_data[id_map[N]]['tpe']==2:
            if change_actor_file:
                actors_data[id_map[N]]['lidar_visible'] = 0
                actors_data[id_map[N]]['camera_visible'] = 0
            continue
        num_lidar_points = get_points_in_rotated_box_3d(full_lidar, corners[N])
        # print(len(num_lidar_points))
        if len(num_lidar_points)> thresh :
            if change_actor_file:
                actors_data[id_map[N]]['{}_visible'.format(mode)] = 1
            visible_actor_ids.append(id_map[N])
        else:
            if change_actor_file:
                actors_data[id_map[N]]['{}_visible'.format(mode)] = 0

    return actors_data, visible_actor_ids

def convert_format(boxes_array):
    """
    Convert boxes array to shapely.geometry.Polygon format.
    Parameters
    ----------
    boxes_array : np.ndarray
        (N, 4, 2) or (N, 8, 3).

    Returns
    -------
        list of converted shapely.geometry.Polygon object.

    """
    polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in
                boxes_array]
    return np.array(polygons)

def torch_tensor_to_numpy(torch_tensor):
    """
    Convert a torch tensor to numpy.

    Parameters
    ----------
    torch_tensor : torch.Tensor

    Returns
    -------
    A numpy array.
    """
    return torch_tensor.numpy() if not torch_tensor.is_cuda else \
        torch_tensor.cpu().detach().numpy()


def compute_iou(box, boxes):
    """
    Compute iou between box and boxes list
    Parameters
    ----------
    box : shapely.geometry.Polygon
        Bounding box Polygon.

    boxes : list
        List of shapely.geometry.Polygon.

    Returns
    -------
    iou : np.ndarray
        Array of iou between box and boxes.

    """
    # Calculate intersection areas
    if np.any(np.array([box.union(b).area for b in boxes])==0):
        print('debug')
    iou = [box.intersection(b).area / (box.union(b).area + 1e-12) for b in boxes]

    return np.array(iou, dtype=np.float32)

def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre

def find_peak_box(data, det_range):
    v = int((det_range[0] + det_range[1])/det_range[4])
    h = int((det_range[2] + det_range[3])/det_range[4])
    det_data = np.zeros((v+2, h+2, 7))
    det_data[1:v+1, 1:h+1] = data
    res = []
    score = []
    for i in range(1, v+1):
        for j in range(1, h+1):
            if det_data[i, j, 0] > 0.9 or (
                det_data[i, j, 0] > 0.4
                and det_data[i, j, 0] > det_data[i, j - 1, 0]
                and det_data[i, j, 0] > det_data[i, j + 1, 0]
                and det_data[i, j, 0] > det_data[i + 1, j + 1, 0]
                and det_data[i, j, 0] > det_data[i - 1, j + 1, 0]
                and det_data[i, j, 0] > det_data[i + 1, j - 1, 0]
                and det_data[i, j, 0] > det_data[i + 1, j + 1, 0]
                and det_data[i, j, 0] > det_data[i - 1, j, 0]
                and det_data[i, j, 0] > det_data[i + 1, j, 0]
            ):
                res.append((i - 1, j - 1))
                score.append(det_data[i - 1, j - 1, 0])
    bbox = res
    score = np.array(score)
    return bbox, score

def find_all_box(data, det_range):
    v = int((det_range[0] + det_range[1])/det_range[4])
    h = int((det_range[2] + det_range[3])/det_range[4])
    det_data = np.zeros((v+2, h+2, 7))
    det_data[1:v+1, 1:h+1] = data
    res = []
    score = []
    for i in range(1, v+1):
        for j in range(1, h+1):
            if det_data[i, j, 0] > 0.2:
                res.append((i - 1, j - 1))
                score.append(det_data[i - 1, j - 1, 0])
    bbox = res
    score = np.array(score)
    return bbox, score

def nms_rotated(boxes, scores, threshold):
    """Performs rorated non-maximum suppression and returns indices of kept
    boxes.

    Parameters
    ----------
    boxes : torch.tensor
        The location preds with shape (N, 4, 2).

    scores : torch.tensor
        The predicted confidence score with shape (N,)

    threshold: float
        IoU threshold to use for filtering.

    Returns
    -------
        An array of index
    """
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int32)
    # boxes = boxes.cpu().detach().numpy()
    # scores = scores.cpu().detach().numpy()

    polygons = convert_format(boxes)

    top = 1000
    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1][:top]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(polygons[i], polygons[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)

    return np.array(pick, dtype=np.int32)


def nms(pred_box3d_tensor,scores):
    # pred_box3d_tensor (N,4,2)
    nms_thresh = 0.00
    keep_index = nms_rotated(pred_box3d_tensor,
                                        scores,
                                        nms_thresh
                                        )

    pred_box3d_tensor = pred_box3d_tensor[keep_index]
    # select cooresponding score
    scores = scores[keep_index]

    return pred_box3d_tensor, scores

def turn_traffic_into_bbox(traffic, det_range):
    # input traffic (20x20x7) tensor
    # output bbox (Nx4x2) tensor
    reweight_array = np.array([1.0, 3.5, 3.5, 2.0, 3.5, 2.0, 8.0])
    v = int((det_range[0] + det_range[1])/det_range[4])
    h = int((det_range[2] + det_range[3])/det_range[4])
    traffic = traffic.reshape(v, h, -1)
    traffic = traffic * reweight_array
    # box_ids, score = find_peak_box(traffic, det_range)
    box_ids, score = find_all_box(traffic, det_range)

    objects_bbox = []
    score_list = []
    for _i, poi in enumerate(box_ids):
        i, j = poi
        # if traffic[i,j,-1] > 0.5:
        #     continue
        l,w = traffic[i,j,4],traffic[i,j,5]
        theta = traffic[i,j,3]
        theta = theta * math.pi + math.pi / 2

        center_x, center_y = convert_grid_to_xy(i, j, det_range)
        center_x = center_x + traffic[i,j,1] 
        center_y = center_y + traffic[i,j,2]
        bbox_x = [l*np.sin(theta)+w*np.cos(theta),l*np.sin(theta)-w*np.cos(theta),-l*np.sin(theta)-w*np.cos(theta),-l*np.sin(theta)+w*np.cos(theta)]
        bbox_y = [l*np.cos(theta)-w*np.sin(theta),l*np.cos(theta)+w*np.sin(theta),-l*np.cos(theta)+w*np.sin(theta),-l*np.cos(theta)-w*np.sin(theta)]

        bbox_x = np.array(bbox_x) + center_x
        bbox_y = np.array(bbox_y) + center_y

        objects_bbox.append(np.array([bbox_x,bbox_y]))
        score_list.append(score[_i])
    objects_bbox = np.array(objects_bbox) # (N,4,2)
    score_list = np.array(score_list)
    if len(objects_bbox) > 0:
        objects_bbox = objects_bbox.transpose(0,2,1)
    objects_bbox, score = nms(objects_bbox, score_list)
    return objects_bbox, score


def turn_traffic_into_bbox_fast(traffic, det_range):
    # input traffic (20x20x7) tensor
    # output bbox (Nx4x2) tensor
    # reweight_array = np.array([1.0, 3.5, 3.5, 2.0, 3.5, 2.0, 8.0])
    reweight_array = np.array([1.0, 3.5, 3.5, 1.0, 3.5, 2.0, 8.0])
    v = int((det_range[0] + det_range[1])/det_range[4])
    h = int((det_range[2] + det_range[3])/det_range[4])
    traffic = traffic.reshape(v, h, -1)
    traffic = traffic * reweight_array
    # box_ids, score = find_peak_box(traffic, det_range)
    
    # box_ids, score = find_all_box(traffic, det_range)

    lw = traffic[:,:,4:6]
    lw = lw.reshape(-1,2,1)

    theta = traffic[:,:,3]
    # theta = theta * math.pi + math.pi / 2
    theta = theta + math.pi / 2
    theta = theta.reshape(-1)

    x_rot_matrix = np.array(
        [[np.sin(theta),np.cos(theta)],
        [np.sin(theta),-np.cos(theta)],
        [-np.sin(theta),-np.cos(theta)],
        [-np.sin(theta),np.cos(theta)],
    ]).transpose(2,0,1)

    y_rot_matrix = np.array([
        [np.cos(theta),-np.sin(theta)],
        [np.cos(theta),np.sin(theta)],
        [-np.cos(theta),np.sin(theta)],
        [-np.cos(theta),-np.sin(theta)],        
    ]).transpose(2,0,1)

    x_bias = np.matmul(x_rot_matrix,lw)
    y_bias = np.matmul(y_rot_matrix,lw)
    xy_bias = np.concatenate([x_bias,y_bias],axis=-1) # (HW,4,2)

    ys, xs = np.meshgrid(np.arange(0, h), np.arange(0, v))

    ys = ys.reshape(-1)
    xs = xs.reshape(-1)

    x_center = det_range[4]*(ys + 0.5) - det_range[2]
    y_center = det_range[0] - det_range[4]*(xs+0.5)

    x_center += traffic[:,:,1].reshape(-1)
    y_center += traffic[:,:,2].reshape(-1)

    x_center = x_center[:,None,None]
    y_center = y_center[:,None,None]
    xy_center = np.concatenate([x_center,y_center],axis=-1) # (HW,1,2)
    
    objects_bbox = xy_center + xy_bias

    score = traffic[:,:,0].reshape(-1)

    objects_bbox, score = filter(objects_bbox, score)
    objects_bbox, score = nms(objects_bbox, score)
    return objects_bbox, score

def turn_traffic_into_bbox_fast_base(traffic, det_range):
    # input traffic (20x20x7) tensor
    # output bbox (Nx4x2) tensor
    # reweight_array = np.array([1.0, 3.5, 3.5, 2.0, 3.5, 2.0, 8.0])
    reweight_array = np.array([1.0, 3.5, 3.5, 1.0, 3.5, 2.0, 8.0])
    v = int((det_range[0] + det_range[1])/det_range[4])
    h = int((det_range[2] + det_range[3])/det_range[4])
    traffic = traffic.reshape(v, h, -1)
    traffic = traffic * reweight_array
    # box_ids, score = find_peak_box(traffic, det_range)
    
    # box_ids, score = find_all_box(traffic, det_range)

    lw = traffic[:,:,4:6]
    lw = lw.reshape(-1,2,1)

    theta = traffic[:,:,3]
    # theta = theta * math.pi + math.pi / 2
    theta = theta + math.pi / 2
    theta = theta.reshape(-1)

    x_rot_matrix = np.array(
        [[np.sin(theta),np.cos(theta)],
        [np.sin(theta),-np.cos(theta)],
        [-np.sin(theta),-np.cos(theta)],
        [-np.sin(theta),np.cos(theta)],
    ]).transpose(2,0,1)

    y_rot_matrix = np.array([
        [np.cos(theta),-np.sin(theta)],
        [np.cos(theta),np.sin(theta)],
        [-np.cos(theta),np.sin(theta)],
        [-np.cos(theta),-np.sin(theta)],        
    ]).transpose(2,0,1)

    x_bias = np.matmul(x_rot_matrix,lw)
    y_bias = np.matmul(y_rot_matrix,lw)
    xy_bias = np.concatenate([x_bias,y_bias],axis=-1) # (HW,4,2)

    ys, xs = np.meshgrid(np.arange(0, h), np.arange(0, v))

    ys = ys.reshape(-1)
    xs = xs.reshape(-1)

    x_center = det_range[4]*(ys + 0.5) - det_range[2]
    y_center = det_range[0] - det_range[4]*(xs+0.5)

    x_center += traffic[:,:,1].reshape(-1)
    y_center += traffic[:,:,2].reshape(-1)

    x_center = x_center[:,None,None]
    y_center = y_center[:,None,None]
    xy_center = np.concatenate([x_center,y_center],axis=-1) # (HW,1,2)
    
    objects_bbox = xy_center + xy_bias

    score = traffic[:,:,0].reshape(-1)

    return objects_bbox, score

def warp_bbox(bbox,transform_matrix):
    # (N,4,2) (2,3)
    N = bbox.shape[0]
    bbox_bias = np.ones((N,4,3)).float()
    bbox_bias[:,:,:2] = bbox
    bbox_bias = bbox_bias.transpose(0,2,1) #(N,3,4)
    output = np.matmul(transform_matrix[None].repeat(N,axis=0),bbox_bias) #(N,2,4)
    output = output.transpose(0,2,1) #(N,4,2)
    return output

def turn_traffic_into_bbox_fast_late(traffic, det_range, transform_matrix):
    # traffic (N,H,W,7)
    N = traffic.shape[0]
    all_bbox = []
    all_score = []
    for i in range(N):
        bbox, score = turn_traffic_into_bbox_fast_base(traffic[i],det_range)
        bbox = warp_bbox(bbox,transform_matrix[0,i])
        all_bbox.append(bbox)
        all_score.append(score)
    objects_bbox = np.concatenate(all_bbox,axis=0)
    score = np.concatenate(all_score,axis=0)

    objects_bbox, score = filter(objects_bbox, score)
    objects_bbox, score = nms(objects_bbox, score)

    return objects_bbox, score

def turn_traffic_into_bbox_fast_vis(traffic, det_range):
    # input traffic (20x20x7) tensor
    # output bbox (Nx4x2) tensor
    reweight_array = np.array([1.0, 3.5, 3.5, 2.0, 3.5, 2.0, 8.0])
    v = int((det_range[0] + det_range[1])/det_range[4])
    h = int((det_range[2] + det_range[3])/det_range[4])
    traffic = traffic.reshape(v, h, -1)
    traffic = traffic * reweight_array
    traffic_flat = traffic.reshape(-1,7)
    idx = (traffic[:,0] > 0.5)
    print(traffic[idx])
    # box_ids, score = find_peak_box(traffic, det_range)
    
    # box_ids, score = find_all_box(traffic, det_range)

    lw = traffic[:,:,4:6]
    lw = lw.reshape(-1,2,1)

    theta = traffic[:,:,3]
    theta = theta * math.pi + math.pi / 2
    theta = theta.reshape(-1)

    x_rot_matrix = np.array(
        [[np.sin(theta),np.cos(theta)],
        [np.sin(theta),-np.cos(theta)],
        [-np.sin(theta),-np.cos(theta)],
        [-np.sin(theta),np.cos(theta)],
    ]).transpose(2,0,1)

    y_rot_matrix = np.array([
        [np.cos(theta),-np.sin(theta)],
        [np.cos(theta),np.sin(theta)],
        [-np.cos(theta),np.sin(theta)],
        [-np.cos(theta),-np.sin(theta)],        
    ]).transpose(2,0,1)

    x_bias = np.matmul(x_rot_matrix,lw)
    y_bias = np.matmul(y_rot_matrix,lw)
    xy_bias = np.concatenate([x_bias,y_bias],axis=-1) # (HW,4,2)

    ys, xs = np.meshgrid(np.arange(0, h), np.arange(0, v))

    ys = ys.reshape(-1)
    xs = xs.reshape(-1)

    x_center = det_range[4]*(ys + 0.5) - det_range[2]
    y_center = det_range[0] - det_range[4]*(xs+0.5)

    x_center += traffic[:,:,1].reshape(-1)
    y_center += traffic[:,:,2].reshape(-1)

    x_center = x_center[:,None,None]
    y_center = y_center[:,None,None]
    xy_center = np.concatenate([x_center,y_center],axis=-1) # (HW,1,2)
    
    objects_bbox = xy_center + xy_bias

    score = traffic[:,:,0].reshape(-1)

    objects_bbox, score = filter(objects_bbox, score)
    objects_bbox, score = nms(objects_bbox, score)
    return objects_bbox, score

def filter(objects_bbox, score):
    idx = (score > 0.2)
    score = score[idx]
    objects_bbox = objects_bbox[idx]
    return objects_bbox, score


def convert_grid_to_xy(i, j, det_range):
    x = det_range[4]*(j + 0.5) - det_range[2]
    y = det_range[0] - det_range[4]*(i+0.5)
    return x, y

def nms_pytorch(boxes: torch.tensor, thresh_iou: float):
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.

    Parameters
    ----------
    boxes : torch.tensor
        The location preds along with the class predscores,
         Shape: [num_boxes,5].
    thresh_iou : float
        (float) The overlap thresh for suppressing unnecessary boxes.
    Returns
    -------
        A list of index
    """

    # we extract coordinates for every
    # prediction box present in P
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # we extract the confidence scores as well
    scores = boxes[:, 4]

    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)

    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()

    # initialise an empty list for
    # filtered prediction boxes
    keep = []

    while len(order) > 0:

        # extract the index of the
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]

        # push S in filtered predictions list
        keep.append(idx.numpy().item()
                    if not idx.is_cuda else idx.cpu().detach().numpy().item())

        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            break

        # select coordinates of BBoxes according to
        # the indices in order
        xx1 = torch.index_select(x1, dim=0, index=order)
        xx2 = torch.index_select(x2, dim=0, index=order)
        yy1 = torch.index_select(y1, dim=0, index=order)
        yy2 = torch.index_select(y2, dim=0, index=order)

        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])

        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1

        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        # find the intersection area
        inter = w * h

        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim=0, index=order)

        # find the union of every prediction T in P
        # with the prediction S
        # Note that areas[idx] represents area of S
        union = (rem_areas - inter) + areas[idx]

        # find the IoU of every prediction in P with S
        IoU = inter / union

        # keep the boxes with IoU less than thresh_iou
        mask = IoU < thresh_iou
        order = order[mask]

    return keep

def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh):
    """
    Calculate the true positive and false positive numbers of the current
    frames.
    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    """
    # fp, tp and gt in the current frame
    fp = []
    tp = []
    gt = gt_boxes.shape[0]
    if det_boxes is not None:
        # convert bounding boxes to numpy array
        # det_boxes = torch_tensor_to_numpy(det_boxes)
        # det_score = torch_tensor_to_numpy(det_score)
        # gt_boxes = torch_tensor_to_numpy(gt_boxes)

        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend] # from high to low
        det_polygon_list = list(convert_format(det_boxes))
        gt_polygon_list = list(convert_format(gt_boxes))

        # match prediction and gt bounding box, in confidence descending order
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = compute_iou(det_polygon, gt_polygon_list)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)

            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)
        result_stat[iou_thresh]['score'] += det_score.tolist()
    result_stat[iou_thresh]['fp'] += fp
    result_stat[iou_thresh]['tp'] += tp
    result_stat[iou_thresh]['gt'] += gt
    return result_stat



def calculate_ap(result_stat, iou):
    """
    Calculate the average precision and recall, and save them into a txt.
    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.
    iou : float
    """
    iou_5 = result_stat[iou]

    fp = np.array(iou_5['fp'])
    tp = np.array(iou_5['tp'])
    score = np.array(iou_5['score'])
    assert len(fp) == len(tp) and len(tp) == len(score)

    sorted_index = np.argsort(-score)
    fp = fp[sorted_index].tolist()
    tp = tp[sorted_index].tolist()

    gt_total = iou_5['gt']

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec


def eval_final_results(result_stat, save_path=None, infer_info=None):
    dump_dict = {}

    ap_30, mrec_30, mpre_30 = calculate_ap(result_stat, 0.30)
    ap_50, mrec_50, mpre_50 = calculate_ap(result_stat, 0.50)
    ap_70, mrec_70, mpre_70 = calculate_ap(result_stat, 0.70)

    dump_dict.update({'ap30': ap_30,
                      'ap_50': ap_50,
                      'ap_70': ap_70,
                      'mpre_50': mpre_50,
                      'mrec_50': mrec_50,
                      'mpre_70': mpre_70,
                      'mrec_70': mrec_70,
                      })
    # if infer_info is None:
    #     save_yaml(dump_dict, os.path.join(save_path, 'eval.yaml'))
    # else:
    #     save_yaml(dump_dict, os.path.join(save_path, f'eval_{infer_info}.yaml'))

    # print('The Average Precision at IOU 0.3 is %.2f, '
    #       'The Average Precision at IOU 0.5 is %.2f, '
    #       'The Average Precision at IOU 0.7 is %.2f' % (ap_30, ap_50, ap_70))

    return ap_30, ap_50, ap_70

def save_yaml(data, save_name):
    """
    Save the dictionary into a yaml file.

    Parameters
    ----------
    data : dict
        The dictionary contains all data.

    save_name : string
        Full path of the output yaml file.
    """

    with open(save_name, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

def vis_detection_result(traffic,det_range,is_pred=False,id_start=0,scene_dict='',frame_id=''):
    # traffic (B,400,7)
    # waypoints (B,10,2)
    batch_size = traffic.shape[0]
    # assert traffic.shape[0] == waypoints.shape[0]
    for i in range(batch_size):
        object_bbox, _ = turn_traffic_into_bbox_fast(traffic[i].reshape(192, 96, -1),det_range)
        draw_result(object_bbox,det_range,idx=i,is_pred=is_pred,scene_dict=scene_dict[i]['ego'],frame_id=frame_id[i])
    return

def draw_result(object_bbox,det_range,idx=0,is_pred=False,scene_dict='',frame_id=''):
    import matplotlib.pyplot as plt

    plt.cla()

    plt.figure(figsize=(6, 12))
    ax = plt.gca()
    ax.set_facecolor("black")

    v = int((det_range[0] + det_range[1])/det_range[4])
    h = int((det_range[2] + det_range[3])/det_range[4])

    plt.xlim((-det_range[2], det_range[3]))
    plt.ylim((-det_range[1], det_range[0]))

    ego_bbox_x = [1,-1,-1,1]
    ego_bbox_y = [2.45,2.45,-2.45,-2.45]

    for i in range(len(object_bbox)):
        plt.fill(object_bbox[i,:,0], object_bbox[i,:,1], color = 'white')

    plt.fill(ego_bbox_x, ego_bbox_y, color = 'yellow')

    # for i in range(len(waypoint)):
    #     plt.scatter(waypoint[i,0],-waypoint[i,1], s=5, color = 'green')

    scene_dict = scene_dict.split('/')
    scene_dict_new = scene_dict[-4] + '_' + scene_dict[-2]

    # if is_pred:
    #     plt.savefig('/GPFS/public/InterFuser/results_cx/vis/detection_vis_gt/'+scene_dict_new+'_'+str(frame_id)+"_pred.png")
    # else:
    #     plt.savefig('/GPFS/public/InterFuser/results_cx/vis/detection_vis_gt/'+scene_dict_new+'_'+str(frame_id)+"_gt.png")
    if is_pred:
        plt.savefig('/dssg/home/acct-agrtkx/agrtkx/cxxu/results_cx/result_vis/'+scene_dict_new+'_'+str(frame_id)+"_pred_check.png")
    else:
        plt.savefig('/dssg/home/acct-agrtkx/agrtkx/cxxu/results_cx/result_vis/'+scene_dict_new+'_'+str(frame_id)+"_gt_check.png")
    return

def generate_bbox(map_input,det_range):
    import matplotlib.pyplot as plt

    object_bbox, _ = turn_traffic_into_bbox_fast(map_input.reshape(192, 96, -1),det_range)
    # plt.cla()

    fig = plt.figure(figsize=(6, 12))
    ax = plt.gca()
    ax.set_facecolor("black")

    v = int((det_range[0] + det_range[1])/det_range[4])
    h = int((det_range[2] + det_range[3])/det_range[4])

    plt.xlim((-det_range[2], det_range[3]))
    plt.ylim((-det_range[1], det_range[0]))

    ego_bbox_x = [1,-1,-1,1]
    ego_bbox_y = [2.45,2.45,-2.45,-2.45]

    print(len(object_bbox))
    for i in range(len(object_bbox)):
        plt.fill(object_bbox[i,:,0], object_bbox[i,:,1], color = 'white')

    # plt.fill(ego_bbox_x, ego_bbox_y, color = 'yellow')

    # for i in range(len(waypoint)):
    #     plt.scatter(waypoint[i,0],-waypoint[i,1], s=5, color = 'green')

    # scene_dict = scene_dict.split('/')
    # scene_dict_new = scene_dict[-4] + '_' + scene_dict[-2]

    # if is_pred:
    #     plt.savefig('/GPFS/public/InterFuser/results_cx/vis/detection_vis_gt/'+scene_dict_new+'_'+str(frame_id)+"_pred.png")
    # else:
    #     plt.savefig('/GPFS/public/InterFuser/results_cx/vis/detection_vis_gt/'+scene_dict_new+'_'+str(frame_id)+"_gt.png")
    # if is_pred:
    #     plt.savefig('/dssg/home/acct-agrtkx/agrtkx/cxxu/results_cx/result_vis/'+scene_dict_new+'_'+str(frame_id)+"_pred_check.png")
    # else:
    #     plt.savefig('/dssg/home/acct-agrtkx/agrtkx/cxxu/results_cx/result_vis/'+scene_dict_new+'_'+str(frame_id)+"_gt_check.png")
    return fig

def generate_bbox_multiclass(map_input,det_range,id=0,name='0'):
    import matplotlib.pyplot as plt

    all_bbox = []
    cls_num = map_input.shape[0]

    for i in range(cls_num):
        object_bbox, _ = turn_traffic_into_bbox_fast(map_input[i].reshape(192, 96, -1),det_range)
        if len(object_bbox) > 0:
            all_bbox.append(object_bbox)

    if len(all_bbox) > 0:
        all_bbox = np.concatenate(all_bbox,axis=0)
    else:
        all_bbox = np.zeros((1,4,2))
    # plt.cla()

    fig = plt.figure(figsize=(6, 12))
    ax = plt.gca()
    ax.set_facecolor("black")

    v = int((det_range[0] + det_range[1])/det_range[4])
    h = int((det_range[2] + det_range[3])/det_range[4])

    plt.xlim((-det_range[2], det_range[3]))
    plt.ylim((-det_range[1], det_range[0]))

    ego_bbox_x = [1,-1,-1,1]
    ego_bbox_y = [2.45,2.45,-2.45,-2.45]

    for i in range(len(all_bbox)):
        plt.fill(all_bbox[i,:,0], all_bbox[i,:,1], color = 'white')

    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig('/dssg/home/acct-agrtkx/agrtkx/cxxu/results_cx/check_warp/'+str(id)+name+'.jpg',bbox_inches='tight', pad_inches = -0.1)
    return fig

def get_pairwise_transformation(base_data_dict, max_cav, proj_first):
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
    pairwise_t_matrix = np.tile(np.eye(4), (1, 1, 1)) # (L, L, 4, 4)

    t_list = []

    # save all transformation matrix in a list in order first.
    for cav_id, cav_content in base_data_dict.items():
        lidar_pose = cav_content['params']['lidar_pose']
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

def x_to_world(pose):
    """
    The transformation matrix from x-coordinate system to carla world system
    Also is the pose in world coordinate: T_world_x

    Parameters
    ----------
    pose : list
        [x, y, z, roll, yaw, pitch], degree

    Returns
    -------
    matrix : np.ndarray
        The transformation matrix.
    """
    x, y, z, roll, yaw, pitch = pose[:]

    # used for rotation matrix
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.identity(4)

    # translation matrix
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix

# def get_pairwise_transformation(pose, max_cav):
#     pairwise_t_matrix = np.tile(np.eye(4), (max_cav, max_cav, 1, 1)) # (L, L, 4, 4)

#     t_list = []

#     # save all transformation matrix in a list in order first.
#     for i in range(max_cav):
#         lidar_pose = pose[i]
#         t_list.append(self.x_to_world(lidar_pose))  # Twx

#     for i in range(len(t_list)):
#         for j in range(len(t_list)):
#             # identity matrix to self
#             if i != j:
#                 # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
#                 # t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
#                 t_matrix = np.linalg.solve(t_list[j], t_list[i])  # Tjw*Twi = Tji
#                 pairwise_t_matrix[i, j] = t_matrix

#     return pairwise_t_matrix

def warp_affine_simple(src, M, dsize,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    B, C, H, W = src.size()
    grid = F.affine_grid(M,
                        [B, C, dsize[0], dsize[1]],
                        align_corners=align_corners).to(src)
    return F.grid_sample(src, grid, align_corners=align_corners)

def warp(rsu_map,pose):
    pairwise_t_matrix = \
        get_pairwise_transformation(pose.cpu(), 3)
    # t_matrix[i, j]-> from i to j
    H, W = rsu_map.shape
    pairwise_t_matrix = pairwise_t_matrix[:,:,[0, 1],:][:,:,:,[0, 1, 3]] # [N, N, 2, 3]
    pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
    pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
    pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (half_w)  #(downsample_rate * discrete_ratio * W) * 2
    pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (half_h) 

    rsu_map = torch.from_numpy(rsu_map)
    t_matrix = torch.from_numpy(pairwise_t_matrix[:N, :N, :, :])
    warp_rsu_map = warp_affine_simple(rsu_map,
                                t_matrix[0, :, :, :],
                                (H, W))
    return warp_rsu_map

def check_warp(idx,pose):
    # pose (3,3)
    ego_map = plt.imread('/dssg/home/acct-agrtkx/agrtkx/cxxu/results_cx/check_warp/'+str(id)+'gt_bbox.jpg')
    rsu_map = plt.imread('/dssg/home/acct-agrtkx/agrtkx/cxxu/results_cx/check_warp/'+str(id)+'rsu_gt_bbox.jpg')
    print(ego_map.shape) #(C,H,W)
    pose_new = torch.cat((pose[0:1],pose[2:3]),dim=0)
    print(pose_new.shape) #(2,3)
    warp_rsu_map = warp(rsu_map,pose)
    print(warp_rsu_map.shape) #(C,H,W)
    plt.imshow(warp_rsu_map)
    plt.savefig('/dssg/home/acct-agrtkx/agrtkx/cxxu/results_cx/check_warp/'+str(id)+'warp_rsu_gt_bbox.jpg',bbox_inches='tight', pad_inches = -0.1)
    return