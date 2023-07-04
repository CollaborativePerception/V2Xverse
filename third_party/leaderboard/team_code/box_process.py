import os
import json
import math
import torch
import numpy as np
import copy

def get_yaw_angle(forward_vector):
    forward_vector = forward_vector / np.linalg.norm(forward_vector)
    yaw = math.acos(forward_vector[0])
    if forward_vector[1] < 0:
        yaw = 2 * np.pi - yaw
    return yaw

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

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False