# We implemented our method on top of AB3DMOT's KITTI tracking open-source code

from __future__ import print_function
import pandas as pd  # for test
import os.path, copy, numpy as np, time, sys
# from numba import jit
# from numba import jit
import sys

sys.path.append('.')

# from sklearn.utils.linear_assignment_ import linear_assignment
# from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from scipy.spatial import ConvexHull
from team_code.utils.tracking.covariance import Covariance
# import json
# from pyquaternion import Quaternion
# from tqdm import tqdm
import argparse
import torch
# from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
# from opencood.utils.box_utils import corner_to_center, project_box3d, boxes_to_corners_3d

####config parameter
# ifdoreid = True
reidlikthr = 0.5
onlyreid = True
# use_mahalanobis = True           # standford is True   ab3d is false
use_angular_velocity = False  # standford is True   ab3d is false


# covariance_id  = 2               # 0 or 2    standford is 2          ab3d is 0
# match_algorithm  = 'greedy'      # 'greedy' or 'pre_threshold' or None  or "reid-greedy" standford is greedy   ab3d is None

# mahalanobis_threshold = 11       # need to test       standford is 11

def linear_assignment(cost_matrix):
    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def poly_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
    return a * b * c


def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        try:
            hull_inter = ConvexHull(inter_p)
            return inter_p, hull_inter.volume
        except:
            return None, 0.0
    else:
        return None, 0.0


def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.
    Args:
        subjectPolygon: a list of (x,y) 2d points, any polygon.
        clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
        **points have to be counter-clockwise ordered**
    Return:
        a list of (x,y) vertex point for the intersection polygon.
    """

    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return (outputList)


def iou3d(corners1, corners2):
    ''' Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]
    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area / (area1 + area2 - inter_area)
    ymax = min(corners1[0, 1], corners2[0, 1])
    ymin = max(corners1[4, 1], corners2[4, 1])
    inter_vol = inter_area * max(0.0, ymax - ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def convert_3dbox_to_8corner(bbox3d_input):
    ''' 
      input x, y, z, rot_y, l, w, h -> 8 * 3
    '''
    # compute rotational matrix around yaw axis
    bbox3d = copy.copy(bbox3d_input)

    R = roty(bbox3d[3])

    # 3d bounding box dimensions  
    l = bbox3d[4]
    w = bbox3d[5]
    h = bbox3d[6]

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + bbox3d[0]
    corners_3d[1, :] = corners_3d[1, :] + bbox3d[1]
    corners_3d[2, :] = corners_3d[2, :] + bbox3d[2]

    return np.transpose(corners_3d)


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox3D, info, bbox_8coner, covariance_id=0, track_score=None, tracking_name='car',
                 use_angular_velocity=False, reid=None):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        if not use_angular_velocity:
            self.kf = KalmanFilter(dim_x=10, dim_z=7)
            self.kf.F = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # state transition matrix
                                  [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

            self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # measurement function,
                                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
        else:
            # with angular velocity
            self.kf = KalmanFilter(dim_x=11, dim_z=7)
            self.kf.F = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # state transition matrix
                                  [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

            self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # measurement function,
                                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])

        # Initialize the covariance matrix, see covariance.py for more details
        if covariance_id == 0:  # exactly the same as AB3DMOT baseline
            # self.kf.R[0:,0:] *= 10.   # measurement uncertainty
            self.kf.P[7:,
            7:] *= 1000.  # state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
            self.kf.P *= 10.

            # self.kf.Q[-1,-1] *= 0.01    # process uncertainty
            self.kf.Q[7:, 7:] *= 0.01
        elif covariance_id == 1:  # for kitti car, not supported
            covariance = Covariance(covariance_id, use_angular_velocity)
            self.kf.P = covariance.P
            self.kf.Q = covariance.Q
            self.kf.R = covariance.R
        elif covariance_id == 2:  # for nuscenes
            covariance = Covariance(covariance_id, use_angular_velocity)
            self.kf.P = covariance.P[tracking_name]
            self.kf.Q = covariance.Q[tracking_name]
            self.kf.R = covariance.R[tracking_name]
            if not use_angular_velocity:
                self.kf.P = self.kf.P[:-1, :-1]
                self.kf.Q = self.kf.Q[:-1, :-1]
        else:
            assert (False)

        self.kf.x[:7] = bbox3D.reshape((7, 1))

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1  # number of total hits including the first detection
        self.hit_streak = 1  # number of continuing hit considering the first detection
        self.first_continuing_hit = 1
        self.still_first = True
        self.age = 0
        self.info = info  # other info
        self.bbox = bbox_8coner  # late
        self.track_score = track_score
        self.tracking_name = tracking_name
        self.use_angular_velocity = use_angular_velocity
        self.reid = reid  # reid from detection

    def update(self, bbox3D, info, uncertainty=None, reid=None):
        """ 
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        # self.history = []
        self.hits += 1
        self.hit_streak += 1  # number of continuing hit
        if self.still_first:
            self.first_continuing_hit += 1  # number of continuing hit in the fist time

        ######################### orientation correction
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2  # make the theta still in the range
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        new_theta = bbox3D[3]
        if new_theta >= np.pi: new_theta -= np.pi * 2  # make the theta still in the range
        if new_theta < -np.pi: new_theta += np.pi * 2
        bbox3D[3] = new_theta

        predicted_theta = self.kf.x[3]
        if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(
                new_theta - predicted_theta) < np.pi * 3 / 2.0:  # if the angle of two theta is not acute angle
            self.kf.x[3] += np.pi
            if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2  # make the theta still in the range
            if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_theta > 0:
                self.kf.x[3] += np.pi * 2
            else:
                self.kf.x[3] -= np.pi * 2

        ######################### 

        if uncertainty is not None:
            R = self.kf.R
            R[0, 0] = uncertainty[0]
            R[1, 1] = uncertainty[1]
            R[3, 3] = uncertainty[2]
            self.kf.update(bbox3D, R=R)
        else:
            self.kf.update(bbox3D)

        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2  # make the theta still in the range
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
        self.info = info
        self.reid = reid  # 更新reid

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
            self.still_first = False
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.history[-1]

    def get_state_covariance(self):
        """
        Returns the current bounding box estimate.
        xest, yest, zest, θ, l, w, h
        """
        return np.trace(self.kf.P[:7, :7])

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x[:7].reshape((7,))


def angle_in_range(angle):
    '''
    Input angle: -2pi ~ 2pi
    Output angle: -pi ~ pi
    '''
    if angle > np.pi:
        angle -= 2 * np.pi
    if angle < -np.pi:
        angle += 2 * np.pi
    return angle


def diff_orientation_correction(det, trk):
    '''
    return the angle diff = det - trk
    if angle diff > 90 or < -90, rotate trk and update the angle diff
    '''
    diff = det - trk
    diff = angle_in_range(diff)
    if diff > np.pi / 2:
        diff -= np.pi
    if diff < -np.pi / 2:
        diff += np.pi
    diff = angle_in_range(diff)
    return diff


def greedy_match(distance_matrix):
    '''
    Find the one-to-one matching using greedy allgorithm choosing small distance
    distance_matrix: (num_detections, num_tracks)
    '''
    matched_indices = []

    num_detections, num_tracks = distance_matrix.shape
    distance_1d = distance_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_tracks, index_1d % num_tracks], axis=1)
    detection_id_matches_to_tracking_id = [-1] * num_detections
    tracking_id_matches_to_detection_id = [-1] * num_tracks
    for sort_i in range(index_2d.shape[0]):
        detection_id = int(index_2d[sort_i][0])
        tracking_id = int(index_2d[sort_i][1])
        if tracking_id_matches_to_detection_id[tracking_id] == -1 and detection_id_matches_to_tracking_id[
            detection_id] == -1:
            tracking_id_matches_to_detection_id[tracking_id] = detection_id
            detection_id_matches_to_tracking_id[detection_id] = tracking_id
            matched_indices.append([detection_id, tracking_id])

    matched_indices = np.array(matched_indices)
    return matched_indices


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.1, \
                                     use_mahalanobis=False, dets=None, trks=None, trks_S=None,
                                     mahalanobis_threshold=0.1, print_debug=False, match_algorithm='greedy', \
                                     doreid=False, det_reid=None, trk_reid=None, ):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    detections:  N x 8 x 3
    trackers:    M x 8 x 3
    dets: N x 7
    trks: M x 7
    trks_S: N x 7 x 7
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 8, 3), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    distance_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    reid_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    if use_mahalanobis:
        assert (dets is not None)
        assert (trks is not None)
        assert (trks_S is not None)

    if use_mahalanobis and print_debug:
        print('dets.shape: ', dets.shape)
        print('dets: ', dets)
        print('trks.shape: ', trks.shape)
        print('trks: ', trks)
        print('trks_S.shape: ', trks_S.shape)
        print('trks_S: ', trks_S)
        S_inv = [np.linalg.inv(S_tmp) for S_tmp in trks_S]  # 7 x 7
        S_inv_diag = [S_inv_tmp.diagonal() for S_inv_tmp in S_inv]  # 7
        print('S_inv_diag: ', S_inv_diag)

    if doreid:
        # 这样比for 循环快了很多
        reid_matrix = (torch.matmul(F.normalize(torch.stack(list(det_reid[:, -1]), 0), p=2, dim=1),
                                    F.normalize(torch.stack(list(trk_reid[:, -1]), 0), p=2,
                                                dim=1).t()).cpu().numpy())  # 注！！！这如果是用npairloss论文方法就要*0.5+1变成0-1后，否则是直接测就行只用0-1的范围
    # print(reid_matrix)
    # for d,det in enumerate(det_reid):
    #   for t,trk in enumerate(trk_reid):
    #     #print(det[-1],trk[-1])# for test ly test ok
    #     #iou_matrix[d,t] = cosine_similarity(list(det[-1].cpu()),list(trk[-1].cpu()))  #for only reid
    #     iou_matrix[d,t] = torch.matmul(det[-1],trk[-1]).cpu()  #for only reid
    # print(iou_matrix)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            if use_mahalanobis:
                S_inv = np.linalg.inv(trks_S[t])  # 7 x 7
                diff = np.expand_dims(dets[d] - trks[t], axis=1)  # 7 x 1
                # manual reversed angle by 180 when diff > 90 or < -90 degree
                corrected_angle_diff = diff_orientation_correction(dets[d][3], trks[t][3])
                diff[3] = corrected_angle_diff
                distance_matrix[d, t] = np.sqrt(np.matmul(np.matmul(diff.T, S_inv), diff)[0][0])  # (0,)

            else:
                # print(d, t)
                iou_matrix[d, t] = iou3d(det, trk)[0]  # det: 8 x 3, trk: 8 x 3
                distance_matrix = -iou_matrix + 1.0  # (0,1)
    if doreid:
        if use_mahalanobis:
            finaldis_matrix = ((
                                           distance_matrix / mahalanobis_threshold) - reid_matrix + 1.0) / 2.0  # (0,1.5)  认为更关注0-1区间内的ReID
        else:
            finaldis_matrix = 1.0 - ((iou_matrix + reid_matrix) / 2.0)  # (0, 1.5) IOU  +  reid
        if onlyreid:
            finaldis_matrix = -reid_matrix + 1.0  # (0, 2.0) reid  贪婪算法要取负数 ,否则为正
    else:
        finaldis_matrix = distance_matrix  # (0,)

    if match_algorithm == 'greedy':
        matched_indices = greedy_match(finaldis_matrix)

    elif match_algorithm == 'pre_threshold':
        if use_mahalanobis:
            to_max_mask = distance_matrix > mahalanobis_threshold
            finaldis_matrix[to_max_mask] = mahalanobis_threshold + 1
            matched_indices = linear_assignment(finaldis_matrix)  # houngarian algorithm
        elif onlyreid:
            to_max_mask = reid_matrix < reidlikthr
            finaldis_matrix[to_max_mask] = 2.0
            matched_indices = linear_assignment(finaldis_matrix)  # houngarian algorithm
        else:
            to_max_mask = iou_matrix < iou_threshold
            distance_matrix[to_max_mask] = 2.0
            iou_matrix[to_max_mask] = 0
            finaldis_matrix[to_max_mask] = 2.0  # if only reid ，需注释
            if doreid:
                to_max_mask = reid_matrix < reidlikthr  # 相当于做了2次prethr
                finaldis_matrix[to_max_mask] = 2.0

            matched_indices = linear_assignment(finaldis_matrix)  # houngarian algorithm
    else:
        matched_indices = linear_assignment(finaldis_matrix)  # houngarian algorithm

    if print_debug:
        print('distance_matrix.shape: ', distance_matrix.shape)
        print('distance_matrix: ', distance_matrix)
        print('matched_indices: ', matched_indices)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if len(matched_indices) == 0 or (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        match = True
        if use_mahalanobis:
            if distance_matrix[m[0], m[1]] > mahalanobis_threshold:
                match = False
        else:
            if (iou_matrix[m[0], m[1]] < iou_threshold):
                match = False
        if not match:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    if print_debug:
        print('matches: ', matches)
        print('unmatched_detections: ', unmatched_detections)
        print('unmatched_trackers: ', unmatched_trackers)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class AB3DMOT(object):
    def __init__(self, covariance_id=0, max_age=1, min_hits=1, tracking_name='car', use_angular_velocity=True,
                 tracking_nuscenes=False):
        """              
        observation: 
        before reorder: [h, w, l, x, y, z, rot_y]
        after reorder:  [x, y, z, rot_y, l, w, h]
        state:
        [x, y, z, rot_y, l, w, h, x_dot, y_dot, z_dot]
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        # self.reorder = [3, 4, 5, 6, 2, 1, 0]
        # self.reorder_back = [6, 5, 4, 0, 1, 2, 3]
        self.reorder = [0, 1, 2, 6, 3, 4, 5]  # before [x, y, z, l, w, h, theta] after: [x, y, z, theta, l, w, h]
        self.reorder_back = [0, 1, 2, 4, 5, 6, 3]
        self.covariance_id = covariance_id
        self.tracking_name = tracking_name
        self.use_angular_velocity = use_angular_velocity
        self.tracking_nuscenes = tracking_nuscenes

    def update(self, dets_all, match_distance, match_threshold, match_algorithm, seq_name=None, doreid=False):
        """
        Params:
        dets_all: dict
            dets - a numpy array of detections in the format [[x,y,z,theta,l,w,h],[x,y,z,theta,l,w,h],...]
            info: a array of other info for each det
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        dets, info, dets_8corner = dets_all['dets'], dets_all['scores'], dets_all[
            'corners']  # dets: N x 7, float numpy array

        if 'uncertainty' in dets_all.keys():
            det_uncertainty = dets_all['uncertainty']
        else:
            det_uncertainty = None

        # if det_uncertainty is not None:
        #   print(det_uncertainty.shape)

        ids = np.zeros((dets.shape[0], 1))  ## 生成一个记载 id 的矩阵
        # det_reids是dets+相应的reid特征
        ## 将相应的reid feature  写至文件
        dets = dets[:, self.reorder] # before [x, y, z, l, w, h, theta] after: [x, y, z, theta, l, w, h]

        self.frame_count += 1

        print_debug = False
        # if False and seq_name == '2f56eb47c64f43df8902d9f88aa8a019' and self.frame_count >= 25 and self.frame_count <= 30:
        #   print_debug = True
        #   print('self.frame_count: ', self.frame_count)
        if print_debug:
            for trk_tmp in self.trackers:
                print('trk_tmp.id: ', trk_tmp.id)

        trks = np.zeros((len(self.trackers), 7))  # N x 7 , #get predicted locations from existing trackers.
        # trk_reid = np.zeros((len(self.trackers),8),dtype=object)   #N x 8 ，trks+reid，dtype=object为了使array里能有其他array或str等多种元素， ly
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict().reshape((-1, 1))
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]
            # trk_reid[t] =[pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6],self.trackers[t].reid] #ly trackingreid      
            # if(np.any(np.isnan(pos))):
            if (pd.isnull(np.isnan(pos.all()))):  # ly
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        if print_debug:
            for trk_tmp in self.trackers:
                print('trk_tmp.id: ', trk_tmp.id)

        dets_8corner = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets]
        if len(dets_8corner) > 0:
            dets_8corner = np.stack(dets_8corner, axis=0)
        else:
            dets_8corner = []

        trks_8corner = [convert_3dbox_to_8corner(trk_tmp) for trk_tmp in trks]
        trks_S = [np.matmul(np.matmul(tracker.kf.H, tracker.kf.P), tracker.kf.H.T) + tracker.kf.R for tracker in
                  self.trackers]  # H*P*H' + R

        if len(trks_8corner) > 0:
            trks_8corner = np.stack(trks_8corner, axis=0)
            trks_S = np.stack(trks_S, axis=0)

        if match_distance == 'iou':  # 如果距离是iou， 那么就
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, trks_8corner,
                                                                                       iou_threshold=match_threshold,
                                                                                       print_debug=print_debug,
                                                                                       match_algorithm=match_algorithm,
                                                                                       doreid=doreid)
        else:
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, trks_8corner,
                                                                                       use_mahalanobis=True, dets=dets,
                                                                                       trks=trks, trks_S=trks_S,
                                                                                       mahalanobis_threshold=match_threshold,
                                                                                       print_debug=print_debug,
                                                                                       match_algorithm=match_algorithm)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]  # a list of index
                if det_uncertainty is not None:
                    trk.update(dets[d, :][0], info[d, :][0], uncertainty=det_uncertainty[d, :][0], reid=None)
                else:
                    trk.update(dets[d, :][0], info[d, :][0], reid=None)
                detection_score = info[d, :][0][-1]
                trk.track_score = detection_score
                ids[d, 0] = trk.id

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:  # a scalar of index
            detection_score = info[i, :]
            track_score = detection_score[0]
            trk = KalmanBoxTracker(dets[i, :], info[i, :], dets_8corner[i, :], self.covariance_id, track_score,
                                   self.tracking_name, self.use_angular_velocity)

            if det_uncertainty is not None:  # 使用det_uncertainty 进行更新。  xyz raw 
                trk.kf.P[0, 0] = det_uncertainty[i, 0] * det_uncertainty[i, 0]
                trk.kf.P[1, 1] = det_uncertainty[i, 1] * det_uncertainty[i, 1]
                trk.kf.P[3, 3] = det_uncertainty[i, 2] * det_uncertainty[i, 2]
            self.trackers.append(trk)
            ids[i, 0] = trk.id

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            # print(trk.id,trk.reid,trk)
            ##输出到文件同一辆车的trackid和reidfe
            ### reid for writing to results
            '''
            with open('../reidferes/sumcar2dthr1-npa-reneg1.txt','a') as feou:
            
            feou.write('%d,car,%s;' % (trk.id,trk.reid))
            '''

            d = trk.get_state()  # bbox location
            d = d[self.reorder_back]

            if ((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
                ret.append(np.concatenate((d, [trk.id + 1], trk.info[:-1], [trk.track_score])).reshape(1,
                                                                                                       -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update >= self.max_age:
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)  # x, y, z, theta, l, w, h, ID, other info(这个版本里面只有score) , confidence
        return np.empty((0, 15 + 7))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="SORT demo")
    parser.add_argument("--mode")  # TODO: what is mode
    # parser.add_argument('--save_path', type=str)
    parser.add_argument(
        "--display",
        dest="display",
        help="Display online tracker output (slow) [False]",
        action="store_true",
    )
    # parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument(
        "--max_age",
        help="Maximum number of frames to keep alive a track without associated detections.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--min_hits",
        help="Minimum number of associated detections before track is initialised.",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3
    )
    parser.add_argument(
        "--det_logs_path", default='', type=str, help="Det logs path (to get the tracking input)"
    )
    parser.add_argument("--split", type=str, help="[test/val]")
    args = parser.parse_args()
    return args


# if __name__ == "__main__":
    # args = parse_args()
    # display = args.display
    # scene_idxes = [x for x in os.listdir(args.det_logs_path) if os.path.exists(os.path.join(args.det_logs_path, x))]
    # print(f'scenes to run: {scene_idxes}')

    # total_time = 0.0
    # total_frames = 0

    # for scene_idx in scene_idxes:
    #     root = os.path.join(args.det_logs_path, scene_idx)
    #     frame_idxs = sorted(list(set([int(x.split('_')[0]) for x in os.listdir(root)])))

    #     save_path = os.path.join(args.det_logs_path.replace('npy', 'track'), 'data')
    #     gt_save_path = os.path.join(args.det_logs_path.replace('npy', 'track'), 'gt', 'OPV2V-test')
    #     save_path_3d = os.path.join(args.det_logs_path.replace('npy', 'track_3D'), 'data')
    #     gt_save_path_3d = os.path.join(args.det_logs_path.replace('npy', 'track_3D'), 'gt', 'OPV2V-test')
    #     os.makedirs(save_path, exist_ok=True)
    #     os.makedirs(gt_save_path, exist_ok=True)
    #     os.makedirs(save_path_3d, exist_ok=True)
    #     os.makedirs(gt_save_path_3d, exist_ok=True)

    #     mot_tracker = AB3DMOT(
    #         covariance_id=0,
    #         use_angular_velocity=True
    #     )  # create instance of the SORT tracker

    #     # save prediction
    #     out_file = open(os.path.join(save_path, '{}.txt'.format(scene_idx)), "w")
    #     out_file_3d = open(os.path.join(save_path_3d, '{}.txt'.format(scene_idx)), "w")
    #     if len(frame_idxs) == 0:
    #         continue
    #     # save gt
    #     gt_out_folder = os.path.join(gt_save_path, '{}'.format(scene_idx), 'gt')
    #     gt_out_folder_3d = os.path.join(gt_save_path_3d, '{}'.format(scene_idx), 'gt')
    #     os.makedirs(gt_out_folder, exist_ok=True)
    #     os.makedirs(gt_out_folder_3d, exist_ok=True)
    #     gt_out_file = open(os.path.join(gt_out_folder, 'gt.txt'), "w")
    #     gt_out_file_3d = open(os.path.join(gt_out_folder_3d, 'gt.txt'), "w")

    #     seqinfo = "\n".join(
    #         [
    #             "[Sequence]",
    #             f"name={scene_idx}",
    #             "imDir=img1",
    #             "frameRate=5",
    #             f"seqLength={len(frame_idxs)}",
    #             "imWidth=255",
    #             "imHeight=255",
    #             "imExt=.jpg",
    #         ])
    #     with open(os.path.join(gt_save_path, '{}'.format(scene_idx), "seqinfo.ini"), "w") as f:
    #         f.write(seqinfo)
    #     print("Processing %s." % (root))
    #     for frame in frame_idxs:
    #         dets = np.load(os.path.join(root, '%04d_pred.npy' % frame))
    #         gts = np.load(os.path.join(root, '%04d_gt.npy' % frame))
    #         trans = np.load(os.path.join(root, '%04d_trans.npy' % frame))

    #         det_scores = dets[:, 1:2]
    #         det_corners = dets[:, 2:].reshape(-1, 8, 3)  # 8 corners

    #         # import ipdb; ipdb.set_trace()
    #         # keep_idx = np.where(det_scores>0.5)[0]
    #         # det_scores = det_scores[keep_idx]
    #         # det_corners = det_corners[keep_idx]

    #         gt_corners = gts[:, 2:].reshape(-1, 8, 3)  # 8 corners

    #         det_corners_3d = project_box3d(det_corners, trans)
    #         gt_corners_3d = project_box3d(gt_corners, trans)

    #         det_centers_3d = corner_to_center(det_corners_3d)  # (N, 7)
    #         gt_centers_3d = corner_to_center(gt_corners_3d)
    #         gt_centers = corner_to_center(gt_corners)
    #         dets_all = OrderedDict()
    #         dets_all = {
    #             'dets': det_centers_3d,
    #             'scores': det_scores,
    #             'corners': det_corners_3d
    #         }
    #         total_frames += 1

    #         start_time = time.time()
    #         trackers = mot_tracker.update(dets_all, match_distance='iou', match_threshold=0.1,
    #                                       match_algorithm='hungar')  # (x,y,z,l,w,h,theta,track_id,score)
    #         cycle_time = time.time() - start_time
    #         total_time += cycle_time

    #         track_corners_3d = boxes_to_corners_3d(trackers[:, :7], 'lwh')
    #         track_corners = project_box3d(track_corners_3d, np.linalg.inv(trans))
    #         track_centers = corner_to_center(track_corners)
    #         for d_idx, d_3d in enumerate(trackers):
    #             print(
    #                 "%d,%d,-1,-1,-1,-1,1,%.2f,%.2f,%.2f"
    #                 % (frame, d_3d[-2], d_3d[0], d_3d[1], d_3d[2]),
    #                 file=out_file,
    #             )
    #             d = track_centers[d_idx]
    #             print(
    #                 "%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f"
    #                 % (frame, d_3d[-2], d[0], d[1], d[2], d[3], d[4], d[5], d[6]),
    #                 file=out_file_3d,
    #             )

    #         for gt_idx, gt_3d in enumerate(gt_centers_3d):
    #             print(
    #                 "%d,%d,-1,-1,-1,-1,1,%.2f,%.2f,%.2f"
    #                 % (frame, gts[gt_idx][1], gt_3d[0], gt_3d[1], gt_3d[2]),
    #                 file=gt_out_file,
    #             )
    #             gt = gt_centers[gt_idx]
    #             print(
    #                 "%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f"
    #                 % (frame, gts[gt_idx][1], gt[0], gt[1], gt[2], gt[3], gt[4], gt[5], gt[6]),
    #                 file=gt_out_file_3d,
    #             )

    # print(
    #     "Total Tracking took: %.3f seconds for %d frames or %.1f FPS"
    #     % (total_time, total_frames, total_frames / total_time)
    # )
