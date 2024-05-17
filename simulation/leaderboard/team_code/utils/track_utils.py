import sys
import torch
import numpy as np
from collections import OrderedDict
import time

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


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


def project_box3d(box3d, transformation_matrix):
    """
    Project the 3d bounding box to another coordinate system based on the
    transfomration matrix.

    Parameters
    ----------
    box3d : torch.Tensor or np.ndarray
        3D bounding box, (N, 8, 3)

    transformation_matrix : torch.Tensor or np.ndarray
        Transformation matrix, (4, 4)

    Returns
    -------
    projected_box3d : torch.Tensor
        The projected bounding box, (N, 8, 3)
    """
    assert transformation_matrix.shape == (4, 4)
    box3d, is_numpy = \
        check_numpy_to_torch(box3d)
    transformation_matrix, _ = \
        check_numpy_to_torch(transformation_matrix)

    # (N, 3, 8)
    box3d_corner = box3d.transpose(1, 2)
    # (N, 1, 8)
    torch_ones = torch.ones((box3d_corner.shape[0], 1, 8))
    torch_ones = torch_ones.to(box3d_corner.device)
    # (N, 4, 8)
    box3d_corner = torch.cat((box3d_corner, torch_ones),
                             dim=1)
    # (N, 4, 8)
    projected_box3d = torch.matmul(transformation_matrix,
                                   box3d_corner)
    # (N, 8, 3)
    projected_box3d = projected_box3d[:, :3, :].transpose(1, 2)

    return projected_box3d if not is_numpy else projected_box3d.numpy()


def corner_to_center(corner3d, order='lwh'):
    """
    Convert 8 corners to x, y, z, dx, dy, dz, yaw.
    yaw in radians

    Parameters
    ----------
    corner3d : np.ndarray
        (N, 8, 3)

    order : str, for output.
        'lwh' or 'hwl'

    Returns
    -------
    box3d : np.ndarray
        (N, 7)
    """
    assert corner3d.ndim == 3
    batch_size = corner3d.shape[0]

    xyz = np.mean(corner3d[:, [0, 3, 5, 6], :], axis=1)
    h = abs(np.mean(corner3d[:, 4:, 2] - corner3d[:, :4, 2], axis=1,
                    keepdims=True))
    l = (np.sqrt(np.sum((corner3d[:, 0, [0, 1]] - corner3d[:, 3, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 2, [0, 1]] - corner3d[:, 1, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 4, [0, 1]] - corner3d[:, 7, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 5, [0, 1]] - corner3d[:, 6, [0, 1]]) ** 2,
                        axis=1, keepdims=True))) / 4

    w = (np.sqrt(
        np.sum((corner3d[:, 0, [0, 1]] - corner3d[:, 1, [0, 1]]) ** 2, axis=1,
               keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 2, [0, 1]] - corner3d[:, 3, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 4, [0, 1]] - corner3d[:, 5, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 6, [0, 1]] - corner3d[:, 7, [0, 1]]) ** 2,
                        axis=1, keepdims=True))) / 4

    theta = (np.arctan2(corner3d[:, 1, 1] - corner3d[:, 2, 1],
                        corner3d[:, 1, 0] - corner3d[:, 2, 0]) +
             np.arctan2(corner3d[:, 0, 1] - corner3d[:, 3, 1],
                        corner3d[:, 0, 0] - corner3d[:, 3, 0]) +
             np.arctan2(corner3d[:, 5, 1] - corner3d[:, 6, 1],
                        corner3d[:, 5, 0] - corner3d[:, 6, 0]) +
             np.arctan2(corner3d[:, 4, 1] - corner3d[:, 7, 1],
                        corner3d[:, 4, 0] - corner3d[:, 7, 0]))[:,
            np.newaxis] / 4

    if order == 'lwh':
        return np.concatenate([xyz, l, w, h, theta], axis=1).reshape(
            batch_size, 7)
    elif order == 'hwl':
        return np.concatenate([xyz, h, w, l, theta], axis=1).reshape(
            batch_size, 7)
    else:
        sys.exit('Unknown order')

def load_det(preds, scores, trans, frame_id=0, gts=None, gt_idx=None):
    '''
    Functionality:
        Preprecess the detector's output to the format Tracker need
    Inputs:
        preds: (N1, 8, 3) predicted 3D object's 8 corners 
        scores: (N1, 1)  predicted 3D object's scores
        trans: (1, 4, 4)    transformation matrixs from ego to world
        gts: (N, 8, 3)  GT 3D object's 8 corners
        gt_idx: (N, 1)  GT 3D object's id
    Outputs:
        dets: (N1, 26)
        gts: (N, 26)
        trans: (1, 4, 4)
    '''
    dets = np.concatenate([np.ones([len(preds),1])*frame_id,scores.reshape([-1,1]), preds.reshape(-1,24)], axis=-1) # frame_id,conf,eight corner
    # gts = np.concatenate([np.ones([len(gts),1])*frame_id,gt_idx, gts.reshape(-1,24)], axis=-1) # frame_id,obj_id,eight corner
    gts = None
    return dets, gts, trans

def reset(mot_tracker):
    '''
    Functionality: 
        Reset the tracker at the begining of each scene
    '''
    mot_tracker.trackers = []
    mot_tracker.frame_count = 0
    return 

def predict(traj_predictor, traj_hist, num_valid=1, scale=50.0):
    '''
    Functionality:
        Predict the future trajectories of the valid agents given the histories with Eqmotion predictor
    Inputs:
        traj_hist: (1, N, T_h, 2)   Historical box centers(trajectories, xy) with length T_h, N is the padded agent number
        num_valid: (K)  Valid agent number
        scale: the resolution used in Eqmotion model
    Outputs:
        traj_pred: (1, N, T_f, 2) Predicted future box centers with length T_f
    '''
    if len(traj_hist.shape) == 2:
        traj_hist = traj_hist.unsqueeze(0).unsqueeze(0)
        num_valid = torch.tensor(1, dtype=torch.int).to(traj_hist.device).reshape(1)
    else:
        traj_hist = traj_hist.unsqueeze(0)

    traj_hist = traj_hist.type(torch.float32)
    traj_hist = traj_hist / scale
    vel = torch.zeros_like(traj_hist).to(traj_hist.device)
    vel[:,:,1:] = traj_hist[:,:,1:] - traj_hist[:,:,:-1]
    vel[:,:,0] = vel[:,:,1]
    nodes = torch.sqrt(torch.sum(vel ** 2, dim=-1)).detach()

    # (1, N, T_f, 2)
    # print('nodes', nodes.shape, 'loc', traj_hist.shape, 'vel', vel.shape, 'num_valid: ', num_valid)
    # print('traj_hist: ', traj_hist.shape, traj_hist)
    traj_pred, _ = traj_predictor(nodes, traj_hist.detach(), vel, num_valid)
    traj_pred = traj_pred * scale
    return traj_pred.detach().squeeze(2).squeeze(0) # (N, T_f, 2)

def predictor(mot_tracker, traj_predictor, past_length, trans, device, ego_trk_hist=None, ego_shape=None):
    '''
    Functionality:
        Fetch the track's history, and estimate the future trajectories.
    Input:
        mot_tracker: tracker
        ego_trk_hist: ego history: (T_f, 2)
        ego_shape: z,l,w,h,theta (5,) same format as track
        past_length: the historical timestamps
        device: the gpu id
    Output:
        ego_preds: (T_f, 8, 3) The predicted ego box
        ego_center_preds: (T_f, 2) The predicted ego centers
    '''
    
    ########## collect all the history #########
    trk_num = len(mot_tracker.trackers)
    trk_hist = np.zeros((trk_num, past_length, 2))
    trk_shape = np.zeros((trk_num, 5))
    
    for t in range(trk_num):
        # print('t/trk_num: ', t, trk_num)
        # print('len(mot_tracker.trackers[t].history):', len(mot_tracker.trackers[t].history))
        if len(mot_tracker.trackers[t].history) <= 1:
            continue
        hist = np.concatenate(mot_tracker.trackers[t].history, axis=-1).T[-past_length:,:2] # (T, 2)
        
        # print('hist: ', hist.shape, hist)
        shape = mot_tracker.trackers[t].get_state().reshape((-1,))[2:] # (5)
        trk_hist[t][-len(hist):] = hist
        trk_shape[t] = shape
        # print('hist_shape: ', shape.shape, shape)
        # print('traj_hist: ', trk_hist.shape, trk_hist)

    ########## Append ego trajectory #########
    if ego_trk_hist is not None:
        trk_num += 1
        # print('ego_trk_hist: ', ego_trk_hist.shape, ego_trk_hist)
        trk_hist = np.concatenate([ego_trk_hist[None,], trk_hist], axis=0)
        trk_shape = np.concatenate([ego_shape[None,], trk_shape], axis=0)
        # print('ego_trk_shape: ', trk_shape.shape, trk_shape)

    ################## predict ################
    trk_hist = torch.tensor(trk_hist).to(device)

    ############ Collaboration ##############
    trk_num = torch.tensor(trk_num, dtype=torch.int).to(device).reshape(1)
    ############ Without Collaboration ##############
    # trk_num = torch.tensor(1, dtype=torch.int).to(device).reshape(1)

    trk_pred = predict(traj_predictor, trk_hist, trk_num).cpu().numpy() # (N, T_f, 2)

    # print('trk_pred: ', trk_pred.shape, trk_pred)
    ###### transform pred to ego coord ########
    ego_center_preds = trk_pred[0]  # (T_f, 2)
    ego_preds = np.concatenate([ego_center_preds, np.repeat(trk_shape[0:1], ego_center_preds.shape[0], axis=0)], axis=-1)

    # print('ego_center_preds: ', ego_center_preds.shape, ego_center_preds)
    ###### transform pred to box ########
    # trks: x,y,z,l,w,h,theta
    ego_preds = ego_preds[:, mot_tracker.reorder_back]
    ego_preds_corners = boxes_to_corners_3d(ego_preds, order='lwh')
    ego_preds_corners_local = project_box3d(ego_preds_corners, np.linalg.inv(trans))
    ego_preds_centers_local = ego_preds_corners_local.mean(axis=1)[:,:2]
    # print('ego_preds_centers_local: ', ego_preds_centers_local.shape, ego_preds_centers_local)
    return ego_preds_corners_local, ego_preds_centers_local


def track_and_predict(mot_tracker, traj_predictor, past_length, preds, scores, trans, ego_trk_hist=None, ego_shape=None, frame_id=0):
    '''
    Functionality:
        Based on the detection results at each timestamp, track them into sequence and predict the future locations at each timestamp.
    Input:
        mot_tracker: tracker
        traj_predictor: traj_predictor eqmotion 
    Output:
        ego_preds: (T_f, 8, 3) The predicted ego box
        ego_center_preds: (T_f, 2) The predicted ego centers
        track_centers: (N, 7) The tracklet center
    '''
    total_time = 0.0
    total_frames = 0

    # reset(mot_tracker)
    # print('preds: ', preds.shape, preds)
        
    dets, gts, trans = load_det(preds, scores, trans, frame_id)

    det_scores = dets[:, 1:2]
    det_corners = dets[:, 2:].reshape(-1, 8, 3)  # 8 corners

    # import ipdb; ipdb.set_trace()
    # keep_idx = np.where(det_scores>0.5)[0]
    # det_scores = det_scores[keep_idx]
    # det_corners = det_corners[keep_idx]

    det_corners_3d = project_box3d(det_corners, trans)
    det_centers_3d = corner_to_center(det_corners_3d)  # (N, 7)

    # print('det_centers_3d: ', det_centers_3d.shape, det_centers_3d)
    dets_all = OrderedDict()
    dets_all = {
        'dets': det_centers_3d,
        'scores': det_scores,
        'corners': det_corners_3d
    }
    total_frames += 1

    start_time = time.time()
    trackers = mot_tracker.update(dets_all, match_distance='iou', match_threshold=0.1,
                                    match_algorithm='hungar')  # (x,y,z,l,w,h,theta,track_id,score)
    cycle_time = time.time() - start_time
    total_time += cycle_time

    track_corners_3d = boxes_to_corners_3d(trackers[:, :7], 'lwh')
    # print('trackers: ', trackers.shape, trackers)
    track_corners = project_box3d(track_corners_3d, np.linalg.inv(trans))
    track_centers = corner_to_center(track_corners)

    # print('track_centers: ', track_centers.shape, track_centers)

    ego_preds, ego_center_preds = predictor(mot_tracker, traj_predictor, past_length, trans, 'cuda:0', ego_trk_hist, ego_shape)

    return ego_center_preds[np.newaxis,]

