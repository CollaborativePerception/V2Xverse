"""
This module is designed for box alignment
It should be used for 1-round communication, maybe adapt to 2-round communication latter
i.e, collaborative agent send the full feature map and noisy pose once together

We will use g2o for pose graph optimization.
"""


from cv2 import threshold
from opencood.models.sub_modules.pose_graph_optim import PoseGraphOptimization2D
from opencood.utils.transformation_utils import pose_to_tfm
from opencood.utils import box_utils
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
import g2o
from icecream import ic
import os

DEBUG = False

def all_pair_l2(A, B):
    """ All pair L2 distance for A and B
    Args:
        A : torch.Tensor
            shape [N_A, D]
        B : torch.Tensor
            shape [N_B, D]
    Returns:
        C : torch.Tensor
            shape [N_A, N_B]
    """
    TwoAB = 2*A@B.T
    C = torch.sqrt(torch.sum(A * A, 1, keepdim=True).expand_as(TwoAB) \
        + torch.sum(B * B, 1, keepdim=True).T.expand_as(TwoAB) \
        - TwoAB)
    return C

def box_alignment_relative_sample(
            pred_corners_list,
            noisy_lidar_pose, 
            clean_lidar_pose=None, 
            uncertainty_list=None, 
            order='hwl', 
            landmark_SE2=True,
            adaptive_landmark=False):
    """ Perform box alignment for one sample. 
    Correcting the relative pose.

    Args:
        pred_corners_list: in each ego coordinate
            [[N_1, 8, 3], ..., [N_cav1, 8, 3]]

        clean_lidar_poses:
            [N_cav1, 6], in degree
        
        noisy_lidar_poses:
            [N_cav1, 6], in degree

        uncertainty_list:
            [[N_1, 3], [N_2, 3], ..., [N_cav1, 3]]

        landmark_SE2:
            if True, the landmark is SE(2), otherwise R^2
        
        adaptive_landmark: (when landmark_SE2 = True)
            if True, landmark will turn to R^2 if yaw angles differ a lot

    Returns: 
        refined_lidar_poses: np.ndarray
            [N_cav1, 3], 
    """

    ## first transform point from ego coordinate to world coordinate, using lidar_pose.
    N = noisy_lidar_pose.shape[0]
    device = pred_corners_list[0].device
    lidar_pose_noisy_tfm = pose_to_tfm(noisy_lidar_pose, dof=6)

    pred_corners_world_list = \
        [box_utils.project_box3d(pred_corners_list[i], lidar_pose_noisy_tfm[i]) for i in range(N)]  # [[N1, 8, 3], [N2, 8, 3],...]
    pred_box3d_list = \
        [box_utils.corner_to_center_torch(corner, order).to(device) for corner in pred_corners_list]   # [[N1, 7], [N2, 7], ...], angle in radius
    pred_box3d_world_list = \
        [box_utils.corner_to_center_torch(corner, order).to(device) for corner in pred_corners_world_list]   # [[N1, 7], [N2, 7], ...], angle in radius

    pred_center_list = \
        [torch.mean(corner_tensor, dim=[1]) for corner_tensor in pred_corners_list] # [[N1,3], [N2,3], ...]

    pred_center_world_list = \
        [pred_box3d_world[:,:3] for pred_box3d_world in pred_box3d_world_list]

    pred_yaw_world_list = \
        [pred_box3d[:, 6] for pred_box3d in pred_box3d_world_list]

    pred_len = \
        [pred_center.shape[0] for pred_center in pred_center_list] 

    


    box_idx_to_agent = []
    for i in range(N):
        box_idx_to_agent += [i] * pred_len[i] 
    

    pred_center_cat = torch.cat(pred_center_list, dim=0)   # [sum(pred_box), 3]
    pred_center_world_cat = torch.cat(pred_center_world_list, dim=0)  # [sum(pred_box), 3]
    pred_box3d_cat = torch.cat(pred_box3d_list, dim=0)  # [sum(pred_box), 7]
    pred_yaw_world_cat = torch.cat(pred_yaw_world_list)  # [sum(pred_box)]


    w_a = 1.6 # width of anchor
    l_a = 3.9 # length of anchor
    d_a_square = w_a ** 2 + l_a ** 2 # anchor's diag


    if uncertainty_list is not None:
        pred_log_sigma2_cat = torch.cat(uncertainty_list)
        pred_certainty_cat = torch.exp(-pred_log_sigma2_cat)
        pred_certainty_cat[:,:2] /= d_a_square # sigma_delta_x -> sigma_x. 


    pred_center_world_cat_cpu = pred_center_world_cat.cpu() # if use gpu, it will get nan.
    pred_center_allpair_dist = all_pair_l2(pred_center_world_cat_cpu, pred_center_world_cat_cpu) # [sum(pred_box), sum(pred_box)]


    # let pair from one vehicle be max distance
    MAX_DIST = 10000
    cum = 0
    for i in range(N):
        pred_center_allpair_dist[cum: cum + pred_len[i], cum: cum +pred_len[i]] = MAX_DIST
        cum += pred_len[i]


    cluster_id = N # let the vertex id of object start from N
    cluster_dict = OrderedDict()
    remain_box = set(range(cum))
    thres = 0.75  # l2 distance within the threshold, can be considered as one object.
    for box_idx in range(cum): 

        if box_idx not in remain_box:  # already assigned
            continue
        within_thres_idx_tensor = (pred_center_allpair_dist[box_idx] < thres).nonzero().flatten()
        within_thres_idx_list = within_thres_idx_tensor.cpu().numpy().tolist()

        if len(within_thres_idx_list) == 0:  # if it's a single box
            continue

        # start from within_thres_idx_list, find new box added to the cluster
        explored = [box_idx]
        unexplored = [idx for idx in within_thres_idx_list if idx in remain_box]

        while unexplored:
            idx = unexplored[0]
            within_thres_idx_tensor = (pred_center_allpair_dist[idx] < thres).nonzero().flatten()
            within_thres_idx_list = within_thres_idx_tensor.cpu().numpy().tolist()
            for newidx in within_thres_idx_list:
                if (newidx not in explored) and (newidx not in unexplored) and (newidx in remain_box):
                    unexplored.append(newidx)
            unexplored.remove(idx)
            explored.append(idx)
        
        if len(explored) == 1: # it's a single box, neighbors have been assigned
            remain_box.remove(box_idx)
            continue
        
        cluster_box_idxs = explored

        cluster_dict[cluster_id] = OrderedDict()
        cluster_dict[cluster_id]['box_idx'] = [idx for idx in cluster_box_idxs]
        cluster_dict[cluster_id]['box_dist'] = [pred_center_cat[idx].norm() for idx in cluster_box_idxs]  # distance to observer 
        cluster_dict[cluster_id]['box_center_world'] = [pred_center_world_cat[idx] for idx in cluster_box_idxs]  # coordinate in world, [3,]
        cluster_dict[cluster_id]['box_yaw'] = [pred_yaw_world_cat[idx] for idx in cluster_box_idxs]

        yaw_var = torch.var(torch.as_tensor(cluster_dict[cluster_id]['box_yaw']), unbiased=False)
        
        if landmark_SE2:
            if adaptive_landmark and yaw_var > 0.2:
                landmark = pred_center_world_cat[box_idx].clone()[:2]
            else:
                landmark = pred_center_world_cat[box_idx].clone()
                landmark[2] = pred_yaw_world_cat[box_idx]
        else:
            landmark = pred_center_world_cat[box_idx].clone()[:2]


        cluster_dict[cluster_id]['landmark'] = landmark.cpu().numpy()  # [x, y, yaw] or [x, y]
        cluster_dict[cluster_id]['landmark_SE2'] = True if landmark.shape[0] == 3 else False

        DEBUG = False
        if DEBUG:
            from icecream import ic
            ic(cluster_dict[cluster_id]['box_idx'])
            ic(cluster_dict[cluster_id]['box_center_world'])
            ic(cluster_dict[cluster_id]['box_yaw'])
            ic(cluster_dict[cluster_id]['landmark'])
        

        cluster_id += 1
        for idx in cluster_box_idxs:
            remain_box.remove(idx)

    vertex_num = cluster_id
    agent_num = N
    landmark_num = cluster_id - N
    # ic(agent_num)
    # ic(landmark_num)

    """
        Now we have clusters for objects. we can create pose graph.
        First we consider center as landmark.
        Maybe set corner as landmarks in the future.
    """
    pgo = PoseGraphOptimization2D(verbose=False)

    # Add agent to vertexs
    for agent_id in range(N):
        v_id = agent_id
        # notice lidar_pose use degree format, translate it to radians.
        pose_np = noisy_lidar_pose[agent_id, [0,1,4]].cpu().numpy()
        pose_np[2] = np.deg2rad(pose_np[2])  # radians
        v_pose = g2o.SE2(pose_np)
        
        if agent_id == 0:
            pgo.add_vertex(id=v_id, pose=v_pose, fixed=True)
        else:
            pgo.add_vertex(id=v_id, pose=v_pose, fixed=False)

    # Add landmark to vertexs
    for landmark_id in range(N, cluster_id):
        v_id = landmark_id
        landmark = cluster_dict[landmark_id]['landmark'] # (3,) or (2,)
        landmark_SE2 = cluster_dict[landmark_id]['landmark_SE2']

        if landmark_SE2:
            v_pose = g2o.SE2(landmark)
        else:
            v_pose = landmark

        pgo.add_vertex(id=v_id, pose=v_pose, fixed=False, SE2=landmark_SE2)

    # Add agent-landmark edge to edge
    for landmark_id in range(N, cluster_id):
        landmark_SE2 = cluster_dict[landmark_id]['landmark_SE2']

        for box_idx in cluster_dict[landmark_id]['box_idx']:
            agent_id = box_idx_to_agent[box_idx]
            if landmark_SE2:
                e_pose = g2o.SE2(pred_box3d_cat[box_idx][[0,1,6]].cpu().numpy().astype(np.float64))
                info = np.identity(3, dtype=np.float64)
                if uncertainty_list is not None:
                    info[[0,1,2],[0,1,2]] = pred_certainty_cat[box_idx].cpu().numpy()
            else:
                e_pose = pred_box3d_cat[box_idx][[0,1]].cpu().numpy().astype(np.float64)
                info = np.identity(2, dtype=np.float64)
                if uncertainty_list is not None:
                    info[[0,1],[0,1]] = pred_certainty_cat[box_idx][:2].cpu().numpy()


            pgo.add_edge(vertices=[agent_id, landmark_id], measurement=e_pose, information=info, SE2=landmark_SE2)
    
    pgo.optimize()

    pose_new_list = []
    for agent_id in range(N):
        # print(pgo.get_pose(agent_id).vector())
        pose_new_list.append(pgo.get_pose(agent_id).vector())

    refined_pose = np.array(pose_new_list)
    refined_pose[:,2] = np.rad2deg(refined_pose[:,2])  # rad -> degree, same as source

    return refined_pose

def box_alignment_sample(pred_corners_list, lidar_poses_for_tfm, noisy_lidar_poses, uncertainty_list=None, order='hwl'):
    """ Perform box alignment for one sample.
    Args:
        pred_corners_list: in each ego coordinate
            [[N_1, 8, 3], ..., [N_cav1, 8, 3]]

        lidar_poses:
            [N_cav1, 6] , in degree

        scores_list:
            [[N_1, 3], [N_2, 3], ..., [N_cav1, 3]]

    Returns: 
        refined_lidar_poses: np.ndarray
            [N_cav1, 3], 
    """
    
    ## first transform point from ego coordinate to world coordinate, using lidar_pose.
    lidar_poses = lidar_poses_for_tfm
    N = lidar_poses.shape[0]
    device = pred_corners_list[0].device
    lidar_pose_tfm = pose_to_tfm(lidar_poses, dof=6)  # Tw_c



    pred_corners_world_list = \
        [box_utils.project_box3d(pred_corners_list[i], lidar_pose_tfm[i]) for i in range(N)]  # [[N1, 8, 3], [N2, 8, 3],...]
    pred_box3d_list = \
        [box_utils.corner_to_center_torch(corner, order).to(device) for corner in pred_corners_list]   # [[N1, 7], [N2, 7], ...], angle in radius
    pred_box3d_world_list = \
        [box_utils.corner_to_center_torch(corner, order).to(device) for corner in pred_corners_world_list]   # [[N1, 7], [N2, 7], ...], angle in radius

    pred_center_list = \
        [torch.mean(corner_tensor, dim=[1]) for corner_tensor in pred_corners_list] # [[N1,3], [N2,3], ...]

    pred_center_world_list = \
        [pred_box3d_world[:,:3] for pred_box3d_world in pred_box3d_world_list]

    pred_yaw_world_list = \
        [pred_box3d[:, 6] for pred_box3d in pred_box3d_world_list]

    pred_len = \
        [pred_center.shape[0] for pred_center in pred_center_list] 


    box_idx_to_agent = []
    for i in range(N):
        box_idx_to_agent += [i] * pred_len[i] 

    if DEBUG:
        vis_corners_list(pred_corners_world_list,filename="/GPFS/rhome/yifanlu/OpenCOOD/box_align_items/gt_box_noisy_pose.png")
    

    pred_center_cat = torch.cat(pred_center_list, dim=0)   # [sum(pred_box), 3]
    pred_center_world_cat = torch.cat(pred_center_world_list, dim=0)  # [sum(pred_box), 3]
    pred_box3d_cat = torch.cat(pred_box3d_list, dim=0)  # [sum(pred_box), 7]
    pred_yaw_world_cat = torch.cat(pred_yaw_world_list)  # [sum(pred_box)]

    pred_center_world_cat_cpu = pred_center_world_cat.cpu() # if use gpu, it will get nan.
    pred_center_allpair_dist = all_pair_l2(pred_center_world_cat_cpu, pred_center_world_cat_cpu) # [sum(pred_box), sum(pred_box)]


    # let pair from one vehicle be max distance
    MAX_DIST = 10000
    cum = 0
    for i in range(N):
        pred_center_allpair_dist[cum: cum + pred_len[i], cum: cum +pred_len[i]] = MAX_DIST
        cum += pred_len[i]


    cluster_id = N # let the vertex id of object start from N
    cluster_dict = OrderedDict()
    remain_box = set(range(cum))
    thres = 1  # l2 distance within the threshold, can be considered as one object.
    for box_idx in range(cum):
        if box_idx not in remain_box:  # already assigned
            continue
        within_thres_idx_tensor = (pred_center_allpair_dist[box_idx] < thres).nonzero().flatten()
        within_thres_idx_list = within_thres_idx_tensor.cpu().numpy().tolist()

        if len(within_thres_idx_list) == 0:  # if it's a single box
            continue

        # start from within_thres_idx_list, find new box added to the cluster
        explored = [box_idx]
        unexplored = [idx for idx in within_thres_idx_list if idx in remain_box]

        while unexplored:
            idx = unexplored[0]
            within_thres_idx_tensor = (pred_center_allpair_dist[idx] < thres).nonzero().flatten()
            within_thres_idx_list = within_thres_idx_tensor.cpu().numpy().tolist()
            for newidx in within_thres_idx_list:
                if (newidx not in explored) and (newidx not in unexplored) and (newidx in remain_box):
                    unexplored.append(newidx)
            unexplored.remove(idx)
            explored.append(idx)
        
        if len(explored) == 1: # it's a single box, neighbors have been assigned
            remain_box.remove(box_idx)
            continue
        
        cluster_box_idxs = explored

        cluster_dict[cluster_id] = OrderedDict()
        cluster_dict[cluster_id]['box_idx'] = [idx for idx in cluster_box_idxs]
        cluster_dict[cluster_id]['box_dist'] = [pred_center_cat[idx].norm() for idx in cluster_box_idxs]  # distance to observer 
        cluster_dict[cluster_id]['box_center_world'] = [pred_center_world_cat[idx] for idx in cluster_box_idxs]  # coordinate in world, [3,]
        cluster_dict[cluster_id]['box_yaw'] = [pred_yaw_world_cat[idx] for idx in cluster_box_idxs]


        box_dist = torch.as_tensor(cluster_dict[cluster_id]['box_dist']).to(device)
        box_weight = F.normalize(1/box_dist, p=1, dim=0) # [n]
        centers = torch.stack(cluster_dict[cluster_id]['box_center_world'], dim=0) # [n, 3]
        yaws = torch.stack(cluster_dict[cluster_id]['box_yaw'])  # [n]

        weighted_center = torch.sum(box_weight.unsqueeze(-1) * centers, dim=0) # [3,]
        weighted_yaw = torch.sum(box_weight * yaws) # [1,]

        weighted_center[2] = weighted_yaw  # just replace z to yaw

        cluster_dict[cluster_id]['se2'] = weighted_center  # [x, y, yaw]

        # DEBUG = True
        if DEBUG:
            from icecream import ic
            ic(cluster_dict[cluster_id]['box_idx'])
            ic(centers)
            ic(yaws)
            ic(box_weight)
            ic(cluster_dict[cluster_id]['se2'])

        cluster_dict[cluster_id].pop('box_dist')
        cluster_dict[cluster_id].pop('box_center_world')
        cluster_dict[cluster_id].pop('box_yaw')

        cluster_id += 1
        for idx in cluster_box_idxs:
            remain_box.remove(idx)

    vertex_num = cluster_id
    agent_num = N
    landmark_num = cluster_id - N
    # ic(agent_num)
    # ic(landmark_num)

    """
        Now we have clusters for objects. we can create pose graph.
        First we consider center as landmark.
        Maybe set corner as landmarks in the future.
    """
    pgo = PoseGraphOptimization2D(verbose=False)
    if DEBUG:
        pgo = PoseGraphOptimization2D(verbose=True)
    # Add agent to vertexs
    for agent_id in range(N):
        v_id = agent_id
        # notice lidar_pose use degree format, translate it to radius.
        # pose_np = lidar_poses[agent_id, [0,1,4]].cpu().numpy()
        pose_np = noisy_lidar_poses[agent_id, [0,1,4]].cpu().numpy()
        pose_np[2] = np.deg2rad(pose_np[2])  # radius
        v_pose = g2o.SE2(pose_np)
        # if agent_id == 0 and DEBUG:
        #     pgo.add_vertex(id=v_id, pose=v_pose, fixed=True)
        # else:
        pgo.add_vertex(id=v_id, pose=v_pose, fixed=False)

    # Add landmark to vertexs
    for landmark_id in range(N, cluster_id):
        v_id = landmark_id
        v_pose = g2o.SE2(cluster_dict[landmark_id]['se2'].cpu().numpy())
        pgo.add_vertex(id=v_id, pose=v_pose, fixed=False)

    # Add agent-landmark edge to edge
    for landmark_id in range(N, cluster_id):
        for box_idx in cluster_dict[landmark_id]['box_idx']:
            agent_id = box_idx_to_agent[box_idx]
            e_pose = g2o.SE2(pred_box3d_cat[box_idx][[0,1,6]].cpu().numpy())
            pgo.add_edge(vertices=[agent_id, landmark_id], measurement=e_pose, information=np.identity(3))
    
    pgo.optimize()

    pose_new_list = []
    for agent_id in range(N):
        # print(pgo.get_pose(agent_id).vector())
        pose_new_list.append(pgo.get_pose(agent_id).vector())

    refined_pose = np.array(pose_new_list)
    refined_pose[:,2] = np.rad2deg(refined_pose[:,2])  # rad -> degree, same as source

    return refined_pose

def box_alignment(pred_corner3d_list, uncertainty_list, lidar_poses, record_len, proj_first=False):
    """
    Args:
        pred_corner3d_list: list of tensors, with shape [[N1_object, 8, 3], [N2_object, 8, 3], ...,[N_sumcav_object, 8, 3]]
            box in each agent's coordinate. (proj_first=False)
        
        pred_box3d_list: not necessary
            list of tensors, with shape [[N1_object, 7], [N2_object, 7], ...,[N_sumcav_object, 7]]

        scores_list: list of tensor, [[N1_object,], [N2_object,], ...,[N_sumcav_object,]]
            box confidence score.

        lidar_poses: torch.Tensor [sum(cav), 6]

        record_len: torch.Tensor
    Returns:
        refined_lidar_pose: torch.Tensor [sum(cav), 6]
    """
    refined_lidar_pose = []
    start_idx = 0
    for b in record_len:
        refined_lidar_pose.append(
            torch.from_numpy(
                box_alignment_relative_sample(
                    pred_corner3d_list[start_idx: start_idx + b],
                    lidar_poses[start_idx: start_idx + b],
                    clean_lidar_pose=None,
                    uncertainty_list= None if uncertainty_list is None else uncertainty_list[start_idx: start_idx + b]
                )
            )
        )
        start_idx += b

    return torch.cat(refined_lidar_pose, dim=0)

def vis_corners_list(corner3d_list, filename="/GPFS/rhome/yifanlu/OpenCOOD/opencood/corners.png"):
    """
    Args:
        corner3d: list of  torch.Tensor, shape [N, 8, 3]

    """
    COLOR = ['red','springgreen','dodgerblue', 'darkviolet']
    box_idx = 0

    for idx in range(len(corner3d_list)):
        corner3d = corner3d_list[idx]
        if torch.is_tensor(corner3d):
            corner3d = corner3d.cpu().numpy()

        corner2d = corner3d[:,:4,:2]
        import matplotlib.pyplot as plt
        for i in range(corner2d.shape[0]):
            plt.scatter(corner2d[i,[0,1],0], corner2d[i,[0,1], 1], s=2, c=COLOR[idx])
            plt.plot(corner2d[i,[0,1,2,3,0],0], corner2d[i,[0,1,2,3,0], 1], linewidth=1, c=COLOR[idx])
            plt.text(corner2d[i,0,0], corner2d[i,0,1], s=str(box_idx), fontsize="xx-small")
            box_idx += 1
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.savefig(filename, dpi=400)
    plt.clf()

def vis_corners(corner3d, filename="/GPFS/rhome/yifanlu/OpenCOOD/opencood/corners.png"):
    """
    Args:
        corner3d: torch.Tensor, shape [N, 8, 3]

        box3d: torch.Tensor shape [N, 7]
    """
    if torch.is_tensor(corner3d):
        corner3d = corner3d.cpu().numpy()


    corner2d = corner3d[:,:4,:2]
    import matplotlib.pyplot as plt
    for i in range(corner2d.shape[0]):
        plt.scatter(corner2d[i,[0,1],0], corner2d[i,[0,1], 1], s=2)
        plt.plot(corner2d[i,[0,1,2,3,0],0], corner2d[i,[0,1,2,3,0], 1])
        # plt.text(corner2d[i,0,0], corner2d[i,0,1], s=f"{box3d[i,0]:.2f},{box3d[i,1]:.2f},{box3d[i,6]:.2f}", fontsize='xx-small')
    plt.axis('equal')
    plt.savefig(filename, dpi=300)
    plt.clf()

def vis_pose(lidar_poses):
    """
    Args:
        lidar_poses: torch.Tensor shape [N_, 6], x,y,z, roll, yaw, pitch
    """
    h = 1.56
    l = 3.9
    w = 1.6
    if torch.is_tensor(lidar_poses):
        lidar_poses = lidar_poses.cpu().numpy()

    box3d = np.zeros((lidar_poses.shape[0], 7))
    box3d[:,0] = lidar_poses[:,0]
    box3d[:,1] = lidar_poses[:,1]
    box3d[:,3] = h # hwl order
    box3d[:,4] = w
    box3d[:,5] = l
    box3d[:,6] = np.deg2rad(lidar_poses[:,4])  # degree -> radius

    corner3d = box_utils.boxes_to_corners_3d(box3d, order='hwl')
    vis_corners(corner3d, box3d, "/GPFS/rhome/yifanlu/OpenCOOD/opencood/pose_corners.png")

def test_pred_gt_box():
    gt_corners_list = torch.load("/GPFS/rhome/yifanlu/workspace/OpenCOOD/box_align_items/previous_items/gt_box_list.pt")
    data = torch.load("/GPFS/rhome/yifanlu/OpenCOOD/box_align_items/stage1_output_02/0.pt")
    pred_corner3d_list, pred_box3d_list, scores_list, record_len, lidar_pose, lidar_pose_clean = data

    lidar_pose_tfm = pose_to_tfm(lidar_pose, dof=6)
    lidar_pose_clean_tfm = pose_to_tfm(lidar_pose_clean, dof=6)  # Tw_c
    N = lidar_pose.shape[0]

    pred_corners_world_list = \
        [box_utils.project_box3d(pred_corner3d_list[i], lidar_pose_tfm[i]) for i in range(N)]  # [[N1, 8, 3], [N2, 8, 3],...]

    gt_corners_world_list = \
        [box_utils.project_box3d(gt_corners_list[i], lidar_pose_clean_tfm[i]) for i in range(N)]

    vis_corners_list([torch.cat(pred_corners_world_list, dim=0), torch.cat(gt_corners_world_list, dim=0)], filename="/GPFS/rhome/yifanlu/OpenCOOD/box_align_items/gt_box_pred_box.png")

    


def test_gt_boxes_world():
    data = torch.load("/GPFS/rhome/yifanlu/OpenCOOD/box_align_items/stage1_output_02/0.pt")
    pred_corner3d_list, pred_box3d_list, scores_list, record_len, lidar_pose, lidar_pose_clean = data
    
    gt_poses_tensor = lidar_pose_clean
    noisy_poses_tensor = lidar_pose

    gt_corners_list = torch.load("/GPFS/rhome/yifanlu/workspace/OpenCOOD/box_align_items/previous_items/gt_box_list.pt")
    
    # refined_poses = box_alignment_sample(gt_corners_list, noisy_poses_tensor, noisy_poses_tensor)
    refined_poses = box_alignment_relative_sample(pred_corner3d_list, noisy_poses_tensor, gt_poses_tensor)
    print("before:\n", noisy_poses_tensor.cpu().numpy()[:,[0,1,4]])
    
    print("after:\n", refined_poses)

    print("gt:\n", gt_poses_tensor.cpu().numpy()[:,[0,1,4]])

    # gt_corners_world_list = \
    #     [box_utils.project_box3d(gt_corners_list[i], lidar_pose_tfm[i]) for i in range(3)]  # [[N1, 8, 3], [N2, 8, 3],...]

    # vis_corners_list(gt_corners_world_list, filename="/GPFS/rhome/yifanlu/workspace/OpenCOOD/box_align_items/gt_corners.png")


def test_box_align_tmp():
    """
        This func input different noise_std pose (load from stored files).
        And run pose graph optimization, compare the localization error w/wo uncertainty/landmark SE2, etc.
    """
    noise_stds = ['02','04','06']
    items = ["16"]
    torch.set_printoptions(precision=3, sci_mode=False)
    np.set_printoptions(precision=3, suppress=True)
    for item in items:
        for noise_std in noise_stds:
            file_dir = f"/GPFS/rhome/yifanlu/workspace/OpenCOOD/box_align_items/stage1_output_{noise_std}_w_uncertainty/{item}.pt"
            data = torch.load(file_dir)
            pred_corner3d_list, pred_box3d_list, uncertainty_list, record_len, lidar_pose, lidar_pose_clean = data
            lidar_pose[0] = lidar_pose_clean[0]
            refined_pose_SE2 = box_alignment_relative_sample(pred_corner3d_list, lidar_pose_clean, lidar_pose, uncertainty_list=uncertainty_list, landmark_SE2=True)
            refined_pose = box_alignment_relative_sample(pred_corner3d_list, lidar_pose_clean, lidar_pose, uncertainty_list=uncertainty_list, landmark_SE2=False)
            # refined_pose = box_alignment_sample(pred_corner3d_list, lidar_pose, lidar_pose)
            lidar_pose_clean = lidar_pose_clean[:,[0,1,4]].cpu().numpy()
            print(f"noise std: {noise_std}: SE2")
            print(np.abs(refined_pose_SE2 - lidar_pose_clean))
            # print(f"PointXY")
            # print(np.abs(refined_pose - lidar_pose_clean))
            print(f"original error:")
            lidar_pose = lidar_pose[:,[0,1,4]].cpu().numpy()
            print(np.abs(lidar_pose - lidar_pose_clean))
            # print(refined_pose_w_u)
            # print(lidar_pose_clean) 


def test_box_align(noise_std="04", relative=True, use_uncertainty=False):
    from glob import glob
    data_dir = f"/GPFS/rhome/yifanlu/workspace/OpenCOOD/box_align_items/stage1_output_{noise_std}_w_uncertainty/*"
    trans_error_before_list = []
    rotate_error_before_list = []
    trans_error_after_list = []
    rotate_error_after_list = []

    full_files = glob(data_dir)
    for filename in full_files:
        data = torch.load(filename)
        
        if relative is False:
            pred_corner3d_list, pred_box3d_list, scores_list, record_len, lidar_pose, lidar_pose_clean = data
            refined_pose = box_alignment_sample(pred_corner3d_list, None, lidar_pose)
        elif relative is True:
            pred_corner3d_list, pred_box3d_list, uncertainty_list, record_len, lidar_pose, lidar_pose_clean = data
            lidar_pose[0] = lidar_pose_clean[0]
            # if not use_uncertainty:
            #     uncertainty_list = None
            refined_pose = box_alignment_relative_sample(pred_corner3d_list, lidar_pose_clean, lidar_pose, uncertainty_list=uncertainty_list)
            uncertainty_list = None
            refined_pose_wo_uncertainty = box_alignment_relative_sample(pred_corner3d_list, lidar_pose_clean, lidar_pose, uncertainty_list=uncertainty_list)

        lidar_pose = lidar_pose.cpu().numpy()[:,[0,1,4]]
        lidar_pose_clean = lidar_pose_clean.cpu().numpy()[:,[0,1,4]]
        np.set_printoptions(suppress=True, precision=4)
        print(lidar_pose[1:])
        print(refined_pose_wo_uncertainty[1:])
        print(refined_pose[1:])
        print(lidar_pose_clean[1:])
        print()

        error_before = np.abs(lidar_pose - lidar_pose_clean)
        error_after = np.abs(refined_pose - lidar_pose_clean)

        trans_error_before_list.append(np.mean(error_before[:,[0,1]]))
        rotate_error_before_list.append(np.mean(error_before[:,2]))

        trans_error_after_list.append(np.mean(error_after[:,[0,1]]))
        rotate_error_after_list.append(np.mean(error_after[:,2]))

    raise


    out_quantile_dict = {0.8:None, 0.5:None, 0.3:None}
    for q in out_quantile_dict.keys():
        out_quantile_dict[q] = (np.quantile(trans_error_before_list, q), 
                                    np.quantile(trans_error_after_list, q), 
                                    np.quantile(rotate_error_before_list, q),
                                    np.quantile(rotate_error_after_list, q))

    return out_quantile_dict
    # return np.mean(trans_error_before_list), np.mean(rotate_error_before_list), np.mean(trans_error_after_list), np.mean(rotate_error_after_list)

def main1():
    """
    This function test the box alignment performance on the subset of training set.
    """
    for noise in ['02', '04', '06']:
        out = test_box_align(noise, relative=True, use_uncertainty=True)
        for k,v in out.items():
            with open(f"/GPFS/rhome/yifanlu/workspace/OpenCOOD/box_align_items/rel_quantile{k*100}_{noise}_w_u.txt", "w") as f:
                f.write(f"trans error before: \t {v[0]}\n")
                f.write(f"trans error after:  \t {v[1]}\n\n")

                f.write(f"rotate error before: \t {v[2]}\n")
                f.write(f"rotate error after: \t {v[3]}\n")


        out = test_box_align(noise, relative=True, use_uncertainty=False)
        for k,v in out.items():
            with open(f"/GPFS/rhome/yifanlu/workspace/OpenCOOD/box_align_items/rel_quantile{k*100}_{noise}_wo_u.txt", "w") as f:
                f.write(f"trans error before: \t {v[0]}\n")
                f.write(f"trans error after:  \t {v[1]}\n\n")

                f.write(f"rotate error before: \t {v[2]}\n")
                f.write(f"rotate error after: \t {v[3]}\n")


def vis_pose_graph(
            poses,
            pred_corner3d, 
            save_dir_path="/GPFS/rhome/yifanlu/OpenCOOD/box_align_items/pose_graph_vis",
            ):
    """
    Args:
        poses: list of np.ndarray
            each item is a pose . [pose_before, ..., pose_refined]

        pred_corner3d: list
            predicted box for each agent.

    """
    COLOR = ['red','springgreen','dodgerblue', 'darkviolet', 'orange']
    from opencood.utils.transformation_utils import get_relative_transformation

    if not os.path.exists(save_dir_path):
        os.mkdir(save_dir_path)

    for iter, pose in enumerate(poses):
        box_idx = 0
        # we first transform other agents' box to ego agent's coordinate
        relative_t_matrix = get_relative_transformation(pose)
        N = pose.shape[0]

        pred_corners3d_in_ego = [box_utils.project_box3d(pred_corner3d[i].cpu().numpy(), relative_t_matrix[i]) for i in range(N)]

        for agent_id in range(len(pred_corners3d_in_ego)):
            corner3d = pred_corners3d_in_ego[agent_id]
            agent_pos = relative_t_matrix[agent_id][:2,3] # agent's position in ego's coordinate
            if torch.is_tensor(corner3d):
                corner3d = corner3d.cpu().numpy()

            corner2d = corner3d[:,:4,:2]
            center2d = np.mean(corner2d, axis=1)
            import matplotlib.pyplot as plt
            for i in range(corner2d.shape[0]):
                plt.scatter(corner2d[i,[0,1],0], corner2d[i,[0,1], 1], s=2, c=COLOR[agent_id])
                plt.plot(corner2d[i,[0,1,2,3,0],0], corner2d[i,[0,1,2,3,0], 1], linewidth=1, c=COLOR[agent_id])
                plt.text(corner2d[i,0,0], corner2d[i,0,1], s=str(box_idx), fontsize="xx-small")
                # add a line connecting box center and agent.
                box_center = center2d[i] # [2,]
                connection_x = [agent_pos[0], box_center[0]]
                connection_y = [agent_pos[1], box_center[1]]
                # print(connection_x)
                # print(connection_y)
                # print()
                plt.plot(connection_x, connection_y,'--', linewidth=0.5, c=COLOR[agent_id], alpha=0.3)
                box_idx += 1
        
        filename = os.path.join(save_dir_path, f"{iter}.png")
        plt.gca().invert_yaxis()
        plt.axis('equal')
        plt.savefig(filename, dpi=400)
        plt.clf()

def vis_pose_graphs():
    noise_stds = ['02','04','06']
    items = ["53", "63", "73", "83"]
    torch.set_printoptions(precision=3, sci_mode=False)
    np.set_printoptions(precision=3, suppress=True)
    for item in items:
        for noise_std in noise_stds:
            file_dir = f"/GPFS/rhome/yifanlu/workspace/OpenCOOD/box_align_items/stage1_output_{noise_std}_w_uncertainty/{item}.pt"
            data = torch.load(file_dir)
            pred_corner3d_list, pred_box3d_list, uncertainty_list, record_len, lidar_pose, lidar_pose_clean = data
            lidar_pose[0] = lidar_pose_clean[0]
            refined_pose_SE2 = box_alignment_relative_sample(pred_corner3d_list, lidar_pose_clean, lidar_pose, uncertainty_list=uncertainty_list, landmark_SE2=True)
            ## visualize pred_corner3d with refined_pose. We can set different iteration to animate
            save_dir_path = f"/GPFS/rhome/yifanlu/OpenCOOD/box_align_items/pose_graph_vis/{item}_{noise_std}"
            poses = [lidar_pose.cpu().numpy(), refined_pose_SE2]
            vis_pose_graph(poses, pred_corner3d_list, save_dir_path)




def main2():
    pass

if __name__ == "__main__":
    # vis_pose_graphs()
    test_box_align_tmp()
    # main1()
    # test_gt_boxes_world()