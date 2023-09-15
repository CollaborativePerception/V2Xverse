import os
import json
import math
import torch
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from utils.det_utils import mask_ego_points

from simple_plot3d import Canvas_3D, Canvas_BEV
import imageio


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

    mask_loc = np.where(mask==True)

    return p_in_box, mask_loc


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


def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:, 2] = 1

    c, s = np.cos(r1), np.sin(r1)
    # 顺时针旋转r1角度，r1车辆坐标转换到world frame
    r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    # world frame -> r2 frame
    # if r1==r2, do nothing
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T
    # reset z-coordinate
    out[:, 2] = xyz[:, 2]

    return out



def filter_one_frame(
    route = "routes_town05_test_w0_02_16_09_22_21",
    ego = "1",
    frame = 0
    ):

    # frame = 282

    bbs_file_path = "{}/ego_vehicle_{}/3d_bbs/{:0>4d}.npy".format(route, ego, frame)  # root_path, weather, 
    bbs = np.load(bbs_file_path, allow_pickle=True)
    # bbs_vehicles = np.array(bbs.item()['vehicles'] + bbs.item()['pedestrians'] + bbs.item()['traffic_lights'])
    bbs_vehicles = np.array(bbs.item()['vehicles'] + bbs.item()['pedestrians'])
    
    actors_file_path = "{}/ego_vehicle_{}/actors_data/{:0>4d}.json".format(route, ego, frame)  # root_path, weather,
    actors_data = json.load(open(actors_file_path))
    # print(bbs)
    # raise ValueError
    # print(bbs_vehicles.shape)
    # N, 3, 3, similar to actors_data, 

    lidar_file_path = "{}/ego_vehicle_{}/lidar/{:0>4d}.npy".format(route, ego, frame)
    lidar_unprocessed = np.load(lidar_file_path, allow_pickle=True)
    lidar_unprocessed = lidar_unprocessed[..., :3]

    lidar_unprocessed[:, 1] *= -1
    full_lidar = lidar_unprocessed

    measurements_file_path = "{}/ego_vehicle_{}/measurements/{:0>4d}.json".format(route, ego, frame)
    measurements = json.load(open(measurements_file_path))

    ###################
    #### ego pose and orientation
    ###################
    ego_x = measurements["lidar_pose_x"]
    ego_y = measurements["lidar_pose_y"]
    ego_z = measurements["lidar_pose_z"]
    ego_theta = measurements["theta"] + np.pi # !note, plus pi in extra.
    # rotate counterclockwise by ego_theta
    R = np.array(
        [
            [np.cos(ego_theta), -np.sin(ego_theta)],
            [np.sin(ego_theta), np.cos(ego_theta)],
        ]
    )
    
    ###################
    #### transform the 3d-bbs from world frame to lidar frame
    ###################

    texts = []
    lidar_visible = []
    for _id in actors_data.keys():
        if actors_data[_id]["tpe"] == 2:
            continue  # FIXME donot add traffix light
        if actors_data[_id]['lidar_visible'] == 1:
            lidar_visible.append(1)
        else:
            lidar_visible.append(0)
        texts.append(str(_id))
        raw_loc = actors_data[_id]['loc'][:2]
        new_loc = R.T.dot(np.array([raw_loc[0] - ego_x , raw_loc[1] - ego_y]))
        new_loc[1] = -new_loc[1]
        actors_data[_id]['loc'][:2] = np.array(new_loc)
        if actors_data[_id]["tpe"] == 1:
            actors_data[_id]['loc'][2] -= actors_data[_id]['box'][2]
        actors_data[_id]['loc'][2] -= (ego_z)
        raw_ori = actors_data[_id]['ori'][:2]
        new_ori = R.T.dot(np.array([raw_ori[0], raw_ori[1]]))
        actors_data[_id]['ori'][:2] = np.array(new_ori)
    
    boxes_corner = [] # pose and orientation of the box,
            # (x, y, z, scale_x, scale_y, scale_z, yaw)
    boxes_draw = []
    for _id in actors_data.keys():
        if actors_data[_id]["tpe"] == 2:
            continue  # FIXME donot add traffix light
        cur_data = actors_data[_id]
        yaw = get_yaw_angle(cur_data['ori'][:2])
        boxes_draw.append(cur_data['loc']+ [i*2 for i in cur_data['box']] + [yaw])
        cur_data['loc'][2] += cur_data['box'][2]
        boxes_corner.append(cur_data['loc']+ [i*2 for i in cur_data['box']] + [yaw])
    boxes_corner = np.array(boxes_corner)   
    boxes_draw = np.array(boxes_draw)   

    re_compute_visibility = False

    colors = np.zeros(np.shape(full_lidar))
    colors[:,:] = (0,255,255)

    if re_compute_visibility:
        lidar_visible = []
        if len(boxes_corner)>0:
            corners = boxes_to_corners_3d(boxes_corner, order='lwh')

            # corners = boxes_to_corners_3d(boxes)
            lidar_visible = []
            # print(lidar_unprocessed[:20])
            for N in range(corners.shape[0]):
                num_lidar_points, mask_loc = get_points_in_rotated_box_3d(full_lidar, corners[N])

                colors[mask_loc] = (255, 0, 255)
                # print(len(num_lidar_points))
                if len(num_lidar_points)>8:
                    lidar_visible += [1]
                else:
                    lidar_visible += [0]
    # print(lidar_visible)


    canvas_3d = Canvas_3D(
        canvas_shape=(600, 800),
        camera_center_coords = (0, 15, 10),
        camera_focus_coords = (0, 0, -4)
    )
    canvas_xy, valid_mask = canvas_3d.get_canvas_coords(full_lidar)
    canvas_3d.draw_canvas_points(canvas_xy[valid_mask], colors=colors[valid_mask], colors_operand=-np.linalg.norm(full_lidar[valid_mask], axis=1), radius=1)  # colors=(0, 255, 255)
    # canvas_xy, valid_mask = canvas_3d.get_canvas_coords(corners.reshape(-1, 3))
    # canvas_3d.draw_canvas_points(canvas_xy, colors=(255, 0, 0), radius=5)
    # canvas_3d.draw_boxes(boxes_draw, texts=texts)


    canvas_3d_visible = Canvas_3D(
        canvas_shape=(600, 800),
        camera_center_coords = (0, 15, 10),
        camera_focus_coords = (0, 0, -4)
    )
    canvas_xy, valid_mask = canvas_3d_visible.get_canvas_coords(full_lidar)
    canvas_3d_visible.draw_canvas_points(canvas_xy[valid_mask], colors=colors[valid_mask], colors_operand=-np.linalg.norm(full_lidar[valid_mask], axis=1), radius=1)
    for N in range(len(lidar_visible)):
        if lidar_visible[N]==0:
            canvas_3d_visible.draw_boxes(boxes_draw[N:N+1], texts=texts[N:N+1], colors=(255, 0, 0))
        else:
            canvas_3d_visible.draw_boxes(boxes_draw[N:N+1], texts=texts[N:N+1], colors=(255, 255, 255))


    canvas_2d_visible = Canvas_BEV(canvas_shape=(800, 800),
                            canvas_x_range=(-40, 40),
                            canvas_y_range=(-40, 40))
    canvas_xy, valid_mask = canvas_2d_visible.get_canvas_coords(full_lidar)
    canvas_2d_visible.draw_canvas_points(canvas_xy[valid_mask], colors=colors[valid_mask], colors_operand=-np.linalg.norm(full_lidar[valid_mask], axis=1), radius=1)
    for N in range(len(lidar_visible)):
        if lidar_visible[N]==0:
            canvas_2d_visible.draw_boxes(boxes_draw[N:N+1], texts=texts[N:N+1], colors=(255, 0, 0))
        else:
            canvas_2d_visible.draw_boxes(boxes_draw[N:N+1], texts=texts[N:N+1], colors=(255, 255, 255))

    draw_lidars_only = False
    if draw_lidars_only:
        fig, ax = plt.subplots(1, 3, figsize=(23, 9), dpi=150)
        fig.suptitle("Frame: {:0>4d}".format(frame),fontsize= 30)
        
        ax[0].imshow(canvas_3d.canvas)
        ax[0].set_title("3D LiDAR with BBOX",fontsize= 10)
        ax[0].axis('off')

        ax[1].imshow(canvas_3d_visible.canvas)
        ax[1].set_title("Visible BBOX under LiDAR sensor",fontsize= 10)
        ax[1].axis('off')

        ax[2].imshow(np.rot90(canvas_2d_visible.canvas, k=1, axes=(1,0)))
        ax[2].set_title("transform LiDAR to BEV",fontsize= 10)
        ax[2].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_root, "vehicle/ego_{}/frame_{}.png".format(ego, frame)), transparent=False)
        plt.close()  
    else:
        rgb_front_file_path = "{}/ego_vehicle_{}/rgb_front/{:0>4d}.jpg".format(route, ego, frame)
        img_front = np.array(Image.open(rgb_front_file_path))
        rgb_left_file_path = "{}/ego_vehicle_{}/rgb_left/{:0>4d}.jpg".format(route, ego, frame)
        img_left = np.array(Image.open(rgb_left_file_path))
        rgb_right_file_path = "{}/ego_vehicle_{}/rgb_right/{:0>4d}.jpg".format(route, ego, frame)
        img_right = np.array(Image.open(rgb_right_file_path))

        fig, ax = plt.subplots(2, 3, figsize=(20, 10), dpi=300)
        fig.suptitle("Frame: {:0>4d}".format(frame),fontsize= 30)
        
        ax[0, 0].imshow(img_left)
        ax[0, 0].set_title("RGB left",fontsize= 10)
        ax[0, 0].axis('off')


        ax[0, 1].imshow(img_front)
        ax[0, 1].set_title("RGB front",fontsize= 10)
        ax[0, 1].axis('off')

        ax[0, 2].imshow(img_right)
        ax[0, 2].set_title("RGB right",fontsize= 10)
        ax[0, 2].axis('off')


        ax[1, 0].imshow(canvas_3d.canvas)
        ax[1, 0].set_title("3D LiDAR",fontsize= 10)
        ax[1, 0].axis('off')

        ax[1, 1].imshow(canvas_3d_visible.canvas)
        ax[1, 1].set_title("Visible BBOX under LiDAR sensor",fontsize= 10)
        ax[1, 1].axis('off')

        ax[1, 2].imshow(np.rot90(canvas_2d_visible.canvas, k=1, axes=(1,0)))
        ax[1, 2].set_title("transform LiDAR to BEV",fontsize= 10)
        ax[1, 2].axis('off')

        plt.subplots_adjust(
                    wspace=0.05, 
                    hspace=0.1)

        result_dir =  os.path.join(save_root, "{}/ego_vehicle_{}".format(route.split('/')[-1],ego))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)        
        plt.savefig(os.path.join(save_root, "{}/ego_vehicle_{}/frame_{}.png".format(route.split('/')[-1], ego, frame)), transparent=False, bbox_inches='tight')
        plt.close()

def filter_lidar_visible_frames(
    route = "routes_town05_test_w0_02_16_09_22_21",
    ego = "1"
    ):
    num_frames = len(os.listdir("{}/ego_vehicle_{}/rgb_front".format(route, ego)))
    for frame in tqdm(range(num_frames)):
        # frame *=10
        filter_one_frame(route, ego, frame)
        # if frame == 1:
        #     raise ValueError
    
    img_path = os.path.join(save_root, "{}/ego_vehicle_{}/frame_{}.png".format(route.split('/')[-1], ego, 0))
    img = cv2.imread(img_path)
    size = img.shape[:2]
    print(size)
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')#MP4格式
    # #完成写入对象的创建，第一个参数是合成之后的视频的名称，第二个参数是可以使用的编码器，第三个参数是帧率即每秒钟展示多少张图片，第四个参数是图片大小信息
    video_path = os.path.join(save_root, "{}/ego_{}_bbs.mp4".format(route.split('/')[-1], ego))
    videowrite = cv2.VideoWriter(video_path,fourcc,2,(size[1],size[0]))

    for frame in range(num_frames):
        
        img_path = os.path.join(save_root, "{}/ego_vehicle_{}/frame_{}.png".format(route.split('/')[-1], ego, frame))

        img = cv2.imread(img_path)
        if img is None:
            print(img_path + " is error!")
            continue

        img = cv2.resize(img,(size[1],size[0]))
        videowrite.write(img)
    videowrite.release()


save_root = 'results/visible2'
route_list = ['/GPFS/public/InterFuser/dataset_cop3_lidarmini/weather-0/data/routes_town01_1_w0_05_01_00_38_26']

if __name__ == "__main__":
    for route_path in route_list:
        for ego in ["0"]:
            filter_lidar_visible_frames(
                route = route_path,
                ego = ego
            )



    # num_frames = 90
    # gif_frames = []
    # for frame in range(num_frames):
    #     path = "results/visible/vehicle/ego_0/frame_"+str(frame)+".png"
    #     gif_frames.append(imageio.imread(path))
    # imageio.mimsave('ego_0_visible_vehicle_only_rgb.gif', gif_frames, 'GIF', duration=0.5)


    # gif_frames = []
    # for frame in range(num_frames):
    #     path = "results/visible/vehicle/ego_1/frame_"+str(frame)+".png"
    #     gif_frames.append(imageio.imread(path))
    # imageio.mimsave('ego_1_visible_vehicle_only_rgb.gif', gif_frames, 'GIF', duration=0.5)