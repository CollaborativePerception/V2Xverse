"""
Functions to render occupancy map from bounding boxes
"""


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
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pygame

def box2occ(infer_result):

    det_range = [36, 12, 12, 12, 0.25]

    attrib_list = ['pred_box_tensor', 'pred_score', 'gt_box_tensor']
    for attrib in attrib_list:
        if isinstance(infer_result[attrib], list):
            infer_result_tensor = []
            for i in range(len(infer_result[attrib])):
                if infer_result[attrib][i] is not None:
                    infer_result_tensor.append(infer_result[attrib][i])
            if len(infer_result_tensor)>0:
                infer_result[attrib] = torch.cat(infer_result_tensor, dim=0)
            else:
                infer_result[attrib] = None

    ### filte out ego box
    if not infer_result['pred_box_tensor'] is None:
        if len(infer_result['pred_box_tensor']) > 0:
            tmp = infer_result['pred_box_tensor'][:,:,0].clone()
            infer_result['pred_box_tensor'][:,:,0]=infer_result['pred_box_tensor'][:,:,1]
            infer_result['pred_box_tensor'][:,:,1] = tmp
        # measurements = car_data_raw[0]['measurements']
        num_object = infer_result['pred_box_tensor'].shape[0]
        # if num_object > 0:
        object_list = []
        # transform from lidar pose to ego pose
        for i in range(num_object):
            transformed_box = infer_result['pred_box_tensor'][i].cpu().numpy()
            transformed_box[:,1] += 1.3


            location_box = np.mean(transformed_box[:4,:2], 0)
            if np.linalg.norm(location_box) < 1.4:
                continue
            object_list.append(torch.from_numpy(transformed_box))
        if len(object_list) > 0:
            processed_pred_box = torch.stack(object_list, dim=0)
        else:
            processed_pred_box = infer_result['pred_box_tensor'][:0]
    else:
        processed_pred_box = [] # infer_result['pred_box_tensor']

    ### turn boxes into occupancy map
    if len(processed_pred_box) > 0:
        occ_map = turn_traffic_into_map(processed_pred_box[:,:4,:2].cpu(), det_range)
    else:
        occ_map = turn_traffic_into_map(processed_pred_box, det_range)

    # # N, K, H, W, C=7
    # occ_map = turn_traffic_into_map(pred_traffic, self.det_range)
    occ_map_shape = occ_map.shape
    occ_map = torch.from_numpy(occ_map).cuda().contiguous().view((-1, 1) + occ_map_shape[1:])

    return occ_map

def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:, 2] = 1

    c, s = np.cos(r1), np.sin(r1)
    # 顺时针旋转r1角度，r1车辆坐标转换到world frame
    r1_to_world = np.matrix([[c, -s, t1_x], [s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, -s, t2_x], [s, c, t2_y], [0, 0, 1]])
    # world frame -> r2 frame
    # if r1==r2, do nothing
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T
    # reset z-coordinate
    out[:, 2] = xyz[:, 2]

    return out

def turn_traffic_into_map(all_bbox, det_range):
    data_total = []
    for idx in range(1):

        if len(all_bbox) == 0:
            all_bbox = np.zeros((1,4,2))
        # plt.cla()

        fig = plt.figure(figsize=(6, 12), dpi=16)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        ax = plt.gca()
        ax.set_facecolor("black")

        plt.xlim((-det_range[2], det_range[3]))
        plt.ylim((-det_range[1], det_range[0]))

        for i in range(len(all_bbox)):
            plt.fill(all_bbox[i,:,0], all_bbox[i,:,1], color = 'white')

        # plt.axis('off')
        # If we haven't already shown or saved the plot, then we need to
        # draw the figure first...
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # H=192, W=96, 3
        data_total.append(data[:, :, 0])
        # plt.savefig('/GPFS/public/InterFuser/results/cop3/pnp/multiclass_finetune_fusion_none/test.png')
        plt.close()

    occ_map = np.stack(data_total, axis=0) # B * T_p, H, W
    return occ_map