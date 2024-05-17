# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os
from torch.utils.data import DataLoader, Subset
from opencood.data_utils import datasets
import torch
from opencood.tools import train_utils, inference_utils
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.visualization import vis_utils, simple_vis
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.camera_utils import coord_3d_to_2d, denormalize_img
from matplotlib import pyplot as plt
import imageio
from opencood.utils.box_utils import boxes_to_corners_3d

history_len = 3
future_len = 3
color_list = [(21/255,101/255,192/255),(164/255,19/255,60/255),(216/255,154/255,158/255)]


def vis_img(image, gt_box2d, gt_box2d_mask, output_path, idx):
    plt.imshow(image)
    N = gt_box2d.shape[0]
    for i in range(N):
        if gt_box2d_mask[i]:
            coord2d = gt_box2d[i]
            for start, end in [(0, 1), (1, 2), (2, 3), (3, 0),
                            (0, 4), (1, 5), (2, 6), (3, 7),
                            (4, 5), (5, 6), (6, 7), (7, 4)]:
                plt.plot(coord2d[[start,end]][:,0], coord2d[[start,end]][:,1], marker="o", c='g')
    plt.savefig(f"{output_path}/{idx}.png", dpi=300)
    plt.clf()

def vis_seq(image, record_len, gt_box2d, gt_box2d_mask, output_path, idx):
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    assert gt_box2d.shape[0] == sum(record_len)
    N = gt_box2d.shape[0]
    alpha = np.arange(0.3, 1.0, 0.7/history_len)
    alphas = []
    colors = []
    for i in range(len(record_len)):
        if i < history_len:
            alphas.extend([alpha[i] for _ in range(record_len[i])])
            colors.extend([color_list[0] for _ in range(record_len[i])])
        else:
            alphas.extend([alpha[-1] for _ in range(record_len[i])])
            colors.extend([color_list[1] for _ in range(record_len[i])])
    for i in range(N):
        if gt_box2d_mask[i]:
            coord2d = gt_box2d[i]
            c_x = coord2d[:,0].mean()
            c_y = coord2d[:,1].mean()
            plt.plot(c_x, c_y, marker="o", c=colors[i], markersize=3, alpha=alphas[i])
    plt.savefig(f"{output_path}/{idx}.png", dpi=300)
    plt.clf()

def save_video(output_path):
    img_array = []
    for idx in range(20, 50):
        image_path = f"{output_path}/{idx}.png"
        if os.path.exists(image_path):
            image = imageio.v2.imread(image_path)
            img_array.append(image)

    gif_out = os.path.join(output_path, "result_GIF.gif")
    imageio.mimsave(gif_out, [x for x in img_array], 'GIF', duration=0.15)

if __name__ == '__main__':
    current_path = os.path.dirname(os.path.realpath(__file__))
    hypes = load_yaml(os.path.join(current_path,
                                    '../hypes_yaml/visualization_dair.yaml'))
    output_path = "YOURPATH"

    save_video(output_path)

    print('Dataset Building')
    opencda_dataset = build_dataset(hypes, visualize=True, train=False)
    dataset_len = len(opencda_dataset)
    subset_idx = range(10, 100)
    subset_dataset = Subset(opencda_dataset, subset_idx)
    
    data_loader = DataLoader(opencda_dataset, batch_size=1, num_workers=2,
                             collate_fn=opencda_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False)
    vis_gt_box = True
    vis_pred_box = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    gt_boxes = []
    record_len = []
    images = []
    int_matrixs = []
    ext_matrixs = []
    key = 1
    comm_rates = []
    comm_rates_ego = []
    for i, batch_data in enumerate(data_loader):
        print(i)
        # import ipdb; ipdb.set_trace()
        for cav_id, cav_content in batch_data.items():
            if cav_id != 'ego':
                comm_rates.append(len(cav_content['origin_lidar'].flatten()))
            else:
                comm_rates_ego.append(len(cav_content['origin_lidar'][0].flatten()))
        continue

        batch_data = train_utils.to_device(batch_data, device)
        gt_box_tensor = boxes_to_corners_3d(batch_data[key]['object_bbx_center'][0,:len(batch_data[key]['object_ids'])],
                                              hypes['postprocess']['order']).cpu().numpy()
        # gt_box_tensor = opencda_dataset.post_processor.generate_gt_bbx(batch_data).cpu().numpy()
        image = np.array(denormalize_img(batch_data[key]['image_inputs']['imgs'][0,0]))  # 3, 480, 640
        int_matrix = batch_data[key]['image_inputs']['intrins'].cpu().numpy()[0,0]  # 1, 4, 3, 3
        ext_matrix = batch_data[key]['image_inputs']['extrinsics'].cpu().numpy()[0,0]  # 1, 4, 3, 3

        # import ipdb; ipdb.set_trace()
        
        gt_boxes.append(gt_box_tensor)
        images.append(image)
        int_matrixs.append(int_matrix)
        ext_matrixs.append(ext_matrix)
        record_len.append(gt_box_tensor.shape[0])

        if i >= (history_len + future_len):
            cur_gt_box = np.concatenate(gt_boxes[-(history_len + future_len):])
            cur_int_matrix = int_matrixs[-future_len]
            cur_ext_matrix = ext_matrixs[-future_len]
            gt_box2d, gt_box2d_mask, _ = coord_3d_to_2d(cur_gt_box, cur_int_matrix, cur_ext_matrix, image_H=1080, image_W=1920, image=None, idx=None)
            cur_image = images[-future_len]
            cur_record_len = record_len[-(history_len + future_len):]
            vis_seq(cur_image, cur_record_len, gt_box2d, gt_box2d_mask, output_path, i)
    print('comm_rates: ', sum(comm_rates)/len(comm_rates))
    print('comm_rates_ego: ', sum(comm_rates_ego)/len(comm_rates))
    # save_video(output_path)
        # vis_save_path = os.path.join(output_path, '3d_%05d.png' % i)
        # simple_vis.visualize(None,
        #                     gt_box_tensor,
        #                     batch_data['ego']['origin_lidar'][0],
        #                     hypes['postprocess']['gt_range'],
        #                     vis_save_path,
        #                     method='3d',
        #                     vis_gt_box = vis_gt_box,
        #                     vis_pred_box = vis_pred_box,
        #                     left_hand=False)
            
        # vis_save_path = os.path.join(output_path, 'bev_%05d.png' % i)
        # simple_vis.visualize(None,
        #                     gt_box_tensor,
        #                     batch_data['ego']['origin_lidar'][0],
        #                     hypes['postprocess']['gt_range'],
        #                     vis_save_path,
        #                     method='bev',
        #                     vis_gt_box = vis_gt_box,
        #                     vis_pred_box = vis_pred_box,
        #                     left_hand=False)