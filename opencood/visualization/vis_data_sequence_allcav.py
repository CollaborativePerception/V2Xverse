# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os
from torch.utils.data import DataLoader
from opencood.data_utils import datasets
import torch
from opencood.tools import train_utils, inference_utils
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.visualization import vis_utils, simple_vis
from opencood.data_utils.datasets.late_fusion_dataset_v2x import \
    LateFusionDatasetV2X
from opencood.data_utils.datasets.late_fusion_dataset import \
    LateFusionDataset


def gen_gt_bbox(cav_content):
    from opencood.utils import box_utils
    gt_range = [-140,-40,-3,140,40,1]
    order = 'hwl'
    gt_box3d_list = []
    object_id_list = []

    object_bbx_center = cav_content['object_bbx_center']
    object_bbx_mask = cav_content['object_bbx_mask']
    object_ids = cav_content['object_ids']
    object_bbx_center = object_bbx_center[object_bbx_mask == 1]

    # convert center to corner
    object_bbx_corner = \
        box_utils.boxes_to_corners_3d(object_bbx_center, order)
    projected_object_bbx_corner = object_bbx_corner.float()

    gt_box3d_list.append(projected_object_bbx_corner)
    # append the corresponding ids
    object_id_list += object_ids

    # gt bbx 3d
    gt_box3d_list = torch.vstack(gt_box3d_list)
    # some of the bbx may be repetitive, use the id list to filter
    gt_box3d_selected_indices = \
        [object_id_list.index(x) for x in set(object_id_list)]
    gt_box3d_tensor = gt_box3d_list[gt_box3d_selected_indices]

    # filter the gt_box to make sure all bbx are in the range
    mask = \
        box_utils.get_mask_for_boxes_within_range_torch(gt_box3d_tensor, gt_range)
    gt_box3d_tensor = gt_box3d_tensor[mask, :, :]

    return gt_box3d_tensor

if __name__ == '__main__':
    current_path = os.path.dirname(os.path.realpath(__file__))
    params = load_yaml(os.path.join(current_path,
                                    '../hypes_yaml/visualization_opv2v.yaml'))  
    output_path = "/GPFS/rhome/yifanlu/OpenCOOD/data_vis/opv2v/all_cav"

    opencda_dataset = LateFusionDataset(params, visualize=True,
                                            train=False)

    data_loader = DataLoader(opencda_dataset, batch_size=1, num_workers=2,
                             collate_fn=opencda_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False)
    vis_gt_box = True
    vis_pred_box = True  ### Modified Here
    hypes = params

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = torch.load("/GPFS/rhome/yifanlu/workspace/OpenCOOD/box_align_items/box_align.pt")  ### Modified Here
    pred_corner3d_list, pred_box3d_list, scores_list, record_len, lidar_poses = data  ### Modified Here

    for i, batch_data in enumerate(data_loader):
        print(i)
        batch_data = train_utils.to_device(batch_data, device)
        gt_box_list = []

        start = 0  ### Modified Here
        for cav_id, cav_content in batch_data.items():
            print(cav_id)
            gt_box_tensor = gen_gt_bbox(cav_content)
            pred_box_tensor = pred_corner3d_list[start]
            gt_box_list.append(gt_box_tensor)

            vis_save_path = os.path.join(output_path, '3d_%05d_%s.png' % (i, cav_id))
            simple_vis.visualize(pred_box_tensor, ### Modified Here
                                gt_box_tensor,
                                cav_content['origin_lidar'][0],
                                hypes['postprocess']['gt_range'],
                                vis_save_path,
                                method='3d',
                                vis_gt_box = vis_gt_box,
                                vis_pred_box = vis_pred_box,
                                left_hand=True)
                
            vis_save_path = os.path.join(output_path, 'bev_%05d_%s.png' % (i, cav_id))
            simple_vis.visualize(pred_box_tensor,   ### Modified Here
                                gt_box_tensor,
                                cav_content['origin_lidar'][0],
                                hypes['postprocess']['gt_range'],
                                vis_save_path,
                                method='bev',
                                vis_gt_box = vis_gt_box,
                                vis_pred_box = vis_pred_box,
                                left_hand=True)
            start += 1   ### Modified Here
        # torch.save(gt_box_list, "/GPFS/rhome/yifanlu/workspace/OpenCOOD/box_align_items/gt_box_list.pt")
        raise