# late fusion dataset
import random
import math
from collections import OrderedDict
import cv2
import numpy as np
import torch
import copy
from icecream import ic
from PIL import Image
import pickle as pkl
from opencood.utils import box_utils as box_utils
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.utils.camera_utils import (
    sample_augmentation,
    img_transform,
    normalize_img,
    img_to_tensor,
)
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.utils.transformation_utils import x1_to_x2, x_to_world, get_pairwise_transformation
from opencood.utils.pose_utils import add_noise_data_dict, add_noise_data_dict_asymmetric
from opencood.utils.pcd_utils import (
    mask_points_by_range,
    mask_ego_points,
    mask_ego_points_v2,
    shuffle_points,
    downsample_lidar_minimum,
)
from opencood.utils.common_utils import merge_features_to_dict

def getLatemulticlassFusionDataset(cls):
    """
    cls: the Basedataset.
    """
    class LatemulticlassFusionDataset(cls):
        def __init__(self, params, visualize, train=True):
            super().__init__(params, visualize, train)
            self.anchor_box = self.post_processor.generate_anchor_box()
            self.anchor_box_torch = torch.from_numpy(self.anchor_box)

            self.heterogeneous = False
            if 'heter' in params:
                self.heterogeneous = True
            
            self.multiclass = params['model']['args']['multi_class']
            
            self.proj_first = False if 'proj_first' not in params['fusion']['args']\
                                         else params['fusion']['args']['proj_first']
            
            # self.proj_first = False
            self.supervise_single = True if ('supervise_single' in params['model']['args'] and params['model']['args']['supervise_single']) \
                                        else False
            self.supervise_single = False
            self.online_eval_only = False


        def __getitem__(self, idx, extra_source=None, data_dir=None):

            if data_dir is not None:
                extra_source=1

            object_bbx_center_list = []
            object_bbx_mask_list = []
            object_id_dict = {}

            object_bbx_center_list_single = []
            object_bbx_mask_list_single = []

            gt_object_bbx_center_list = []
            gt_object_bbx_mask_list = []
            gt_object_id_dict = {}

            gt_object_bbx_center_list_single = []
            gt_object_bbx_mask_list_single = []

            output_dict = {}
            for tpe in ['all', 0, 1, 3]:
                output_single_class = self.__getitem_single_class__(idx, tpe, extra_source, data_dir)
                output_dict[tpe] = output_single_class
                if tpe == 'all' and extra_source is None:
                    continue
                elif tpe == 'all' and extra_source is not None:
                    break
                object_bbx_center_list.append(output_single_class['ego']['object_bbx_center'])
                object_bbx_mask_list.append(output_single_class['ego']['object_bbx_mask'])
                object_id_dict[tpe] = output_single_class['ego']['object_ids']

                gt_object_bbx_center_list.append(output_single_class['ego']['gt_object_bbx_center'])
                gt_object_bbx_mask_list.append(output_single_class['ego']['gt_object_bbx_mask'])
                gt_object_id_dict[tpe] = output_single_class['ego']['gt_object_ids']

            if self.multiclass and extra_source is None:
                output_dict['all']['ego']['object_bbx_center'] = np.stack(object_bbx_center_list, axis=0)
                output_dict['all']['ego']['object_bbx_mask'] = np.stack(object_bbx_mask_list, axis=0)
                output_dict['all']['ego']['object_ids'] = object_id_dict

                output_dict['all']['ego']['gt_object_bbx_center'] = np.stack(gt_object_bbx_center_list, axis=0)
                output_dict['all']['ego']['gt_object_bbx_mask'] = np.stack(gt_object_bbx_mask_list, axis=0)
                output_dict['all']['ego']['gt_object_ids'] = gt_object_id_dict
            

            return output_dict['all']

        def __getitem_single_class__(self, idx, tpe=None, extra_source=None, data_dir=None):

            if extra_source is None and data_dir is None:
                base_data_dict = self.retrieve_base_data(idx, tpe) ## {id:{'ego':True/False, 'params': {'lidar_pose','speed','vehicles','ego_pos',...}, 'lidar_np': array (N,4)}}
            elif data_dir is not None:
                base_data_dict = self.retrieve_base_data(idx=None, tpe=tpe, data_dir=data_dir)
            elif extra_source is not None:
                base_data_dict = self.retrieve_base_data(idx=None, tpe=tpe, extra_source=extra_source)

            # base_data_dict = add_noise_data_dict(base_data_dict,self.params['noise_setting'])
            base_data_dict = add_noise_data_dict_asymmetric(base_data_dict,self.params['noise_setting'])
            processed_data_dict = OrderedDict()
            processed_data_dict['ego'] = {}
            ego_id = -1
            ego_lidar_pose = []
            ego_cav_base = None
            cav_id_list = []
            lidar_pose_list = []
            too_far = []
            # first find the ego vehicle's lidar pose
            for cav_id, cav_content in base_data_dict.items():
                
                if cav_content['ego']:
                    ego_id = cav_id
                    ego_lidar_pose = cav_content['params']['lidar_pose']
                    ego_lidar_pose_clean = cav_content['params']['lidar_pose_clean']
                    ego_cav_base = cav_content
                    break

            assert ego_id != -1
            assert len(ego_lidar_pose) > 0

            agents_image_inputs = []
            processed_features = []
            object_stack = []
            object_id_stack = []

            gt_object_stack = []
            gt_object_id_stack = []

            single_label_list = []
            single_object_bbx_center_list = []
            single_object_bbx_mask_list = []
            too_far = []
            lidar_pose_list = []
            lidar_pose_clean_list = []
            cav_id_list = []
            projected_lidar_clean_list = [] # disconet

            if self.visualize:
                projected_lidar_stack = []    

            # loop over all CAVs to process information
            for cav_id, selected_cav_base in base_data_dict.items():
                distance = \
                    math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                            ego_lidar_pose[0]) ** 2 + (
                                    selected_cav_base['params'][
                                        'lidar_pose'][1] - ego_lidar_pose[
                                        1]) ** 2)
                if distance > self.params['comm_range']:
                    too_far.append(cav_id)
                    continue
                cav_id_list.append(cav_id)
                lidar_pose_list.append(selected_cav_base['params']['lidar_pose'])
                lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])                

            for cav_id in too_far:
                base_data_dict.pop(cav_id)
            
            pairwise_t_matrix = \
                get_pairwise_transformation(base_data_dict,
                                                self.max_cav,
                                                self.proj_first)
            cav_num = len(cav_id_list)
            cav_id_list_newname = []

            lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)  # [N_cav, 6]
            lidar_poses_clean = np.array(lidar_pose_clean_list).reshape(-1, 6)  # [N_cav, 6]

            for cav_id in cav_id_list:
                selected_cav_base = base_data_dict[cav_id]
                # find the transformation matrix from current cav to ego.
                cav_lidar_pose = selected_cav_base['params']['lidar_pose']
                transformation_matrix = x1_to_x2(cav_lidar_pose, ego_lidar_pose)
                cav_lidar_pose_clean = selected_cav_base['params']['lidar_pose_clean']
                transformation_matrix_clean = x1_to_x2(cav_lidar_pose_clean, ego_lidar_pose_clean)

                selected_cav_processed = \
                    self.get_item_single_car(selected_cav_base, 
                                            ego_cav_base,
                                            tpe,
                                            extra_source!=None)
                selected_cav_processed.update({'transformation_matrix': transformation_matrix,
                                            'transformation_matrix_clean': transformation_matrix_clean})
                if extra_source is None:
                    object_stack.append(selected_cav_processed['object_bbx_center'])
                    object_id_stack += selected_cav_processed['object_ids']
                
                    gt_object_stack.append(selected_cav_processed['gt_object_bbx_center'])
                    gt_object_id_stack += selected_cav_processed['gt_object_ids']

                if tpe == 'all':
                        
                    if self.load_lidar_file:
                        processed_features.append(
                            selected_cav_processed['processed_lidar'])
                       
                    if self.load_camera_file:
                        agents_image_inputs.append(
                            selected_cav_processed['image_inputs'])
                    
                    if self.visualize:
                        projected_lidar_stack.append(
                            selected_cav_processed['projected_lidar'])
                    

                if self.supervise_single  and extra_source is None :
                    single_label_list.append(selected_cav_processed['single_label_dict'])
                    single_object_bbx_center_list.append(selected_cav_processed['single_object_bbx_center'])
                    single_object_bbx_mask_list.append(selected_cav_processed['single_object_bbx_mask'])
                
                update_cav = "ego" if cav_id == ego_id else cav_id
                processed_data_dict.update({update_cav: selected_cav_processed})
                cav_id_list_newname.append(update_cav)
            
            if self.supervise_single and extra_source is None:
                single_label_dicts = {}
                if tpe == 'all':
                    # unused label
                    if False:
                        single_label_dicts = self.post_processor.collate_batch(single_label_list)
                single_object_bbx_center = torch.from_numpy(np.array(single_object_bbx_center_list))
                single_object_bbx_mask = torch.from_numpy(np.array(single_object_bbx_mask_list))
                processed_data_dict['ego'].update({
                    "single_label_dict_torch": single_label_dicts,
                    "single_object_bbx_center_torch": single_object_bbx_center,
                    "single_object_bbx_mask_torch": single_object_bbx_mask,
                    })

            # heterogeneous
            if self.heterogeneous:
                processed_data_dict['ego']['idx'] = idx
                processed_data_dict['ego']['cav_list'] = cav_id_list_newname
            
            if extra_source is None:
                unique_indices = \
                    [object_id_stack.index(x) for x in set(object_id_stack)]
                object_stack = np.vstack(object_stack)
                object_stack = object_stack[unique_indices]

                # make sure bounding boxes across all frames have the same number
                object_bbx_center = \
                    np.zeros((self.params['postprocess']['max_num'], 7))
                mask = np.zeros(self.params['postprocess']['max_num'])
                object_bbx_center[:object_stack.shape[0], :] = object_stack
                mask[:object_stack.shape[0]] = 1

                gt_unique_indices = \
                    [gt_object_id_stack.index(x) for x in set(gt_object_id_stack)]
                gt_object_stack = np.vstack(gt_object_stack)
                gt_object_stack = gt_object_stack[gt_unique_indices]

                # make sure bounding boxes across all frames have the same number
                gt_object_bbx_center = \
                    np.zeros((self.params['postprocess']['max_num'], 7))
                gt_mask = np.zeros(self.params['postprocess']['max_num'])
                gt_object_bbx_center[:gt_object_stack.shape[0], :] = gt_object_stack
                gt_mask[:gt_object_stack.shape[0]] = 1

                processed_data_dict['ego'].update(
                    {'object_bbx_center': object_bbx_center,  # (100,7)
                    'object_bbx_mask': mask, # (100,)
                    'object_ids': [object_id_stack[i] for i in unique_indices],     
                    }   
                )

            # generate targets label
            label_dict = {}
            # if tpe == 'all':
            # unused label
            if extra_source is None:
                label_dict = \
                    self.post_processor.generate_label(
                        gt_box_center=object_bbx_center,
                        anchors=self.anchor_box,
                        mask=mask)
                gt_label_dict = \
                    self.post_processor.generate_label(
                        gt_box_center=gt_object_bbx_center,
                        anchors=self.anchor_box,
                        mask=gt_mask)


                processed_data_dict['ego'].update(
                    {'gt_object_bbx_center': gt_object_bbx_center,  # (100,7)
                    'gt_object_bbx_mask': gt_mask, # (100,)
                    'gt_object_ids': [gt_object_id_stack[i] for i in gt_unique_indices],
                    'gt_label_dict': gt_label_dict})

            processed_data_dict['ego'].update(
                {
                'anchor_box': self.anchor_box,
                'label_dict': label_dict,
                'cav_num': cav_num,
                'pairwise_t_matrix': pairwise_t_matrix,
                'lidar_poses_clean': lidar_poses_clean,
                'lidar_poses': lidar_poses})
        
            if tpe == 'all':
                if self.load_lidar_file:
                    merged_feature_dict = merge_features_to_dict(processed_features)
                    processed_data_dict['ego'].update({'processed_lidar': merged_feature_dict})
                
                if self.load_camera_file:
                    merged_image_inputs_dict = merge_features_to_dict(agents_image_inputs, merge='stack')
                    processed_data_dict['ego'].update({'image_inputs': merged_image_inputs_dict})
                
                if self.visualize:
                    processed_data_dict['ego'].update({'origin_lidar':
                                                    #    projected_lidar_stack})
                        np.vstack(
                            projected_lidar_stack)})
                    processed_data_dict['ego'].update({'lidar_len': [len(projected_lidar_stack[i]) for i in range(len(projected_lidar_stack))]})
                

                processed_data_dict['ego'].update({'sample_idx': idx,
                                                    'cav_id_list': cav_id_list})

                img_front_list = []
                img_left_list = []
                img_right_list = []
                BEV_list = []

                if self.visualize:
                    for car_id in base_data_dict:
                        if not base_data_dict[car_id]['ego'] == True:
                            continue
                        if 'rgb_front' in base_data_dict[car_id] and 'rgb_left' in base_data_dict[car_id] and 'rgb_right' in base_data_dict[car_id] and 'BEV' in base_data_dict[car_id] :
                            img_front_list.append(base_data_dict[car_id]['rgb_front'])
                            img_left_list.append(base_data_dict[car_id]['rgb_left'])
                            img_right_list.append(base_data_dict[car_id]['rgb_right'])
                            BEV_list.append(base_data_dict[car_id]['BEV'])
                processed_data_dict['ego'].update({'img_front': img_front_list,
                                                    'img_left': img_left_list,
                                                    'img_right': img_right_list,
                                                    'BEV': BEV_list})
            processed_data_dict['ego'].update({'scene_dict': base_data_dict['car_0']['scene_dict'],
                                                    'frame_id': base_data_dict['car_0']['frame_id'],
                                                    })

            return processed_data_dict

        def get_item_single_car(self, selected_cav_base, ego_cav_base, tpe, online_eval=False):
            """
            Process a single CAV's information for the train/test pipeline.


            Parameters
            ----------
            selected_cav_base : dict
                The dictionary contains a single CAV's raw information.
                including 'params', 'camera_data'

            Returns
            -------
            selected_cav_processed : dict
                The dictionary contains the cav's processed information.
            """
            selected_cav_processed = {}

            if not online_eval:
                # label
                object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center_single(
                    [selected_cav_base], selected_cav_base["params"]["lidar_pose_clean"]
                )
            
            ego_pose, ego_pose_clean = ego_cav_base['params']['lidar_pose'], ego_cav_base['params']['lidar_pose_clean']

            
            # calculate the transformation matrix
            transformation_matrix = \
                x1_to_x2(selected_cav_base['params']['lidar_pose'],
                        ego_pose) # T_ego_cav
            transformation_matrix_clean = \
                x1_to_x2(selected_cav_base['params']['lidar_pose_clean'],
                        ego_pose_clean)
            
            # lidar
            if tpe == 'all':
                if self.load_lidar_file or self.visualize:
                    lidar_np = selected_cav_base['lidar_np']
                    lidar_np = shuffle_points(lidar_np)
                    lidar_np = mask_points_by_range(lidar_np,
                                                self.params['preprocess'][
                                                    'cav_lidar_range'])
                    # remove points that hit ego vehicle
                    lidar_np = mask_ego_points_v2(lidar_np)

                    # data augmentation, seems very important for single agent training, because lack of data diversity.
                    # only work for lidar modality in training.
                    if not self.heterogeneous and not online_eval:
                        lidar_np, object_bbx_center, object_bbx_mask = \
                        self.augment(lidar_np, object_bbx_center, object_bbx_mask)
                    
                    projected_lidar = \
                        box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)
                      
                    if self.proj_first:
                        lidar_np[:, :3] = projected_lidar
                    
                    if self.visualize:
                        # filter lidar
                        selected_cav_processed.update({'projected_lidar': projected_lidar})
                    
                    lidar_dict = self.pre_processor.preprocess(lidar_np)
                    selected_cav_processed.update({'processed_lidar': lidar_dict})
        
                if self.visualize:
                    selected_cav_processed.update({'origin_lidar': lidar_np})
            
            if not online_eval:
                object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center(
                    [selected_cav_base], selected_cav_base['params']['lidar_pose']
                )

                gt_object_bbx_center, gt_object_bbx_mask, gt_object_ids = self.generate_object_center(
                    [selected_cav_base], selected_cav_base['params']['lidar_pose']
                )
                
                label_dict = self.post_processor.generate_label(
                    gt_box_center=object_bbx_center, anchors=self.anchor_box, mask=object_bbx_mask
                )

                gt_label_dict = self.post_processor.generate_label(
                    gt_box_center=gt_object_bbx_center, anchors=self.anchor_box, mask=gt_object_bbx_mask
                )

                selected_cav_processed.update({
                                    "single_label_dict": label_dict,
                                    "single_object_bbx_center": object_bbx_center,
                                    "single_object_bbx_mask": object_bbx_mask})

            # camera
            if tpe == 'all':
                if self.load_camera_file:
                    # adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/data.py
                    camera_data_list = selected_cav_base["camera_data"]

                    params = selected_cav_base["params"]
                    imgs = []
                    rots = []
                    trans = []
                    intrins = []
                    extrinsics = [] # cam_to_lidar
                    post_rots = []
                    post_trans = []
    
                    for idx, img in enumerate(camera_data_list):
                        camera_to_lidar, camera_intrinsic = self.get_ext_int(params, idx)

                        intrin = torch.from_numpy(camera_intrinsic)
                        rot = torch.from_numpy(
                            camera_to_lidar[:3, :3]
                        )  # R_wc, we consider world-coord is the lidar-coord
                        tran = torch.from_numpy(camera_to_lidar[:3, 3])  # T_wc
    
                        post_rot = torch.eye(2)
                        post_tran = torch.zeros(2)

                        img_src = [img]

                        # depth
                        if self.load_depth_file:
                            depth_img = selected_cav_base["depth_data"][idx]
                            img_src.append(depth_img)
                        else:
                            depth_img = None

                        # data augmentation
                        resize, resize_dims, crop, flip, rotate = sample_augmentation(
                            self.data_aug_conf, self.train
                        )
                        img_src, post_rot2, post_tran2 = img_transform(
                            img_src,
                            post_rot,
                            post_tran,
                            resize=resize,
                            resize_dims=resize_dims,
                            crop=crop,
                            flip=flip,
                            rotate=rotate,
                        )
                        # for convenience, make augmentation matrices 3x3
                        post_tran = torch.zeros(3)
                        post_rot = torch.eye(3)
                        post_tran[:2] = post_tran2
                        post_rot[:2, :2] = post_rot2

                        img_src[0] = normalize_img(img_src[0])
                        if self.load_depth_file:
                            img_src[1] = img_to_tensor(img_src[1]) * 255

                        imgs.append(torch.cat(img_src, dim=0))
                        intrins.append(intrin)
                        extrinsics.append(torch.from_numpy(camera_to_lidar))
                        rots.append(rot)
                        trans.append(tran)
                        post_rots.append(post_rot)
                        post_trans.append(post_tran)

                    selected_cav_processed.update(
                        {
                        "image_inputs": 
                            {
                                "imgs": torch.stack(imgs), # [N, 3or4, H, W]
                                "intrins": torch.stack(intrins),
                                "extrinsics": torch.stack(extrinsics),
                                "rots": torch.stack(rots),
                                "trans": torch.stack(trans),
                                "post_rots": torch.stack(post_rots),
                                "post_trans": torch.stack(post_trans),
                            }
                        }
                    )
            
                selected_cav_processed.update({"anchor_box": self.anchor_box})

            if not online_eval:
                object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center([selected_cav_base],
                                                            ego_pose_clean)

                gt_object_bbx_center, gt_object_bbx_mask, gt_object_ids = self.generate_object_center([selected_cav_base],
                                                            ego_pose_clean)
                selected_cav_processed.update(
                    {
                        "object_bbx_center": object_bbx_center,
                        "object_bbx_mask": object_bbx_mask,
                        "object_ids": object_ids,
                    }
                )

                selected_cav_processed.update(
                    {
                        "gt_object_bbx_center": gt_object_bbx_center[gt_object_bbx_mask == 1],
                        "gt_object_bbx_mask": gt_object_bbx_mask,
                        "gt_object_ids": gt_object_ids
                    }
                )

                # generate targets label
                label_dict = self.post_processor.generate_label(
                    gt_box_center=object_bbx_center, anchors=self.anchor_box, mask=object_bbx_mask
                )
                selected_cav_processed.update({"label_dict": label_dict})

            selected_cav_processed.update(
                {
                    'transformation_matrix': transformation_matrix,
                    'transformation_matrix_clean': transformation_matrix_clean
                }
            )

            return selected_cav_processed


        def collate_batch_train(self, batch, online_eval_only=False):
            """
            Customized collate function for pytorch dataloader during training
            for early and late fusion dataset.

            Parameters
            ----------
            batch : dict

            Returns
            -------
            batch : dict
                Reformatted batch.
            """
            # during training, we only care about ego.
            output_dict = {'ego': {}}

            object_bbx_center = []
            object_bbx_mask = []
            processed_lidar_list = []
            label_dict_list = []
            origin_lidar = []

            gt_object_bbx_center = []
            gt_object_bbx_mask = []
            gt_object_ids = []
            gt_label_dict_list = []
            record_len = []
            
            object_ids = []
            image_inputs_list = []
            # used to record different scenario
            record_len = []
            label_dict_list = []
            lidar_pose_list = []
            origin_lidar = []
            lidar_len = []
            lidar_pose_clean_list = []

            # heterogeneous
            lidar_agent_list = []
            
            # pairwise transformation matrix
            pairwise_t_matrix_list = []

            # disconet
            teacher_processed_lidar_list = []

            # image
            img_front = []
            img_left = []
            img_right = []
            BEV = []

            dict_list = []

            if self.supervise_single:
                pos_equal_one_single = []
                neg_equal_one_single = []
                targets_single = []
                object_bbx_center_single = []
                object_bbx_mask_single = []

            for i in range(len(batch)):
                ego_dict = batch[i]['ego']
                
                if not online_eval_only:
                    object_bbx_center.append(ego_dict['object_bbx_center'])
                    object_bbx_mask.append(ego_dict['object_bbx_mask'])
                    object_ids.append(ego_dict['object_ids'])
                    
                    gt_object_bbx_center.append(ego_dict['gt_object_bbx_center'])
                    gt_object_bbx_mask.append(ego_dict['gt_object_bbx_mask'])
                
                    gt_object_ids.append(ego_dict['gt_object_ids'])

                    label_dict_list.append(ego_dict['label_dict'])

                    gt_label_dict_list.append(ego_dict['gt_label_dict'])

                else:
                    object_ids.append(None)    
                    gt_object_ids.append(None)

                lidar_pose_list.append(ego_dict['lidar_poses']) # ego_dict['lidar_pose'] is np.ndarray [N,6]
                lidar_pose_clean_list.append(ego_dict['lidar_poses_clean'])

                if self.load_lidar_file:
                    processed_lidar_list.append(ego_dict['processed_lidar'])
                if self.load_camera_file:
                    image_inputs_list.append(ego_dict['image_inputs']) # different cav_num, ego_dict['image_inputs'] is dict.

                record_len.append(ego_dict['cav_num'])
                pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])
                
                dict_list.append([ego_dict['scene_dict'], ego_dict['frame_id']])

                if self.visualize:
                    origin_lidar.append(ego_dict['origin_lidar'])
                    # lidar_len.append(ego_dict['lidar_len'])
                    if len(ego_dict['img_front']) > 0 and len(ego_dict['img_right']) > 0 and len(ego_dict['img_left']) > 0 and len(ego_dict['BEV']) > 0:
                        img_front.append(ego_dict['img_front'][0])
                        img_left.append(ego_dict['img_left'][0])
                        img_right.append(ego_dict['img_right'][0])
                        BEV.append(ego_dict['BEV'][0])

                if self.supervise_single and not online_eval_only:
                    # unused label
                    if False:
                        pos_equal_one_single.append(ego_dict['single_label_dict_torch']['pos_equal_one'])
                        neg_equal_one_single.append(ego_dict['single_label_dict_torch']['neg_equal_one'])
                        targets_single.append(ego_dict['single_label_dict_torch']['targets'])
                    object_bbx_center_single.append(ego_dict['single_object_bbx_center_torch'])
                    object_bbx_mask_single.append(ego_dict['single_object_bbx_mask_torch'])

                # heterogeneous
                if self.heterogeneous:
                    lidar_agent_list.append(ego_dict['lidar_agent'])

            # convert to numpy, (B, max_num, 7)
            if not online_eval_only:
                object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
                object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))
                gt_object_bbx_center = torch.from_numpy(np.array(gt_object_bbx_center))
                gt_object_bbx_mask = torch.from_numpy(np.array(gt_object_bbx_mask))
            else:
                object_bbx_center = None
                object_bbx_mask = None
                gt_object_bbx_center = None
                gt_object_bbx_mask = None


            # unused label
            label_torch_dict = {}
            if False:
                label_torch_dict = \
                    self.post_processor.collate_batch(label_dict_list)

            record_len = torch.from_numpy(np.array(record_len))
            record_len = torch.from_numpy(np.array(record_len, dtype=int))
            pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))
            label_torch_dict['record_len'] = record_len
            label_torch_dict['pairwise_t_matrix'] = pairwise_t_matrix
            # for centerpoint
            if not online_eval_only:
                label_torch_dict.update({'object_bbx_center': object_bbx_center,
                                        'object_bbx_mask': object_bbx_mask})
                output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                        'object_bbx_mask': object_bbx_mask,})
            output_dict['ego'].update({
                                    'anchor_box': torch.from_numpy(self.anchor_box),
                                    'label_dict': label_torch_dict,
                                    'record_len': record_len,
                                    'pairwise_t_matrix': pairwise_t_matrix})
            if self.visualize:
                origin_lidar = \
                    np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict['ego'].update({'origin_lidar': origin_lidar})

            if self.load_lidar_file:
                merged_feature_dict = merge_features_to_dict(processed_lidar_list)
                if self.heterogeneous:
                    lidar_agent = np.concatenate(lidar_agent_list)
                    lidar_agent_idx = lidar_agent.nonzero()[0].tolist()
                    for k, v in merged_feature_dict.items(): # 'voxel_features' 'voxel_num_points' 'voxel_coords'
                        merged_feature_dict[k] = [v[index] for index in lidar_agent_idx]

                if not self.heterogeneous or (self.heterogeneous and sum(lidar_agent) != 0):
                    processed_lidar_torch_dict = \
                        self.pre_processor.collate_batch(merged_feature_dict)
                    output_dict['ego'].update({'processed_lidar': processed_lidar_torch_dict})
                
            if self.load_camera_file:
                # collate ego camera information
                imgs_batch = []
                rots_batch = []
                trans_batch = []
                intrins_batch = []
                extrinsics_batch = []
                post_trans_batch = []
                post_rots_batch = []
                for i in range(len(batch)):
                    ego_dict = batch[i]["ego"]["image_inputs"]
                    imgs_batch.append(ego_dict["imgs"])
                    rots_batch.append(ego_dict["rots"])
                    trans_batch.append(ego_dict["trans"])
                    intrins_batch.append(ego_dict["intrins"])
                    extrinsics_batch.append(ego_dict["extrinsics"])
                    post_trans_batch.append(ego_dict["post_trans"])
                    post_rots_batch.append(ego_dict["post_rots"])

                output_dict["ego"].update({
                    "image_inputs":
                        {
                            "imgs": torch.stack(imgs_batch),  # [B, N, C, H, W]
                            "rots": torch.stack(rots_batch),
                            "trans": torch.stack(trans_batch),
                            "intrins": torch.stack(intrins_batch),
                            "post_trans": torch.stack(post_trans_batch),
                            "post_rots": torch.stack(post_rots_batch),
                        }
                    }
                )

                merged_image_inputs_dict = merge_features_to_dict(image_inputs_list, merge='cat')

                if self.heterogeneous:
                    lidar_agent = np.concatenate(lidar_agent_list)
                    camera_agent = 1 - lidar_agent
                    camera_agent_idx = camera_agent.nonzero()[0].tolist()
                    if sum(camera_agent) != 0:
                        for k, v in merged_image_inputs_dict.items(): # 'imgs' 'rots' 'trans' ...
                            merged_image_inputs_dict[k] = torch.stack([v[index] for index in camera_agent_idx])

                if not self.heterogeneous or (self.heterogeneous and sum(camera_agent) != 0):
                    output_dict['ego'].update({'image_inputs': merged_image_inputs_dict})
            
            record_len = torch.from_numpy(np.array(record_len, dtype=int))
            pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))
            label_torch_dict['record_len'] = record_len
            label_torch_dict['pairwise_t_matrix'] = pairwise_t_matrix
            lidar_pose = torch.from_numpy(np.concatenate(lidar_pose_list, axis=0))
            lidar_pose_clean = torch.from_numpy(np.concatenate(lidar_pose_clean_list, axis=0))

            if not online_eval_only:
                label_torch_dict = \
                    self.post_processor.collate_batch(label_dict_list)

                gt_label_torch_dict = \
                    self.post_processor.collate_batch(gt_label_dict_list)

                # for centerpoint
                label_torch_dict.update({'object_bbx_center': object_bbx_center,
                                        'object_bbx_mask': object_bbx_mask})          

                gt_label_torch_dict.update({'gt_object_bbx_center': gt_object_bbx_center,
                                        'gt_object_bbx_mask': gt_object_bbx_mask})
            else:
                gt_label_torch_dict = {}

            gt_label_torch_dict['pairwise_t_matrix'] = pairwise_t_matrix
            gt_label_torch_dict['record_len'] = record_len

            # object id is only used during inference, where batch size is 1.
            # so here we only get the first element.
            output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                    'object_bbx_mask': object_bbx_mask,
                                    'record_len': record_len,
                                    'label_dict': label_torch_dict,
                                    'object_ids': object_ids[0],
                                    'pairwise_t_matrix': pairwise_t_matrix,
                                    'lidar_pose_clean': lidar_pose_clean,
                                    'lidar_pose': lidar_pose,
                                    'anchor_box': self.anchor_box_torch})
            
            output_dict['ego'].update({'gt_object_bbx_center': gt_object_bbx_center,
                                    'gt_object_bbx_mask': gt_object_bbx_mask,
                                    'gt_label_dict': gt_label_torch_dict,
                                    'gt_object_ids': gt_object_ids[0]})

            output_dict['ego'].update({'dict_list': dict_list})
            output_dict['ego'].update({'record_len': record_len,
                                       'pairwise_t_matrix': pairwise_t_matrix
                })

            if self.visualize:
                origin_lidar = torch.from_numpy(np.array(origin_lidar))
                output_dict['ego'].update({'origin_lidar': origin_lidar})
                output_dict['ego'].update({'img_front': img_front})
                output_dict['ego'].update({'img_right': img_right})
                output_dict['ego'].update({'img_left': img_left})
                output_dict['ego'].update({'BEV': BEV})

            if self.supervise_single and not online_eval_only:
                output_dict['ego'].update({
                    "label_dict_single":{
                            # "pos_equal_one": torch.cat(pos_equal_one_single, dim=0),
                            # "neg_equal_one": torch.cat(neg_equal_one_single, dim=0),
                            # "targets": torch.cat(targets_single, dim=0),
                            # for centerpoint
                            "object_bbx_center_single": torch.cat(object_bbx_center_single, dim=0),
                            "object_bbx_mask_single": torch.cat(object_bbx_mask_single, dim=0)
                        },
                    "object_bbx_center_single": torch.cat(object_bbx_center_single, dim=0),
                    "object_bbx_mask_single": torch.cat(object_bbx_mask_single, dim=0)
                })

            if self.heterogeneous:
                output_dict['ego'].update({
                    "lidar_agent_record": torch.from_numpy(np.concatenate(lidar_agent_list)) # [0,1,1,0,1...]
                })
        

            return output_dict

        def collate_batch_test(self, batch, online_eval_only=False):
            """
            Customized collate function for pytorch dataloader during testing
            for late fusion dataset.

            Parameters
            ----------
            batch : dict

            Returns
            -------
            batch : dicn
                Reformatted batch.
            """
            # currently, we only support batch size of 1 during testing
            assert len(batch) <= 1, "Batch size 1 is required during testing!"
            
            self.online_eval_only = online_eval_only

            output_dict = self.collate_batch_train(batch, online_eval_only)
            if output_dict is None:
                return None
            
            batch = batch[0]
            
            if batch['ego']['anchor_box'] is not None:
                output_dict['ego'].update({'anchor_box':
                    self.anchor_box_torch})
            
            record_len = torch.from_numpy(np.array([batch['ego']['cav_num']]))
            pairwise_t_matrix = torch.from_numpy(np.array([batch['ego']['pairwise_t_matrix']]))

            output_dict['ego'].update({'record_len': record_len,
                'pairwise_t_matrix': pairwise_t_matrix
                })

            # heterogeneous
            if self.heterogeneous:
                idx = batch['ego']['idx']
                cav_list = batch['ego']['cav_list'] # ['ego', '650' ..]
                cav_num = len(batch)
                lidar_agent, camera_agent = self.selector.select_agent(idx)
                lidar_agent = lidar_agent[:cav_num] # [1,0,0,1,0]
                lidar_agent_idx = lidar_agent.nonzero()[0].tolist()
                lidar_agent_cav_id = [cav_list[index] for index in lidar_agent_idx] # ['ego', ...]
        

            # for late fusion, we also need to stack the lidar for better
            # visualization
            if self.visualize:
                projected_lidar_list = []
                origin_lidar = []
            
            for cav_id, cav_content in batch.items():
                if cav_id != 'ego':
                    output_dict.update({cav_id: {}})
                # output_dict.update({cav_id: {}})

                if not online_eval_only:
                    object_bbx_center = \
                        torch.from_numpy(np.array([cav_content['object_bbx_center']]))
                    object_bbx_mask = \
                        torch.from_numpy(np.array([cav_content['object_bbx_mask']]))
                    object_ids = cav_content['object_ids']

                # the anchor box is the same for all bounding boxes usually, thus
                # we don't need the batch dimension.
                output_dict[cav_id].update(
                    {"anchor_box": self.anchor_box_torch}
                )

                transformation_matrix = cav_content['transformation_matrix']
                
                if self.visualize:
                    origin_lidar = [cav_content['origin_lidar']]
                    if (self.params['only_vis_ego'] is False) or (cav_id=='ego'):
                        projected_lidar = copy.deepcopy(cav_content['origin_lidar'])
                        projected_lidar[:, :3] = \
                            box_utils.project_points_by_matrix_torch(
                                projected_lidar[:, :3],
                                transformation_matrix)
                        projected_lidar_list.append(projected_lidar)
                
                
                if self.load_lidar_file:
                    # processed lidar dictionary
                    #if 'processed_features' in cav_content.keys():
                    
                    merged_feature_dict = merge_features_to_dict([cav_content['processed_lidar']])
                    processed_lidar_torch_dict = \
                        self.pre_processor.collate_batch(merged_feature_dict)
                    output_dict[cav_id].update({'processed_lidar': processed_lidar_torch_dict})

                if self.load_camera_file:
                    imgs_batch = [cav_content["image_inputs"]["imgs"]]
                    rots_batch = [cav_content["image_inputs"]["rots"]]
                    trans_batch = [cav_content["image_inputs"]["trans"]]
                    intrins_batch = [cav_content["image_inputs"]["intrins"]]
                    extrinsics_batch = [cav_content["image_inputs"]["extrinsics"]]
                    post_trans_batch = [cav_content["image_inputs"]["post_trans"]]
                    post_rots_batch = [cav_content["image_inputs"]["post_rots"]]

                    output_dict[cav_id].update({
                        "image_inputs":
                            {
                                "imgs": torch.stack(imgs_batch),
                                "rots": torch.stack(rots_batch),
                                "trans": torch.stack(trans_batch),
                                "intrins": torch.stack(intrins_batch),
                                "extrinsics": torch.stack(extrinsics_batch),
                                "post_trans": torch.stack(post_trans_batch),
                                "post_rots": torch.stack(post_rots_batch),
                            }
                        }
                    )

                # heterogeneous
                if self.heterogeneous:
                    if cav_id in lidar_agent_cav_id:
                        output_dict[cav_id].pop('image_inputs')
                    else:
                        output_dict[cav_id].pop('processed_lidar')

                if not online_eval_only:
                    # label dictionary
                    label_torch_dict = \
                        self.post_processor.collate_batch([cav_content['label_dict']])
                        
                    # for centerpoint
                    label_torch_dict.update({'object_bbx_center': object_bbx_center,
                                            'object_bbx_mask': object_bbx_mask})

                # save the transformation matrix (4, 4) to ego vehicle
                transformation_matrix_torch = \
                    torch.from_numpy(
                        np.array(cav_content['transformation_matrix'])).float()
                
                # late fusion training, no noise
                transformation_matrix_clean_torch = \
                    torch.from_numpy(
                        np.array(cav_content['transformation_matrix_clean'])).float()

                if not online_eval_only:
                    output_dict[cav_id].update({'object_bbx_center': object_bbx_center,
                                                'object_bbx_mask': object_bbx_mask,
                                                'label_dict': label_torch_dict,
                                                # 'record_len': record_len,
                                                'object_ids': object_ids,})                    
                output_dict[cav_id].update({
                                            'transformation_matrix': transformation_matrix_torch,
                                            'transformation_matrix_clean': transformation_matrix_clean_torch})


                if 'cav_num' in cav_content.keys():
                    record_len = torch.from_numpy(np.array([cav_content['cav_num']]))
                    output_dict[cav_id].update({'record_len': record_len})

                if 'pairwise_t_matrix' in cav_content.keys():
                    pairwise_t_matrix = torch.from_numpy(np.array([cav_content['pairwise_t_matrix']]))
                    output_dict[cav_id].update({'pairwise_t_matrix': pairwise_t_matrix})


                
                if self.visualize:
                    origin_lidar = \
                        np.array(
                            downsample_lidar_minimum(pcd_np_list=origin_lidar))
                    origin_lidar = torch.from_numpy(origin_lidar)
                    output_dict[cav_id].update({'origin_lidar': origin_lidar})
                
            if self.visualize:
                projected_lidar_stack = [torch.from_numpy(
                    np.vstack(projected_lidar_list))]
                output_dict['ego'].update({'origin_lidar': projected_lidar_stack})

            output_dict['ego'].update({
                "sample_idx": batch['ego']['sample_idx'],
                "cav_id_list": batch['ego']['cav_id_list']
            })
            batch_record_len = output_dict['ego']['record_len']

            for cav_id in output_dict.keys():
                if 'record_len' in output_dict[cav_id].keys():
                    continue
                output_dict[cav_id].update({'record_len': batch_record_len})
            
            
            return output_dict


        def post_process(self, data_dict, output_dict):
            """
            Process the outputs of the model to 2D/3D bounding box.

            Parameters
            ----------
            data_dict : dict
                The dictionary containing the origin input data of model.

            output_dict :dict
                The dictionary containing the output of the model.

            Returns
            -------
            pred_box_tensor : torch.Tensor
                The tensor of prediction bounding box after NMS.
            gt_box_tensor : torch.Tensor
                The tensor of gt bounding box.
            """
            pred_box_tensor, pred_score = self.post_processor.post_process(
                data_dict, output_dict
            )
            gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

            return pred_box_tensor, pred_score, gt_box_tensor

        def post_process_no_fusion(self, data_dict, output_dict_ego):
            data_dict_ego = OrderedDict()
            data_dict_ego["ego"] = data_dict["ego"]
            gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

            pred_box_tensor, pred_score = self.post_processor.post_process(
                data_dict_ego, output_dict_ego
            )
            return pred_box_tensor, pred_score, gt_box_tensor
        
        def post_process_multiclass(self, data_dict, output_dict, online_eval_only=False):
            """
            Process the outputs of the model to 2D/3D bounding box.

            Parameters
            ----------
            data_dict : dict
                The dictionary containing the origin input data of model.

            output_dict :dict
                The dictionary containing the output of the model.

            Returns
            -------
            pred_box_tensor : torch.Tensor
                The tensor of prediction bounding box after NMS.
            gt_box_tensor : torch.Tensor
                The tensor of gt bounding box.
            """

            if online_eval_only == False:
                online_eval_only = self.online_eval_only

            num_class = output_dict['ego']['cls_preds'].shape[1]
            pred_box_tensor_list = []
            pred_score_list = []
            gt_box_tensor_list = []

            num_list = [0,1,3]

            for i in range(num_class):
                data_dict_single = copy.deepcopy(data_dict)
                gt_dict_single = {'ego': {}}
                gt_dict_single['ego'] = copy.deepcopy(data_dict['ego'])
                output_dict_single = copy.deepcopy(output_dict)
                if not online_eval_only:
                    data_dict_single['ego']['object_bbx_center'] = data_dict['ego']['object_bbx_center'][:,i,:,:]
                    data_dict_single['ego']['object_bbx_mask'] = data_dict['ego']['object_bbx_mask'][:,i,:]
                    data_dict_single['ego']['object_ids'] = data_dict['ego']['object_ids'][num_list[i]]
                    gt_dict_single['ego']['object_bbx_center'] = data_dict['ego']['gt_object_bbx_center'][:,i,:,:]
                    gt_dict_single['ego']['object_bbx_mask'] = data_dict['ego']['gt_object_bbx_mask'][:,i,:]
                    gt_dict_single['ego']['object_ids'] = data_dict['ego']['gt_object_ids'][num_list[i]]
                

                for cav in output_dict_single.keys():
                    output_dict_single[cav]['cls_preds'] = output_dict[cav]['cls_preds'][:,i:i+1,:,:]
                    output_dict_single[cav]['reg_preds'] = output_dict[cav]['reg_preds_multiclass'][:,i,:,:]

                pred_box_tensor, pred_score = \
                    self.post_processor.post_process(data_dict_single, output_dict_single)

                if not online_eval_only:
                    gt_box_tensor = self.post_processor.generate_gt_bbx(gt_dict_single)
                else:
                    gt_box_tensor = None

                pred_box_tensor_list.append(pred_box_tensor)
                pred_score_list.append(pred_score)
                gt_box_tensor_list.append(gt_box_tensor)

            return pred_box_tensor_list, pred_score_list, gt_box_tensor_list

        def post_process_multiclass_no_fusion(self, data_dict, output_dict_ego, online_eval_only=False):
            """
            Process the outputs of the model to 2D/3D bounding box.

            Parameters
            ----------
            data_dict : dict
                The dictionary containing the origin input data of model.

            output_dict :dict
                The dictionary containing the output of the model.

            Returns
            -------
            pred_box_tensor : torch.Tensor
                The tensor of prediction bounding box after NMS.
            gt_box_tensor : torch.Tensor
                The tensor of gt bounding box.
            """

            online_eval_only = self.online_eval_only

            num_class = data_dict['ego']['object_bbx_center'].shape[1]


            pred_box_tensor_list = []
            pred_score_list = []
            gt_box_tensor_list = []

            num_list = [0,1,3]
            
            for i in range(num_class):
                data_dict_single = copy.deepcopy(data_dict)
                gt_dict_single = {'ego': {}}
                gt_dict_single['ego'] = copy.deepcopy(data_dict['ego'])
                output_dict_single = copy.deepcopy(output_dict_ego)
                data_dict_single['ego']['object_bbx_center'] = data_dict['ego']['object_bbx_center'][:,i,:,:]
                data_dict_single['ego']['object_bbx_mask'] = data_dict['ego']['object_bbx_mask'][:,i,:]
                data_dict_single['ego']['object_ids'] = data_dict['ego']['object_ids'][num_list[i]]
                gt_dict_single['ego']['object_bbx_center'] = data_dict['ego']['gt_object_bbx_center'][:,i,:,:]
                gt_dict_single['ego']['object_bbx_mask'] = data_dict['ego']['gt_object_bbx_mask'][:,i,:]
                gt_dict_single['ego']['object_ids'] = data_dict['ego']['gt_object_ids'][num_list[i]]
                output_dict_single['ego']['cls_preds'] = output_dict_ego['ego']['cls_preds'][:,i:i+1,:,:]
                output_dict_single['ego']['reg_preds'] = output_dict_ego['ego']['reg_preds_multiclass'][:,i,:,:]
                data_dict_single_ego = OrderedDict()
                data_dict_single_ego["ego"] = data_dict_single["ego"]
                pred_box_tensor, pred_score = \
                    self.post_processor.post_process(data_dict_single_ego, output_dict_single)
                gt_box_tensor = self.post_processor.generate_gt_bbx(gt_dict_single)
                             

                pred_box_tensor_list.append(pred_box_tensor)
                pred_score_list.append(pred_score)
                gt_box_tensor_list.append(gt_box_tensor)

            return pred_box_tensor_list, pred_score_list, gt_box_tensor_list

        def post_process_no_fusion_uncertainty(self, data_dict, output_dict_ego):
            data_dict_ego = OrderedDict()
            data_dict_ego['ego'] = data_dict['ego']
            gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

            pred_box_tensor, pred_score, uncertainty = \
                self.post_processor.post_process(data_dict_ego, output_dict_ego, return_uncertainty=True)
            return pred_box_tensor, pred_score, gt_box_tensor, uncertainty

    return LatemulticlassFusionDataset
