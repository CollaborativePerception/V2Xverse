from matplotlib import pyplot as plt
import numpy as np
import copy

from opencood.tools.inference_utils import get_cav_box
import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev

def visualize(infer_result, pcd, pcd_len, pc_range, save_path, method='all', left_hand=False, img_dict=None):
        """
        Visualize the prediction, ground truth with point cloud together.
        They may be flipped in y axis. Since carla is left hand coordinate, while kitti is right hand.

        Parameters
        ----------
        infer_result:
            pred_box_tensor : torch.Tensor
                (N, 8, 3) prediction.

            gt_tensor : torch.Tensor
                (N, 8, 3) groundtruth bbx
            
            uncertainty_tensor : optional, torch.Tensor
                (N, ?)

            lidar_agent_record: optional, torch.Tensor
                (N_agnet, )


        pcd : torch.Tensor
            PointCloud, (N, 4).

        pc_range : list
            [xmin, ymin, zmin, xmax, ymax, zmax]

        save_path : str
            Save the visualization results to given path.

        dataset : BaseDataset
            opencood dataset object.

        method: str, 'bev' or '3d'

        """

        pc_range = [int(i) for i in pc_range]
        pcd_np = pcd.cpu().numpy()

        split_list = [0]
        for length in range(len(pcd_len)):
            split_list.append(split_list[length]+pcd_len[length])
            

        pred_box_tensor = infer_result.get("pred_box_tensor", None)
        gt_box_tensor = infer_result.get("gt_box_tensor", None)

        if pred_box_tensor is not None:
            pred_box_np = pred_box_tensor.cpu().numpy()
            pred_name = ['pred'] * pred_box_np.shape[0]

            score = infer_result.get("score_tensor", None)
            if score is not None:
                score_np = score.cpu().numpy()
                pred_name = [f'score:{score_np[i]:.3f}' for i in range(score_np.shape[0])]

            uncertainty = infer_result.get("uncertainty_tensor", None)
            if uncertainty is not None:
                uncertainty_np = uncertainty.cpu().numpy()
                uncertainty_np = np.exp(uncertainty_np)
                d_a_square = 1.6**2 + 3.9**2
                
                if uncertainty_np.shape[1] == 3:
                    uncertainty_np[:,:2] *= d_a_square
                    uncertainty_np = np.sqrt(uncertainty_np) 
                    # yaw angle is in radian, it's the same in g2o SE2's setting.

                    pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:.3f} a_u:{uncertainty_np[i,2]:.3f}' \
                                    for i in range(uncertainty_np.shape[0])]

                elif uncertainty_np.shape[1] == 2:
                    uncertainty_np[:,:2] *= d_a_square
                    uncertainty_np = np.sqrt(uncertainty_np) # yaw angle is in radian

                    pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:3f}' \
                                    for i in range(uncertainty_np.shape[0])]

                elif uncertainty_np.shape[1] == 7:
                    uncertainty_np[:,:2] *= d_a_square
                    uncertainty_np = np.sqrt(uncertainty_np) # yaw angle is in radian

                    pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:3f} a_u:{uncertainty_np[i,6]:3f}' \
                                    for i in range(uncertainty_np.shape[0])]                    

        if gt_box_tensor is not None:
            gt_box_np = gt_box_tensor.cpu().numpy()
            gt_name = ['gt'] * gt_box_np.shape[0]

        if method == 'bev' or method == 'all':
            canvas_b = canvas_bev.Canvas_BEV_heading_right(canvas_shape=((pc_range[4]-pc_range[1])*10, (pc_range[3]-pc_range[0])*10),
                                            canvas_x_range=(pc_range[0], pc_range[3]), 
                                            canvas_y_range=(pc_range[1], pc_range[4]),
                                            left_hand=left_hand)
            
            canvas_b_single = []
            for k in range(len(pcd_len)):
                canvas_b_single.append(canvas_bev.Canvas_BEV_heading_right(canvas_shape=((pc_range[4]-pc_range[1])*10, (pc_range[3]-pc_range[0])*10),
                                            canvas_x_range=(pc_range[0], pc_range[3]), 
                                            canvas_y_range=(pc_range[1], pc_range[4]),
                                            left_hand=left_hand))

            color_list = [(255,200,200), (200,255,200), (200,200,255), (225,225,200), (225,200,225)]
            for k in range(len(pcd_len)):
                pcd_np_single = pcd_np[split_list[k]:split_list[k+1]]
                canvas_xy, valid_mask = canvas_b.get_canvas_coords(pcd_np_single.copy()) # Get Canvas Coords
                canvas_b.draw_canvas_points(canvas_xy[valid_mask], colors=color_list[k],  radius=-1) # Only draw valid points

                canvas_xy, valid_mask = canvas_b_single[k].get_canvas_coords(pcd_np_single.copy()) # Get Canvas Coords
                canvas_b_single[k].draw_canvas_points(canvas_xy[valid_mask], colors=color_list[k],  radius=1) # Only draw valid points    

                if gt_box_tensor is not None:
                    canvas_b_single[k].draw_boxes(gt_box_np.copy(),colors=(0,255,0), texts=gt_name)
                if pred_box_tensor is not None:
                    canvas_b_single[k].draw_boxes(pred_box_np.copy(), colors=(255,0,0), texts=pred_name)

            if gt_box_tensor is not None:
                canvas_b.draw_boxes(gt_box_np.copy(),colors=(0,255,0), texts=gt_name)
            if pred_box_tensor is not None:
                canvas_b.draw_boxes(pred_box_np.copy(), colors=(255,0,0), texts=pred_name)

            # heterogeneous
            lidar_agent_record = infer_result.get("lidar_agent_record", None)
            cav_box_np = infer_result.get("cav_box_np", None)
            if lidar_agent_record is not None:
                cav_box_np = copy.deepcopy(cav_box_np)
                for i, islidar in enumerate(lidar_agent_record):
                    text = ['lidar'] if islidar else ['camera']
                    color = (0,191,255) if islidar else (255,185,15)
                    canvas_b.draw_boxes(cav_box_np[i:i+1].copy(), colors=color, texts=text)
                    for k in range(len(pcd_len)):
                        canvas_b_single[k].draw_boxes(cav_box_np[i:i+1].copy(), colors=color, texts=text)


        if method == '3d' or method == 'all':
            canvas_3 = canvas_3d.Canvas_3D(left_hand=left_hand,                
                                           canvas_shape=(1000, 1000),
                                           camera_center_coords = (-20, 0, 16), # (-20, 0, 16)
                                           camera_focus_coords = (0, 0, -4))  # (0, 0, -4)

            canvas_3_single = []
            for k in range(len(pcd_len)):
                canvas_3_s = canvas_3d.Canvas_3D(left_hand=left_hand,                
                                canvas_shape=(1000, 1000),
                                camera_center_coords = (-20, 0, 16),
                                camera_focus_coords = (0, 0, -4))
                canvas_3_single.append(canvas_3_s)

            color_list = [(255,200,200), (200,255,200), (200,200,255), (225,225,200), (225,200,225)]
            for k in range(len(pcd_len)):
                pcd_np_single = pcd_np[split_list[k]:split_list[k+1]]
                canvas_xy, valid_mask = canvas_3.get_canvas_coords(pcd_np_single.copy()) # Get Canvas Coords
                canvas_3.draw_canvas_points(canvas_xy[valid_mask], colors=color_list[k],  radius=-1) # Only draw valid points

                canvas_xy, valid_mask = canvas_3_single[k].get_canvas_coords(pcd_np_single.copy()) # Get Canvas Coords
                canvas_3_single[k].draw_canvas_points(canvas_xy[valid_mask], colors=color_list[k],  radius=1) # Only draw valid points    

                if gt_box_tensor is not None:
                    canvas_3_single[k].draw_boxes(gt_box_np.copy(),colors=(0,255,0), texts=gt_name)
                if pred_box_tensor is not None:
                    canvas_3_single[k].draw_boxes(pred_box_np.copy(), colors=(255,0,0), texts=pred_name)

            # canvas_xy, valid_mask = canvas_3.get_canvas_coords(pcd_np)
            # canvas_3.draw_canvas_points(canvas_xy[valid_mask])

            if gt_box_tensor is not None:
                canvas_3.draw_boxes(gt_box_np.copy(),colors=(0,255,0), texts=gt_name)
            if pred_box_tensor is not None:
                canvas_3.draw_boxes(pred_box_np.copy(), colors=(255,0,0), texts=pred_name)

            # heterogeneous
            lidar_agent_record = infer_result.get("lidar_agent_record", None)
            cav_box_np = infer_result.get("cav_box_np", None)
            if lidar_agent_record is not None:
                cav_box_np = copy.deepcopy(cav_box_np)
                for i, islidar in enumerate(lidar_agent_record):
                    text = ['lidar'] if islidar else ['camera']
                    color = (0,191,255) if islidar else (255,185,15)
                    canvas_3.draw_boxes(cav_box_np[i:i+1].copy(), colors=color, texts=text)
                    for k in range(len(pcd_len)):
                        canvas_3_single[k].draw_boxes(cav_box_np[i:i+1], colors=color, texts=text)

        else:
            raise(f"Not Completed for f{method} visualization.")

        
        # plt.subplot(331)
        ax1 = plt.subplot(331)
        ax1.imshow(img_dict['img_left'])
        ax1.set_title('left image', fontsize= 5)
        ax1.axis('off')        

        # plt.subplot(332)
        ax2 = plt.subplot(332)
        ax2.imshow(img_dict['img_front'])
        ax2.set_title('front image', fontsize= 5)
        ax2.axis('off')        

        # plt.subplot(333)
        ax3 = plt.subplot(333)
        ax3.imshow(img_dict['img_right'])
        ax3.set_title('right image', fontsize= 5)
        ax3.axis('off')

        ax6 = plt.subplot(3,3,4)
        ax6.imshow(canvas_3_single[0].canvas)  # np.rot90(canvas_3.canvas, k=1, axes=(1,0))
        ax6.set_title('3d point cloud', fontsize= 5)
        ax6.axis('off')

        ax7 = plt.subplot(3,3,5)
        if len(canvas_3_single)>2:
            ax7.imshow(canvas_3_single[1].canvas)  # np.rot90(canvas_3.canvas, k=1, axes=(1,0))
            ax7.set_title('3d point cloud', fontsize= 5)
        ax7.axis('off')

        ax8 = plt.subplot(3,3,6)
        if len(canvas_3_single)>2:
            ax8.imshow(canvas_3_single[2].canvas)  # np.rot90(canvas_3.canvas, k=1, axes=(1,0))
            ax8.set_title('3d point cloud', fontsize= 5)
        ax8.axis('off')

        ax4 = plt.subplot2grid((3,3),(2,1),colspan = 1)  # colspan = 2
        ax4.imshow(np.rot90(canvas_b.canvas, k=-1, axes=(1,0)))  # np.rot90(canvas_b.canvas, k=1, axes=(1,0))
        ax4.set_title('bev point cloud', fontsize= 5)
        ax4.axis('off')

        ax5 = plt.subplot(337)
        ax5.imshow(canvas_3.canvas)  # np.rot90(canvas_3.canvas, k=1, axes=(1,0))
        ax5.set_title('3d point cloud', fontsize= 5)
        ax5.axis('off')

        ax5 = plt.subplot(339)
        ax5.imshow(np.rot90(img_dict['BEV'], k=0, axes=(1,0)))  # np.rot90(canvas_3.canvas, k=1, axes=(1,0))
        ax5.set_title('BEV gt', fontsize= 5)
        ax5.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, transparent=False, dpi=1000)
        plt.clf()
        plt.close()


