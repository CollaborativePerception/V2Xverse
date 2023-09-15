"""
Written by Jinhyung Park

Simple BEV visualization for 3D points & boxes.
"""

import numpy as np
import cv2
import copy
from functools import partial
import matplotlib

class Canvas_BEV(object):
    def __init__(self, 
                 canvas_shape=(1000, 1000),
                 canvas_x_range=(-50, 50),
                 canvas_y_range=(-50, 50),
                 canvas_bg_color=(0, 0, 0)):
        """
        Args:
            canvas_shape (Tuple[int]): Shape of BEV Canvas image. First element
                corresponds to X range, the second element to Y range.
            canvas_x_range (Tuple[int]): Range of X-coords to visualize. X is
                vertical: negative ~ positive is top ~ down.
            canvas_y_range (Tuple[int]): Range of Y-coords to visualize. Y is
                horizontal: negative ~ positive is left ~ right.
            canvas_bg_color (Tuple[int]): RGB (0 ~ 255) of Canvas background
                color.
        """

        # Sanity check ratios
        if ((canvas_shape[0] / canvas_shape[1]) != 
            ((canvas_x_range[0] - canvas_x_range[1]) / 
             (canvas_y_range[0] - canvas_y_range[1]))):

            print("Not an error, but the x & y ranges are not "\
                  "proportional to canvas height & width.")
        
        self.canvas_shape = canvas_shape
        self.canvas_x_range = canvas_x_range
        self.canvas_y_range = canvas_y_range
        self.canvas_bg_color = canvas_bg_color
        
        self.clear_canvas()
    
    def get_canvas(self):
        return self.canvas

    def clear_canvas(self):
        self.canvas = np.zeros((*self.canvas_shape, 3), dtype=np.uint8)
        self.canvas[..., :] = self.canvas_bg_color

    def get_canvas_coords(self, xy):
        """
        Args:
            xy (ndarray): (N, 2+) array of coordinates. Additional columns
                beyond the first two are ignored.
        
        Returns:
            canvas_xy (ndarray): (N, 2) array of xy scaled into canvas 
                coordinates. Invalid locations of canvas_xy are clipped into 
                range. "x" is dim0, "y" is dim1 of canvas.
            valid_mask (ndarray): (N,) boolean mask indicating which of 
                canvas_xy fits into canvas.
        """
        xy = np.copy(xy) # prevent in-place modifications

        x = xy[:, 0]
        y = xy[:, 1]

        # Get valid mask
        valid_mask = ((x > self.canvas_x_range[0]) & 
                      (x < self.canvas_x_range[1]) &
                      (y > self.canvas_y_range[0]) & 
                      (y < self.canvas_y_range[1]))
        
        # Rescale points
        x = ((x - self.canvas_x_range[0]) / 
             (self.canvas_x_range[1] - self.canvas_x_range[0]))
        x = x * self.canvas_shape[0]
        x = np.clip(np.around(x), 0, 
                    self.canvas_shape[0] - 1).astype(np.int32)
                    
        y = ((y - self.canvas_y_range[0]) / 
             (self.canvas_y_range[1] - self.canvas_y_range[0]))
        y = y * self.canvas_shape[1]
        y = np.clip(np.around(y), 0, 
                    self.canvas_shape[1] - 1).astype(np.int32)
        
        # Return
        canvas_xy = np.stack([x, y], axis=1)

        return canvas_xy, valid_mask
                                      

    def draw_canvas_points(self, 
                           canvas_xy,
                           radius=-1,
                           colors=None,
                           colors_operand=None):
        """
        Draws canvas_xy onto self.canvas.

        Args:
            canvas_xy (ndarray): (N, 2) array of *valid* canvas coordinates.
                "x" is dim0, "y" is dim1 of canvas.
            radius (Int): 
                -1: Each point is visualized as a single pixel.
                r: Each point is visualized as a circle with radius r.
            colors: 
                None: colors all points white.
                Tuple: RGB (0 ~ 255), indicating a single color for all points.
                ndarray: (N, 3) array of RGB values for each point.
                String: Such as "Spectral", uses a matplotlib cmap, with the
                    operand (the value cmap is called on for each point) being 
                    colors_operand. If colors_operand is None, uses normalized
                    distance from (0, 0) of XY point coords.
            colors_operand (ndarray | None): (N,) array of values cooresponding
                to canvas_xy, to be used only if colors is a cmap.
        """
        if len(canvas_xy) == 0:
            return 
            
        if colors is None:
            colors = np.full(
                (len(canvas_xy), 3), fill_value=255, dtype=np.uint8)
        elif isinstance(colors, tuple):
            assert len(colors) == 3
            colors_tmp = np.zeros((len(canvas_xy), 3), dtype=np.uint8)
            colors_tmp[..., :] = np.array(colors)
            colors = colors_tmp
        elif isinstance(colors, np.ndarray):
            assert len(colors) == len(canvas_xy)
            colors = colors.astype(np.uint8)
        elif isinstance(colors, str):
            colors = matplotlib.cm.get_cmap(colors)
            if colors_operand is None:
                # Get distances from (0, 0) (albeit potentially clipped)
                origin_center = self.get_canvas_coords(np.zeros((1, 2)))[0][0]
                colors_operand = np.sqrt(
                    ((canvas_xy - origin_center) ** 2).sum(axis=1))
                    
            # Normalize 0 ~ 1 for cmap
            colors_operand = colors_operand - colors_operand.min()
            colors_operand = colors_operand / colors_operand.max()
        
            # Get cmap colors - note that cmap returns (*input_shape, 4), with
            # colors scaled 0 ~ 1
            colors = (colors(colors_operand)[:, :3] * 255).astype(np.uint8)
        else:
            raise Exception(
                "colors type {} was not an expected type".format(type(colors)))

        if radius == -1:
            self.canvas[canvas_xy[:, 0], canvas_xy[:, 1], :] = colors
        else:
            for color, (x, y) in zip(colors.tolist(), canvas_xy.tolist()):
                self.canvas = cv2.circle(self.canvas, (y, x), radius, color, 
                                         -1, lineType=cv2.LINE_AA)

    def draw_boxes(self,
                   boxes,
                   colors=None,
                   texts=None,
                   box_line_thickness=2,
                   box_text_size=0.5,
                   text_corner=0):
        """
        Draws a set of boxes onto the canvas.
        Args:
            boxes (ndarray): Can either be of shape:
                (N, 7): Then, assumes (x, y, z, x_size, y_size, z_size, yaw)
                (N, 5): Then, assumes (x, y, x_size, y_size, yaw)
                Everything is in the same coordinate system as points 
                (not canvas coordinates)
            colors: 
                None: colors all points white.
                Tuple: RGB (0 ~ 255), indicating a single color for all points.
                ndarray: (N, 3) array of RGB values for each point.
            texts (List[String]): Length N; text to write next to boxes.
            box_line_thickness (int): cv2 line/text thickness
            box_text_size (float): cv2 putText size
            text_corner (int): 0 ~ 3. Which corner of 3D box to write text at.
        """
        # Setup colors
        if colors is None:
            colors = np.full((len(boxes), 3), fill_value=255, dtype=np.uint8)
        elif isinstance(colors, tuple):
            assert len(colors) == 3
            colors_tmp = np.zeros((len(boxes), 3), dtype=np.uint8)
            colors_tmp[..., :len(colors)] = np.array(colors)
            colors = colors_tmp
        elif isinstance(colors, np.ndarray):
            assert len(colors) == len(boxes)
            colors = colors.astype(np.uint8)
        else:
            raise Exception(
                "colors type {} was not an expected type".format(type(colors)))

        boxes = np.copy(boxes) # prevent in-place modifications
        assert len(boxes.shape) == 2
        
        if boxes.shape[-1] == 7:
            boxes = boxes[:, [0, 1, 3, 4, 6]]
        else:
            assert boxes.shape[-1] == 5
        
        ## Get the BEV four corners
        # Get BEV 4 corners in box canonical coordinates
        bev_corners = np.array([[
            [0.5, 0.5],
            [-0.5, 0.5],
            [-0.5, -0.5],
            [0.5, -0.5]
        ]]) * boxes[:, None, [2, 3]] # N x 4 x 2

        # Get rotation matrix from yaw
        rot_sin = np.sin(boxes[:, -1])
        rot_cos = np.cos(boxes[:, -1])
        rot_matrix = np.stack([
            [rot_cos, -rot_sin],
            [rot_sin, rot_cos]
        ]) # 2 x 2 x N

        # Rotate BEV 4 corners. Result: N x 4 x 2
        bev_corners = np.einsum('aij,jka->aik', bev_corners, rot_matrix)

        # Translate BEV 4 corners
        bev_corners = bev_corners + boxes[:, None, [0, 1]]

        ## Transform BEV 4 corners to canvas coords
        bev_corners_canvas, valid_mask = \
            self.get_canvas_coords(bev_corners.reshape(-1, 2))
        bev_corners_canvas = bev_corners_canvas.reshape(*bev_corners.shape)
        valid_mask = valid_mask.reshape(*bev_corners.shape[:-1])

        # At least 1 corner in canvas to draw.
        valid_mask = valid_mask.sum(axis=1) > 0
        bev_corners_canvas = bev_corners_canvas[valid_mask]
        if texts is not None:
            texts = np.array(texts)[valid_mask]

        ## Draw onto canvas
        # Draw the outer boundaries
        idx_draw_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for i, (color, curr_box_corners) in enumerate(
                zip(colors.tolist(), bev_corners_canvas)):
                
            curr_box_corners = curr_box_corners.astype(np.int32)
            for start, end in idx_draw_pairs:
                self.canvas = cv2.line(self.canvas,
                                       tuple(curr_box_corners[start][::-1]\
                                        .tolist()),
                                       tuple(curr_box_corners[end][::-1]\
                                        .tolist()),
                                       color=color,
                                       thickness=box_line_thickness)
            if texts is not None:
                self.canvas = cv2.putText(self.canvas,
                                          str(texts[i]),
                                          tuple(curr_box_corners[text_corner]\
                                            [::-1].tolist()),
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          box_text_size,
                                          color=color,
                                          thickness=box_line_thickness)
