'''
Functions: Transform waypoints to bev maps.
'''

import numpy as np
import torch.nn.functional as F
import torch

def global2grid(waypoints, grid_coord=[96,288,1/2,3/4], det_range=[-36,-12,-10,36,12,10]):
    X, Y, r_x, r_y = grid_coord
    center_y, center_x = Y * r_y, X * r_x
    waypoints *= grid_coord[1]/(det_range[3]-det_range[0])
    waypoints[:,:,0] = waypoints[:,:,0] + center_x
    waypoints[:,:,1] = waypoints[:,:,1] + center_y
    return waypoints

def waypoints2map(waypoints, grid_coord=[192,96,3/4,1/2]):
    Y, X, r_y, r_x = grid_coord
    B, N, _ = waypoints.shape # [B, N, 2]
    bev_map = np.zeros([B, Y, X])
    grids = global2grid(waypoints)
    grids = np.array(grids, dtype=np.uint8)
    batch_idx = np.repeat(np.arange(B),N)
    x_idx = grids[:,:,0].flatten()
    y_idx = grids[:,:,1].flatten()
    valid_mask = (y_idx > (-1)) * (y_idx < Y) * (x_idx > (-1)) * (x_idx < X)
    valid_idx = np.where(valid_mask*1)[0]
    bev_map[batch_idx[valid_idx], y_idx[valid_idx], x_idx[valid_idx]] = 1
    # print(bev_map.sum())
    # print(len(valid_idx))
    return bev_map

def gradcam_resize(bev_map, scale=50):
    '''
    bev_map: [B,Y,X] torch.tensor
    '''
    bev_map = torch.Tensor(bev_map)
    bev_map = bev_map.unsqueeze(1)
    bev_map_expand = F.max_pool2d(bev_map, scale, stride=1, padding=(scale-1)//2)
    return bev_map_expand.squeeze(1)


def gaussian_2d(shape, sigma=1):
    """Generate gaussian map.

    Args:
        shape (list[int]): Shape of the map.
        sigma (float): Sigma to generate gaussian map.
            Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_radius(det_size, min_overlap=0.5):
    """Get radius of gaussian.

    Args:
        det_size (tuple[torch.Tensor]): Size of the detection result.
        min_overlap (float): Gaussian_overlap. Defaults to 0.5.

    Returns:
        torch.Tensor: Computed radius.
    """
    height, width = det_size
    
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)

 
def draw_gaussian(heatmap, center, radius, ratio=5, k=1):
    """Get gaussian masked heatmap.

    Args:
        heatmap (torch.Tensor): Heatmap to be masked.
        center (torch.Tensor): Center coord of the heatmap.
        radius (int): Radius of gausian.
        K (int): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        torch.Tensor: Masked heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter/ratio )

    # x, y = int(center[0]), int(center[1])
    x, y = int(center[1]), int(center[0])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                                radius - left:radius + right]
    
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        # torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    #     masked_heatmap = np.max([masked_heatmap[None,], (masked_gaussian * k)[None,]], axis=0)[0]
    # heatmap[y - top:y + bottom, x - left:x + right] = masked_heatmap
    return heatmap

def draw_heatmap(heatmap, x, y, radius=50, sigma=5):
    feature_map_size = heatmap.shape

    # throw out not in range objects to avoid out of array
    # area when creating the heatmap
    if not (0 <= x < feature_map_size[0]
            and 0 <= y < feature_map_size[1]):
        return heatmap

    heatmap = draw_gaussian(heatmap, (x,y), radius, sigma) 
    return heatmap


def waypoints2map_radius(waypoints, radius=40, sigma_reverse=5, grid_coord=[96,288,1/2,3/4], det_range=[-36,-12,-10,36,12,10]):

    waypoints[:,:,1] *= -1
    X, Y, r_x, r_y = grid_coord
    B, N, _ = waypoints.shape # [B, N, 2]
    bev_map = np.zeros([B, X, Y])
    grids = global2grid(waypoints, grid_coord=grid_coord, det_range=det_range)
    # grids = np.array(grids, dtype=np.uint8)
    batch_idx = np.repeat(np.arange(B),N)
    x_idx = grids[:,:,0].flatten()
    y_idx = grids[:,:,1].flatten()
    valid_mask = (y_idx > (-1)) * (y_idx < Y) * (x_idx > (-1)) * (x_idx < X)
    valid_idx = np.where(valid_mask*1)[0]

    radius *= grid_coord[0]/96*24/(det_range[4]-det_range[1])
    radius = int(radius)
    
    for i in valid_idx:
        b = batch_idx[i]
        x = x_idx[i]
        y = y_idx[i]
        bev_map[b] = draw_heatmap(bev_map[b], x, y, radius, sigma_reverse)

    return bev_map
