from icecream import ic
import torch
import math
import torch.nn as nn
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.utils.transformation_utils import pose_to_tfm, tfm_to_pose_torch, tfm_to_xycs_torch, xycs_to_tfm_torch

def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x

# Part1, Pose Regression Module

class PoseRegression(nn.Module):
    """
    Args:
        in_ch: 2*C

    forward:
        x: [N,2C,H,W] concatenated feature

    Returns:
        [N, 3]: x, y, yaw
    
    """
    def __init__(self, in_ch=512, hidden_ch=256):
        super(PoseRegression, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                hidden_ch, hidden_ch, kernel_size=(3, 3), stride=(2, 2), padding=1
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_features=hidden_ch, out_features=hidden_ch, bias=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(in_features=hidden_ch, out_features=hidden_ch, bias=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(in_features=hidden_ch, out_features=3, bias=True),
        )

    def forward(self, x):

        pose_reg = self.model(x)
        return pose_reg



class PoseRegressionWraper(nn.Module):
    """
    Args:
        features: [sum(cav), C, H, W], 
        record_len: list
        pairwise_t_matrix: [B, L, L, 4, 4], original pairwise_t_matrix, noise contains
    Retuens:
        pairwise_t_matrix_new: [B, L, L, 4, 4], the relative pose after correction.
    """
    def __init__(self, in_ch, hidden_ch, affine_parameter):
        super(PoseRegressionWraper, self).__init__()
        self.pose_regression = PoseRegression(
            in_ch=in_ch, hidden_ch=hidden_ch
        )
        self.H = affine_parameter['H']
        self.W = affine_parameter['W']
        self.downsample_rate = affine_parameter['downsample_rate']
        self.discrete_ratio = affine_parameter['discrete_ratio']

    def forward(self, features, record_len, pairwise_t_matrix):
        _, C, H, W = features.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(features, record_len)
        pairwise_t_matrix_new = torch.eye(4, device=pairwise_t_matrix.device).view(1,1,1,4,4).repeat(B,L,L,1,1)
        pose_corr_matrix = torch.zeros((B,L,L,3),device=pairwise_t_matrix.device)
        for b in range(B):
            N = record_len[b]
            agent_features = split_x[b]
            for i in range(N):
                t_matrix = pairwise_t_matrix[b]
                t_matrix = t_matrix[:,:,[0, 1],:][:,:,:,[0, 1, 3]] # [L, L, 2, 3]
                t_matrix[...,0,1] = t_matrix[...,0,1] * H / W
                t_matrix[...,1,0] = t_matrix[...,1,0] * W / H
                t_matrix[...,0,2] = t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
                t_matrix[...,1,2] = t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2
                # (N,C,H,W)
                neighbors = warp_affine_simple(agent_features, t_matrix[i, :N, :, :], (self.H, self.W))
                # (N,C,H,W)
                ego_agent_feature = agent_features[i].unsqueeze(0).repeat(N, 1, 1, 1)
                # (N,2C,H,W)
                neighbor_feature = torch.cat(
                        [neighbors, ego_agent_feature], dim=1)
                # (N,3)
                pose_corr = self.pose_regression(neighbor_feature)
                pose_corr_matrix[b,i,:N] = pose_corr

                # (N, 4, 4)
                pose_corr_tfm = pose_to_tfm(pose_corr)
                pairwise_t_matrix_new[b,i,:N] = pose_corr_tfm @ pairwise_t_matrix[b,i,:N]

        return pose_corr_matrix, pairwise_t_matrix_new



# Part 2, Global Consistent Module
def get_intersection(pairwise_t_matrix, affine_parameter):
    """ get intersection from pairwise_t_matrix

    Args: 
        pairwise_t_matrix: torch.Tensor, shape [L, L, 4, 4]
            pairwise transformation matrix for one frame.
            pairwise_t_matrix[i,j] = Tji, i is ego
        affine_parameter: dict
            H, W, etc.


    Returns:
        intersection: torch.Tensor, shape [L, L]
    """
    H = affine_parameter['H']
    W = affine_parameter['W']
    downsample_rate = affine_parameter['downsample_rate']
    discrete_ratio = affine_parameter['discrete_ratio']
    intersections = []

    L = pairwise_t_matrix.shape[0]
    one_tensor = torch.zeros((L,1,H,W), device=pairwise_t_matrix.device)
    for i in range(L):
        t_matrix = pairwise_t_matrix[:,:,[0, 1],:][:,:,:,[0, 1, 3]] # [L, L, 2, 3]
        t_matrix[...,0,1] = t_matrix[...,0,1] * H / W
        t_matrix[...,1,0] = t_matrix[...,1,0] * W / H
        t_matrix[...,0,2] = t_matrix[...,0,2] / (downsample_rate * discrete_ratio * W) * 2
        t_matrix[...,1,2] = t_matrix[...,1,2] / (downsample_rate * discrete_ratio * H) * 2

        # [L,1,H,W]
        neighbors = warp_affine_simple(one_tensor, t_matrix[i, :L, :, :], (H, W))
        intersection = torch.sum(neighbors, dim=[1,2,3]) / (H * W)  # [L,]
        intersections.append(intersection)

    # [L, L], intersections[i,:], ego is i
    intersections = torch.stack(intersections)

    # if intersection is zero, may meet nan later.
    eps = 0.01
    intersections += eps
    
    return intersections




def WeightedMLE(pose, pairwise_t_matrix, weight):
    """ Weighted MLE for estimate mu and sigma of multivariate student t distribution.
        simutanously for all nodes
    Args:
        pose: [N,3]
        pairwise_t_matrix: [L, L, 4, 4]
        weight: [L, L]

    Returns:
        pose_mu: [N, 3] , but [N, 4] now
        pose_sigma: [N, 3, 3], but [N, 4] now
    """

    N = pose.shape[0]
    mu_list = []
    sigma_list = []

    for i in range(N):

        neighbor_ids = list(range(N))
        neighbor_ids.remove(i)

        weights = weight[i,neighbor_ids].repeat(2) # [2(N-1)]
        relative_pose1 = pairwise_t_matrix[i,neighbor_ids] # [N-1, 4, 4]  Tji
        relative_pose2 = pairwise_t_matrix[neighbor_ids,i] # [N-1, 4, 4]  Tij
        relative_pose2 = torch.inverse(relative_pose2)
        relative_pose = torch.cat([relative_pose1,relative_pose2], dim=0) # [2(N-1), 4, 4]

        tfm = pose_to_tfm(pose[neighbor_ids]).repeat(2,1,1) # [2(N-1), 4, 4]
        samples = tfm @ relative_pose # [2(N-1), 4, 4]
        # here is one problem, -179 and +179 degree. They are close actually.
        # so we use cos and sin to replace angle
        samples = tfm_to_xycs_torch(samples).to(torch.float64) # [N, 4]


        mu = samples.median(0).values
        Sigma = torch.eye(4, device=pose.device, dtype=torch.float64)
        small_identity = torch.eye(4, device=pose.device, dtype=torch.float64) * 0.05

        diff = mu[None] - samples

        v = 2
        for _ in range(15):
            eta = (v + mu.size(0)) / (
                v + torch.einsum("ni,ij,nj->n", diff, Sigma.inverse(), diff)
            )
            mu = torch.einsum("n,n,ni->i", weights, eta, samples) / (weights * eta).sum()
            diff = mu[None] - samples
            # Sigma = torch.einsum('n,n,ni,nj->ij', weights, w, diff, diff) / weights.sum()
            Sigma = (
                torch.einsum("n,ni,nj->ij", eta, diff, diff) / diff.size(0) + small_identity
            )

        mu_list.append(mu.to(torch.float32))
        sigma_list.append(Sigma.to(torch.float32))

    pose_mu = torch.stack(mu_list)
    pose_sigma = torch.stack(sigma_list)

    return pose_mu, pose_sigma


def WeightedEM(lidar_pose, pairwise_t_matrix, intersection):
    """Weighted EM algorithm, for a single frame, not batch data
    Args:
        lidar_pose : torch.Tenosr
            shape [N, 3]
        pairwise_t_matrix: torch.Tensor
            shape [L, L, 4, 4]
        intersection: torch.Tensor
            shape [L, L]

    Returns:
        pose_mu : torch.Tensor
            new lidar pose after correction. shape [N, 3]
    """
    num_iters = 10
    pose = lidar_pose
    weight = torch.ones_like(intersection, device=intersection.device)
    
    for k in range(num_iters):
        pose_mu, pose_sigma = WeightedMLE(pose, pairwise_t_matrix, weight) # [N, 4], [N, 4, 4]
        weight = update_weight(pose_mu, pose_sigma, pairwise_t_matrix, intersection)
    
    N = lidar_pose.shape[0]
    pose_new = torch.zeros((N,3), device=lidar_pose.device, dtype=lidar_pose.dtype)
    pose_new[:,:2] = pose_mu[:,:2]
    pose_new[:,2] = torch.rad2deg(torch.atan2(pose_mu[:,3], pose_mu[:,2])) # sin, cos

    return pose_new

def update_weight(pose_mu, pose_sigma, pairwise_t_matrix, intersection):
    """ using the close form to update weight w.
    Args:
        pose_mu: [N,3], but [N, 4] now 
        pose_sigma: [N, 3, 3], but [N, 4, 4] now 
        pairwise_t_matrix: [L,L,4,4]
        interesection: [L, L]
    """
    k = 120
    df = 2 # degree of freedom
    L = intersection.shape[0]
    N = pose_mu.shape[0]
    weight = torch.zeros_like(intersection, device=intersection.device)
    for i in range(N):
        for j in range(N):
            if i!=j:
                pose_estimate1 = xycs_to_tfm_torch(pose_mu[[j]])[0] @ pairwise_t_matrix[i,j] # [4,4]
                pose_estimate2 = xycs_to_tfm_torch(pose_mu[[i]])[0] @ torch.inverse(pairwise_t_matrix[i,j]) # [4,4]
                pose_estimate = torch.stack([pose_estimate1, pose_estimate2]) # [2, 4, 4]
                pose_estimate = tfm_to_xycs_torch(pose_estimate) # [2, 4]
                weight[i,j] = k * intersection[i,j] / (k - log_t(pose_estimate, pose_mu[i], pose_sigma[i], df).sum())

    return weight



def log_t(x, mu, Sigma, df): 
    """ log pdf of t distribution
    Args:
        x: [N, 3]
        mu: [3,]
        Sigma: [3,3]
        df: int, degree of freedom

    Returns:
        log_pdf: log of the pdf
    """

    assert len(x.shape) == 2
    n, p = x.shape
    # assert mu.shape[0] == p   # for now, allow multiple mu
    assert Sigma.shape == (p, p)

    v = torch.as_tensor(df, dtype=x.dtype, device=x.device)
    p = torch.as_tensor(p, dtype=x.dtype, device=x.device)
    pi = torch.tensor(math.pi, dtype=x.dtype, device=x.device)
    half_v = v / 2.0
    half_p = p / 2.0

    log_num = (half_v + half_p).lgamma()
    log_denom = half_v.lgamma() + half_p * (v.log() + pi.log()) + 0.5 * Sigma.logdet()

    d = x - mu
    log_val = -(half_p + half_v) * torch.log(
        1 + torch.einsum("ni,ij,nj->n", d, Sigma.inverse(), d) / v
    )

    log_pdf = log_num - log_denom + log_val

    return log_pdf


# Part 3, Attention Module

class Attention(nn.Module):
    """
    Args:
        in_ch: 2*C

    forward:
        x: [N,2C,H,W] concatenated feature
    
    """
    def __init__(self, in_ch, hidden_ch=160):
        super(Attention, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hidden_ch, hidden_ch, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveMaxPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_features=hidden_ch, out_features=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.model(x)
        return out

class AttentionWrapper(nn.Module):
    """ wrapper of attention scoring
    Args:
        features: [sum(cav), C, H, W], 
        record_len: list
        pairwise_t_matrix: [B, L, L, 4, 4], original pairwise_t_matrix, noise contains
    Retuens:
        pairwise_score: [B, L, L]
            pairwise_score[i,j], ego is i.
    """
    def __init__(self, in_ch, hidden_ch, affine_parameter, learnable_alpha=True):
        super(AttentionWrapper, self).__init__()
        self.attention_net = Attention(in_ch, hidden_ch)
        self.H = affine_parameter['H']
        self.W = affine_parameter['W']
        self.downsample_rate = affine_parameter['downsample_rate']
        self.discrete_ratio = affine_parameter['discrete_ratio']
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.Tensor([0.15]))
        else:
            self.alpha = 0.35
    
    def forward(self, features, record_len, pairwise_t_matrix):
        _, C, H, W = features.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(features, record_len)
        pairwise_score = torch.zeros((B, L, L), device=features.device)
        # mask = torch.eye(L, device=features.device).expand(B,L,L)


        for b in range(B):
            N = record_len[b]
            agent_features = split_x[b]
            for i in range(N):
                t_matrix = pairwise_t_matrix[b]
                t_matrix = t_matrix[:,:,[0, 1],:][:,:,:,[0, 1, 3]] # [L, L, 2, 3]
                t_matrix[...,0,1] = t_matrix[...,0,1] * self.H / self.W
                t_matrix[...,1,0] = t_matrix[...,1,0] * self.W / self.H
                t_matrix[...,0,2] = t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
                t_matrix[...,1,2] = t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2

                # (N,C,H,W)
                neighbors = warp_affine_simple(agent_features, t_matrix[i, :N, :, :], (self.H, self.W))
                # (N,C,H,W)
                ego_agent_feature = agent_features[i].unsqueeze(0).repeat(N, 1, 1, 1)
                # (N,2C,H,W)
                neighbor_feature = torch.cat(
                        [neighbors, ego_agent_feature], dim=1)
                # (N,1)
                pairwise_score[b,i,:N] = self.attention_net(neighbor_feature).flatten()
        
        # pairwise_score *= mask

        scores = pairwise_score
        eps = 1e-4
        # pairwise_score (B, L, L). pairwise_score[b,i,j] is agent j' feature warping to agent i's coordinate
        # weight (B, L, L), normalized at dim=2
        weight = pairwise_score / (torch.sum(pairwise_score, dim=2, keepdim=True) + self.alpha + eps)

        return scores, weight
