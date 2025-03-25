#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
from math import exp
import numpy as np
from tqdm import tqdm

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 0)
    rho = torch.tensor([rho] * len(rho_hat)).cuda()
    return torch.mean(
        rho * torch.log(rho / (rho_hat + 1e-5)) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-5)))


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def compute_reliability_map_batched(rgbs: torch.Tensor, device="cuda:0") -> torch.Tensor:
    '''
    rgbs: "N H W 3" tensor
    
    Returns: "N H W 1" tensor
    '''
    def preprocess(batch):
        transforms = T.Compose(
            [
                T.ConvertImageDtype(torch.float32),
                T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
                T.Resize(size=(256, 256)),
            ]
        )
        batch = transforms(batch)
        return batch
    
    rgbs = preprocess(rgbs.permute(0, 3, 1, 2)) # "N 3 H W" tensor
    rgbs_1 = rgbs[:-1] # frames 0 to N-1
    rgbs_2 = rgbs[1:]  # frames 1 to N
    
    # Compute optical flow
    from torchvision.models.optical_flow import raft_large

    model = raft_large(pretrained=True, progress=False).to(device)
    model = model.eval()

    forward_flow = model(rgbs_1.to(device), rgbs_2.to(device))
    predicted_forward_flow = forward_flow[-1] # "N-1 2 H W" tensor 
    
    backward_flow = model(rgbs_2.to(device), rgbs_1.to(device))
    predicted_backward_flow = backward_flow[-1] # "N-1 2 H W" tensor 
    
    B, _, H, W = predicted_forward_flow.shape # B = N-1
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=0).float().unsqueeze(0).expand(B, -1, -1, -1) # "B 2 H W"
    
    coords_forward = grid + predicted_forward_flow
    
    norm_X = 2.0 * coords_forward[:, 0, :, :] / (W - 1) - 1.0
    norm_Y = 2.0 * coords_forward[:, 1, :, :] / (H - 1) - 1.0
    norm_grid = torch.stack((norm_X, norm_Y), dim=3) # "B H W 2"
    
    backward_flow_warped = F.grid_sample(predicted_backward_flow, norm_grid, align_corners=True)
    fb_error = (predicted_forward_flow + backward_flow_warped).norm(dim=1, keepdim=True)
    
    lambda_val = 1.0
    reliability = torch.exp(-lambda_val * fb_error) # "B 1 H W"
    
    reliability_list = [reliability[i] for i in range(B)]
    reliability_list.append(reliability[-1])  # duplicate last reliability for frame N-1
    reliability_tensor = torch.stack(reliability_list, dim=0)  # shape: (N, 1, H, W)
    
    # Permute to the desired output shape: (N, H, W, 1)
    reliability_tensor = reliability_tensor.permute(0, 2, 3, 1).contiguous()
    
    return reliability_tensor # N, H, W, 1
    
    
def compute_reliability_map(video_frames: torch.tensor, device="cuda:0") -> torch.tensor:
    """Computes per-frame reliability map based on optical flow consistency"""
    from torchvision.models.optical_flow import raft_large

    model = raft_large(pretrained=True, progress=False).to(device)
    model = model.eval()

    def preprocess(batch):
        transforms = T.Compose(
            [
                T.ConvertImageDtype(torch.float32),
                T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
                T.Resize(size=(256, 256)),
            ]
        )
        batch = transforms(batch)
        return batch
    
    
    reliability_maps = []
    for i, (frame1, frame2) in enumerate(zip(video_frames, video_frames[1:])):
        frame1 = preprocess(frame1[None].permute(0, 3, 1, 2)).to(device)
        frame2 = preprocess(frame2[None].permute(0, 3, 1, 2)).to(device)
        
        predicted_flow_forward = model(frame1, frame2)
        flow_forward = predicted_flow_forward[-1][0].permute(1, 2, 0).detach().cpu().numpy() # 2, H, W -> H, W, 2
        
        predicted_flow_backward = model(frame2, frame1)
        flow_backward = predicted_flow_backward[-1][0].permute(1, 2 ,0).detach().cpu().numpy()  # 2, H, W -> H, W, 2
        
        H, W = flow_forward.shape[:2]
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        X_warped = np.clip(X + flow_forward[..., 0], 0, W-1).astype(np.int32)
        Y_warped = np.clip(Y + flow_forward[..., 1], 0, H-1).astype(np.int32)

        flow_backward_warped = flow_backward[Y_warped, X_warped]
        
        consistency_error = np.linalg.norm(flow_forward + flow_backward_warped, axis=-1)
        
        reliability_map = np.exp(-consistency_error)
        reliability_maps.append(reliability_map)
        
    reliability_maps.append(reliability_maps[-1]) # duplicate last reliability map for frame N-1
    
    return torch.tensor(np.stack(reliability_maps)).float().to(device) # N, H, W


def weighted_mse_loss(gt_rgb: torch.Tensor, pred_rgb: torch.Tensor, reliability_map: torch.Tensor) -> torch.Tensor:
    """Computes weighted MSE loss"""
    # Reliability map: 4, 256, 256, 1
    # GT RGB: 4, 256, 256, 3
    # Pred RGB: 4, 256, 256, 3
    return torch.mean(reliability_map * (gt_rgb - pred_rgb) ** 2)


def temporal_smoothness_loss(vertices: torch.Tensor) -> torch.Tensor:
    """
    Computes a temporal smoothness loss by penalizing the per-vertex 
    displacement between consecutive time steps.

    Args:
        vertices (list): A list of vertices, one per time step.
    Returns:
        torch.Tensor: Scalar tensor representing the smoothness loss.
    """
    L, N, _ = vertices.shape # L: num_frames, N: num_vertices, _: 3
    #Make sure vertices are on the gpu
    diff = torch.diff(vertices, dim=0).abs() # Compute the difference between consecutive vertices
    return diff.mean() # Return the mean of the absolute difference

cosine_sim = torch.nn.CosineSimilarity()

def cosine_sum(features, targets):
    return -cosine_sim(features, targets).sum()

def cosine_avg(features, targets):
    return -cosine_sim(features, targets).mean()