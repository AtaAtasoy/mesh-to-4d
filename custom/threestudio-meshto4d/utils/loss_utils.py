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
from threestudio.utils.typing import *


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
    reliability_tensor = reliability_tensor.permute(0, 2, 3, 1).contiguous().cpu()
    
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
    
    return torch.tensor(np.stack(reliability_maps)).float() # N, H, W

def compute_forward_optical_flow(rgbs: torch.Tensor, device="cuda:0") -> torch.Tensor:
    """
    Computes the forward optical flow magnitude from each frame to the next using the RAFT model.
    The flow magnitude is computed as the L2 norm of the flow vector.
    
    Args:
        rgbs (torch.Tensor): Batch of RGB images with shape (N, H, W, 3) and values in [0, 1].
        device (str): Device to run the model on (default "cuda:0").
    
    Returns:
        torch.Tensor: Optical flow magnitude maps with shape (N, H, W).
                    The flow is computed for pairs [frame_i, frame_i+1] and the last map is duplicated.
    """
    from torchvision.models.optical_flow import raft_large

    # Initialize RAFT.
    model = raft_large(pretrained=True, progress=False).to(device)
    model.eval()

    # Preprocessing function: converts (H, W, 3) to normalized tensor (C, H, W).
    def preprocess(frame):
        # Convert shape: (H, W, 3) -> (3, H, W)
        if frame.dim() == 3 and frame.shape[-1] == 3:
            frame = frame.permute(2, 0, 1)
        transform_pipeline = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # Map [0,1] -> [-1,1]
        ])
        return transform_pipeline(frame)

    N, H, W, _ = rgbs.shape
    flows = []

    # Compute forward optical flow for each consecutive pair.
    with torch.no_grad():
        for i in range(N - 1):
            frame1 = preprocess(rgbs[i]).unsqueeze(0).to(device)  # shape: (1, 3, H, W)
            frame2 = preprocess(rgbs[i+1]).unsqueeze(0).to(device)  # shape: (1, 3, H, W)
            predicted_flow = model(frame1, frame2)
            # predicted_flow is a list; take the last prediction, first element in batch.
            flow_vector = predicted_flow[-1][0].permute(1, 2, 0)  # shape: (H, W, 2)
            # Compute flow magnitude (L2 norm over the 2 channels)
            flow_mag = torch.linalg.norm(flow_vector, dim=-1)  # shape: (H, W)
            flows.append(flow_mag.cpu())

    # Duplicate the last computed flow magnitude to match the input frame count.
    if flows:
        flows.append(flows[-1])
    else:
        # In case there is only one frame, create a zeros flow.
        flows.append(torch.zeros(H, W))
    
    flow_stack = torch.stack(flows, dim=0)  # shape: (N, H, W)
    return flow_stack

def select_key_frames(reliability_maps: torch.Tensor, top_k=4):
    motion_scores = 1 - reliability_maps.mean(dim=(1, 2))  # Higher score = more motion
    topk_indices = torch.topk(motion_scores, k=top_k).indices
    return sorted(topk_indices.tolist())

def load_novel_frames(novel_frames_path: str, white_background: bool = False, number_of_novel_frames: int = 6, video_length: int = 81) -> tuple[torch.Tensor, torch.Tensor]:
    import cv2
    import os
    import re
    """
    Loads key frame RGB images and their corresponding masks.
    
    Args:
        novel_frames_path (str): Path to the directory containing novel frame images.
    
    Returns:
        tuple: A tuple containing two tensors:
            - Novel frame RGB images (shape: [N, M, H, W, 3]), where M is number_of_novel_frames and N is video_length
            - Corresponding masks (shape: [N, M, H, W, 1])
    """
    
    print(f"INFO: Loading novel frames from {novel_frames_path}, white_background={white_background}, number_of_novel_frames={number_of_novel_frames}")
    
    # List and filter out files which don't follow the pattern or have novel index >= number_of_novel_frames
    novel_img_paths = sorted(os.listdir(novel_frames_path))
    novel_img_paths = [
        f for f in novel_img_paths
        if re.match(r"\d{4}_\d{4}\.png", f) and int(f.split('_')[1].split('.')[0]) < number_of_novel_frames
    ]
    
    all_novel_rgsbs = []  # Will hold all the novel frames. 6 frames for each key frame
    all_novel_masks = []  # Will hold all the masks. 6 frames for each key frame
    novel_imgs = []
    novel_masks = []
    for i, img_path in enumerate(novel_img_paths):
        rgba_path = os.path.join(novel_frames_path, img_path)
        print(f"Loading {rgba_path}")
        
        rgba = cv2.cvtColor(
            cv2.imread(rgba_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
        ).astype(np.float32) / 255.0
        rgb = rgba[..., :3]
        rgb: Float[torch.Tensor, "1 H W 3"] = (
            torch.from_numpy(rgb).unsqueeze(0).contiguous()
        )
        mask: Float[torch.Tensor, "1 H W 1"] = (
            torch.from_numpy(rgba[..., 3:] > 0.5).unsqueeze(0)
        )
        if white_background:
            rgb[~mask[..., 0], :] = 1.0
        
        novel_imgs.append(rgb)
        novel_masks.append(mask)
        if (i + 1) % number_of_novel_frames == 0:  # Group every number_of_novel_frames
            novel_imgs = torch.cat(novel_imgs, dim=0)
            novel_masks = torch.cat(novel_masks, dim=0)
            all_novel_rgsbs.append(novel_imgs)
            all_novel_masks.append(novel_masks)
            
            novel_imgs = []
            novel_masks = []
        
        if len(all_novel_rgsbs) == video_length:
            break
                    
    all_novel_rgsbs = torch.stack(all_novel_rgsbs, dim=0)  # Shape: (N, number_of_novel_frames, H, W, 3)
    all_novel_masks = torch.stack(all_novel_masks, dim=0)    # Shape: (N, number_of_novel_frames, H, W, 1)
    
    print(f"INFO: Loaded {len(all_novel_rgsbs)} key frames with {number_of_novel_frames} novel frames each.")
    
    return all_novel_rgsbs, all_novel_masks

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

def safe_cosine_similarity(features, targets, eps=1e-8):
    # Compute the dot product along the appropriate dimension
    dot = (features * targets).sum(dim=-1)
    # Compute norms
    norm_features = features.norm(p=2, dim=-1)
    norm_targets = targets.norm(p=2, dim=-1)
    # Add epsilon to denominator to avoid division by zero
    return dot / (norm_features * norm_targets + eps)

def cosine_sum(features, targets):
    return -safe_cosine_similarity(features, targets).sum()

def cosine_avg(features, targets):
    return -safe_cosine_similarity(features, targets).mean()

def robust_mask_loss(pred_mask, gt_mask, logits, lambda_reg=0.1):
    """
    pred_mask, gt_mask: [B, H, W, 1]
    logits: [1, H, W, 1] - learnable, passed through sigmoid to get weights
    """
    weights = torch.sigmoid(logits)  # values in (0, 1)
    residual = (pred_mask - gt_mask) ** 2
    weighted_loss = (weights ** 2) * residual
    reg = lambda_reg * ((1 - weights ** 2) ** 2)
    return weighted_loss.mean() + reg.mean()


def robust_mask_loss_vectorized(pred_mask, gt_mask, frame_indices, mask_logits_all, lambda_reg=0.1):
    """
    pred_mask, gt_mask: [B, H, W, 1]
    frame_indices: [B] int tensor with values in [0, 80]
    mask_logits_all: [81, H, W, 1]
    """

    B, H, W, _ = pred_mask.shape

    # Gather weights for each sample in batch: shape [B, H, W, 1]
    weights = torch.sigmoid(mask_logits_all[frame_indices])  # advanced indexing

    residual = (pred_mask - gt_mask) ** 2
    weighted_loss = (weights ** 2) * residual
    reg_term = lambda_reg * ((1 - weights ** 2) ** 2)

    return weighted_loss.mean() + reg_term.mean()

def robust_rgb_loss(pred_rgb, gt_rgb, frame_indices, rgb_logits_all, lambda_reg=0.1):
    """
    pred_rgb, gt_rgb: [B, 3, H, W]
    frame_indices: [B] int tensor in [0, 80]
    rgb_logits_all: [81, H, W, 1]
    """
    # [B, H, W, 1] → [B, 1, H, W] → [B, 3, H, W]
    weights = torch.sigmoid(rgb_logits_all[frame_indices])  # [B, H, W, 1]
    weights = weights.permute(0, 3, 1, 2)  # → [B, 1, H, W]
    weights = weights.expand(-1, 3, -1, -1)  # → [B, 3, H, W]
    weights = weights.permute(0, 2, 3, 1) # B, H, W, 3

    residual = (pred_rgb - gt_rgb) ** 2
    weighted_loss = (weights ** 2) * residual
    reg_term = lambda_reg * ((1 - weights ** 2) ** 2)

    return weighted_loss.mean() + reg_term.mean()


def to_colormap_image(tensor_2d):
    import matplotlib.pyplot as plt
    import io
    from PIL import Image
    
    fig, ax = plt.subplots()
    ax.imshow(tensor_2d.numpy(), cmap='viridis')
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    plt.close(fig)
    return image


def save_rgb_weights_as_png(logits, save_dir="rgb_weights", current_epoch=0):
    import os
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)

    weights_all = torch.sigmoid(logits).detach().cpu()  # [81, H, W, 1]

    for i in range(weights_all.shape[0]):
        weights = weights_all[i].squeeze()  # [H, W]
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
        plt.imsave(f"{save_dir}/frame_{i:04d}_epoch_{current_epoch:04d}.png", weights.numpy(), cmap="plasma")