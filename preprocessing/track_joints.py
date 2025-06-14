import torch
import numpy as np
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform
)
from argparse import ArgumentParser
import os
from PIL import Image
from tqdm import tqdm
import cv2

def parse_joints(joints_path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    joints_list = []
    bone_pairs = []
    with open(joints_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5 and parts[0] == "joints":
                joints_list.append([float(parts[2]), float(parts[3]), float(parts[4])])
            elif parts[0] == "hier":
                parent_name, child_name = parts[1], parts[2]
                parent_idx = int(parent_name.replace("joint", ""))
                child_idx = int(child_name.replace("joint", ""))
                bone_pairs.append((parent_idx, child_idx))
                
    joints = np.array(joints_list, dtype=np.float32)
    bone_pairs = np.array(bone_pairs, dtype=np.int32)
    
    rest_lengths = np.linalg.norm(joints[bone_pairs[:, 0]] - joints[bone_pairs[:, 1]], axis=1)
    
    return torch.from_numpy(joints), torch.from_numpy(bone_pairs), torch.from_numpy(rest_lengths)

def project_skeletons_to_pixels(joints: torch.tensor, video_res: tuple[int, int] = (960, 960), device: torch.device = torch.device("cuda")) -> torch.tensor:
    fixed_fov = 30.0
    distance = 0.5 / np.sin(np.radians(fixed_fov/2))
    # distance = (0.5 / np.tan(np.radians(fov/2)))
    
    R, T = look_at_view_transform(dist=distance, elev=0.0, azim=0.0)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fixed_fov)
    
    projected_joints = cameras.transform_points_screen(joints, image_size=(480, 480))
    
    projected_joints_xy = projected_joints[..., :2]  # Keep only x and y coordinates
    # queried points are time, x, y
    # time is 0 for all queries 
    # x and y are projected coordinates
    query_time = torch.zeros(size=(projected_joints.shape[0], 1), device=device)
    queries = torch.cat((query_time, projected_joints_xy), dim=-1)  # Shape: (N, 3) where N is the number of joints
    
    return queries

def visualize_tracked_joints(pred_tracks: torch.Tensor,
                                pred_visibility: torch.Tensor,
                                video_frames: torch.Tensor,
                                output_path: str = "tracked_video.mp4",
                                fps: int = 8):
    """
    Draws tracked joint points on each frame and writes out a video.

    Args:
        pred_tracks: Tensor of shape (T, N, 2) containing x,y pixel coords per joint.
        pred_visibility: Tensor of shape (T, N, 1) with visibility scores (0–1).
        video_frames: Tensor of shape (T, H, W, 3) with uint8 or float frames in [0,255].
        output_path: Path to write the output .mp4 video.
        fps: Frames per second for output video.
    """
    # Move everything to CPU numpy
    tracks_np = pred_tracks.detach().cpu().numpy()       # (T, N, 2)
    vis_np    = pred_visibility.squeeze(-1).cpu().numpy()  # (T, N)
    frames_np = video_frames.detach().cpu().numpy()      # (T, H, W, 3)

    T, H, W, _ = frames_np.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    # create a directory to save individual frames
    frames_dir = os.path.splitext(output_path)[0] + "_frames"
    os.makedirs(frames_dir, exist_ok=True)

    for t in range(T):
        frame = frames_np[t].astype("uint8").copy()
        # convert RGB to BGR for correct colors in OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for n, (x, y) in enumerate(tracks_np[t]):
            if vis_np[t, n] > 0.3:
                cv2.circle(
                    frame,
                    center=(int(x), int(y)),
                    radius=4,
                    color=(255, 255, 0),
                    thickness=-1
                )
        writer.write(frame)
        # also save each frame as an image
        frame_path = os.path.join(frames_dir, f"{t:04d}.png")
        cv2.imwrite(frame_path, frame)

    writer.release()
    print(f"Video saved at {output_path}")

def combine_video_frames_to_tensor(video_frames: str, device: torch.device = torch.device("cuda")) -> torch.tensor:
    frames = []
    for frame_path in tqdm(sorted(os.listdir(video_frames))):
        if frame_path.endswith('.jpg') or frame_path.endswith('.png'):
            frame = np.array(Image.open(os.path.join(video_frames, frame_path)).resize((480, 480)))
            frame = frame[..., :3]  # Ensure we only take RGB channels
            frames.append(frame)
    frames_np = np.array(frames, dtype=np.float32)
    print(frames_np.shape) # B T H W C
    return torch.from_numpy(frames_np)[None].to(device).permute(0, 1, 4, 2, 3).float()  # B T C H W
    
if __name__ == "__main__":
    parser = ArgumentParser(description="Project skeleton joints to pixel coordinates.")
    parser.add_argument("--joints_path", type=str, required=True, help="Path to the joints file.")
    parser.add_argument("--video_frames", type=str, required=True, help="Path to the video frames directory.")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    frames = combine_video_frames_to_tensor(args.video_frames).to(device)
    print(f"Frames shape: {frames.shape}")  # Debugging line to check frames shape
    
    video_res = (frames.shape[2], frames.shape[3])  # Extract resolution from frames
    
    joints, _, _= parse_joints(args.joints_path).to(device)
    projected_joints = project_skeletons_to_pixels(joints, video_res=video_res, device=device)
    
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    
    # # Run Offline CoTracker:
    pred_tracks, pred_visibility = cotracker(frames, queries=projected_joints[None]) # B T N 2,  B T N 1
    visualize_tracked_joints(pred_tracks=pred_tracks[0], pred_visibility=pred_visibility[0], video_frames=frames[0].permute(0, 2, 3, 1))  # Visualize the first batch of frames
