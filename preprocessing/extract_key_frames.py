import cv2
import torch
import numpy as np
from torchvision import transforms as T
from torchvision.models.optical_flow import raft_large
import argparse
import os

@torch.no_grad()
def compute_reliability_map(video_frames: torch.tensor, device="cuda:0") -> torch.tensor:
    model = raft_large(pretrained=True, progress=False).to(device).eval()

    def preprocess(batch):
        transforms = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),
            T.Resize(size=(256, 256)),
        ])
        return transforms(batch)

    reliability_maps = []
    for frame1, frame2 in zip(video_frames, video_frames[1:]):
        frame1 = preprocess(frame1.permute(2, 0, 1).unsqueeze(0)).to(device)
        frame2 = preprocess(frame2.permute(2, 0, 1).unsqueeze(0)).to(device)

        flow_forward = model(frame1, frame2)[-1][0].permute(1, 2, 0).detach().cpu().numpy()
        flow_backward = model(frame2, frame1)[-1][0].permute(1, 2, 0).detach().cpu().numpy()

        H, W = flow_forward.shape[:2]
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        X_warped = np.clip(X + flow_forward[..., 0], 0, W - 1).astype(np.int32)
        Y_warped = np.clip(Y + flow_forward[..., 1], 0, H - 1).astype(np.int32)

        flow_backward_warped = flow_backward[Y_warped, X_warped]
        consistency_error = np.linalg.norm(flow_forward + flow_backward_warped, axis=-1)

        reliability_map = np.exp(-consistency_error)
        reliability_maps.append(reliability_map)

    reliability_maps.append(reliability_maps[-1])
    return torch.tensor(np.stack(reliability_maps)).float().to(device)

def extract_frames(video_path: str, max_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(frame))
    cap.release()
    return frames

def select_key_frames(reliability_maps: torch.Tensor, top_k=4):
    motion_scores = 1 - reliability_maps.mean(dim=(1, 2))  # Higher score = more motion
    topk_indices = torch.topk(motion_scores, k=top_k).indices
    return sorted(topk_indices.tolist())

def save_selected_frames(frames, indices, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for i in indices:
        frame = frames[i].numpy()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_folder, f"keyframe_{i:04d}.png"), frame_bgr)

def main(video_path, output_dir, device="cuda:0"):
    frames = extract_frames(video_path)
    if len(frames) < 2:
        print("Not enough frames.")
        return

    video_tensor = torch.stack(frames)
    reliability_maps = compute_reliability_map(video_tensor, device=device)
    key_indices = select_key_frames(reliability_maps, top_k=8)

    print(f"Selected key frames: {key_indices}")
    save_selected_frames(frames, key_indices, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Path to video file")
    parser.add_argument("--output", help="Directory to save keyframes")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    main(args.video, args.output, args.device)
