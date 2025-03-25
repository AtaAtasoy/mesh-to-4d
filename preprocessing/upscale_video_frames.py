import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

class VideoFrameDataset(Dataset):
    def __init__(self, frames_path, transform=None):
        self.frames_path = frames_path
        self.frames = sorted(os.listdir(frames_path))
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.frames_path, self.frames[idx])
        image = Image.open(img_name).convert("RGBA")
        if self.transform:
            image = self.transform(image)
        return image

def upscale_frames(frames_path, upscale_factor, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = VideoFrameDataset(frames_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    upscale_transform = transforms.Compose([
        transforms.Resize((256 * upscale_factor, 256 * upscale_factor), interpolation=Image.Resampling.LANCZOS),
        transforms.ToPILImage()
    ])

    output_path = os.path.join(frames_path, f"upscaled_{upscale_factor}x")
    os.makedirs(output_path, exist_ok=True)

    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        for j, image in enumerate(batch):
            upscaled_image = upscale_transform(image.cpu())
            upscaled_image.save(os.path.join(output_path, f"frame_{i * batch_size + j:04d}.png"))

if __name__ == "__main__":
    frames_path = "/home/atasoy/mesh-to-4d/input/video_frames/spiderman_walking"
    upscale_factor = 2  # Change this to your desired upscale factor
    upscale_frames(frames_path, upscale_factor)