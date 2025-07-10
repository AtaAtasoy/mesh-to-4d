import os
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np

def combine_video_frames_to_tensor(video_frames: str, output_path: str, width: int = 480, height: int = 480):
    for frame_path in tqdm(sorted(os.listdir(video_frames))):
        if frame_path.endswith('.jpg') or frame_path.endswith('.png'):
            frame = Image.open(os.path.join(video_frames, frame_path)).convert("RGBA").resize((width, height))
            white_bg = Image.new("RGBA", (width, height), (255, 255, 255, 255))
            composite = Image.alpha_composite(white_bg, frame)
            composite.save(os.path.join(output_path, frame_path))
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video frames to add white background')
    parser.add_argument('--input', required=True, help='Input directory containing video frames')
    parser.add_argument('--output', required=True, help='Output directory for processed frames')
    parser.add_argument('--width', type=int, default=480, help='Width of the output frames')
    parser.add_argument('--height', type=int, default=480, help='Height of the output frames')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    combine_video_frames_to_tensor(args.input, output_path=args.output, width=args.width, height=args.height)
    print("All frames processed and saved with white background.")