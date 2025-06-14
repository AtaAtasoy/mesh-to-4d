import os
from tqdm import tqdm
from PIL import Image
import numpy as np



def combine_video_frames_to_tensor(video_frames: str, output_path: str):
    frames = []
    for frame_path in tqdm(sorted(os.listdir(video_frames))):
        if frame_path.endswith('.jpg') or frame_path.endswith('.png'):
            frame = Image.open(os.path.join(video_frames, frame_path)).convert("RGBA").resize((480, 480))
            white_bg = Image.new("RGBA", (480, 480), (255, 255, 255, 255))
            composite = Image.alpha_composite(white_bg, frame)
            
            composite.save(os.path.join(output_path, frame_path))  # Save the composite image
            

if __name__ == "__main__":
    video_frames_path = "input/video_frames/humanoid_robot/dancing_1"  # Replace with your actual path
    output_path = "input/video_frames/humanoid_robot/dancing_1_with_white_bg"  # Output path for processed frames
    os.makedirs(output_path, exist_ok=True)  # Ensure output directory exists
    combine_video_frames_to_tensor(video_frames_path, output_path=output_path)
    print("All frames processed and saved with white background.") 