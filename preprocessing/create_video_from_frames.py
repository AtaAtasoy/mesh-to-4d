import imageio
import numpy as np
from PIL import Image
import os
import argparse
from tqdm import tqdm
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create video from rendered images')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing the inpainted images')
    parser.add_argument('--output_video_path', type=str, required=True, help='Path to save the output video')
    parser.add_argument('--frame_limit', type=int, default=32, help='Number of frames to include in the video')
    parser.add_argument('--fps', type=int, default=8, help='fps of the output video')
    args = parser.parse_args()
    
    # Read the input directory
    input_dir = args.input_dir
    output_video_path = args.output_video_path
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    batch_size = args.frame_limit
    
    images = []
    for i in range(args.frame_limit):
        image_path = os.path.join(input_dir, f"{i}.png")
        images.append(np.array(Image.open(image_path)))
        shutil.copyfile(image_path, os.path.join(os.path.dirname(output_video_path), f"{i}.png"))
        
    # Save the images as a video
    with imageio.get_writer(output_video_path, mode='I', fps=args.fps) as writer:
        for j in tqdm(range(batch_size)):
            writer.append_data(images[j])
