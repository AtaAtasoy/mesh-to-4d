import cv2
import os
import argparse


def extract_frames(video_path, output_dir):
    """
    Extract frames from a video file and save them as images.
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory to save the extracted frames
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    video = cv2.VideoCapture(video_path)
    
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"FPS: {fps}")
    print(f"Total frames: {frame_count}")
    
    # Read frames
    frame_number = 0
    while True:
        success, frame = video.read()
        if not success:
            break
            
        frame_path = os.path.join(output_dir, f"{frame_number:04d}.png")
        cv2.imwrite(frame_path, frame)
        
        frame_number += 1
        if frame_number % 100 == 0:
            print(f"Processed {frame_number} frames")
    
    video.release()
    print("Frame extraction completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("--video_path", type=str, help="Path to the input video file")
    parser.add_argument("--output_path", type=str, help="Directory to save the extracted frames")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    extract_frames(args.video_path, args.output_path)