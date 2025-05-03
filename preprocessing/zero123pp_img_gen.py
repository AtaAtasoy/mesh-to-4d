import cv2
import copy
import numpy
import torch
from PIL import Image
from diffusers import DiffusionPipeline, ControlNetModel
from matting_postprocess import postprocess
import numpy as np
from argparse import ArgumentParser
import os


def rescale(single_res, input_image, ratio=0.95):
    # Rescale and recenter
    image_arr = numpy.array(input_image)
    ret, mask = cv2.threshold(numpy.array(input_image.split()[-1]), 0, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(mask)
    max_size = max(w, h)
    side_len = int(max_size / ratio)
    padded_image = numpy.zeros((side_len, side_len, 4), dtype=numpy.uint8)
    center = side_len//2
    padded_image[center-h//2:center-h//2+h, center-w//2:center-w//2+w] = image_arr[y:y+h, x:x+w]
    rgba = Image.fromarray(padded_image).resize((single_res, single_res), Image.LANCZOS)
    return rgba

def generate_multiview_img(cond_img_path: str) -> Image:
    # Load the pipeline
    pipeline: DiffusionPipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2", custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16
    )
    normal_pipeline = copy.copy(pipeline)
    normal_pipeline.add_controlnet(ControlNetModel.from_pretrained(
        "sudo-ai/controlnet-zp12-normal-gen-v1", torch_dtype=torch.float16
    ), conditioning_scale=1.0)
    pipeline.to("cuda:0", torch.float16)
    normal_pipeline.to("cuda:0", torch.float16)

    cond = Image.open(cond_img_path)
    cond = cond.resize((320, 320))
    genimg = pipeline(
        cond,
        prompt='', guidance_scale=4, num_inference_steps=50, width=640, height=960
    ).images[0]
    
    normalimg = normal_pipeline(
        cond, depth_image=genimg,
        prompt='', guidance_scale=4, num_inference_steps=50, width=640, height=960
    ).images[0]
    
    genimg, normalimg = postprocess(genimg, normalimg)
    normalimg.save("normals_100step.png")
    return genimg

def extract_sub_images_from_result(result_img: np.ndarray) -> list:
    img_bins = []
    for i in range(6):
        row_idx = i // 2
        col_idx = i % 2
        
        # Calculate the pixel ranges for slicing
        start_row = 320 * row_idx
        end_row = start_row + 320
        start_col = 320 * col_idx
        end_col = start_col + 320
        
        print(f"Extracting sub-image {i+1}: ({start_row}:{end_row}, {start_col}:{end_col})")
        
        # Extract the sub-image
        sub_img = result_img[start_row:end_row, start_col:end_col]
        img_bins.append(sub_img)
        
    return img_bins


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = ArgumentParser()
    parser.add_argument("--video_frames_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    video_frame_paths = sorted([os.path.join(args.video_frames_dir, img) for img in os.listdir(args.video_frames_dir)])[32:]
    depth_elevation_azimuth_pairs = [(20, 30), (-10, 90), (20, 150), (-10, 210), (20, 270), (-10, 330)]
    os.makedirs(args.output_dir, exist_ok=True)
    
    
    for video_frame_path in video_frame_paths:
        img_name = os.path.basename(video_frame_path)
        img_name = img_name[:-4]
        video_frame_out_dir = os.path.join(args.output_dir, img_name)
        # os.makedirs(video_frame_out_dir, exist_ok=True)
        
        multi_view_img = generate_multiview_img(cond_img_path=video_frame_path)
        multi_view_img_np = np.array(multi_view_img)
        sub_images = extract_sub_images_from_result(result_img=multi_view_img_np)
        for idx, sub_img in enumerate(sub_images):
            elevation, azimuth = depth_elevation_azimuth_pairs[idx]            
            Image.fromarray(sub_img.astype(np.uint8)).save(f'{args.output_dir}/{img_name}_{idx:04d}.png')