import os
import shutil
import argparse

def copy_images_with_stride(input_dir, output_dir, stride=4):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get sorted list of image files in the input directory
    images = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # Copy images with the specified stride
    for idx, image in enumerate(images[::stride]):
        src_path = os.path.join(input_dir, image)
        dst_path = os.path.join(output_dir, f"{idx:04d}.png")
        shutil.copy(src_path, dst_path)
        print(f"Copied: {src_path} -> {dst_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Copy images with a specified stride.")
    parser.add_argument("--input_dir", type=str, help="Path to the input directory containing images.")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory to save images.")
    parser.add_argument("--stride", type=int, default=4, help="Stride value for selecting images (default: 4).")

    args = parser.parse_args()

    copy_images_with_stride(args.input_dir, args.output_dir, args.stride)