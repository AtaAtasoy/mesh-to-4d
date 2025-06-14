import argparse
import sys
import numpy as np
from PIL import Image

#!/usr/bin/env python3

def main():
    parser = argparse.ArgumentParser(
        description="Compute absolute pixel-wise difference between two images and save the result."
    )
    parser.add_argument("-img1", help="Path to the first image")
    parser.add_argument("-img2", help="Path to the second image")
    parser.add_argument("-out", help="Path to save the difference image")
    args = parser.parse_args()

    img1 = Image.open(args.img1)
    img2 = Image.open(args.img2)

    if img1.size != img2.size or img1.mode != img2.mode:
        print("Error: images must have the same dimensions and mode", file=sys.stderr)
        sys.exit(1)

    arr1 = np.array(img1, dtype=np.int16)
    arr2 = np.array(img2, dtype=np.int16)
    diff = np.abs(arr1 - arr2).astype(np.uint8)

    diff_img = Image.fromarray(diff, mode=img1.mode)
    diff_img.save(args.out)
    print(f"Saved difference image to {args.out}")

if __name__ == "__main__":
    main()