from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from os.path import join
import argparse


def inpaint(img_path: str, mask_img_path: str, output_path: str, prompt: str, negative_prompt: str=None, num_images_per_prompt: int=1):
    #image and mask_image should be PIL images.
    #The mask structure is white for inpainting and black for keeping as is
    try:
        image = Image.open(img_path)

        mask_image = Image.open(mask_img_path)
        mask_np = np.array(mask_image)
        inverted_mask = 255 - mask_np
        mask_image = Image.fromarray(inverted_mask)
    except:
        print("Mask image not found")
        return

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float32,
    )
    pipe.to("cuda")

    inpainted_image = pipe(prompt=prompt, image=image, mask_image=mask_image, negative_prompt=negative_prompt, num_images_per_prompt=num_images_per_prompt).images

    for i, img in tqdm(enumerate(inpainted_image)):
        img.save(join(output_path, f"inpainted_{i:06d}.png"))
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='inpainting with stable diffusion')
    parser.add_argument('--image', type=str, default='image.png', help='image path')
    parser.add_argument('--mask', type=str, default='mask.png', help='mask path')
    parser.add_argument('--output', type=str, default='output', help='output folder. The output will be <output>/inpainted_<index>.png')
    parser.add_argument('--prompt', type=str, default='A photo of a cat', help='prompt')
    parser.add_argument('--negative_prompt', type=str, default="", help='negative prompt')
    parser.add_argument('--num_images_per_prompt', type=int, default=1, help='num images per prompt')
    parser.parse_args()
    
    inpaint(parser.image, parser.mask, parser.output, parser.prompt, parser.negative_prompt, parser.num_images_per_prompt)