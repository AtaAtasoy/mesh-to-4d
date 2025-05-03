import torch
from diffusers import FluxPipeline, AutoPipelineForText2Image
from argparse import ArgumentParser
import os
from random import randint


def t2i(prompt: str, num_images_per_prompt: int=5, num_inference_steps: int=4, max_sequence_length: int=256, guidance_scale: float=0.0, output_dir: str="output"):

    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power    pipe.to("cuda")
    os.makedirs(output_dir, exist_ok=True)
    
    image = pipe(
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=max_sequence_length,
        generator=torch.Generator("cpu").manual_seed(randint(0, 1000000)),
        num_images_per_prompt=num_images_per_prompt,
        height=480,
        width=720,
    ).images
    
    
    for i, img in enumerate(image):
        img.save(f"{output_dir}/{prompt}_{i}.png")


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate images from text prompts.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to generate images from.")
    parser.add_argument("--num_images_per_prompt", type=int, default=5, help="Number of images to generate per prompt.")
    parser.add_argument("--num_inference_steps", type=int, default=4, help="Number of inference steps.")
    parser.add_argument("--max_sequence_length", type=int, default=256, help="Maximum sequence length.")
    parser.add_argument("--guidance_scale", type=float, default=0.0, help="Guidance scale.")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory.")
    args = parser.parse_args()
    
    t2i(args.prompt, args.num_images_per_prompt, args.num_inference_steps, args.max_sequence_length, args.guidance_scale, args.output_dir)    
    