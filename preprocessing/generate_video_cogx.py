# To get started, PytorchAO needs to be installed from the GitHub source and PyTorch Nightly.
# Source and nightly installation is only required until the next release.

import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, CogVideoXPipeline, CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import T5EncoderModel
from torchao.quantization import quantize_, int8_weight_only
from argparse import ArgumentParser
import os 

# @torch.compile()
def generate_video(image_path: str, prompt: str, negative_prompt: str="Excessive streching. Unrealistic. Streching.", num_videos_per_prompt: int=4, output_dir: str="output"):
    quantization = int8_weight_only

    text_encoder = T5EncoderModel.from_pretrained("THUDM/CogVideoX-5b-I2V", subfolder="text_encoder", torch_dtype=torch.bfloat16)
    quantize_(text_encoder, quantization())

    transformer = CogVideoXTransformer3DModel.from_pretrained("THUDM/CogVideoX-5b-I2V",subfolder="transformer", torch_dtype=torch.bfloat16)
    quantize_(transformer, quantization())

    vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-5b-I2V", subfolder="vae", torch_dtype=torch.bfloat16)
    quantize_(vae, quantization())

    # Create pipeline and run inference
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
       "THUDM/CogVideoX1.5-5B-I2V",
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        torch_dtype=torch.bfloat16
    )

    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    # pipe = CogVideoXPipeline.from_pretrained(
    #     "THUDM/CogVideoX-2b",
    #     torch_dtype=torch.float16
    # )

    # pipe.to("cuda")

    image = load_image(image=image_path)
    videos = pipe(
        prompt=prompt,
        image=image,
        num_videos_per_prompt=num_videos_per_prompt,
        num_inference_steps=50,
        num_frames=81,
        guidance_scale=6,
        generator=torch.Generator(device="cuda").manual_seed(42),
        negative_prompt=negative_prompt,
    ).frames

    for i, video in enumerate(videos):
        export_to_video(video, f"{output_dir}/output_{i}.mp4", fps=8)
        

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate videos from text prompts.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to generate videos from.")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt.")
    parser.add_argument("--num_videos_per_prompt", type=int, default=4, help="Number of videos to generate per prompt.")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory.")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    generate_video(args.image_path, args.prompt, args.negative_prompt, args.num_videos_per_prompt, args.output_dir)