import os
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    AmbientLights,
    BlendParams,
)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate 3D mesh animation video')
parser.add_argument('--mesh_dir', type=str, default='./meshes',
                    help='Directory containing the mesh files')
parser.add_argument('--fps', type=int, default=8, help='fps of the output video')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for rendering')
args = parser.parse_args()

mesh_dir = args.mesh_dir
batch_size = args.batch_size
fps = args.fps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get all mesh filenames and sort
mesh_filenames = [os.path.join(mesh_dir, f) for f in os.listdir(mesh_dir) if f.endswith(".obj")]
mesh_filenames = sorted(mesh_filenames, key=lambda x: int(os.path.basename(x).split(".")[0]))
total_meshes = len(mesh_filenames)

# Rendering settings
raster_settings = RasterizationSettings(
    image_size=1024,
    blur_radius=0.0,
    faces_per_pixel=1,
)
blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))

azimuth_angles = [-75, 15, 105, 195, 0]

for i, azimuth_angle in tqdm(enumerate(azimuth_angles)):
    all_images = []
    for start in range(0, total_meshes, batch_size):
        end = min(start + batch_size, total_meshes)
        batch_meshes = load_objs_as_meshes(mesh_filenames[start:end], device=device)

        dist = torch.tensor([3.8] * len(batch_meshes), device=device)
        elev = torch.tensor([5.0] * len(batch_meshes), device=device)
        azim = torch.tensor([azimuth_angle] * len(batch_meshes), device=device)

        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=20.0)

        lights = AmbientLights(device=device)

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=HardPhongShader(
                device=device,
                cameras=cameras,
                lights=lights,
                blend_params=blend_params
            )
        )

        images = renderer(batch_meshes).cpu().numpy()
        all_images.extend(images)

    # Save the images and video
    if azimuth_angle == 0:
        images_out_dir = os.path.join(mesh_dir, "train_angle")
    else:
        images_out_dir = os.path.join(mesh_dir, f"eval_{i+1}")

    os.makedirs(images_out_dir, exist_ok=True)
    output_video_path = os.path.join(images_out_dir, f"mesh_animation_{azimuth_angle}.mp4")

    for idx, image in enumerate(all_images):
        plt.imsave(os.path.join(images_out_dir, f"{idx}.png"), image.squeeze())

    with imageio.get_writer(output_video_path, mode='I', fps=fps) as writer:
        for image in tqdm(all_images):
            writer.append_data((255 * image).astype(np.uint8))
