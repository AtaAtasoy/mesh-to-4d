import os
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
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
from pytorch3d.renderer.mesh.textures import TexturesVertex

import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate 3D mesh animation video')
parser.add_argument('--mesh_dir', type=str, default='./meshes',
                    help='Directory containing the mesh files')
args = parser.parse_args()

# Directory containing the mesh files
mesh_dir = args.mesh_dir
output_video_path = os.path.join(os.path.dirname(mesh_dir), "mesh_animation.mp4")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load all meshes
mesh_filenames = [os.path.join(mesh_dir, f) for f in os.listdir(mesh_dir) if f.endswith(".obj")]

# Mesh index is the integer before the .obj extension, sort accordingly
mesh_filenames = sorted(mesh_filenames, key=lambda x: int(os.path.basename(x).split(".")[0]))


meshes = load_objs_as_meshes(mesh_filenames, device=device)
batch_size = len(meshes)

# Rendering settings
raster_settings = RasterizationSettings(
    image_size=256,   # 256x256 resolution
    blur_radius=0.0,
    faces_per_pixel=1,
)
blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))  # Purple (magenta) background

distance = torch.tensor([3.8] * batch_size, device=device)
elevation = torch.tensor([5.0] * batch_size, device=device)
azimuth = torch.zeros(batch_size, device=device)

# Camera settings (distance=2.7, elev=0.0, azim=0.0)
R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth, device=device)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=20.0)

# Setup lights
lights = AmbientLights(device=device)

# Create renderer
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=HardPhongShader(
        device=device,
        cameras=cameras,
        lights=lights,
        blend_params=blend_params
    )
)

# Render the image
images = renderer(meshes).cpu().numpy()

# Save the images under "mesh_renders" directory
os.makedirs(os.path.join(mesh_dir, "mesh_renders"), exist_ok=True)
for i in range(batch_size):
    plt.imsave(os.path.join(mesh_dir, "mesh_renders", f"frame_{i:04d}.png"), images[i].squeeze())

# Save the images as a video
with imageio.get_writer(output_video_path, mode='I', fps=10) as writer:
    for i in tqdm(range(batch_size)):
        writer.append_data((255*images[i]).astype(np.uint8))
