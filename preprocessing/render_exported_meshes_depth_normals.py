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
    MeshRendererWithFragments,
    PointLights,
    BlendParams,
)
from PIL import Image

def compute_normal_map(depth_map):
    # Invert the depth map (foreground = 255, background = 0)
    inverted_depth_map = 255 - depth_map

    # Exclude background (originally 255 in depth_map)
    valid_mask = depth_map < 255

    # Compute gradients (set gradients to 0 for background)
    dzdx = np.zeros_like(inverted_depth_map, dtype=np.float32)
    dzdy = np.zeros_like(inverted_depth_map, dtype=np.float32)

    dzdx[valid_mask] = np.gradient(inverted_depth_map.astype(np.float32), axis=1)[valid_mask]
    dzdy[valid_mask] = np.gradient(inverted_depth_map.astype(np.float32), axis=0)[valid_mask]

    # Compute normals
    normal_map = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.float32)
    normal_map[..., 0] = -dzdx  # x component
    normal_map[..., 1] = -dzdy  # y component
    normal_map[..., 2] = 1.0    # z component

    # Normalize the vectors (avoid division by zero)
    norm = np.linalg.norm(normal_map, axis=2, keepdims=True)
    normal_map /= (norm + 1e-8)

    # Map to [0, 255] for visualization
    normal_map = ((normal_map + 1) / 2 * 255).astype(np.uint8)

    # Set background pixels to black
    normal_map[~valid_mask] = [0, 0, 0]

    return normal_map

def create_depth_and_normal_map(depth_map):
    valid_depth_mask = depth_map > 0 # Exclude background
    normalized_depth = np.zeros_like(depth_map, dtype=np.float32)
        
    valid_depth = depth_map[valid_depth_mask]
    normalized_valid_depth = (valid_depth - valid_depth.min()) / (valid_depth.max() - valid_depth.min() + 1e-8)
        
    inverted_normalized_depth = 1.0 - normalized_valid_depth
    normalized_depth[valid_depth_mask] = inverted_normalized_depth

    assert np.isclose(inverted_normalized_depth.max(), 1.0, atol=1e-6), "Max normalized depth should be 1.0"
    assert np.isclose(inverted_normalized_depth.min(), 0.0, atol=1e-6), "Min normalized depth should be 0.0"
        
    normalized_depth_rgb = (normalized_depth * 255).astype(np.uint8).squeeze(0).squeeze(-1)    # should be(H, W)
    normal_map = compute_normal_map(normalized_depth_rgb) # should be (H, W, 3)
    normal_map = normal_map * (normalized_depth_rgb > 0).reshape(normal_map.shape[0], normal_map.shape[1], 1)

    return normalized_depth_rgb, normal_map


# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate 3D mesh animation video')
parser.add_argument('--mesh_dir', type=str, default='./meshes',
                    help='Directory containing the mesh files')
parser.add_argument('--fps', type=int, default=15, help='fps of the output video')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for rendering')
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
    image_size=480,
    blur_radius=0.0,
    faces_per_pixel=1,
)
blend_params = BlendParams(background_color=(1.0, 1.0, 1.0))

azimuth_angles = [-75, 15, 45, 105, 195, 0]

for i, azimuth_angle in tqdm(enumerate(azimuth_angles)):
    all_images = []
    all_depths = []
    all_normals = []
    for start in range(0, total_meshes, batch_size):
        end = min(start + batch_size, total_meshes)
        batch_meshes = load_objs_as_meshes(mesh_filenames[start:end], device=device)
        
        R, T = look_at_view_transform(dist=2.8, elev=5.0, azim=azimuth_angle, device=device)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=20.0)

        # lights = PointLights(device=device, location=[[0.0, 0.0, 1.0]])
        light_location = torch.tensor([[0.0, 0.0, 0.0]], device=device)
        transformed_light_loc = (torch.bmm(R, light_location.unsqueeze(-1)) + T.unsqueeze(-1)).squeeze(-1)
        lights = PointLights(device=device, location=transformed_light_loc)


        renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=HardPhongShader(
                device=device,
                cameras=cameras,
                lights=lights,
                blend_params=blend_params
            )
        )

        images, fragments = renderer(batch_meshes)
        images = images.cpu().numpy()
        depths = fragments.zbuf.cpu().numpy()
        depth_images, normal_images = create_depth_and_normal_map(depths)
        depth_images = depth_images[np.newaxis, ...]  # Add batch dimension
        normal_images = normal_images[np.newaxis, ...]  # Add batch dimension
        
        all_images.extend(images)
        all_depths.extend(depth_images)
        all_normals.extend(normal_images)
        

    # Save the images and video
    if azimuth_angle == 0:
        images_out_dir = os.path.join(mesh_dir, "train_angle")
    else:
        images_out_dir = os.path.join(mesh_dir, f"eval_{i+1}")

    os.makedirs(images_out_dir, exist_ok=True)
    output_video_path = os.path.join(images_out_dir, f"mesh_animation_{azimuth_angle}.mp4")
    output_gif_path = os.path.join(images_out_dir, f"mesh_animation_{azimuth_angle}.gif")
    depth_gif_path = os.path.join(images_out_dir, f"depth_animation_{azimuth_angle}.gif")
    normal_gif_path = os.path.join(images_out_dir, f"normal_animation_{azimuth_angle}.gif")

    for idx, image in enumerate(all_images):
        Image.fromarray((255 * image).astype(np.uint8).clip(0, 255)).save(os.path.join(images_out_dir, f"{idx:04d}.png"))
        Image.fromarray(all_depths[idx]).save(os.path.join(images_out_dir, f"{idx:04d}_depth.png"))
        Image.fromarray(all_normals[idx]).save(os.path.join(images_out_dir, f"{idx:04d}_normal.png"))
        
    images_rgb = []
    for image in all_images:
        # Convert to uint8 and take only RGB channels (first 3)
        img_uint8 = (255 * image[:, :, :3]).astype(np.uint8).clip(0, 255)
        images_rgb.append(img_uint8)
        

    with imageio.get_writer(output_video_path, mode='I', fps=fps) as writer:
        for image in tqdm(images_rgb):
            writer.append_data(image)

    with imageio.get_writer(output_gif_path, mode='I', fps=fps, loop=0) as writer:
        for image in tqdm(images_rgb):
            writer.append_data(image)
            
    with imageio.get_writer(depth_gif_path, mode='I', fps=fps, loop=0) as writer:
        for depth in tqdm(all_depths):
            writer.append_data(depth)
            
    with imageio.get_writer(normal_gif_path, mode='I', fps=fps, loop=0) as writer:
        for normal in tqdm(all_normals):
            writer.append_data(normal)
