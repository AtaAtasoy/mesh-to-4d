import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from PIL import Image
from pytorch3d.io import load_objs_as_meshes, IO
from pytorch3d.io.experimental_gltf_io import MeshGlbFormat

from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    RasterizationSettings,
    MeshRendererWithFragments,
    MeshRasterizer,  
    SoftPhongShader,
    AmbientLights,
    look_at_view_transform,
    HardPhongShader,
)
from dreifus.camera import CameraCoordinateConvention
from dreifus.matrix import Pose
import cv2
import shutil
from rembg import remove

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Render multi-view images.")
parser.add_argument('--mesh_path', type=str, required=False, help='Path to the mesh file.', default="meshes/cow/cow.obj")
parser.add_argument('--output_path', type=str, required=False, default='./ns', help='Path to the output.')
parser.add_argument('--num_views', type=int, required=False, default=128, help='Number of views to render.')
parser.add_argument('--resolution', type=int, required=False, default=512, help='Resolution of the rendered images.')
parser.add_argument('--batch_size', type=int, required=False, default=128, help='Batch size for rendering.')
args = parser.parse_args()

print(f"Running.")

mesh_name = os.path.splitext(os.path.basename(args.mesh_path))[0]
output_path = args.output_path

# Constants
RESOLUTION = args.resolution
RESULTS_PATH = f"{output_path}"

# Configuration
NUM_VIEWS = args.num_views
BATCH_SIZE = args.batch_size

# Load the mesh
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mesh_io = IO()
mesh_io.register_meshes_format(MeshGlbFormat())

mesh = mesh_io.load_mesh(path=args.mesh_path, include_textures=True, device=device)

print(f"Mesh bounding boxes: {mesh.get_bounding_boxes()}")

# Set up lights and renderer settings
lights = AmbientLights(device=device)
img_raster_settings = RasterizationSettings(
    image_size=RESOLUTION, 
    blur_radius = 0, 
    faces_per_pixel=1,
)

normal_raster_settings = RasterizationSettings(
    image_size=RESOLUTION, 
    blur_radius=0, 
    faces_per_pixel=1,
)

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

def export_dataset(data_list: list, split: str, mesh_name: str) -> None:
    """
    Exports the camera parameters and renders based on the dataset style.
    """
    split_dir = os.path.join(RESULTS_PATH, mesh_name, split)
    os.makedirs(split_dir, exist_ok=True)
    print(f"split_dir: {split_dir}")

    print(f"Exporting the data in Mesh-to-4D Style for {split} split.")

    images_output_path = os.path.join(split_dir, "images")
    mask_output_path = os.path.join(split_dir, "masks")
    depth_output_path = os.path.join(split_dir, "depths")
    normal_output_path = os.path.join(split_dir, "normals")
    os.makedirs(depth_output_path, exist_ok=True)
    os.makedirs(images_output_path, exist_ok=True)
    os.makedirs(normal_output_path, exist_ok=True)
    os.makedirs(mask_output_path, exist_ok=True)

    params = {}
    for data_entry in data_list:
        idx = data_entry['idx']
        param = data_entry['param']
        image = data_entry['image']
        depth = data_entry['depth'][:, :, 0]

        image_name = f"{idx:04d}.png"
        image_path = os.path.join(images_output_path, image_name)
        depth_path = os.path.join(depth_output_path, image_name)
        normal_path = os.path.join(normal_output_path, image_name)
        mask_path = os.path.join(mask_output_path, image_name)

        # Save images
        image = (image * 255).astype(np.uint8)
        Image.fromarray(image).save(image_path)
    
        # Create mask from alpha channel
        alpha_channel = image[:, :, 3] if image.shape[2] == 4 else np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
        Image.fromarray(alpha_channel).save(mask_path)
        
        # Save depths
        valid_depth_mask = depth > 0 # Exclude background
        normalized_depth = np.zeros_like(depth, dtype=np.float32)
        
        valid_depth = depth[valid_depth_mask]
        normalized_valid_depth = (valid_depth - valid_depth.min()) / (valid_depth.max() - valid_depth.min() + 1e-8)
        
        inverted_normalized_depth = 1.0 - normalized_valid_depth
        normalized_depth[valid_depth_mask] = inverted_normalized_depth

        assert np.isclose(inverted_normalized_depth.max(), 1.0, atol=1e-6), "Max normalized depth should be 1.0"
        assert np.isclose(inverted_normalized_depth.min(), 0.0, atol=1e-6), "Min normalized depth should be 0.0"
        
        normalized_depth_rgb = (normalized_depth * 255).astype(np.uint8)
        Image.fromarray(normalized_depth_rgb).save(depth_path)
        
        # Save normals
        normal_map = compute_normal_map(normalized_depth_rgb)
        
        # Mask out the background
        normal_map = normal_map * (normalized_depth_rgb > 0).reshape(normal_map.shape[0], normal_map.shape[1], 1)
        Image.fromarray(normal_map).save(normal_path)
        
        # Collect parameters
        params[image_name] = param

        # Save parameters to JSON file
        params_path = os.path.join(split_dir, f"{split}_camera_parameters.json")
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=4)

def generate_random_views_spherical(num_views, dist=3.8, upper_views=True, bottom_views=True):
    # Generate random points on unit sphere
    if upper_views and bottom_views:
        v = np.random.uniform(-1, 1, size=num_views)
    elif upper_views:
        v = np.random.uniform(0, 1, size=num_views)
    elif bottom_views:
        v = np.random.uniform(-1, 0, size=num_views)
    else:
        raise ValueError("At least one of upper_views or bottom_views must be True.")
    theta = np.arccos(v)
    phi = np.random.uniform(0, 2 * np.pi, size=num_views)
        
    # Convert to elevation and azimuth angles
    elevations = 90 - np.degrees(theta)
    azimuths = np.degrees(phi)
    distances = np.full(num_views, dist)
    
    elevations = torch.tensor(elevations, dtype=torch.float32)
    azimuths = torch.tensor(azimuths, dtype=torch.float32)
    distances = torch.tensor(distances, dtype=torch.float32)
    
    R, T = look_at_view_transform(dist=distances, elev=elevations, azim=azimuths)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device, fov=20.0)
    
    return cameras, R, T

# Generate all views first
cameras, rotations, translations = generate_random_views_spherical(NUM_VIEWS)

# Split indices
indices = np.arange(NUM_VIEWS)
np.random.shuffle(indices)
train_size = int(0.8 * NUM_VIEWS)
val_size = int(0.05 * NUM_VIEWS)
train_idx = indices[:train_size]
val_idx = indices[train_size:train_size+val_size]
test_idx = indices[train_size+val_size:]

indices = {
    'train': train_idx,
    'val': val_idx,
    'test': test_idx
}

# Render images and save parameters
for split, split_indices in indices.items():
    num_batches = int(np.ceil(len(split_indices) / BATCH_SIZE))
    print(f"Number of batches for {split}: {num_batches}")
    
    data_list = []  # Collect all data entries here
    
    for batch in range(num_batches):
        batch_idx = split_indices[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
        print(f"Rendering batch {batch+1}/{num_batches} for {split} split.")
        
        batch_cameras = cameras[batch_idx.tolist()]
        
        print(f"Batch cameras: {batch_cameras.R.shape}")
        
        # Extend mesh and set up renderer
        meshes_batch = mesh.extend(len(batch_idx))

        rgb_renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=batch_cameras, 
                raster_settings=img_raster_settings
            ),
            shader=HardPhongShader(
                device=device, 
                cameras=batch_cameras,
                lights=lights,
            )
        )
        
        # Render images
        images_batch, fragments = rgb_renderer(meshes_batch, cameras=batch_cameras)

        images_batch = images_batch.detach().cpu().numpy()
        depths_batch = fragments.zbuf.detach().cpu().numpy()

        # Calculate focal length
        fov = batch_cameras.fov[0].item()
        focal_length = RESOLUTION / (2 * np.tan(np.radians(fov / 2)))

        # Collect data for batch
        for i, idx in enumerate(batch_idx):
            R = batch_cameras.R[i].cpu()
            T = batch_cameras.T[i].cpu()
            w2c = torch.eye(4)
            w2c[:3, :3] = R.T
            w2c[:3, 3] = T

            world_2_cam_pose = Pose(w2c, camera_coordinate_convention=CameraCoordinateConvention.PYTORCH_3D)
            cam_2_world_torch3d = world_2_cam_pose.invert()
            cam_2_world_opencv = cam_2_world_torch3d.change_camera_coordinate_convention(
                new_camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV, 
                inplace=False
            )
            
            w2c_opencv = cam_2_world_opencv.invert()
            
            cam_2_world_opengl = cam_2_world_torch3d.change_camera_coordinate_convention(
                new_camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL, 
                inplace=False
            )
            
            full_proj_transform = batch_cameras[i].get_full_projection_transform().get_matrix().cpu().numpy().tolist()
            camera_center = batch_cameras[i].get_camera_center().cpu().numpy().tolist()
            

            param = {
                "intrinsics": {
                    "height": RESOLUTION,
                    "width": RESOLUTION,
                    "focal": focal_length,
                },
                "K": [
                    [focal_length, 0, RESOLUTION / 2],
                    [0, focal_length, RESOLUTION / 2],
                    [0, 0, 1]
                ],
                "c2w_torch3d": cam_2_world_torch3d.numpy().tolist(),
                "c2w_opencv": cam_2_world_opencv.numpy().tolist(),
                "c2w_opengl": cam_2_world_opengl.numpy().tolist(),
                "w2c_opencv": w2c_opencv.numpy().tolist(),
                "full_proj_transform": full_proj_transform,
                "camera_center": camera_center,
            }

            data_entry = {
                'idx': idx,  # Keep track of the original index
                'param': param,
                'image': images_batch[i],
                'depth': depths_batch[i],
            }
            data_list.append(data_entry)

        # Clear batch memory
        meshes_batch.cpu()
        batch_cameras.cpu()
        torch.cuda.empty_cache()

    # Proceed to export the dataset using the collected data
    export_dataset(
        data_list=data_list,
        split=split,
        mesh_name=mesh_name,
    )
