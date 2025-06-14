import os
import glob
import numpy as np
import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.shader import HardPhongShader
import argparse

import matplotlib.pyplot as plt
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesVertex
)

def parse_skeleton_file(file_path):
    """
    Parse a skeleton file to extract joint positions and hierarchy.
    """
    joints = {}
    hierarchy = {}
    root_joint = None
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            
            if parts[0] == 'joints':
                # Parse joint position
                joint_name = parts[1]
                position = [float(parts[2]), float(parts[3]), float(parts[4])]
                joints[joint_name] = position
            
            elif parts[0] == 'root':
                # Parse root joint
                root_joint = parts[1]
            
            elif parts[0] == 'hier':
                # Parse hierarchy
                parent = parts[1]
                child = parts[2]
                
                if parent not in hierarchy:
                    hierarchy[parent] = []
                
                hierarchy[parent].append(child)
    
    return joints, hierarchy, root_joint

def visualize_mesh_with_skeleton(
    mesh_path, 
    skeleton_file, 
    output_path=None, 
    device="cuda" if torch.cuda.is_available() else "cpu",
    image_size=512
):
    # Parse skeleton file
    joints, hierarchy, root_joint = parse_skeleton_file(skeleton_file)
    
    # Load the mesh
    verts, faces, _ = load_obj(mesh_path)
    verts = verts.to(device)
    faces = faces.verts_idx.to(device)
    
    # Create a textures object
    verts_rgb = torch.ones_like(verts)[None]  # (1, N_V, 3)
    textures = TexturesVertex(verts_features=verts_rgb)
    
    # Create a Meshes object
    mesh = Meshes(
        verts=[verts],
        faces=[faces],
        textures=textures
    )
    
    # Camera settings (distance=2.7, elev=0.0, azim=0.0)
    fixed_fov = 30.0
    distance = 0.5 / np.sin(np.radians(fixed_fov/2))
    # distance = (0.5 / np.tan(np.radians(fov/2)))
    
    R, T = look_at_view_transform(dist=distance, elev=0.0, azim=270.0)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fixed_fov)

    # Define the rasterization settings
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0, 
        faces_per_pixel=1,
    )
    
    # Place a point light in front of the object
    lights = PointLights(device=device, location=[[0.0, 0.0, 1.0]])
    
    # Create a renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
    )
    
    # Render the mesh
    mesh_image = renderer(mesh)
    
    # Create a figure and render the image
    plt.figure(figsize=(10, 10))
    plt.imshow(mesh_image[0, ..., :3].cpu().numpy())
    
    # Convert joint positions to a tensor
    joint_positions = []
    joint_names = []
    for name, pos in joints.items():
        joint_positions.append(pos)
        joint_names.append(name)
    
    joint_positions = torch.tensor([joint_positions], device=device)
    
    # Project joints to 2D image space
    projected_joints = cameras.transform_points_screen(
        joint_positions, 
        image_size=(image_size, image_size)
    )[0]
    
    # Draw the joints as dots
    for i, (name, pos_2d) in enumerate(zip(joint_names, projected_joints)):
        plt.scatter(pos_2d[0].item(), pos_2d[1].item(), c='r', s=30)
    
    # Draw the skeleton as lines
    for parent, children in hierarchy.items():
        if parent in joint_names:
            parent_idx = joint_names.index(parent)
            parent_pos = projected_joints[parent_idx]
            
            for child in children:
                if child in joint_names:
                    child_idx = joint_names.index(child)
                    child_pos = projected_joints[child_idx]
                    
                    plt.plot([parent_pos[0].item(), child_pos[0].item()],
                            [parent_pos[1].item(), child_pos[1].item()],
                            'g-', linewidth=2)
    
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize skeleton on meshes')
    parser.add_argument('--mesh_dir', required=True, help='Directory containing mesh files')
    parser.add_argument('--skeleton_file', required=True, help='Path to skeleton file')
    parser.add_argument('--output_dir', default=None, help='Output directory for visualizations')
    parser.add_argument('--mesh_ext', default='.obj', help='Mesh file extension')
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Find all mesh files
    mesh_files = glob.glob(os.path.join(args.mesh_dir, f'*{args.mesh_ext}'))
    
    if not mesh_files:
        print(f"No mesh files found in {args.mesh_dir} with extension {args.mesh_ext}")
        return
    
    for mesh_file in mesh_files:
        print(f"Processing {mesh_file}...")
        base_name = os.path.basename(mesh_file).replace(args.mesh_ext, '')
        
        if args.output_dir:
            output_path = os.path.join(args.output_dir, f"{int(base_name):04d}_skeleton.png")
        else:
            output_path = None
        
        visualize_mesh_with_skeleton(
            mesh_file,
            args.skeleton_file,
            output_path
        )

if __name__ == "__main__":
    main()