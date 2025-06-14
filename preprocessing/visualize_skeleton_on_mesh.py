import os
import argparse
import numpy as np
import torch
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)
import matplotlib.pyplot as plt


def load_mesh_and_texture(obj_path):
    """Load mesh and texture from .obj file"""
    if os.path.exists(obj_path):
        verts, faces, aux = load_obj(obj_path)
        
        # Create a Textures object
        if aux.texture_images:
            # Handle textured mesh
            texture_image = next(iter(aux.texture_images.values()))
            faces_uvs = faces.textures_idx if hasattr(faces, "textures_idx") else None
            verts_uvs = aux.verts_uvs if hasattr(aux, "verts_uvs") else None
            
            mesh = load_objs_as_meshes([obj_path], device=device)
        else:
            # No texture available, use vertex colors
            verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
            textures = TexturesVertex(verts_features=verts_rgb)
            mesh = Meshes(
                verts=[verts],
                faces=[faces.verts_idx],
                textures=textures
            )
            
        return mesh
    else:
        raise FileNotFoundError(f"File {obj_path} not found")


def load_skeleton(skeleton_path, color=[1.0, 0.0, 0.0]):
    """Load skeleton mesh and highlight it with a specific color"""
    if os.path.exists(skeleton_path):
        verts, faces, _ = load_obj(skeleton_path)
        
        # Create a red color for the skeleton to make it stand out
        verts_rgb = torch.tensor([color] * verts.shape[0])[None]  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb)
        
        skeleton_mesh = Meshes(
            verts=[verts],
            faces=[faces.verts_idx],
            textures=textures
        )
        
        return skeleton_mesh
    else:
        raise FileNotFoundError(f"File {skeleton_path} not found")


def create_renderer(image_size=512, dist=2.5, elev=5.0, azim=0):
    """Create a renderer with specified parameters"""
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=20.0)

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    lights = PointLights(
        device=device,
        location=[[0.0, 0.0, 3.0]],
        ambient_color=[[0.7, 0.7, 0.7]],
        diffuse_color=[[0.3, 0.3, 0.3]],
        specular_color=[[0.0, 0.0, 0.0]],
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
    )

    return renderer


def visualize_mesh_with_skeleton(mesh_path, skeleton_path, output_path=None, num_views=4, device=torch.device("cuda:0")):
    """Visualize mesh with skeleton overlay from multiple angles"""
    # Load mesh and skeleton
    mesh = load_mesh_and_texture(mesh_path).to(device)
    skeleton_mesh = load_skeleton(skeleton_path).to(device)
    
    # Create subplots for multiple views
    fig, axs = plt.subplots(1, num_views, figsize=(5*num_views, 5))
    
    for i in range(num_views):
        azim = i * 90  # Rotate around the model
        renderer = create_renderer(dist=2.5, elev=30, azim=azim)
        
        # Render the original mesh
        mesh_image = renderer(mesh)
        
        # Render the skeleton (will be on top due to z-buffer)
        skeleton_image = renderer(skeleton_mesh)
        
        # Combine the images (simple overlay with transparency)
        image = mesh_image[0, ..., :3]  # RGB channels only
        skeleton_rgb = skeleton_image[0, ..., :3]
        skeleton_alpha = (skeleton_image[0, ..., 3:] > 0).float()
        
        # Overlay skeleton on mesh image
        blended_image = image * (1 - skeleton_alpha) + skeleton_rgb * skeleton_alpha
        
        # Display the rendered image
        if num_views > 1:
            ax = axs[i]
        else:
            ax = axs
        ax.imshow(blended_image.cpu().numpy())
        ax.set_title(f"View {i+1}")
        ax.axis("off")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize 3D mesh with skeleton overlay")
    parser.add_argument("--mesh", required=True, help="Path to the source mesh .obj file")
    parser.add_argument("--skeleton", required=True, help="Path to the skeleton .obj file")
    parser.add_argument("--output", help="Path to save the visualization (optional)")
    parser.add_argument("--views", type=int, default=4, help="Number of views to render")
    
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    visualize_mesh_with_skeleton(args.mesh, args.skeleton, args.output, args.views, device)