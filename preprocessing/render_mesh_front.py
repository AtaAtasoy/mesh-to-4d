import os
import argparse
from PIL import Image
import numpy as np
import torch
from pytorch3d.io import load_objs_as_meshes, load_ply, IO
from pytorch3d.io.experimental_gltf_io import MeshGlbFormat

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    look_at_view_transform,
    RasterizationSettings,
    BlendParams,
    AmbientLights, 
    TexturesVertex,
    HardPhongShader
)
from rembg import remove

def main():
    parser = argparse.ArgumentParser(description="Render a mesh from the front view.")
    parser.add_argument('--mesh_path', type=str, required=True, help='Path to the mesh file (e.g., .obj).')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the rendered image.')
    parser.add_argument('--remove_bg', action='store_true', help='Remove the background from the rendered image.')
    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh_io = IO()
    mesh_io.register_meshes_format(MeshGlbFormat())

    # Load and normalize the mesh
    mesh = mesh_io.load_mesh(path=args.mesh_path, include_textures=True, device=device)
    print(f"Mesh bounding boxes: {mesh.get_bounding_boxes()}")
    
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    # Rendering settings
    raster_settings = RasterizationSettings(
        image_size=1024, 
        blur_radius=0, 
        faces_per_pixel=150, 
    )

    blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))  

    # Camera settings (distance=2.7, elev=0.0, azim=0.0)
    R, T = look_at_view_transform(dist=3.8, elev=5.0, azim=0.0)
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
    images = renderer(mesh)
    image = images[0].cpu().numpy()
    image = (image * 255).astype(np.uint8)
    image_pil = Image.fromarray(image)
        
    # Save the image
    os.makedirs(args.output_path, exist_ok=True)
    mesh_name = os.path.splitext(os.path.basename(args.mesh_path))[0]
    output_file = os.path.join(args.output_path, f"{mesh_name}.png")
    image_pil.save(output_file)
    
    # Create mask from alpha channel
    alpha_channel = image[:, :, 3] if image.shape[2] == 4 else np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
    mask_pil = Image.fromarray(alpha_channel)
    
    # Save the mask
    mask_file = os.path.join(args.output_path, f"{mesh_name}_mask.png")
    mask_pil.save(mask_file)
    print(f"Mask saved to {mask_file}")
    
    print(f"Rendered image saved to {output_file}")

if __name__ == "__main__":
    main()