import os
import argparse
from PIL import Image
import numpy as np
import torch
from pytorch3d.io import IO
from pytorch3d.io.experimental_gltf_io import MeshGlbFormat

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRenderer,
    MeshRasterizer,
    look_at_view_transform,
    RasterizationSettings,
    BlendParams,
    AmbientLights,
    HardPhongShader
)

def render_front(mesh_path: str, output_path: str, img_width: int=512, img_height: int=512, azimuth: float=0.0):

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh_io = IO()
    mesh_io.register_meshes_format(MeshGlbFormat())

    # Load and normalize the mesh
    mesh = mesh_io.load_mesh(path=mesh_path, include_textures=True, device=device)
    print(f"Mesh bounding boxes: {mesh.get_bounding_boxes()}")
    
    # Rendering settings
    raster_settings = RasterizationSettings(
        image_size=(img_height, img_width), 
        blur_radius=0, 
        faces_per_pixel=1, 
    )

    blend_params = BlendParams(background_color=(1.0, 1.0, 1.0))  

    # Camera settings (distance=2.7, elev=0.0, azim=0.0)
    fixed_fov = 33.8
    distance = 0.5 / np.sin(np.radians(fixed_fov/2))
    # distance = (0.5 / np.tan(np.radians(fov/2)))
    
    R, T = look_at_view_transform(dist=distance, elev=0.0, azim=azimuth)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fixed_fov)

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
    
    image_rgb = image[:, :, :3]
    image_rgb_pil = Image.fromarray(image_rgb)
    
    # Save the image
    os.makedirs(output_path, exist_ok=True)
    mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
    output_file = os.path.join(output_path, f"{mesh_name}_{azimuth}.png")
    image_pil.save(output_file)
    image_rgb_pil.save(output_file.replace('.png', '_rgb.png'))
    
    # Create mask from alpha channel
    alpha_channel = image[:, :, 3] if image.shape[2] == 4 else np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
    mask_pil = Image.fromarray(alpha_channel)
    
    # Save the mask
    mask_file = os.path.join(output_path, f"{mesh_name}_{azimuth}_mask.png")
    mask_pil.save(mask_file)
    print(f"Mask saved to {mask_file}")
    
    print(f"Rendered image saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a mesh from the front view.")
    parser.add_argument('--mesh_path', type=str, required=True, help='Path to the mesh file (e.g., .obj).')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the rendered image.')
    parser.add_argument('--img_width', type=int, default=576, help='Width of the rendered image.')
    parser.add_argument('--img_height', type=int, default=576, help='Height of the rendered image.')
    parser.add_argument('--azim', type=float, default=0, help='Azimuth angle of the camera')
    args = parser.parse_args()
    
    render_front(args.mesh_path, args.output_path, args.img_width, args.img_height, args.azim)