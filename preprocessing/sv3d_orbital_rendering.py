import torch
import numpy as np
from pytorch3d.io import load_objs_as_meshes
import os
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    AmbientLights,
    look_at_view_transform,
    
)
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Render orbital photos of a 3D mesh. Imitates the output of SV3D.")
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to the input mesh file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save rendered images.")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    mesh = load_objs_as_meshes([args.mesh_path], device=device)
    
    image_size = 576
    cameras = FoVPerspectiveCameras(
        device=device,
        fov=20.0
    )

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Place a point light in front of the object
    lights = AmbientLights(device=device)

    # Create a phong renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )

    # Generate 21 views from -180 to 180 degrees
    # azimuths = np.linspace(0, 360, 21)
    azimuths_deg = np.linspace(0, 360, 21 + 1)[1:] % 360
    azimuths_rad = [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
    azimuths_rad[:-1].sort()

    elevation = 5.0
    distance = 3.8

    for azim in azimuths_deg:
        R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azim)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=20.0)
        
        images = renderer(mesh, cameras=cameras)
        image = images[0, ..., :3].cpu().numpy()
        
        image = image * 255
        image = image.astype(np.uint8)
        
        plt.imsave(
            os.path.join(output_dir, f"render_azimuth_{azim:.1f}.png"),
            image,
            dpi=1
        )

    print("Rendering complete!")