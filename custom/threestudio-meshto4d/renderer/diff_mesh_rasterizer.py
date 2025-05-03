import math
from dataclasses import dataclass

import numpy as np
import threestudio
import torch
import torch.nn.functional as F

from pytorch3d.renderer import (
    BlendParams,
    SoftSilhouetteShader,
    MeshRendererWithFragments,
    MeshRasterizer,
    RasterizationSettings,
    FoVPerspectiveCameras,
    MeshRenderer,
    AmbientLights,
)
from pytorch3d.renderer.mesh.shader import SoftDepthShader


from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer
from threestudio.utils.typing import *

from .gaussian_batch_renderer import GaussianBatchRenderer
from ..geometry.dynamic_sugar import DynamicSuGaRModel
import time
from dreifus.camera import CameraCoordinateConvention, PoseType
from dreifus.matrix import Pose

@threestudio.register("diff-mesh-rasterizer")
class DiffMesh(Rasterizer, GaussianBatchRenderer):
    @dataclass
    class Config(Rasterizer.Config):
        debug: bool = False
        back_ground_color: Tuple[float, float, float] = (1, 1, 1)
        invert_bg_prob: float = 1.0

    cfg: Config
    
    def configure(
        self,
        geometry: BaseGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        

    def forward(
        self,
        **kwargs
    ) -> Dict[str, Any]:
        batch = kwargs  # Access the unpacked dictionary
        """
        Render the silhouette and depth of the mesh.
        Should adjust dataloaders to avoid the data operations in the forward function.
        Current example call:
        out = self.forward(**batch)
        """
        
        phong_blend_params = BlendParams(background_color=(0.0, 0.0, 0.0), sigma=1e-4/3)    
        
        raster_settings_soft_silhouette = RasterizationSettings(
            image_size=(batch["height"], batch["width"]), 
            blur_radius=np.log(1. / 1e-4 - 1.)*(phong_blend_params.sigma), 
            faces_per_pixel=50, 
        )

        c2w_opencv = batch["c2w"]
        
        # Multiply the first two columns of the matrix by -1
        c2w_pytorch3d = torch.zeros_like(c2w_opencv)
        c2w_pytorch3d[:, 0, :] = -c2w_opencv[:, 0, :]
        c2w_pytorch3d[:, 1, :] = -c2w_opencv[:, 1, :]
        c2w_pytorch3d[:, 2, :] = c2w_opencv[:, 2, :]
        c2w_pytorch3d[:, 3, :] = c2w_opencv[:, 3, :]
        
        R = c2w_pytorch3d[:, :3, :3]
        R_inv = torch.linalg.inv(R)
        
        translation_c2w_torch3d = c2w_pytorch3d[:, :3, 3]
        translation_w2c_torch3d = -R_inv @ translation_c2w_torch3d.unsqueeze(-1)
        T = translation_w2c_torch3d.squeeze(-1)
                
        camera = FoVPerspectiveCameras(R=R, T=T, device=self.geometry.device, fov=30.0)
        
        renderer_silhouette = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=camera, 
                raster_settings=raster_settings_soft_silhouette
            ),
            shader=SoftSilhouetteShader()
        )
        
        geometry: DynamicSuGaRModel = self.geometry
        meshes = geometry.get_timed_surface_mesh(batch['timestamp'], batch['frame_indices'])
                            
        mask, fragments = renderer_silhouette(meshes)
        depth = fragments.zbuf[..., 0]
        mask = mask[..., 3].unsqueeze(-1)
        
        depth_mask = depth != -1
        depth[depth_mask] = depth[depth_mask].detach()
        
        return {
            "mesh_comp_mask": mask,
            "mesh_comp_depth": depth,
        }