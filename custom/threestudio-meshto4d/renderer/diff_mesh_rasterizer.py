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
    SoftPhongShader,
    PointLights,
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
        
        
    # def project_points(self,
    #                    points: torch.Tensor,
    #                    camera: FoVPerspectiveCameras,
    #                    image_size: Tuple[int, int],
    #                    ) -> torch.Tensor:
    #     camera_space_transformation = camera.get_world_to_view_transform()
    #     points_camera_space = camera_space_transformation.transform_points(points)
    #     depth = points_camera_space[..., 2]
        
    #     points_perspective_division = points_camera_space[..., :2] / (depth.unsqueeze(-1) + 1e-8)
    #     intrinsic_matrix = camera.(get_projection_transform)
        

    def forward(
        self,
        **batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Render the silhouette and depth of the mesh.
        Should adjust dataloaders to avoid the data operations in the forward function.
        Current example call:
        out = self.forward(**batch)
        """
        
        phong_blend_params = BlendParams(background_color=(1.0, 1.0, 1.0), sigma=1e-4)    
        
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
        lights = AmbientLights(device=self.geometry.device)
        
        renderer_silhouette = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=camera, 
                raster_settings=raster_settings_soft_silhouette,
            ),
            shader=SoftPhongShader(
                lights=lights,
                cameras= camera,
                blend_params=phong_blend_params,
                device=self.geometry.device,
            ),
        )
        
        # geometry: DynamicSuGaRModel = self.geometry
        meshes_t = batch["timed_surface_mesh"]
        # meshes_t1 = batch["timed_surface_mesh_next"]
                            
        rgba, fragments = renderer_silhouette(meshes_t)
        depth = fragments.zbuf[..., 0]
        rgb = rgba[..., :3]
        
        mask = rgba[..., 3] > 0.5
        mask = mask.unsqueeze(-1)
    
        depth_mask = depth != -1
        depth[depth_mask] = depth[depth_mask].detach()
        
        pix_vert_t = camera.transform_points_screen(
            meshes_t.verts_padded(), 
            image_size=(batch["height"], batch["width"])
        )
        
        # pix_vert_t1 = camera.transform_points_screen(
        #     meshes_t1.verts_padded(), 
        #     image_size=(batch["height"], batch["width"])
        # )
        
        
        return {
            "mesh_comp_mask": mask,
            "mesh_comp_depth": depth,
            "mesh_pix_vert_t": pix_vert_t[..., :2],
            "mesh_comp_rgb": rgb,
        }