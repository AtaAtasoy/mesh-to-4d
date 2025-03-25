import torch
from threestudio.utils.ops import get_cam_info_gaussian
from torch.amp import autocast

from ..geometry.gaussian_base import Camera
import numpy as np
from PIL import Image
import os

from dreifus.camera import CameraCoordinateConvention, PoseType
from dreifus.matrix import Pose

from pytorch3d.renderer import (
    RasterizationSettings, MeshRendererWithFragments, MeshRenderer ,MeshRasterizer, BlendParams,
    SoftSilhouetteShader, AmbientLights, SoftPhongShader, FoVPerspectiveCameras
)

import torch

class GaussianBatchRenderer:
    def batch_forward(self, batch):
        torch.cuda.empty_cache()
        bs = batch["c2w"].shape[0]
        renders = []
        mesh_masks = []
        mesh_renders = []
        viewspace_points = []
        visibility_filters = []
        radiis = []
        normals = []
        normals_from_dist = []
        pred_normals = []
        depths = []
        masks = []
        bgs = []
        outputs = {}
        
        if batch.__contains__("timed_surface_mesh"):
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            for batch_idx in range(bs):
            
                # timed_mesh = batch["timed_surface_mesh"][batch_idx]
                timed_mesh = batch["timed_surface_mesh"][batch_idx]
                # R_all, T_all = [], []
                c2w = batch["c2w"][batch_idx]
                pose = Pose(matrix_or_rotation=c2w.cpu().numpy(), pose_type=PoseType.CAM_2_WORLD, camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV)
                pose.change_camera_coordinate_convention(CameraCoordinateConvention.PYTORCH_3D, inplace=True)
                # R_all.append(pose.get_rotation_matrix().numpy())
                R = torch.from_numpy(pose.get_rotation_matrix()).to(device)
                
                pose.change_pose_type(PoseType.WORLD_2_CAM, inplace=True)
                # T_all.append(pose.get_translation())
                T = torch.from_numpy(pose.get_translation()).to(device)     
                
                # R = torch.from_numpy(np.stack(R_all, axis=0)).to(device)
                # T = torch.from_numpy(np.stack(T_all, axis=0)).to(device)
                    
                camera = FoVPerspectiveCameras(R=R[None], T=T[None], device=device, fov=20.0)
                
                phong_blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))                                    

                if "render_mesh_mask" in batch:
                    raster_settings_soft_silhouette = RasterizationSettings(
                        image_size=(batch["height"], batch["width"]), 
                        blur_radius=np.log(1. / 1e-4 - 1.)*(phong_blend_params.sigma), 
                        faces_per_pixel=1, 
                    )
                    
                    renderer_silhouette = MeshRenderer(
                        rasterizer=MeshRasterizer(
                            cameras=camera, 
                            raster_settings=raster_settings_soft_silhouette
                        ),
                            shader=SoftSilhouetteShader()
                    )
                    
                    mask = renderer_silhouette(timed_mesh, cameras=camera)[0, ..., 3].unsqueeze(-1)
                    mesh_masks.append(mask)

                if "render_mesh_rgb" in batch: 
                    raster_settings_soft_rgb = RasterizationSettings(
                        image_size=(batch["height"], batch["width"]), 
                        blur_radius=np.log(1. / 1e-4 - 1.)*(phong_blend_params.sigma), 
                        faces_per_pixel=150, 
                        perspective_correct=False, 
                    )
                    lights = AmbientLights(device="cuda:0")

                    # Differentiable soft renderer using per vertex RGB colors for texture
                    renderer_textured = MeshRenderer(
                        rasterizer=MeshRasterizer(
                            cameras=camera, 
                            raster_settings=raster_settings_soft_rgb
                        ),
                        shader=SoftPhongShader(device=device, 
                            cameras=camera,
                            lights=lights,
                            blend_params=phong_blend_params)
                    )
                    
                    rgb = renderer_textured(timed_mesh, cameras=camera)[0, ..., :3]
                    # Threshold the black background
                    # mask = (rgb > 0.01).any(dim=-1)            
                    mesh_renders.append(rgb)
                    # mesh_masks[batch_idx] = mask.unsqueeze(-1)
            
            if "render_mesh_mask" in batch:
                outputs.update({"mesh_comp_mask": torch.stack(mesh_masks, dim=0)})
            if "render_mesh_rgb" in batch:
                outputs.update({"mesh_comp_rgb": torch.stack(mesh_renders, dim=0)})
                
        if batch.__contains__("render_gaussians"):
            for batch_idx in range(bs):
                batch["batch_idx"] = batch_idx
                fovy = batch["fovy"][batch_idx]
                c2w = batch["c2w"][batch_idx]
                
                w2c, proj, cam_p = get_cam_info_gaussian(
                    c2w=c2w, fovx=fovy, fovy=fovy, znear=1.0, zfar=100
                ) # pytorch3d uses znear=1.0 and zfar=100
                # w2c, proj, cam_p = batch["w2c"], batch["full_proj_transform"], batch["camera_center"]
                
                if batch.__contains__("timestamp"):
                    timestamp = batch["timestamp"][batch_idx]
                else:
                    timestamp = None
                
                if batch.__contains__("frame_indices"):
                    frame_idx = batch["frame_indices"][batch_idx]
                else:
                    frame_idx = None

                # import pdb; pdb.set_trace()
                viewpoint_cam = Camera(
                    FoVx=fovy,
                    FoVy=fovy,
                    image_width=batch["width"],
                    image_height=batch["height"],
                    world_view_transform=w2c,
                    full_proj_transform=proj,
                    camera_center=cam_p,
                    timestamp=timestamp,
                    frame_idx=frame_idx
                )

                with autocast('cuda', enabled=False):
                    render_pkg = self.forward(
                        viewpoint_cam, self.background_tensor, **batch
                    )
                    renders.append(render_pkg["render"])
                    viewspace_points.append(render_pkg["viewspace_points"])
                    visibility_filters.append(render_pkg["visibility_filter"])
                    radiis.append(render_pkg["radii"])
                    if render_pkg.__contains__("normal") and render_pkg["normal"] is not None:
                        normals.append(render_pkg["normal"])
                    if (
                        render_pkg.__contains__("normal_from_dist")
                        and render_pkg["normal_from_dist"] is not None
                    ):
                        normals_from_dist.append(render_pkg["normal_from_dist"])
                    if (
                        render_pkg.__contains__("pred_normal")
                        and render_pkg["pred_normal"] is not None
                    ):
                        pred_normals.append(render_pkg["pred_normal"])
                    if render_pkg.__contains__("depth"):
                        depths.append(render_pkg["depth"])
                    if render_pkg.__contains__("mask"):
                        masks.append(render_pkg["mask"])
                    if render_pkg.__contains__("comp_rgb_bg"):
                        bgs.append(render_pkg["comp_rgb_bg"])
    
        if len(mesh_renders) > 0: 
            outputs.update(
                {
                    "mesh_comp_rgb": torch.stack(mesh_renders, dim=0),
                    "mesh_comp_mask": torch.stack(mesh_masks, dim=0),
                    # "mesh_comp_depth": torch.stack(mesh_depths, dim=0),
                }
            )       
        
        if len(renders) > 0:
            outputs.update({
                "comp_rgb": torch.stack(renders, dim=0).permute(0, 2, 3, 1),
                "viewspace_points": viewspace_points,
                "visibility_filter": visibility_filters,
                "radii": radiis,
            })
        if len(normals) > 0:
            outputs.update(
                {
                    "comp_normal": torch.stack(normals, dim=0).permute(0, 2, 3, 1),
                }
            )
        if len(normals_from_dist) > 0:
            outputs.update(
                {
                    "comp_normal_from_dist": torch.stack(normals_from_dist, dim=0).permute(0, 2, 3, 1),
                }
            )
        if len(pred_normals) > 0:
            outputs.update(
                {
                    "comp_pred_normal": torch.stack(pred_normals, dim=0).permute(
                        0, 2, 3, 1
                    ),
                }
            )
        if len(depths) > 0:
            outputs.update(
                {
                    "comp_depth": torch.stack(depths, dim=0).permute(0, 2, 3, 1),
                }
            )
        if len(masks) > 0:
            outputs.update(
                {
                    "comp_mask": torch.stack(masks, dim=0).permute(0, 2, 3, 1),
                }
            )
        if len(bgs) > 0:
            outputs.update(
                {
                    "comp_rgb_bg": torch.cat(bgs, dim=0).permute(0, 2, 3, 1),
                }
            )
        return outputs
