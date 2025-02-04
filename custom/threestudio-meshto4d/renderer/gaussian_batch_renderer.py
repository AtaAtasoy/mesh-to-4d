import torch
from threestudio.utils.ops import get_cam_info_gaussian
from torch.amp import autocast

from ..geometry.gaussian_base import BasicPointCloud, Camera
from ..geometry.sugar import convert_camera_from_gs_to_pytorch3d, GSCamera
import numpy as np
from PIL import Image
import os

from dreifus.camera import CameraCoordinateConvention, PoseType
from dreifus.matrix import Pose, Intrinsics

from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, AmbientLights, SoftPhongShader, FoVPerspectiveCameras
)

from pytorch3d.io import save_obj

def swap_columns(matrix, i, j):
    matrix[:, [i, j]] = matrix[:, [j, i]]
    return matrix


class GaussianBatchRenderer:
    def batch_forward(self, batch):
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
        for batch_idx in range(bs):
            batch["batch_idx"] = batch_idx
            fovy = batch["fovy"][batch_idx]
            c2w = batch["c2w"][batch_idx]
            timed_mesh = batch["timed_surface_mesh"][batch_idx]
            
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
            
            # If there is a surface mesh, we also render it.
            if batch.__contains__("timed_surface_mesh"):
                '''
                To be consistent with SoftRasterizer, 
                initialize the RasterizationSettings for the rasterizer with blur_radius = np.log(1. / 1e-4 - 1.) * blend_params.sigma
                '''
                
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                
                pose = Pose(matrix_or_rotation=c2w.cpu().numpy(), pose_type=PoseType.CAM_2_WORLD, camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV)
                pose.change_camera_coordinate_convention(CameraCoordinateConvention.PYTORCH_3D, inplace=True)
                R = torch.from_numpy(pose.get_rotation_matrix()).to(device)
                
                # Rotate Z axis 90 degrees
                # rot_z_90_ccw = torch.tensor([
                #     [0, -1, 0],  # x -> -y
                #     [1, 0, 0],   # y -> -x
                #     [0, 0, 1]    # z unchanged
                # ], device=device, dtype=torch.float32)
                # R = R @ rot_z_90_ccw
                # R[:, 1] = -R[:, 1]
                
                pose.change_pose_type(PoseType.WORLD_2_CAM, inplace=True)
                T = torch.from_numpy(pose.get_translation()).to(device)        
                camera = FoVPerspectiveCameras(R=R[None], T=T[None], device=device, fov=20.0)

                faces_per_pixel = 100
                phong_blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))
                silhouette_shader_blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
                                
                lights = AmbientLights(device="cuda:0")
                mesh_raster_settings = RasterizationSettings(
                    image_size=(batch["width"], batch["height"]),
                    blur_radius=0.0,
                    faces_per_pixel=faces_per_pixel,
                )
                mesh_renderer = MeshRenderer(
                    rasterizer=MeshRasterizer(cameras=camera, raster_settings=mesh_raster_settings),
                    shader=SoftPhongShader(
                        device=device,
                        cameras=camera,
                        lights=lights,
                        blend_params=phong_blend_params
                    )
                )
                
                silhouette_raster_settings = RasterizationSettings(
                    image_size=(batch["width"], batch["height"]),
                    blur_radius=np.log(1. / 1e-4 - 1.) * silhouette_shader_blend_params.sigma,
                    faces_per_pixel=faces_per_pixel,
                )
                silhouette_rasterizer = MeshRasterizer(
                    cameras=camera,
                    raster_settings=silhouette_raster_settings
                )
                silhouette_renderer = MeshRenderer(
                    rasterizer=silhouette_rasterizer,
                    shader=SoftSilhouetteShader(blend_params=silhouette_shader_blend_params)
                )
                
                rgb = mesh_renderer(timed_mesh, cameras=camera)[0, ..., :3]
                mask = silhouette_renderer(timed_mesh, cameras=camera)[0, ..., 3]                                
                mesh_renders.append(rgb)
                mesh_masks.append(mask)
              
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
                }
            )       
            
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
