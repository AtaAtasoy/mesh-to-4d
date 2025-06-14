import torch
from threestudio.utils.ops import get_cam_info_gaussian
from torch.amp import autocast

from ..geometry.gaussian_base import Camera

import torch

class GaussianBatchRenderer:
    def batch_forward(self, batch):
        bs = batch["c2w"].shape[0]
        renders = []
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
        
        if batch.get("timed_surface_mesh", None) is not None:
            render_pkg = self.forward(**batch)
            outputs.update({
                "mesh_comp_mask": render_pkg["mesh_comp_mask"],
                "mesh_comp_depth": render_pkg["mesh_comp_depth"],
                "mesh_pix_vert_t": render_pkg["mesh_pix_vert_t"],
                "mesh_pix_vert_t1": render_pkg["mesh_pix_vert_t1"],
            })
        else:                   
            for batch_idx in range(bs):
                batch["batch_idx"] = batch_idx
                fovy = batch["fovy"][batch_idx]
                c2w = batch["c2w"][batch_idx]
                
                w2c, proj, cam_p = get_cam_info_gaussian(
                    c2w=c2w, fovx=fovy, fovy=fovy, znear=1.0, zfar=100
                ) # pytorch3d uses znear=1.0 and zfar=100
                
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
