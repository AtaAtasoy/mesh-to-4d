import time
from dataclasses import dataclass, field


import numpy as np
import threestudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from simple_knn._C import distCUDA2
from threestudio.utils.misc import C
from threestudio.utils.typing import *

from einops import rearrange
import pypose as pp
import open3d as o3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.io import load_objs_as_meshes

from .sugar import SuGaRModel
from .deformation import DeformationNetwork, ModelHiddenParams
from ..utils.dual_quaternions import DualQuaternion
from .mesh_utils import calculate_volume

from tqdm import trange, tqdm

import potpourri3d as pp3d


def strain_tensor_to_matrix(strain_tensor: Float[Tensor, "... 6"]):
    strain_matrix = torch.zeros(*strain_tensor.shape[:-1], 3, 3)
    strain_matrix[..., 0, 0] += 1.
    strain_matrix[..., 1, 1] += 1.
    strain_matrix[..., 2, 2] += 1.
    strain_matrix = strain_matrix.to(strain_tensor).flatten(-2, -1)
    strain_matrix[..., [0, 4, 8]] += strain_tensor[..., :3]
    strain_matrix[..., [1, 2, 5]] += strain_tensor[..., 3:]
    strain_matrix[..., [3, 6, 7]] += strain_tensor[..., 3:]
    strain_matrix = strain_matrix.reshape(*strain_tensor.shape[:-1], 3, 3)
    return strain_matrix


@threestudio.register("dynamic-sugar")
class DynamicSuGaRModel(SuGaRModel):
    @dataclass
    class Config(SuGaRModel.Config):
        num_frames: int = 14
        static_learnable: bool = False
        use_deform_graph: bool = True
        deformation_driver: str = "dg" # 'dg', 'skeleton'
        dynamic_mode: str = "deformation"  # 'discrete', 'deformation'

        n_dg_nodes: int = 1000
        dg_node_connectivity: int = 8

        dg_trans_lr: Any = 0.001
        dg_rot_lr: Any = 0.001
        dg_scale_lr: Any = 0.001

        vert_trans_lr: Any = 0.001
        vert_rot_lr: Any = 0.001
        vert_scale_lr: Any = 0.001

        deformation_lr: Any = 0.001
        grid_lr: Any = 0.001

        d_xyz: bool = True
        d_rotation: bool = True
        d_opacity: bool = False
        d_scale: bool = True
        use_shear_matrix: bool = True

        dist_mode: str = 'eucdisc'

        skinning_method: str = "hybrid"  # "lbs"(linear blending skinning) or "dqs"(dual-quaternion skinning) or "hybrid"
        use_extra_features: bool = False

        skeleton_pred_path: str = "" # path of the skeleton

    cfg: Config

    def configure(self) -> None:
        super().configure()
        if not self.cfg.static_learnable:
            self._points.requires_grad_(False)
            self._quaternions.requires_grad_(False)
            self.all_densities.requires_grad_(False)
            self._sh_coordinates_dc.requires_grad_(False)
            self._sh_coordinates_rest.requires_grad_(False)
            self._scales.requires_grad_(False)
            self._quaternions.requires_grad_(False)
            self.surface_mesh_thickness.requires_grad_(False)

        self.num_frames = self.cfg.num_frames
        self.dynamic_mode = self.cfg.dynamic_mode

        if self.cfg.use_deform_graph:
            if self.cfg.deformation_driver == "dg":
                self.build_deformation_graph(
                    self.cfg.n_dg_nodes,
                    None,
                    nodes_connectivity=self.cfg.dg_node_connectivity,
                    mode=self.cfg.dist_mode
                )
            elif self.cfg.deformation_driver == "skeleton":
                self.parse_joints_pred_with_skinning(self.cfg.skeleton_pred_path)
                # self.parse_joints_pred(self.cfg.skeleton_pred_path)
                # self.build_skeleton_binding_refined(nodes_connectivity=self.cfg.dg_node_connectivity)
                # self.build_skeleton_binding_refined_project_joints(self.cfg.dg_node_connectivity)
                self.build_skeleton_binding_with_skinning()

        if self.dynamic_mode == "discrete":
            # xyz
            if self.cfg.use_deform_graph:  # True
                self._dg_node_trans = nn.Parameter(
                    torch.zeros((self.num_frames, self.cfg.n_dg_nodes, 3), device="cuda"), requires_grad=True
                )
                dg_node_rots = torch.zeros((self.num_frames, self.cfg.n_dg_nodes, 4), device="cuda")
                dg_node_rots[..., -1] = 1
                self._dg_node_rots = nn.Parameter(dg_node_rots, requires_grad=True)
                if self.cfg.d_scale:
                    self._dg_node_scales = nn.Parameter(
                        torch.zeros((self.num_frames, self.cfg.n_dg_nodes, 3), device="cuda"), requires_grad=True
                    )
                else:
                    self._dg_node_scales = None

                if self.cfg.skinning_method == "hybrid":
                    self._dg_node_lbs_weights = nn.Parameter(
                        torch.zeros((self.num_frames, self.cfg.n_dg_nodes, 1), device="cuda"), reuqires_grad=True
                    )
                else:
                    self._dg_node_lbs_weights = None

            else:
                self._vert_trans = nn.Parameter(
                    torch.zeros(
                        (self.num_frames, *self._points.shape), device=self.device
                    ).requires_grad_(True)
                )
                vert_rots = torch.zeros((self.num_frames, self.n_verts, 4), device="cuda")
                vert_rots[..., -1] = 1
                self._vert_rots = nn.Parameter(vert_rots, requires_grad=True)
                if self.cfg.d_scale:
                    self._vert_scales = nn.Parameter(
                        torch.zeros((self.num_frames, self.n_verts, 3), device=self.device),
                        requires_grad=True
                    )
                else:
                    self._vert_scales = None

        elif self.dynamic_mode == "deformation":
            deformation_args = ModelHiddenParams(None)
            deformation_args.no_dr = False
            deformation_args.no_ds = not ((self.cfg.d_scale or self.cfg.skinning_method == "hybrid" or self.cfg.skinning_method == "lbs") and self.cfg.use_shear_matrix)
            deformation_args.no_do = not (self.cfg.skinning_method == "hybrid")

            self._deformation = DeformationNetwork(deformation_args)
            self._deformation_table = torch.empty(0)
        else:
            raise ValueError(f"Unimplemented dynamic mode {self.dynamic_mode}.")

        self.training_setup_dynamic()

        self._gs_bary_weights = torch.cat(
            [self.surface_triangle_bary_coords] * self.n_faces, dim=0
        )
        self._gs_vert_connections = self._surface_mesh_faces.repeat_interleave(
            self.cfg.n_gaussians_per_surface_triangle, dim=0
        )

        # debug
        self.val_step = 0
        self.global_step = 0
        self.save_path = None


    def training_setup_dynamic(self):
        training_args = self.cfg

        l = []
        if self.dynamic_mode == "discrete":
            if self.cfg.use_deform_graph:
                l += [
                    # xyz
                    {
                        "params": [self._dg_node_trans],
                        "lr": C(training_args.dg_trans_lr, 0, 0) * self.spatial_lr_scale,
                        "name": "dg_trans"
                    },
                    {
                        "params": [self._dg_node_rots],
                        "lr": C(training_args.dg_rot_lr, 0, 0),
                        "name": "dg_rotation"
                    },
                ]
                if self.cfg.d_scale:
                    l += [
                        {
                            "params": [self._dg_node_scales],
                            "lr": C(training_args.dg_scale_lr, 0, 0),
                            "name": "dg_scale"
                        },
                    ]
            else:
                l += [
                    {
                        "params": [self._vert_trans],
                        "lr": C(training_args.vert_trans_lr, 0, 0) * self.spatial_lr_scale,
                        "name": "vert_trans",
                    },
                    {
                        "params": [self._vert_rots],
                        "lr": C(training_args.vert_rot_lr, 0, 0),
                        "name": "vert_rotation",
                    },
                ]
                if self.cfg.d_scale:
                    l += [
                        {
                            "params": [self._vert_scales],
                            "lr": C(training_args.vert_scale_lr, 0, 0),
                            "name": "vert_scale"
                        }
                    ]

        elif self.dynamic_mode == "deformation":
            l += [
                {
                    "params": list(self._deformation.get_mlp_parameters()),
                    "lr": C(training_args.deformation_lr, 0, 0) * self.spatial_lr_scale,
                    "name": "deformation"
                },
                {
                    "params": list(self._deformation.get_grid_parameters()),
                    "lr": C(training_args.grid_lr, 0, 0) * self.spatial_lr_scale,
                    "name": "grid"
                }
            ]

        # a = self._deformation.get_grid_parameters()
        # b = self._deformation.get_mlp_parameters()

        self.optimize_list = l
        self.optimize_params = [d["name"] for d in l]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if not ("name" in param_group):
                continue

            if self.dynamic_mode == "discrete":
                if self.cfg.use_deform_graph:
                    if param_group["name"] == "dg_trans":
                        param_group["lr"] = C(
                            self.cfg.dg_trans_lr, 0, iteration, interpolation="exp"
                        ) * self.spatial_lr_scale
                    if param_group["name"] == "dg_rotation":
                        param_group["lr"] = C(
                            self.cfg.dg_rot_lr, 0, iteration, interpolation="exp"
                        )
                    if param_group["name"] == "dg_scale":
                        param_group["lr"] = C(
                            self.cfg.dg_scale_lr, 0, iteration, interpolation="exp"
                        )
                else:
                    if param_group["name"] == "vert_trans":
                        param_group["lr"] = C(
                            self.cfg.vert_trans_lr, 0, iteration, interpolation="exp"
                        ) * self.spatial_lr_scale
                    if param_group["name"] == "vert_rotation":
                        param_group["lr"] = C(
                            self.cfg.vert_rot_lr, 0, iteration, interpolation="exp"
                        )
                    if param_group["name"] == "vert_scale":
                        param_group["lr"] = C(
                            self.cfg.vert_scale_lr, 0, iteration, interpolation="exp"
                        )
            elif self.dynamic_mode == "deformation":
                if "grid" in param_group["name"]:
                    param_group["lr"] = C(
                        self.cfg.grid_lr, 0, iteration, interpolation="exp"
                    ) * self.spatial_lr_scale
                elif param_group["name"] == "deformation":
                    param_group["lr"] = C(
                        self.cfg.deformation_lr, 0, iteration, interpolation="exp"
                    ) * self.spatial_lr_scale

        self.color_clip = C(self.cfg.color_clip, 0, iteration)

    def get_timed_vertex_xyz(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None
    ) -> Float[Tensor, "N_t N_v 3"]:
        n_t = len(timestamp) if timestamp is not None else len(frame_idx)
        deformed_vert_pos = []
        for i in range(n_t):
            t = timestamp[i] if timestamp is not None else None
            f = frame_idx[i] if frame_idx is not None else None
            key = dict_temporal_key(t, f)
            if self._deformed_vert_positions.__contains__(key):
                vert_pos = self._deformed_vert_positions[key]
            else:
                vert_pos = self.get_timed_vertex_attributes(
                    t[None] if t is not None else t,
                    f[None] if f is not None else f
                )["xyz"][0]
            deformed_vert_pos.append(vert_pos)
        deformed_vert_pos = torch.stack(deformed_vert_pos, dim=0)
        return deformed_vert_pos
    
    def get_timed_dg_xyz_attributes(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None,
    ): 
        n_t = len(timestamp) if timestamp is not None else len(frame_idx)
        deformed_dg_xyz = []
        for i in range(n_t):
            t = timestamp[i] if timestamp is not None else None
            f = frame_idx[i] if frame_idx is not None else None
            key = dict_temporal_key(t, f)
            if self._cached_timed_dg_xyz.__contains__(key): # Should always be true
                dg_xyz = self._cached_timed_dg_xyz[key]
            else:
                dg_xyz = self.get_timed_dg_w_forward_pass(
                    t[None] if t is not None else t,
                    f[None] if f is not None else f
                )
            deformed_dg_xyz.append(dg_xyz)
        deformed_dg_xyz = torch.stack(deformed_dg_xyz, dim=0)
        return deformed_dg_xyz

    def get_timed_vertex_rotation(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None,
        return_matrix: bool = False,
    ) -> Union[Float[pp.LieTensor, "N_t N_v 4"], Float[Tensor, "N_t N_v 3 3"]]:
        n_t = len(timestamp) if timestamp is not None else len(frame_idx)
        deformed_vert_rot = []
        for i in range(n_t):
            t = timestamp[i] if timestamp is not None else None
            f = frame_idx[i] if frame_idx is not None else None
            key = dict_temporal_key(t, f)
            if self._deformed_vert_rotations.__contains__(key):
                vert_rot = self._deformed_vert_rotations[key]
            else:
                vert_rot = self.get_timed_vertex_attributes(
                    t[None] if t is not None else t,
                    f[None] if f is not None else f
                )["rotation"][0]
            deformed_vert_rot.append(vert_rot)
        deformed_vert_rot = torch.stack(deformed_vert_rot, dim=0)
        if return_matrix:
            return deformed_vert_rot.matrix()
        else:
            return deformed_vert_rot
        
    def get_timed_deformation(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None
    ) -> Float[Tensor, "N_t N_v 3"]:
        # Calculate the difference of the mesh at given timestamp with the rest pose
        
        rest_pose = self.get_timed_vertex_xyz(torch.zeros_like(timestamp), torch.zeros_like(frame_idx))
        deformed_pose = self.get_timed_vertex_xyz(timestamp, frame_idx)
        
        return deformed_pose - rest_pose
        

    # ============= Functions to compute normals ============= #
    def get_timed_surface_mesh(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None
    ) -> Meshes:
        n_t = len(timestamp) if timestamp is not None else len(frame_idx)
        deformed_vert_pos = self.get_timed_vertex_xyz(timestamp, frame_idx)
        # if torch.all(self._vertex_colors == 0.5):
        #     textures = self.torch3d_mesh.extend(n_t).textures # TexturesUV
        # else:
        #     textures = TexturesVertex(
        #         verts_features=torch.stack([self._vertex_colors] * n_t, dim=0).clamp(0, 1).to(self.device)
        #     )
        textures = self.torch3d_mesh.extend(n_t).textures.to(self.device)
        
        surface_mesh = Meshes(
            # verts=self.get_timed_xyz_vertices(timestamp, frame_idx),
            verts=deformed_vert_pos,
            faces=torch.stack([self._surface_mesh_faces] * n_t, dim=0),
            textures=textures
        )
        return surface_mesh

    def get_timed_face_normals(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None
    ) -> Float[Tensor, "N_t N_faces 3"]:
        return F.normalize(
            self.get_timed_surface_mesh(timestamp, frame_idx).faces_normals_padded(),
            dim=-1
        )

    def get_timed_gs_normals(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None
    ) -> Float[Tensor, "N_t N_gs 3"]:
        return self.get_timed_face_normals(timestamp, frame_idx).repeat_interleave(
            self.cfg.n_gaussians_per_surface_triangle, dim=1
        )

    def get_timed_vertex_normals(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None
    ) -> Float[Tensor, "N_t N_v 3"]:
        return self.get_timed_surface_mesh(timestamp, frame_idx).verts_normals_padded()
    
    # ========= Compute WKS attributes ======== #    
    def get_timed_mesh_volume(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None
    ) -> Float[Tensor, "N_t 1"]:
        timed_meshes = self.get_timed_surface_mesh(timestamp, frame_idx)
        
        verts, faces = timed_meshes.verts_padded(), timed_meshes.faces_padded()
        
        volumes = calculate_volume(verts, faces)
        
        return volumes
        
        
    
    # ========= Compute deformation nodes' attributes ======== #
    def get_timed_dg_attributes(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None,
    ) -> Dict[str, Union[Float[Tensor, "N_t N_p C"], pp.LieTensor]]:
        n_t = timestamp.shape[0] if timestamp is not None else frame_idx.shape[0]
        timed_attr_list = []
        for i in range(n_t):
            key_timestamp = timestamp[i].item() if timestamp is not None else 0
            key_frame = frame_idx[i].float().item() if frame_idx is not None else 0
            key = key_timestamp + key_frame

            if self.dg_timed_attrs.__contains__(key):
                attrs = self.dg_timed_attrs[key]
            else:
                attrs = self._get_timed_dg_attributes(
                    timestamp=timestamp[i:i + 1] if timestamp is not None else None,
                    frame_idx=frame_idx[i:i + 1] if frame_idx is not None else None,
                )
                self.dg_timed_attrs[key] = attrs
            timed_attr_list.append(attrs)

        timed_attrs = {}
        timed_attrs["xyz"] = torch.cat(
            [attr_dict["xyz"] for attr_dict in timed_attr_list], dim=0
        )
        timed_attrs["rotation"] = pp.SO3(
            torch.cat(
                [attr_dict["rotation"].tensor() for attr_dict in timed_attr_list],
                dim=0
            )
        )
        timed_attrs["scale"] = torch.cat(
            [attr_dict["scale"] for attr_dict in timed_attr_list], dim=0
        ) if (self.cfg.d_scale or self.cfg.skinning_method == "hybrid" or self.cfg.skinning_method == "lbs") and self.cfg.use_shear_matrix else None
        timed_attrs["opacity"] = torch.cat(
            [attr_dict["opacity"] for attr_dict in timed_attr_list], dim=0
        ) if self.cfg.skinning_method == "hybrid" else None
        return timed_attrs


    def _get_timed_dg_attributes(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None
    ):
        if self.dynamic_mode == "discrete":
            assert frame_idx is not None
            trans = self._dg_node_trans[frame_idx]
            rot = self._dg_node_rots[frame_idx]
            d_scale = self._dg_node_scales[frame_idx] if self.cfg.d_scale else None
            d_opacity = None

        elif self.dynamic_mode == "deformation":
            assert timestamp is not None
            pts = self._deform_graph_node_xyz
        
            num_pts = pts.shape[0]
            num_t = timestamp.shape[0]
            pts = torch.cat([pts] * num_t, dim=0)
            ts = timestamp.unsqueeze(-1).repeat_interleave(num_pts, dim=0)
            
            trans, rot, d_scale, d_opacity = self._deformation.forward_dynamic_delta(pts, ts * 2 - 1)

            # trans, rot = self._deformation.forward_dg_trans_and_rotation(pts, ts * 2 - 1)
            trans = trans.reshape(num_t, num_pts, 3)
            rot = rot.reshape(num_t, num_pts, 4)

            # NOTE: why do this?
            idt_quaternion = torch.zeros((1, num_pts, 4)).to(rot)
            idt_quaternion[..., -1] = 1
            rot = rot + idt_quaternion

            if d_scale is not None:
                d_scale = d_scale.reshape(num_t, num_pts, 6)
                # to shear matrix
                d_scale = strain_tensor_to_matrix(d_scale)
            if d_opacity is not None:
                d_opacity = d_opacity.reshape(num_t, num_pts, 1)
                d_opacity = F.sigmoid(d_opacity)
        # rot = rot / rot.norm(dim=-1, keepdim=True)
        rot = F.normalize(rot, dim=-1)
        attrs = {
            "xyz": trans, "rotation": pp.SO3(rot), "scale": d_scale, "opacity": d_opacity
        }
        return attrs
    
    def get_timed_dg(self, dg_node_attrs: Dict[str, Union[torch.Tensor, pp.LieTensor]]) -> Float[Tensor, "N_t N_dg_nodes 3"]:
        '''
        Calculate the deformed positions of the deform graph nodes based on their timed attributes.
        Transformation order: Scale -> Rotate -> Translate (delta).
        '''
        rest_dg_node_xyz: Float[Tensor, "N_dg_nodes 3"] = self._deform_graph_node_xyz
        
        # Determine n_t (number of time steps) from the input attributes
        # Assuming 'xyz' (translations) is always present and indicates N_t
        if "xyz" not in dg_node_attrs or dg_node_attrs["xyz"] is None:
            raise ValueError("dg_node_attrs must contain 'xyz' tensor to determine N_t.")
        n_t: int = dg_node_attrs["xyz"].shape[0]
        
        if n_t == 0:
            # Handle case with no time steps, return empty or appropriately shaped tensor
            return torch.empty((0, rest_dg_node_xyz.shape[0], 3), device=rest_dg_node_xyz.device, dtype=rest_dg_node_xyz.dtype)

        # current_xyz starts as rest positions, expanded for batch operations over N_t
        # Shape: (1, N_dg_nodes, 3) -> expanded to (N_t, N_dg_nodes, 3)
        current_xyz = rest_dg_node_xyz.unsqueeze(0).expand(n_t, -1, -1) # Shape: (N_t, N_dg_nodes, 3)

        # 1. Apply scale transformation (if scale is available)
        # dg_node_attrs["scale"] has shape (N_t, N_dg_nodes, 3, 3) or is None
        dg_node_scales = dg_node_attrs.get("scale")
        if dg_node_scales is not None:
            # Reshape for batch matrix multiplication:
            # Input tensors for bmm: (B, N, M) and (B, M, P)
            # Here, B = N_t * N_dg_nodes (number of nodes across all time steps)
            # dg_node_scales reshaped: (N_t * N_dg_nodes, 3, 3)
            # current_xyz reshaped: (N_t * N_dg_nodes, 3, 1)
            num_dg_nodes = current_xyz.shape[1]
            scaled_xyz = torch.bmm(
                dg_node_scales.reshape(n_t * num_dg_nodes, 3, 3),
                current_xyz.reshape(n_t * num_dg_nodes, 3, 1)
            ).reshape(n_t, num_dg_nodes, 3) # Reshape back to (N_t, N_dg_nodes, 3)
            current_xyz = scaled_xyz
        
        # 2. Apply rotation transformation
        # dg_node_attrs["rotation"] is pp.LieTensor of shape (N_t, N_dg_nodes, ...)
        # .matrix() gives shape (N_t, N_dg_nodes, 3, 3)
        dg_node_rot_matrices = dg_node_attrs["rotation"].matrix()
        num_dg_nodes = current_xyz.shape[1] # Potentially re-evaluate if current_xyz changed
        
        rotated_xyz = torch.bmm(
            dg_node_rot_matrices.reshape(n_t * num_dg_nodes, 3, 3), 
            current_xyz.reshape(n_t * num_dg_nodes, 3, 1)           
        ).reshape(n_t, num_dg_nodes, 3) # Reshape back to (N_t, N_dg_nodes, 3)
        current_xyz = rotated_xyz
        
        # 3. Apply translation (delta)
        # dg_node_attrs["xyz"] (translations) has shape (N_t, N_dg_nodes, 3)
        dg_node_translations = dg_node_attrs["xyz"]
        updated_dg_node_xyz = current_xyz + dg_node_translations
        
        return updated_dg_node_xyz
    
    
    def get_timed_dg_w_forward_pass(self, timestamp: Float[Tensor, "N_t"] = None, frame_idx: Int[Tensor, "N_t"] = None) -> Float[Tensor, "N_t N_dg_nodes 3"]:
        """
        Calculate the updated positions of the deformation graph nodes.
        This is done by applying their timed deformation attributes (scale, rotation, translation)
        to their rest positions. The transformation order is Scale -> Rotate -> Translate.
        """
        # Get all timed attributes for the deformation graph nodes
        # dg_node_attrs will contain 'xyz', 'rotation', and potentially 'scale' and 'opacity'.
        # - 'xyz': delta translation for the DG nodes.
        # - 'rotation': rotation of the DG nodes.
        # - 'scale': scale transformation of the DG nodes.
        dg_node_attrs = self.get_timed_dg_attributes(timestamp=timestamp, frame_idx=frame_idx)
        
        rest_dg_node_xyz = self._deform_graph_node_xyz  # Shape: (N_dg_nodes, 3)
        
        # Determine n_t (number of time steps)
        if timestamp is not None:
            n_t = timestamp.shape[0]
        elif frame_idx is not None:
            n_t = frame_idx.shape[0]
        else:
            # This case should ideally be prevented by get_timed_dg_attributes
            raise ValueError("Either timestamp or frame_idx must be provided and be non-empty.")

        # current_xyz starts as rest positions, expanded for batch operations over N_t
        # Shape: (1, N_dg_nodes, 3) -> expanded to (N_t, N_dg_nodes, 3)
        current_xyz = rest_dg_node_xyz.unsqueeze(0).expand(n_t, -1, -1) # Shape: (N_t, N_dg_nodes, 3)

        # 1. Apply scale transformation (if scale is available)
        # dg_node_attrs["scale"] has shape (N_t, N_dg_nodes, 3, 3) or is None
        dg_node_scales = dg_node_attrs.get("scale")
        if dg_node_scales is not None:
            # Reshape for batch matrix multiplication:
            # Input tensors for bmm: (B, N, M) and (B, M, P)
            # Here, B = N_t * N_dg_nodes (number of nodes across all time steps)
            # dg_node_scales reshaped: (N_t * N_dg_nodes, 3, 3)
            # current_xyz reshaped: (N_t * N_dg_nodes, 3, 1)
            scaled_xyz = torch.bmm(
                dg_node_scales.reshape(-1, 3, 3),
                current_xyz.reshape(-1, 3, 1)
            ).reshape(n_t, -1, 3) # Reshape back to (N_t, N_dg_nodes, 3)
            current_xyz = scaled_xyz
        
        # 2. Apply rotation transformation
        # dg_node_attrs["rotation"] is pp.SO3 of shape (N_t, N_dg_nodes, 4 for quat)
        # .matrix() gives shape (N_t, N_dg_nodes, 3, 3)
        dg_node_rot_matrices = dg_node_attrs["rotation"].matrix()
        
        rotated_xyz = torch.bmm(
            dg_node_rot_matrices.reshape(-1, 3, 3), # (N_t * N_dg_nodes, 3, 3)
            current_xyz.reshape(-1, 3, 1)           # (N_t * N_dg_nodes, 3, 1)
        ).reshape(n_t, -1, 3) # Reshape back to (N_t, N_dg_nodes, 3)
        current_xyz = rotated_xyz
        
        # 3. Apply translation (delta)
        # dg_node_attrs["xyz"] (translations) has shape (N_t, N_dg_nodes, 3)
        dg_node_translations = dg_node_attrs["xyz"]
        updated_dg_node_xyz = current_xyz + dg_node_translations
        
        return updated_dg_node_xyz

    # =========== Compute mesh vertices' attributes ========== #    
    def get_timed_vertex_attributes(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None,
    ) -> Dict[str, Union[Float[Tensor, "N_t N_p C"], pp.LieTensor]]:
        if self.cfg.use_deform_graph:
            vert_attrs = self._get_timed_vertex_attributes_from_dg(timestamp, frame_idx)
        else:
            vert_attrs = self._get_timed_vertex_attributes(timestamp, frame_idx)
        # cache deformed mesh vert positions
        for i in range(vert_attrs["xyz"].shape[0]):
            t = timestamp[i] if timestamp is not None else None
            f = frame_idx[i] if frame_idx is not None else None
            key = dict_temporal_key(t, f)
            self._deformed_vert_positions[key] = vert_attrs["xyz"][i]
            self._deformed_vert_rotations[key] = vert_attrs["rotation"][i]
            self._cached_timed_dg_xyz[key] = vert_attrs["deformed_dg"][i] if "deformed_dg" in vert_attrs else None

        return vert_attrs

    def _get_timed_vertex_attributes_from_dg(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None,
    ) -> Dict[str, Union[Float[Tensor, "N_t N_p C"], pp.LieTensor]]:
        n_t = len(timestamp) if timestamp is not None else len(frame_idx)
        neighbor_nodes_xyz: Float[Tensor, "N_p N_n 3"]
        neighbor_nodes_xyz = self._deform_graph_node_xyz[self._xyz_neighbor_node_idx]
        
        # debug time
        # start_time = time.time_ns()
        dg_node_attrs = self.get_timed_dg_attributes(timestamp, frame_idx)
        
        dg_node_trans, dg_node_rots = dg_node_attrs["xyz"], dg_node_attrs["rotation"]

        neighbor_nodes_trans: Float[Tensor, "N_t N_p N_n 3"]
        neighbor_nodes_rots: Float[pp.LieTensor, "N_t N_p N_n 4"]
        neighbor_nodes_trans = dg_node_trans[:, self._xyz_neighbor_node_idx]
        neighbor_nodes_rots = dg_node_rots[:, self._xyz_neighbor_node_idx]

        # debug time
        # start_time = time.time_ns()
        if (self.cfg.d_scale or self.cfg.skinning_method == "hybrid" or self.cfg.skinning_method == "lbs") and self.cfg.use_shear_matrix:
            dg_node_scales = dg_node_attrs.get("scale")
            assert dg_node_scales is not None

            neighbor_nodes_scales: Float[Tensor, "N_t N_p N_n 3 3"]
            neighbor_nodes_scales = dg_node_scales[:, self._xyz_neighbor_node_idx]

        # deform vertex xyz
        if self.cfg.skinning_method == "lbs" or self.cfg.skinning_method == "hybrid":
            num_pts = self.get_xyz_verts.shape[0]
            if self.cfg.use_shear_matrix:
                # neighbor_nodes_scales becomes: 1, N_v, N_n, 3, 3
                neighbor_nodes_scales = neighbor_nodes_scales.reshape(-1, 3, 3)
                xyz_verts = self.get_xyz_verts.unsqueeze(0).unsqueeze(2).unsqueeze(-1).repeat(n_t, 1, neighbor_nodes_xyz.shape[1], 1, 1).reshape(-1, 3, 1)
                
                deformed_xyz = torch.bmm(
                    neighbor_nodes_scales, xyz_verts
                )
                # Deformed xyz has shape N_v * N_n, 3, 1
            else:
                deformed_xyz = self.get_xyz_verts.unsqueeze(0).unsqueeze(2).unsqueeze(-1).repeat(
                    n_t, 1, neighbor_nodes_xyz.shape[1], 1, 1
                ).reshape(-1, 3, 1)
                
            deformed_xyz = torch.bmm(
                neighbor_nodes_rots.matrix().reshape(-1, 3, 3), deformed_xyz
            ).squeeze(-1).reshape(n_t, num_pts, -1, 3)
            deformed_xyz = deformed_xyz + neighbor_nodes_trans

            nn_weights = self._xyz_neighbor_nodes_weights[None, :, :, None]
            deformed_vert_xyz_lbs = (nn_weights * deformed_xyz).sum(dim=2)

        if self.cfg.skinning_method == "dqs" or self.cfg.skinning_method == "hybrid":
            dual_quat = DualQuaternion.from_quat_pose_array(
                torch.cat([neighbor_nodes_rots.tensor(), neighbor_nodes_trans], dim=-1)
            )
            q_real: Float[Tensor, "N_t N_p N_n 4"] = dual_quat.q_r.tensor()
            q_dual: Float[Tensor, "N_t N_p N_n 4"] = dual_quat.q_d.tensor()
            nn_weights = self._xyz_neighbor_nodes_weights[None, :, :, None]
            weighted_sum_q_real: Float[Tensor, "N_t N_p 4"] = (q_real * nn_weights).sum(dim=-2)
            weighted_sum_q_dual: Float[Tensor, "N_t N_p 4"] = (q_dual * nn_weights).sum(dim=-2)
            weighted_sum_dual_quat = DualQuaternion.from_dq_array(
                torch.cat([weighted_sum_q_real, weighted_sum_q_dual], dim=-1)
            )
            dq_normalized = weighted_sum_dual_quat.normalized()
            deformed_vert_xyz_dqs = dq_normalized.transform_point_simple(self.get_xyz_verts)


        if self.cfg.skinning_method == "lbs":
            deformed_vert_xyz = deformed_vert_xyz_lbs
        elif self.cfg.skinning_method == "dqs":
            deformed_vert_xyz = deformed_vert_xyz_dqs
        elif self.cfg.skinning_method == "hybrid":
            neighbor_nodes_opacity = dg_node_attrs["opacity"][:, self._xyz_neighbor_node_idx]
            vert_lbs_weight = (
                self._xyz_neighbor_nodes_weights[None, ..., None] * neighbor_nodes_opacity
            ).sum(dim=-2)
            # debug
            vert_lbs_weight = torch.clamp(vert_lbs_weight + 0.4, max=1.0)

            deformed_vert_xyz = vert_lbs_weight * deformed_vert_xyz_lbs + (1 - vert_lbs_weight) * deformed_vert_xyz_dqs

        # deform vertex rotation
        deformed_vert_rots: Float[pp.LieTensor, "N_t N_p 4"]
        deformed_vert_rots = (
            self._xyz_neighbor_nodes_weights[None, ..., None] * neighbor_nodes_rots.Log()
        ).sum(dim=-2)
        deformed_vert_rots = pp.so3(deformed_vert_rots).Exp()

        outs = {"xyz": deformed_vert_xyz, "rotation": deformed_vert_rots}
        if self.cfg.d_scale:
            deformed_vert_scales: Float[Tensor, "N_t N_p 3 3"]
            if self.cfg.skinning_method == "lbs":
                deformed_vert_scales = (
                    self._xyz_neighbor_nodes_weights[None, ..., None, None] * neighbor_nodes_scales
                ).sum(dim=-3)
                outs["scale"] = deformed_vert_scales
            elif self.cfg.skinning_method == "hybrid":
                deformed_vert_scales = (
                    self._xyz_neighbor_nodes_weights[None, ..., None, None]
                    * neighbor_nodes_opacity[..., None]
                    * neighbor_nodes_scales
                ).sum(dim=-3)
                deformed_vert_scales = (
                    deformed_vert_scales
                    + (1 - vert_lbs_weight)[..., None] * torch.eye(3).to(deformed_vert_scales)
                )
                outs["scale"] = deformed_vert_scales
        
        if self.cfg.deformation_driver == "skeleton":
            outs["deformed_dg"] = self.get_timed_dg(dg_node_attrs=dg_node_attrs)
            
        return outs

    def _get_timed_vertex_attributes(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None
    ) -> Dict[str, Union[Float[Tensor, "N_t N_p C"], pp.LieTensor]]:
        if self.dynamic_mode == "discrete":
            assert frame_idx is not None
            trans = self._vert_trans[frame_idx]
            rot = self._vert_rots[frame_idx]
            d_scale = self._vert_scales[frame_idx] if self.cfg.d_scale else None
            d_opacity = None

        elif self.dynamic_mode == "deformation":
            assert timestamp is not None
            pts = self._points

            num_pts = pts.shape[0]
            num_t = timestamp.shape[0]
            pts = torch.cat([pts] * num_t, dim=0)
            ts = timestamp.unsqueeze(-1).repeat_interleave(num_pts, dim=0)
            trans, rot, d_scale, d_opacity = self._deformation.forward_dynamic_delta(pts, ts * 2 - 1)
            # trans, rot = self._deformation.forward_dg_trans_and_rotation(pts, ts * 2 - 1)
            trans = trans.reshape(num_t, num_pts, 3)
            rot = rot.reshape(num_t, num_pts, 4)

            # NOTE: why do this?
            idt_quaternion = torch.zeros((1, num_pts, 4)).to(rot)
            idt_quaternion[..., -1] = 1
            rot = rot + idt_quaternion

            if d_scale is not None:
                d_scale = d_scale.reshape(num_t, num_pts, 3)
            if d_opacity is not None:
                d_opacity = d_opacity.reshape(num_t, num_pts, 1)
        # rot = rot / rot.norm(dim=-1, keepdim=True)
        rot = F.normalize(rot, dim=-1)
        attrs = {
            "xyz": trans, "rotation": pp.SO3(rot), "scale": d_scale
        }
        return attrs

    # ========= Compute gaussian kernals' attributes ========= #
    def get_timed_gs_attributes(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"] = None,
    ) -> Dict[str, Union[Float[Tensor, "N_t N_p C"], pp.LieTensor]]:
        vert_attrs = self.get_timed_vertex_attributes(timestamp, frame_idx)
        # xyz
        # debug time
        # start_time = time.time_ns()
        gs_timed_xyz = self._get_gs_xyz_from_vertex(vert_attrs["xyz"])

        # rotations
        gs_drots_q: Float[pp.LieTensor, "N_t N_g 4"] = fuse_rotations(
            self._gs_vert_connections, self._gs_bary_weights, vert_attrs["rotation"]
        )
        gs_rots_q_orig = pp.SO3(
            self.get_rotation[None, :, [1, 2, 3, 0]])  # NOTE: the quaternion order should be considered
        gs_timed_rots = gs_drots_q @ gs_rots_q_orig
        gs_timed_rots = gs_timed_rots.tensor()[..., [3, 0, 1, 2]]
        gs_timed_rots = F.normalize(gs_timed_rots, dim=-1)
        
        # debug time
        # print(f"Skinning GS time: {time.time_ns() - start_time} ns")


        gs_attrs = {"xyz": gs_timed_xyz, "rotation": gs_timed_rots}
        # scales
        if self.cfg.d_scale:
            # gs_scales_orig = self._scales
            # gs_scales_orig = torch.cat([
            #     torch.zeros(len(self._scales), 1, device=self.device),
            #     self._scales], dim=-1)
            # vert_timed_dscales = vert_attrs["scale"][:, self._gs_vert_connections, :]
            # gs_timed_dscale = (self._gs_bary_weights[None] * vert_timed_dscales).sum(dim=-2)
            # # gs_timed_scales = gs_scales_orig + gs_timed_dscale
            # gs_timed_scales = gs_scales_orig * (gs_timed_dscale + 1)
            # gs_timed_scales = torch.cat([
            #     self.surface_mesh_thickness * torch.ones(*gs_timed_scales.shape[:-1], 1, device=self.device),
            #     self.scale_activation(gs_timed_scales[..., 1:])], dim=-1)
            # gs_attrs["scale"] = gs_timed_scales

            gs_scales_orig = torch.stack([self.scaling]*gs_timed_xyz.shape[0], dim=0)
            vert_timed_dscales = vert_attrs["scale"][:, self._gs_vert_connections, ...]
            gs_timed_dscale = (self._gs_bary_weights[None, ..., None] * vert_timed_dscales).sum(dim=-3)
            gs_timed_scales = torch.einsum(
                "tpij,tpjk->tpik", gs_timed_dscale, gs_scales_orig.unsqueeze(-1)
            ).squeeze(-1)
            gs_attrs["scale"] = gs_timed_scales

        return gs_attrs

    def get_timed_gs_all_single_time(self, timestamp=None, frame_idx=None):
        if timestamp is not None and timestamp.ndim == 0:
            timestamp = timestamp[None]
        if frame_idx is not None and frame_idx.ndim == 0:
            frame_idx = frame_idx[None]

        gs_timed_attrs = self.get_timed_gs_attributes(timestamp, frame_idx)
        means3D = gs_timed_attrs["xyz"][0]
        rotations = gs_timed_attrs["rotation"][0]
        if self.cfg.d_scale:
            scales = gs_timed_attrs["scale"][0]
        else:
            scales = self.get_scaling

        opacity = self.get_opacity
        colors_precomp = self.get_points_rgb()
        return means3D, scales, rotations, opacity, colors_precomp
    
    def _get_gs_xyz_from_vertex(self, xyz_vert=None) -> Float[Tensor, "N_t N_gs 3"]:
        if self.binded_to_surface_mesh:
            if xyz_vert is None:
                xyz_vert = self._points
            # First gather vertices of all triangles
            if xyz_vert.ndim == 2:
                xyz_vert = xyz_vert[None]  # (n_t, n_v, 3)
            faces_verts = xyz_vert[:, self._surface_mesh_faces]  # (n_t, n_faces, 3, 3)

            # Then compute the points using barycenter coordinates in the surface triangles
            points = faces_verts[:, :, None] * self.surface_triangle_bary_coords[
                None, None]  # n_t, n_faces, n_gaussians_per_face, 3, n_coords
            points = points.sum(dim=-2)  # n_t, n_faces, n_gaussians_per_face, n_coords
            # points = points.reshape(self._n_points, 3)
            points = rearrange(points, "t f n c -> t (f n) c")
        else:
            raise ValueError("No vertices when with no mesh binded.")
        return points

    def build_deformation_graph(self, n_nodes, xyz_nodes=None, nodes_connectivity=6, mode="geodisc"):
        threestudio.info(f"Building deformation graph with `{mode}`...")
        device = self.device
        xyz_verts = self.get_xyz_verts
        self._xyz_cpu = xyz_verts.cpu().numpy()
        torch3d_mesh = load_objs_as_meshes([self.cfg.surface_mesh_to_bind_path], device=self.device)
            
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices =  o3d.utility.Vector3dVector(torch3d_mesh.verts_list()[0].cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(torch3d_mesh.faces_list()[0].cpu().numpy())
        mesh.vertex_normals = o3d.utility.Vector3dVector(torch3d_mesh.verts_normals_list()[0].cpu().numpy())

        if xyz_nodes is None:
            downpcd = mesh.sample_points_uniformly(number_of_points=n_nodes)
            # downpcd = mesh.sample_points_poisson_disk(number_of_points=1000, pcl=downpcd)
        else:
            downpcd = o3d.geometry.PointCloud()
            downpcd.points = o3d.utility.Vector3dVector(xyz_nodes.cpu().numpy())

        downpcd.paint_uniform_color([0.5, 0.5, 0.5])
        self._deform_graph_node_xyz = torch.from_numpy(np.asarray(downpcd.points)).float().to(device)

        downpcd_tree = o3d.geometry.KDTreeFlann(downpcd)

        if mode == "eucdisc":
            downpcd_size, _ = self._deform_graph_node_xyz.size()
            # TODO delete unused connectivity attr
            deform_graph_connectivity = [
                torch.from_numpy(
                    np.asarray(
                        downpcd_tree.search_knn_vector_3d(downpcd.points[i], nodes_connectivity + 1)[1][1:]
                    )
                ).to(device)
                for i in range(downpcd_size)
            ]
            self._deform_graph_connectivity = torch.stack(deform_graph_connectivity).long().to(device)
            self._deform_graph_tree = downpcd_tree

            # build connections between the original point cloud to deformation graph node
            xyz_neighbor_node_idx = [
                torch.from_numpy(
                    np.asarray(downpcd_tree.search_knn_vector_3d(self._xyz_cpu[i], nodes_connectivity)[1])
                ).to(device)
                for i in range(self._xyz_cpu.shape[0])
            ]
            xyz_neighbor_nodes_weights = [
                torch.from_numpy(
                    np.asarray(downpcd_tree.search_knn_vector_3d(self._xyz_cpu[i], nodes_connectivity)[2])
                ).float().to(device)
                for i in range(self._xyz_cpu.shape[0])
            ]
        elif mode == "geodisc":
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)

            # init geodisc calculation algorithm (geodesic version)
            # geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, faces)

            # init geodisc calculation algorithm (potpourri3d  version)
            geoalg = pp3d.MeshHeatMethodDistanceSolver(vertices, faces)

            # 1. build a kd tree for all vertices in mesh
            mesh_pcl = o3d.geometry.PointCloud()
            mesh_pcl.points = o3d.utility.Vector3dVector(mesh.vertices)
            mesh_kdtree = o3d.geometry.KDTreeFlann(mesh_pcl)

            # 2. find the nearest vertex of all downsampled points and get their index.
            nearest_vertex = [
                np.asarray(mesh_kdtree.search_knn_vector_3d(downpcd.points[i], 1)[1])[0]
                for i in range(n_nodes)
            ]
            target_index = np.array(nearest_vertex)

            # 3. find k nearest neighbors(geodistance) of mesh vertices and downsample pointcloud
            xyz_neighbor_node_idx = []
            xyz_neighbor_nodes_weights = []
            downpcd_points = np.asarray(downpcd.points)
            for i in trange(self._xyz_cpu.shape[0], leave=False):
                source_index = np.array([i])
                # geodesic distance calculation
                # distances = geoalg.geodesicDistances(source_index, target_index)[0]

                # potpourri3d distance calculation
                distances = geoalg.compute_distance(source_index)[target_index]

                sorted_index = np.argsort(distances)

                k_n_neighbor = sorted_index[:nodes_connectivity]
                k_n_plus1_neighbor = sorted_index[:nodes_connectivity + 1]
                vert_to_neighbor_dists = np.linalg.norm(
                    vertices[i] - downpcd_points[k_n_plus1_neighbor], axis=-1
                )

                xyz_neighbor_node_idx.append(torch.from_numpy(k_n_neighbor).to(device))

                xyz_neighbor_nodes_weights.append(
                    torch.from_numpy(
                        (1 - vert_to_neighbor_dists[:nodes_connectivity] / vert_to_neighbor_dists[-1]) ** 2
                    ).float().to(device)
                )
        else:
            raise ValueError("The mode must be eucdisc or geodisc!")

        self._xyz_neighbor_node_idx = torch.stack(xyz_neighbor_node_idx).long().to(device)

        print(torch.max(self._xyz_neighbor_node_idx))
        print(torch.min(self._xyz_neighbor_node_idx))

        self._xyz_neighbor_nodes_weights = torch.stack(xyz_neighbor_nodes_weights).to(device)
        # normalize
        self._xyz_neighbor_nodes_weights = (self._xyz_neighbor_nodes_weights
                                            / self._xyz_neighbor_nodes_weights.sum(dim=-1, keepdim=True)
                                            )
    # ======== Skeleton binding ======== #
    def parse_joints_pred(self, joints_pred_path: str):
        positions = []
        bone_pairs = []
        with open(joints_pred_path, "r") as file:
            for line in file:
                tokens = line.strip().split()
                if tokens:
                    if tokens[0] == "joints":
                        x, y, z = map(float, tokens[2:5]) # Example line: joints joint0 -0.26757812 -0.35693359 -0.16052246
                        positions.append([x, y, z])
                    if tokens[0] == "hier":
                        parent_name, child_name = tokens[1], tokens[2]
                        parent_idx = int(parent_name.replace("joint", ""))
                        child_idx  = int(child_name .replace("joint", ""))
                        bone_pairs.append((parent_idx, child_idx))
                

        self.joints_xyz = torch.from_numpy(np.array(positions)).float().to(self.device)
        # self.bone_pairs = bone_pairs
        print("Joint positions shape:", self.joints_xyz.shape) # J x 3
        bone_pairs = torch.tensor(bone_pairs, dtype=torch.long)
        p_idx, c_idx = bone_pairs.t()
        
        rest_vec = self.joints_xyz[p_idx] - self.joints_xyz[c_idx]
        rest_len = rest_vec.norm(dim=-1)
        
        eps = 1e-6
        if (rest_len < eps).any():
            from warnings import warn
            bad = (rest_len < eps).nonzero(as_tuple=False).flatten()
            warn(
                f"{bad.numel()} bones have near-zero length; "
                f"clamping them to {eps:.1e} to keep training stable."
            )
            rest_len[bad] = eps    
        
        
        # rest_lengths = []
        # for (p, c) in self.bone_pairs:
        #     rest_lengths.append(np.linalg.norm(self.joints_xyz[p] - self.joints_xyz[c]))
        # self.rest_bone_lengths = torch.from_numpy(np.array(rest_lengths)).float().to(self.device) # Number of bones x 1
        
        self.register_buffer("bone_parent_idx", p_idx)
        self.register_buffer("bone_child_idx",  c_idx)
        self.register_buffer("rest_bone_lengths", rest_len)
        
    def parse_joints_pred_with_skinning(self, rig_file_path: str):
        """
        Parses a rig file and sets up joint positions, bone relationships, and skinning information.
        
        Args:
            rig_file_path: Path to the *_rig.txt file containing joint and skinning info
        """
        # Data structures to store joint info
        joints_dict = {}  # Store joint name -> position mapping
        joint_name_to_idx = {}  # Store joint name -> index mapping
        bone_pairs = []  # Store parent-child relationships
        root_joint = None
        
        # Data structures to store skinning info
        vertex_to_joints_weights = {}  # Store vertex -> (joint_indices, weights) mapping
        max_connectivity = 0  # Track maximum number of influences per vertex
        
        # Parse the file once and extract all information
        with open(rig_file_path, "r") as file:
            joint_idx = 0
            for line in file:
                line = line.strip()
                if not line or line.startswith("//"):  # Skip empty lines and comments
                    continue
                
                tokens = line.split()
                
                if tokens[0] == "joints":
                    # Format: joints joint_name x y z
                    joint_name = tokens[1]
                    x, y, z = map(float, tokens[2:5])
                    
                    joints_dict[joint_name] = [x, y, z]
                    joint_name_to_idx[joint_name] = joint_idx
                    joint_idx += 1
                    
                elif tokens[0] == "root" and len(tokens) > 1:
                    # Format: root root_joint_name
                    root_joint = tokens[1]
                    
                elif tokens[0] == "hier" and len(tokens) >= 3:
                    # Format: hier parent_name child_name
                    parent_name, child_name = tokens[1], tokens[2]
                    if parent_name in joint_name_to_idx and child_name in joint_name_to_idx:
                        parent_idx = joint_name_to_idx[parent_name]
                        child_idx = joint_name_to_idx[child_name]
                        bone_pairs.append((parent_idx, child_idx))
                    
                elif tokens[0] == "skin" and len(tokens) >= 4:
                    # Format: skin vertex_id joint_name weight [joint_name weight ...]
                    vertex_id = int(tokens[1])
                    
                    # Parse all joint-weight pairs
                    joint_indices = []
                    weights = []
                    i = 2
                    while i < len(tokens):
                        joint_name = tokens[i]
                        if i+1 < len(tokens) and joint_name in joint_name_to_idx:
                            try:
                                weight = float(tokens[i+1])
                                joint_idx = joint_name_to_idx[joint_name]
                                joint_indices.append(joint_idx)
                                weights.append(weight)
                                i += 2
                            except ValueError:
                                break  # If we can't convert to float, stop parsing
                        else:
                            break
                    
                    if joint_indices:  # Only store if we have valid indices
                        vertex_to_joints_weights[vertex_id] = (joint_indices, weights)
                        # Update max_connectivity
                        max_connectivity = max(max_connectivity, len(joint_indices))
        
        # Sort joint names to ensure consistent ordering
        joint_names = sorted(joints_dict.keys(), key=lambda x: joint_name_to_idx[x])
        
        # Create a list of joint positions in the correct order
        positions = [joints_dict[name] for name in joint_names]
        
        # Store the joint positions
        self.joints_xyz = torch.tensor(positions, dtype=torch.float32).to(self.device)
        self.joint_names = joint_names
        self.joint_name_to_idx = joint_name_to_idx
        print(f"Joint positions shape: {self.joints_xyz.shape}")  # J x 3
        print(f"Maximum connectivity found: {max_connectivity} joints per vertex")
        
        # Store skin weights for later use
        self.vertex_to_joints_weights = vertex_to_joints_weights
        self.max_connectivity = max_connectivity
        
        # Process bone pairs if they exist
        if bone_pairs:
            bone_pairs = torch.tensor(bone_pairs, dtype=torch.long)
            p_idx, c_idx = bone_pairs.t()
            
            # Calculate rest bone lengths
            rest_vec = self.joints_xyz[p_idx] - self.joints_xyz[c_idx]
            rest_len = rest_vec.norm(dim=-1)
            
            # Handle zero-length bones
            eps = 1e-6
            if (rest_len < eps).any():
                from warnings import warn
                bad = (rest_len < eps).nonzero(as_tuple=False).flatten()
                warn(
                    f"{bad.numel()} bones have near-zero length; "
                    f"clamping them to {eps:.1e} to keep training stable."
                )
                rest_len[bad] = eps
            
            # Register bone relationship buffers
            self.register_buffer("bone_parent_idx", p_idx)
            self.register_buffer("bone_child_idx", c_idx)
            self.register_buffer("rest_bone_lengths", rest_len)
        else:
            print("Warning: No bone hierarchy information found in rig file")

        
    def build_skeleton_binding(self, nodes_connectivity: int = 4):
        device = self.device
        k = nodes_connectivity

        torch3d_mesh = load_objs_as_meshes(
            [self.cfg.surface_mesh_to_bind_path],
            device=device
        )
        verts_np = torch3d_mesh.verts_list()[0].cpu().numpy()   # [V×3]
        faces_np = torch3d_mesh.faces_list()[0].cpu().numpy()   # [F×3]

        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(verts_np)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces_np)

        geo_solver = pp3d.MeshHeatMethodDistanceSolver(verts_np, faces_np)

        # joints_np = np.asarray(self.joints_xyz, dtype=np.float64)
        # self._deform_graph_node_xyz = torch.from_numpy(joints_np).float().to(device)  # [J×3] # naming it like this to be consistent with the rest of the code
        self._deform_graph_node_xyz = self.joints_xyz
        J = self._deform_graph_node_xyz.shape[0]
        
        joints_np = self.joints_xyz.cpu().numpy()

        # Build Open3D point cloud for mesh vertices (to snap joints)
        mesh_pcd = o3d.geometry.PointCloud()
        mesh_pcd.points = o3d.utility.Vector3dVector(verts_np)
        mesh_tree = o3d.geometry.KDTreeFlann(mesh_pcd)

        # For each joint, find the nearest mesh-vertex index
        nearest_vertex = [
            np.asarray(mesh_tree.search_knn_vector_3d(joints_np[i], 1)[1])[0]
            for i in range(J)
        ] # nearest_vertex is a list of indices of the nearest mesh vertices to each joint
        target_index = np.array(nearest_vertex)

        # For each vertex, find its nearest k joints (by geodesic rank) & compute weights
        V = verts_np.shape[0]
        neighbor_idx_list = []
        neighbor_wgt_list = []
        for i in trange(V, desc="Binding vertices to skeleton", leave=False):
            source_index = np.array([i])
            # Compute distances from vertex i to all mesh vertices
            dist_to_joints = geo_solver.compute_distance(source_index)[target_index]  # numpy array [V]
            # Extract distances at the snapped‐joint indices

            # Sort joints by geodesic distance
            sorted_j = np.argsort(dist_to_joints)
            k_n_neighbor = sorted_j[:nodes_connectivity]  # Get the indices of the k nearest joints
            k_n_plus1_neighbor = sorted_j[:nodes_connectivity + 1]  # Get the indices of the k+1 nearest joints
            verts_to_neighbor_dists = np.linalg.norm(
                verts_np[i] - joints_np[k_n_plus1_neighbor], axis=-1
            )
            
            neighbor_idx_list.append(torch.LongTensor(k_n_neighbor).to(device))  # [k]
            neighbor_wgt_list.append(
                torch.from_numpy(
                    (1 - verts_to_neighbor_dists[:nodes_connectivity] / verts_to_neighbor_dists[-1]) ** 2
                ).float().to(device)  # [k] 
            )

        # Stack into [V×k] tensors
        self._xyz_neighbor_node_idx = torch.stack(neighbor_idx_list).long().to(device)    # LongTensor [V×k]
        self._xyz_neighbor_nodes_weights = torch.stack(neighbor_wgt_list).to(device)    # FloatTensor [V×k] # naming it like this to be consistent with the rest of the code
        
        # normalize joint weights
        self._xyz_neighbor_nodes_weights = (self._xyz_neighbor_nodes_weights / self._xyz_neighbor_nodes_weights.sum(dim=-1, keepdim=True))
        
    def build_skeleton_binding_refined(self, nodes_connectivity: int = 4):
        device = self.device
        k = nodes_connectivity

        torch3d_mesh = load_objs_as_meshes(
            [self.cfg.surface_mesh_to_bind_path],
            device=device
        )
        verts_np = torch3d_mesh.verts_list()[0].cpu().numpy()   # [V×3]
        faces_np = torch3d_mesh.faces_list()[0].cpu().numpy()   # [F×3]
        V = verts_np.shape[0]

        # The potpourri3d solver is efficient and takes numpy arrays directly.
        geo_solver = pp3d.MeshHeatMethodDistanceSolver(verts_np, faces_np)

        self._deform_graph_node_xyz = self.joints_xyz
        joints_np = self.joints_xyz.cpu().numpy()
        J = joints_np.shape[0]

        # Build a KDTree on mesh vertices to snap joints to the surface
        mesh_pcd = o3d.geometry.PointCloud()
        mesh_pcd.points = o3d.utility.Vector3dVector(verts_np)
        mesh_tree = o3d.geometry.KDTreeFlann(mesh_pcd)

        # For each joint, find the index of the nearest mesh vertex
        nearest_vertex_indices = np.array([
            mesh_tree.search_knn_vector_3d(joints_np[i], 1)[1][0]
            for i in range(J)
        ])

        # --- OPTIMIZED WEIGHTING LOGIC ---

        # 1. Compute geodesic distances from each snapped joint to all vertices.
        # This is much faster than looping over all V vertices.
        all_joint_dists = np.zeros((J, V))
        for i, vert_idx in enumerate(tqdm(nearest_vertex_indices, desc="Computing geodesic distances from joints")):
            all_joint_dists[i] = geo_solver.compute_distance(vert_idx)

        # Transpose to get a [V, J] matrix where dists[v, j] is the distance
        # from vertex v to joint j.
        dists_from_joints = all_joint_dists.T

        # 2. For each vertex, find the k-nearest joints using vectorized operations.
        k_nearest_joint_indices = np.argsort(dists_from_joints, axis=1)[:, :k]

        # 3. Get the corresponding geodesic distances for the k-nearest joints.
        k_nearest_joint_dists = np.take_along_axis(dists_from_joints, k_nearest_joint_indices, axis=1)

        # 4. Compute weights using inverse square distance. Add epsilon for stability.
        epsilon = 1e-8
        weights = 1.0 / (k_nearest_joint_dists**2 + epsilon)

        # 5. Convert to PyTorch tensors.
        self._xyz_neighbor_node_idx = torch.from_numpy(k_nearest_joint_indices).long().to(device)
        self._xyz_neighbor_nodes_weights = torch.from_numpy(weights).float().to(device)

        # 6. Normalize joint weights so they sum to 1 for each vertex.
        self._xyz_neighbor_nodes_weights = (
            self._xyz_neighbor_nodes_weights / self._xyz_neighbor_nodes_weights.sum(dim=-1, keepdim=True)
        )
        
    def build_skeleton_binding_refined_project_joints(self, nodes_connectivity: int = 4):
        import trimesh
        device = self.device
        k = nodes_connectivity

        # --- 1. Load mesh --------------------------------------------------------
        torch3d_mesh = load_objs_as_meshes(
            [self.cfg.surface_mesh_to_bind_path],
            device=device
        )
        verts_np = torch3d_mesh.verts_list()[0].cpu().numpy()   # [V×3]
        faces_np = torch3d_mesh.faces_list()[0].cpu().numpy()   # [F×3]
        V = verts_np.shape[0]

        # --- 2. Build the heat‐method solver -------------------------------------
        geo_solver = pp3d.MeshHeatMethodDistanceSolver(verts_np, faces_np)

        # --- 3. Project joints to surface with barycentric coords ----------------
        joints_np = self.joints_xyz.cpu().numpy()  # [J×3]
        self._deform_graph_node_xyz = self.joints_xyz # for consistency with the rest of the code
        J = joints_np.shape[0]
        mesh_tm   = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
        
        # find the closest point on the mesh to each random point
        closest_pts, distances, triangle_id = mesh_tm.nearest.on_surface(joints_np)
        # closest_pts: [J×3], distances: [J], triangle_id: [J,]  # closest points on the mesh to each joint
        # distance from point to surface of mesh distances
        tri_verts = verts_np[faces_np]   # shape [F, 3, 3]

        # Compute barycentric coords of each closest_pts within its triangle
        # returns array [J×3] of weights (w0, w1, w2) for v0,v1,v2
        bary_coords = trimesh.triangles.points_to_barycentric(
            tri_verts[triangle_id],  # select the triangles for each joint
            closest_pts              # the projected points
        )

        # Allocate the blended-distance array
        dists_from_joints = np.zeros((V, J), dtype=np.float32)

        for j in tqdm(range(J), desc="Geo‐solves & blend"):
            fidx = int(triangle_id[j])        # index of face under joint j
            bc   = bary_coords[j]             # e.g. [w0, w1, w2]
            v0, v1, v2 = faces_np[fidx]       # the three mesh-vertex indices

            # Compute three vertex-seeded geodesic fields
            d0 = geo_solver.compute_distance(v0)  # [V,]
            d1 = geo_solver.compute_distance(v1)
            d2 = geo_solver.compute_distance(v2)

            # Blend by barycentric weights to approximate
            # geodesic from the true projected point
            dists_from_joints[:, j] = bc[0]*d0 + bc[1]*d1 + bc[2]*d2

        # --- 5. Find k nearest joints & compute normalized weights ---------------
        # argsort along J to pick the k smallest distances per vertex
        k_idx  = np.argsort(dists_from_joints, axis=1)[:, :k]            # [V×k]
        k_dist = np.take_along_axis(dists_from_joints, k_idx, axis=1)    # [V×k]

        # inverse‐square weighting (with epsilon for stability)
        eps     = 1e-8
        weights = 1.0 / (k_dist**2 + eps)        # [V×k]
        weights = weights / weights.sum(axis=1, keepdims=True)

        # stash in torch
        self._xyz_neighbor_node_idx      = torch.from_numpy(k_idx).long().to(device)
        self._xyz_neighbor_nodes_weights = torch.from_numpy(weights).float().to(device)

    def build_skeleton_binding_with_skinning(self):
        device = self.device
        k = self.max_connectivity

        torch3d_mesh = load_objs_as_meshes(
            [self.cfg.surface_mesh_to_bind_path],
            device=device
        )
        verts_np = torch3d_mesh.verts_list()[0].cpu().numpy()   # [V×3]
        V = verts_np.shape[0]

        self._deform_graph_node_xyz = self.joints_xyz

        # Build a KDTree on mesh vertices to snap joints to the surface
        mesh_pcd = o3d.geometry.PointCloud()
        mesh_pcd.points = o3d.utility.Vector3dVector(verts_np)
        
        xyz_neighbor_node_idx = np.zeros((V, k), dtype=np.int64)
        xyz_neighbor_nodes_weights = np.zeros((V, k), dtype=np.float32)

        # --- OPTIMIZED WEIGHTING LOGIC ---
        vertices_with_weights = np.array(list(self.vertex_to_joints_weights.keys()))
        
        for v_idx in tqdm(vertices_with_weights, desc="building skeleton binding with explicit weights"):
            joint_indices, weights = self.vertex_to_joints_weights[v_idx]
                        
            # If we have less than k influences, pad with dummy joints with zero weight
            if len(joint_indices) < k:
                # Use the last joint index as padding (with zero weight)
                pad_joint_idx = 0 if len(joint_indices) == 0 else joint_indices[-1]
                joint_indices.extend([pad_joint_idx] * (k - len(joint_indices)))
                weights.extend([0.0] * (k - len(weights)))
            
            # Store the joint indices and weights
            xyz_neighbor_node_idx[v_idx, :] = joint_indices[:k]
            xyz_neighbor_nodes_weights[v_idx, :] = weights[:k]
                    
        self._xyz_neighbor_node_idx = torch.from_numpy(xyz_neighbor_node_idx).long().to(device)
        self._xyz_neighbor_nodes_weights = torch.from_numpy(xyz_neighbor_nodes_weights).float().to(device)
        
        # 10. Final normalization to ensure weights sum to 1 for each vertex
        self._xyz_neighbor_nodes_weights = (
            self._xyz_neighbor_nodes_weights / self._xyz_neighbor_nodes_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        )
        
    # def calculate_timed_bone_lengths(self, timestamp: Float[Tensor, "N_t"] = None, frame_idx: Int[Tensor, "N_t"] = None):
    #     """
    #     Calculate the lengths of the bones based on the joint positions
    #     """
    #     # timed_skeleton = self._cached_timed_dg_xyz[dict_temporal_key(timestamp, frame_idx)]
    #     timed_skeleton = self.get_timed_dg_xyz_attributes(timestamp=timestamp, frame_idx=frame_idx)
    #     timed_skeleton_lengths = []
    #     for (p, c) in self.bone_pairs:
    #         bone_length = torch.norm(timed_skeleton[:, p] - timed_skeleton[:, c], dim=-1)
    #         timed_skeleton_lengths.append(bone_length.unsqueeze(-1))
    #     timed_skeleton_lengths = torch.cat(timed_skeleton_lengths, dim=-1)  # Shape: (N_t, N_bones)
        
    #     return timed_skeleton_lengths
    
    def calculate_timed_bone_lengths(
        self,
        timestamp: Float[Tensor, "N_t"] = None,
        frame_idx: Int[Tensor, "N_t"]  = None,
    ) -> Float[Tensor, "N_t N_bones"]:
        """
        Return current bone lengths for every requested time step.
        """
        skel_xyz = self.get_timed_dg_xyz_attributes(timestamp=timestamp,
                                                    frame_idx=frame_idx)   # (N_t, J, 3)

        p, c     = self.bone_parent_idx, self.bone_child_idx               # (N_bones,)
        diff     = skel_xyz[:, p] - skel_xyz[:, c]                         # (N_t, N_bones, 3)
        lengths  = diff.norm(dim=-1)                                       # (N_t, N_bones)

        return lengths
    
    # ========= Update step ========= #
        
    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        super().update_step(epoch, global_step, on_load_weights)

        self.dg_timed_attrs = {}
        # self._have_comp_dg_attrs_this_step = False

        self._deformed_vert_positions = {}
        self._deformed_vert_rotations = {}

        # debug
        self.global_step = global_step
        
        self._cached_timed_dg_xyz = {}



def fuse_rotations(
    neighbor_node_idx: Int[Tensor, "N_p N_n"],
    weights: Float[Tensor, "N_p N_n"],
    rotations: Float[pp.LieTensor, "N_t N_v 4"]
):
    """
    q'_i = Exp(\Sigma_{j\in \mathcal{N}(i)} w_{ij} * Log(q_ij))
    """
    rots_log: Float[pp.LieTensor, "N_t N_p N_n 3"] = rotations[:, neighbor_node_idx].Log()
    weighted_rots: Float[pp.LieTensor, "N_t N_p 4"]
    weighted_rots = (weights[None] * rots_log).sum(dim=-2)
    weighted_rots = pp.so3(weighted_rots).Exp()
    return weighted_rots


def dict_temporal_key(timestamp: float = None, frame_idx: int = None):
    if timestamp is None:
        assert frame_idx is not None
        return f"f{frame_idx}"
    elif frame_idx is None:
        return f"t{timestamp}"
    else:
        return f"t{timestamp}_f{frame_idx}"