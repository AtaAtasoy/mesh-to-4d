import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import threestudio
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from threestudio.systems.utils import parse_optimizer
from threestudio.utils.loss import tv_loss
from threestudio.utils.typing import *
from threestudio.utils.misc import C
from torchmetrics import PearsonCorrCoef

from pytorch3d.structures import Meshes
from pytorch3d.io import save_obj
from pytorch3d.loss import mesh_normal_consistency, mesh_laplacian_smoothing, mesh_edge_loss

from ..utils.arap_utils import ARAPCoach
from .base import BaseSuGaRSystem
from torchmetrics import PeakSignalNoiseRatio
from torchvision.utils import save_image
from ..utils.loss_utils import weighted_mse_loss, temporal_smoothness_loss, cosine_avg, robust_mask_loss, to_colormap_image, robust_rgb_loss, save_rgb_weights_as_png
import clip
from ..utils.clip_spatial import CLIPVisualEncoder
from ..utils.resize_right import resize, cubic


@threestudio.register("sugar-4dgen-system")
class SuGaR4DGen(BaseSuGaRSystem):
    @dataclass
    class Config(BaseSuGaRSystem.Config):
        stage: str = "static"  # ["static", "motion", "refine"]

        guidance_zero123_type: str = "stale-zero123-guidance"
        guidance_zero123: dict = field(default_factory=dict)

        prompt_processor_3d_type: Optional[str] = ""
        prompt_processor_3d: dict = field(default_factory=dict)
        guidance_3d_type: Optional[str] = "image-dream-guidance"
        guidance_3d: dict = field(default_factory=dict)

        prompt_processor_vid_type: Optional[str] = ""
        prompt_processor_vid: dict = field(default_factory=dict)
        guidance_vid_type: Optional[str] = ""
        guidance_vid: dict = field(default_factory=dict)

        freq: dict = field(default_factory=dict)
        refinement: bool = False
        ambient_ratio_min: float = 0.5
        back_ground_color: Tuple[float, float, float] = (0, 0, 0)

        # Intermediate frames
        num_inter_frames: int = 10
        length_inter_frames: float = 0.2
        
        dynamic_stage: bool = True
        render_mesh: bool = False
        render_gaussians: bool = True
        
        # CLIP
        text_prompt: str = "standing up"
        base_text_prompt: str = "walking"
        clip_model: str = "ViT-B/32"
        consistency_clip_model: str = "ViT-B/32"
        consistency_vit_stride: int = 8
        consistency_vit_layer: int = 11

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.automatic_optimization = True
        # self.mask_logits = nn.Parameter(torch.zeros(1, 320, 320, 1)) 
        # self.ref_rgb_logits = nn.Parameter(torch.zeros(81, 512, 512, 3))
        self.stage = self.cfg.stage
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        
    def configure_optimizers(self):
        optim = self.geometry.optimizer
        if hasattr(self, "merged_optimizer"):
            return [optim]
        if hasattr(self.cfg.optimizer, "name"):
            net_optim = parse_optimizer(self.cfg.optimizer, self)
            optim = self.geometry.merge_optimizer(net_optim)
            self.merged_optimizer = True
        else:
            self.merged_optimizer = False
        return [optim]

    def forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        self.geometry.update_learning_rate(self.global_step)
        if self.cfg.render_mesh:
            batch["timed_surface_mesh"] = self.geometry.get_timed_surface_mesh(
                    timestamp=batch["timestamp"], frame_idx=batch["frame_indices"]
                )
        if self.cfg.render_gaussians:
            batch["render_gaussians"] = True
        if self.cfg.loss.lambda_mesh_rgb > 0:
            batch["render_mesh_rgb"] = True
        if self.cfg.loss.lambda_mesh_mask > 0:
            batch["render_mesh_mask"] = True

        outputs = self.renderer.batch_forward(batch)
        return outputs

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # no prompt processor
        print(f"Fit start. Stage: {self.stage}")
        if self.cfg.loss.lambda_sds_zero123 > 0 or self.cfg.loss.lambda_sds_mesh_zero123 > 0:
            self.guidance_zero123 = threestudio.find(self.cfg.guidance_zero123_type)(self.cfg.guidance_zero123)
        # Maybe use ImageDream
        self.enable_imagedream = self.cfg.guidance_3d_type is not None and C(self.cfg.loss.lambda_sds_3d, 0, 0) > 0
        if self.enable_imagedream:
            self.guidance_3d = threestudio.find(self.cfg.guidance_3d_type)(self.cfg.guidance_3d)
        else:
            self.guidance_3d = None

        # Maybe use video diffusion models
        self.enable_vid = (
            self.stage == "refine"
            and self.cfg.guidance_vid_type is not None
            and C(self.cfg.loss.lambda_sds_video, 0, 0) > 0
        )
        if self.enable_vid:
            threestudio.info(f"Using video diffusion guidance: {self.cfg.guidance_vid_type}")
            self.guidance_vid = threestudio.find(self.cfg.guidance_vid_type)(self.cfg.guidance_vid)
            self.prompt_processor_video = threestudio.find(self.cfg.prompt_processor_vid_type)(self.cfg.prompt_processor_vid)
            self.prompt_utils_video = self.prompt_processor_video()
        else:
            self.guidance_vid = None

        # visualize all training images
        all_images = self.trainer.datamodule.train_dataloader().dataset.get_all_images()
        self.save_image_grid(
            "all_training_images.png",
            [
                {"type": "rgb", "img": image, "kwargs": {"data_format": "HWC"}}
                for image in all_images
            ],
            name="on_fit_start",
            step=self.true_global_step,
        )

        self.pearson = PearsonCorrCoef().to(self.device)

        # ARAP
        self.arap_coach = None

        # debug
        self.geometry.save_path = self.get_save_dir()
        
        self.timestamps = torch.as_tensor(
            np.linspace(0, 1,  self.geometry.num_frames+2, endpoint=True), dtype=torch.float32, device=self.device
        )[1:-1]
        self.frame_indices = torch.arange(self.geometry.num_frames, dtype=torch.long, device=self.device)
        
        # CLIP
        if self.cfg.loss.lambda_clip_loss > 0:
            self.clip_model, _ = clip.load(self.cfg.clip_model, device=self.device)
            self.fe = CLIPVisualEncoder(self.cfg.consistency_clip_model, self.cfg.consistency_vit_stride, self.device)

            self.clip_mean = torch.tensor([0.48154660, 0.45782750, 0.40821073], device=self.device)
            self.clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device)
            
            threestudio.info(f'Base text prompt is {self.cfg.base_text_prompt}')
            threestudio.info(f'Target text prompt is {self.cfg.text_prompt}')
            with torch.no_grad():
                self.text_embeds = clip.tokenize(self.cfg.text_prompt).to(self.device)
                self.base_text_embeds = clip.tokenize(self.cfg.base_text_prompt).to(self.device)
                self.text_embeds = self.clip_model.encode_text(self.text_embeds).detach()
                self.target_text_embeds = self.text_embeds.clone() / self.text_embeds.norm(dim=1, keepdim=True)

                self.delta_text_embeds = self.text_embeds - self.clip_model.encode_text(self.base_text_embeds)
                self.delta_text_embeds = self.delta_text_embeds / self.delta_text_embeds.norm(dim=1, keepdim=True)
                
    def training_substep(self, batch, batch_idx, guidance: str):
        """
        Args:
            guidance: one of "ref" (reference image supervision), "zero123"
        """
        torch.cuda.empty_cache()
        if guidance == "ref":
            ambient_ratio = 1.0
            shading = "diffuse"
            batch["shading"] = shading
        elif guidance == "zero123":
            # default store the reference view camera config, switch to random camera for zero123 guidance
            batch = batch["random_camera"]
            ambient_ratio = (
                self.cfg.ambient_ratio_min
                + (1 - self.cfg.ambient_ratio_min) * random.random()
            )    
        elif guidance == "clip":
            batch = batch["video_random_camera"]
            ambient_ratio = 1.0
        elif guidance == "multiview":
            batch = batch["multiview_camera"]
            ambient_ratio = 1.0
                                                        
        batch["ambient_ratio"] = ambient_ratio
        out = self(batch)

        loss_prefix = f"loss_{guidance}_"

        loss_terms = {}

        def set_loss(name, value):
            loss_terms[f"{loss_prefix}{name}"] = value

        guidance_eval = (
            self.cfg.freq.guidance_eval > 0
            and self.true_global_step % self.cfg.freq.guidance_eval == 0
        )

        if guidance == "multiview":
            out_rgb = out["comp_rgb"]
            out_mask = out["comp_mask"]
            
            gt_rgb = batch["rgb"]
            gt_mask = batch["mask"]
            gt_rgb = gt_rgb * gt_mask.float()
            
            gt_rgb = gt_rgb * gt_mask.float()
            if self.C(self.cfg.loss.lambda_novel_rgb) > 0:
                set_loss("novel_rgb", F.mse_loss(gt_rgb, out_rgb))
            if self.C(self.cfg.loss.lambda_novel_mask) > 0:
                set_loss("novel_mask", F.mse_loss(gt_mask.float(), out_mask))
                # set_loss("novel_mask", robust_mask_loss(out_mask, gt_mask.float(), self.mask_logits, lambda_reg=0.1))

            
            if guidance_eval:
                imgs_data = [
                            {"type": "rgb", "img": img, "kwargs": {"data_format": "HWC"}}
                            for img in out_rgb
                        ] 
                imgs_data_gt = [
                                {"type": "rgb", "img": img, "kwargs": {"data_format": "HWC"}}
                                for img in gt_rgb
                            ]
                
                masks_data = [
                            {"type": "grayscale", "img": mask[..., 0], "kwargs": {"cmap": None, "data_range": (0, 1)}}
                            for mask in out_mask
                        ]
                
                masks_data_gt = [
                            {"type": "grayscale", "img": mask[..., 0], "kwargs": {"cmap": None, "data_range": (0, 1)}}
                            for mask in gt_mask
                        ]
                
                # weights = torch.sigmoid(self.mask_logits).detach().squeeze().cpu()
                # weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
                # image = to_colormap_image(tensor_2d=weights)
                # # self.log.add_image("mask_weights_colormap", transforms.ToTensor()(image), self.current_epoch)
                # image.save(os.path.join(self.get_save_dir(), f"mask_weights_colormap_{self.true_global_step}.png"))
               
                self.save_image_grid(f"fixed_camera_gaussian_{self.true_global_step}.png", imgs=imgs_data)
                self.save_image_grid(f"fixed_camera_gt_{self.true_global_step}.png", imgs=imgs_data_gt)
                self.save_image_grid(f"fixed_camera_mask_{self.true_global_step}.png", imgs=masks_data)
                self.save_image_grid(f"fixed_camera_gt_mask_{self.true_global_step}.png", imgs=masks_data_gt)

        elif guidance == "ref":
            gt_mask = batch["mask"]
            gt_rgb = batch["rgb"]         
            # color loss
            # gt_rgb = gt_rgb * gt_mask.float()
            if self.C(self.cfg.loss.lambda_rgb) > 0:
                if not self.C(self.cfg.loss.lambda_flow) > 0:
                    set_loss("rgb", F.mse_loss(gt_rgb, out["comp_rgb"]))
                    # set_loss("novel_rgb", robust_rgb_loss(out["comp_rgb"], gt_rgb, batch["frame_indices"], self.ref_rgb_logits, lambda_reg=0.1))
                    # if guidance_eval:
                    #     save_rgb_weights_as_png(self.ref_rgb_logits, save_dir=self.get_save_dir(), current_epoch=self.true_global_step)
                else:
                    # save reliability maps
                    # Need to be B, H, W, C and torch.float32
                    reliability_maps = batch["reliability_map"].detach()
                    if len(reliability_maps.shape) == 3:
                        reliability_maps = reliability_maps.unsqueeze(-1)
                        
                    if self.true_global_step % 200 == 0: 
                        save_image(reliability_maps.permute(0, 3, 1, 2), f"reliability_map_{self.true_global_step}.png")
                        
                    set_loss("rgb", weighted_mse_loss(gt_rgb, out["comp_rgb"], reliability_maps * gt_mask.float()))
                if guidance_eval:
                    comp_rgb = [
                        {"type": "rgb", "img": img, "kwargs": {"data_range": (0, 1)}}
                        for img in out["comp_rgb"]
                    ]
                    
                    gt_rgb = [
                        {"type": "rgb", "img": img, "kwargs": {"data_range": (0, 1)}}
                        for img in gt_rgb
                    ]
                    
                    self.save_image_grid(f"ref_rgb_{self.true_global_step}.png", imgs=comp_rgb)
                    self.save_image_grid(f"comp_rgb_{self.true_global_step}.png", imgs=comp_rgb)

            if self.C(self.cfg.loss.lambda_mask) > 0:
                # mask loss
                if guidance_eval:
                    comp_mask_data = [
                        {"type": "grayscale", "img": mask[..., 0], "kwargs": {"cmap": None, "data_range": (0, 1)}}
                        for mask in out["comp_mask"]
                    ]
                    self.save_image_grid(f"comp_mask_{self.true_global_step}.png", imgs=comp_mask_data)
                    
                    ref_mask_data = [
                        {"type": "grayscale", "img": mask[..., 0], "kwargs": {"cmap": None, "data_range": (0, 1)}}
                        for mask in gt_mask
                    ]
                    
                    self.save_image_grid(f"ref_mask_{self.true_global_step}.png", imgs=ref_mask_data)
                    
                    
                set_loss("mask", F.mse_loss(gt_mask.float(), out["comp_mask"]))
                
            if self.C(self.cfg.loss.lambda_mesh_mask) > 0:
                if guidance_eval:
                    mesh_mask_data = [
                        {"type": "grayscale", "img": mask[..., 0], "kwargs": {"cmap": None, "data_range": (0, 1)}}
                        for mask in out["mesh_comp_mask"]
                    ]
                    self.save_image_grid(f"ref_mesh_mask_{self.true_global_step}.png", imgs=mesh_mask_data)
                    
                    gt_mesh_mask_data = [
                        {"type": "grayscale", "img": mask[..., 0], "kwargs": {"cmap": None, "data_range": (0, 1)}}
                        for mask in gt_mask
                    ]
                    self.save_image_grid(f"ref_gt_mesh_mask_{self.true_global_step}.png", imgs=gt_mesh_mask_data)
                    
                set_loss("mesh_mask", F.mse_loss(gt_mask.float(), out["mesh_comp_mask"]))
            
            if self.C(self.cfg.loss.lambda_mesh_rgb) > 0:
                if not self.C(self.cfg.loss.lambda_flow) > 0: 
                    set_loss("mesh_rgb", F.mse_loss(gt_rgb, out["mesh_comp_rgb"]))
                else:
                    reliability_maps = batch["reliability_map"].detach()
                    if len(reliability_maps.shape) == 3:
                        reliability_maps = reliability_maps.unsqueeze(-1)
                    if self.true_global_step % 200 == 0: 
                        save_image(reliability_maps.permute(0, 3, 1, 2), f"reliability_map_{self.true_global_step}.png")
                    set_loss("mesh_rgb", weighted_mse_loss(gt_rgb, out["mesh_comp_rgb"], reliability_maps * gt_mask.float()))

            # depth loss
            if self.C(self.cfg.loss.lambda_depth) > 0:
                # gt_mask : 16, 480, 720, 1 : bool
                # batch["ref_depth"]: 16, 480, 720, 1
                # batch["comp_depth"]: 16, 480, 720, 1
                
                masked_gt_depth = batch["ref_depth"] 
                masked_pred_depth = out["comp_depth"] 
                
                valid_gt_depth = batch["ref_depth"][gt_mask].unsqueeze(1)
                valid_pred_depth = out["comp_depth"][gt_mask].unsqueeze(1)

                with torch.no_grad():
                    A = torch.cat(
                        [valid_pred_depth, torch.ones_like(valid_pred_depth)], dim=-1
                    )  # [B, 2]
                    X = torch.linalg.lstsq(A, valid_gt_depth).solution  # [2, 1]
                    valid_pred_depth = A @ X  # [B, 1]
                set_loss("depth", F.mse_loss(valid_gt_depth, valid_pred_depth))

                if guidance_eval:
                    aligned_pred_depth_img = torch.zeros_like(batch["ref_depth"])
                    aligned_pred_depth_img[gt_mask] = valid_pred_depth.squeeze(1)
                                        
                    pred_depth_data = [
                        {"type": "grayscale", "img": img[:, :, 0], "kwargs": {"cmap": None}}
                        for img in masked_pred_depth
                    ]
                    
                    gt_depth_data = [
                        {"type": "grayscale", "img": img[:, :, 0], "kwargs": {"cmap": None}}
                        for img in masked_gt_depth
                    ] 
                    
                    self.save_image_grid(
                        filename=f"pred_depth_{self.true_global_step}.png",
                        imgs=pred_depth_data,
                    )
                    
                    self.save_image_grid(
                        filename=f"gt_depth_{self.true_global_step}.png",
                        imgs=gt_depth_data,
                    )
                                   
            # relative depth loss
            if self.C(self.cfg.loss.lambda_depth_rel) > 0:
                valid_gt_depth = batch["ref_depth"][gt_mask]  # [B,]
                if "comp_depth" in out:
                    valid_pred_depth = out["comp_depth"][gt_mask]
                else:
                    valid_pred_depth = out["mesh_comp_depth"][gt_mask]
                                    
                # valid_pred_depth = out["comp_depth"][gt_mask]  # [B,]
                set_loss(
                    "depth_rel", 1 - self.pearson(valid_pred_depth, valid_gt_depth)
                )

            # normal loss
            if self.C(self.cfg.loss.lambda_normal) > 0:
                if guidance_eval:
                    
                    # Load and mask normals by broadcasting the mask over the 3 channels
                    comp_normal_masked = out["comp_normal"]
                    gt_normal_masked = batch["ref_normal"] * gt_mask.float()
                    
                    gt_normal_data = [
                        {"type": "rgb", "img": normal, "kwargs": {"data_range": (0, 1)}}
                        for normal in gt_normal_masked
                    ]
                    
                    comp_normal_data = [
                        {"type": "rgb", "img": normal, "kwargs": {"data_range": (0, 1)}}
                        for normal in comp_normal_masked
                    ]
                    
                    self.save_image_grid(
                        filename=f"pred_normal_{self.true_global_step}.png",
                        imgs=comp_normal_data,
                    )
                    
                    self.save_image_grid(
                        filename=f"gt_normal_{self.true_global_step}.png",
                        imgs=gt_normal_data,
                    )
                    
                valid_gt_normal = (
                    1 - 2 * batch["ref_normal"][gt_mask.squeeze(-1)]
                )  # [B, 3]
                valid_pred_normal = (
                    2 * out["comp_normal"][gt_mask.squeeze(-1)] - 1
                )  # [B, 3]
                set_loss(
                    "normal",
                    1 - F.cosine_similarity(valid_pred_normal, valid_gt_normal).mean(),
                )

            if (
                self.C(self.cfg.loss.lambda_normal_consistency) > 0
                or self.C(self.cfg.loss.lambda_laplacian_smoothing) > 0
            ):
                surface_meshes = self.geometry.get_timed_surface_mesh(
                    timestamp=batch["timestamp"], frame_idx=batch["frame_indices"]
                )
                if self.C(self.cfg.loss.lambda_normal_consistency) > 0:
                    set_loss(
                        "normal_consistency",
                        mesh_normal_consistency(surface_meshes)
                    )
                if self.C(self.cfg.loss.lambda_laplacian_smoothing) > 0:
                    set_loss(
                        "laplacian_smoothing",
                        mesh_laplacian_smoothing(surface_meshes, "uniform")
                    )
                if self.C(self.cfg.loss.lambda_mesh_edge) > 0:
                    set_loss(
                        "mesh_edge",
                        mesh_edge_loss(surface_meshes)
                    )
        elif guidance == "zero123":        
            # zero123
            if self.C(self.cfg.loss.lambda_sds_zero123) > 0:
                guidance_out, guidance_eval_out = self.guidance_zero123(
                    out["comp_rgb"],
                    **batch,
                    rgb_as_latents=False,
                    guidance_eval=guidance_eval,
                )
                set_loss("sds_zero123", guidance_out["loss_sds"])
                # guidance_eval = guidance_out["guidance_eval"]
                if guidance_eval:
                    self.save_guidance_eval_images(guidance_eval_out, "zero123", self.true_global_step)
            
            if self.C(self.cfg.loss.lambda_sds_mesh_zero123) > 0:
                guidance_mesh_out, guidance_eval_out = self.guidance_zero123(
                    out["mesh_comp_rgb"],
                    **batch,
                    rgb_as_latents=False,
                    guidance_eval=guidance_eval,
                )
                set_loss("sds_mesh_zero123", guidance_mesh_out["loss_sds"])   
                     
                if guidance_eval:
                    self.save_guidance_eval_images(guidance_eval_out, "zero123", self.true_global_step)
            
                    if self.C(self.cfg.loss.lambda_sds_zero123) > 0:
                        zero123_cond_imgs = out["comp_rgb"]
                        imgs_data = [
                            {"type": "rgb", "img": img, "kwargs": {"data_format": "HWC"}}
                            for img in zero123_cond_imgs
                        ]                    
                        self.save_image_grid(f"rand_camera_gaussian_{self.true_global_step}.png", imgs=imgs_data)
                    
                    if self.cfg.loss.lambda_mesh_rgb > 0:
                        threestudio.info(f"Saving mesh images at {self.true_global_step}")
                        mesh_rgbs = out["mesh_comp_rgb"]
                        imgs_data = [
                            {"type": "rgb", "img": img, "kwargs": {"data_format": "HWC"}}
                            for img in mesh_rgbs
                        ]
                        self.save_image_grid(f"rand_camera_mesh_{self.true_global_step}.png", imgs=imgs_data)
                    if self.cfg.loss.lambda_mesh_mask > 0:
                        threestudio.info(f"Saving mesh masks at {self.true_global_step}")
                        mesh_masks = out["mesh_comp_mask"]
                        imgs_data = [
                            {"type": "grayscale", "img": mask[..., 0],
                            "kwargs": {"cmap": None, "data_range": (0, 1)}}
                            for mask in mesh_masks
                        ]
                        self.save_image_grid(f"rand_camera_mesh_mask_{self.true_global_step}.png", imgs=imgs_data)
        elif guidance == "video_diffusion" and self.stage == "refine":
            guidance_inp = out["comp_rgb"]
            
            vid_guidance = self.guidance_vid
            prompt_utils = self.prompt_utils_video

            # random camera
            guidance_out, guidance_eval_out = vid_guidance(
                guidance_inp, prompt_utils, **batch, rgb_as_latents=False, guidance_eval=guidance_eval
            )
            if guidance_eval:
                self.save_guidance_eval_images(guidance_eval_out, "video_diffusion", self.true_global_step)
                imgs_data = [
                    {"type": "rgb", "img": img, "kwargs": {"data_format": "HWC"}}
                    for img in guidance_inp
                ]                    
                self.save_image_grid(f"video_diffusion_gaussian_{self.true_global_step}.png", imgs=imgs_data)

            loss = guidance_out['loss_sds_video']
            set_loss("sds_video", loss)
     
        elif guidance == "clip":
            clip_renders  = out["comp_rgb"].permute(0, 3, 1, 2) # BxHxWxC
            clip_renders = resize(clip_renders, out_shape=(224, 224), interp_method=cubic)            
            if self.true_global_step % 200 == 0:
                clip_imgs_data = [
                            {"type": "rgb", "img": img, "kwargs": {"data_format": "CHW"}}
                            for img in clip_renders
                        ]
                self.save_image_grid(f"clip_renders{self.true_global_step}.png", imgs=clip_imgs_data)

            # CLIP similarity bas
            normalized_clip_render = (clip_renders - self.clip_mean[None, :, None, None]) / self.clip_std[None, :, None, None]
            image_embeds = self.clip_model.encode_image(normalized_clip_render)

            orig_image_embeds = image_embeds.clone() / image_embeds.norm(dim=1, keepdim=True)

            clip_loss = cosine_avg(orig_image_embeds, self.target_text_embeds)
            set_loss("clip", clip_loss)

        self._apply_regularization_losses(out, guidance, loss_terms, batch.get("timestamp"), batch.get("frame_indices"), guidance_eval=guidance_eval)

        if guidance == "ref" and self.C(self.cfg.loss.lambda_arap_reg_key_frame) > 0 and self.arap_coach is not None:
            set_loss(
                "arap_reg_key_frame",
                self._compute_arap_energy(batch.get("timestamp"), batch.get("frame_indices"))
            )

        loss = 0.0
        for name, value in loss_terms.items():
            self.log(f"train/{name}", value)
            if name.startswith(loss_prefix):
                loss_weighted = value * self.C(
                    self.cfg.loss[name.replace(loss_prefix, "lambda_")]
                )
                self.log(f"train/{name}_w", loss_weighted)
                loss += loss_weighted

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        self.log(f"train/loss_{guidance}", loss)

        out.update({"loss": loss})
        return out

    def training_substep_inter_frames(self, batch, batch_idx):
        loss_terms = {}
        loss_prefix = "loss_interf_"

        def set_loss(name, value):
            loss_terms[f"{loss_prefix}{name}"] = value
            
        # Densely sample frames in a smaller time range
        rand_range_start = np.random.rand() * (1 - self.cfg.length_inter_frames)
        rand_timestamps = torch.as_tensor(
            np.linspace(
                rand_range_start,
                rand_range_start + self.cfg.length_inter_frames,
                self.cfg.num_inter_frames,
                endpoint=True
            ),
            dtype=torch.float32,
            device=self.device
        )

        # ARAP regularization
        if self.C(self.cfg.loss.lambda_arap_reg_inter_frame) > 0 and self.arap_coach is not None:
            set_loss("arap_reg_inter_frame", self._compute_arap_energy(rand_timestamps))

        loss = 0.0
        for name, value in loss_terms.items():
            self.log(f"train/{name}", value)
            if name.startswith(loss_prefix):
                loss_weighted = value * self.C(
                    self.cfg.loss[name.replace(loss_prefix, "lambda_")]
                )
                self.log(f"train/{name}_w", loss_weighted)
                loss += loss_weighted

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        self.log(f"train/loss_interf", loss)

        return loss

    def _compute_arap_energy(self, tgt_timestamp=None, tgt_frame_idx=None):
        vert_timed_xyz = self.geometry.get_timed_vertex_xyz(tgt_timestamp, tgt_frame_idx)
        vert_timed_rot = self.geometry.get_timed_vertex_rotation(
            tgt_timestamp, tgt_frame_idx, return_matrix=True)

        loss_arap = 0.
        for i in range(vert_timed_xyz.shape[0]):
            loss_arap += self.arap_coach.compute_arap_energy(
                xyz_prime=vert_timed_xyz[i], vert_rotations=vert_timed_rot[i]
            )
        return loss_arap

    def _apply_regularization_losses(self, out: Dict[str, Any], guidance: str, loss_terms: Dict[str, Any], tgt_timestamp=None, tgt_frame_idx=None, guidance_eval=False):
        def set_loss(name, value):
            loss_terms[f"loss_{guidance}_{name}"] = value

        # Normal smoothness regularization
        if (
            out.__contains__("comp_normal")
            and self.C(self.cfg.loss.lambda_normal_smooth) > 0
        ):
            normal = out["comp_normal"]
            set_loss(
                "normal_smooth",
                (normal[:, 1:, :, :] - normal[:, :-1, :, :]).square().mean()
                + (normal[:, :, 1:, :] - normal[:, :, :-1, :]).square().mean(),
            )

        # Temporal smoothness regularization
        if self.C(self.cfg.loss.lambda_temporal_smoothness) > 0:
            number_of_frames = self.geometry.num_frames
            timestamps = torch.as_tensor(
                np.linspace(0, 1, number_of_frames+2, endpoint=True), dtype=torch.float32, device=self.device
            )[1:-1]
            frame_indices = torch.arange(number_of_frames, dtype=torch.long, device=self.device)
            
            all_vertices = self.geometry.get_timed_vertex_xyz(timestamps, frame_indices)    
            
            temporal_smoothness_loss_val = temporal_smoothness_loss(all_vertices)      
            set_loss("temporal_smoothness", temporal_smoothness_loss_val)

        # Total variation regularization for RGB
        if self.cfg.loss["lambda_rgb_tv"] > 0.0:
            loss_rgb_tv = tv_loss(out["comp_rgb"].permute(0, 3, 1, 2))
            set_loss("rgb_tv", loss_rgb_tv)

        # Total variation regularization for depth
        if (
            out.__contains__("comp_depth")
            and self.cfg.loss["lambda_depth_tv"] > 0.0
        ):
            loss_depth_tv = tv_loss(out["comp_depth"].permute(0, 3, 1, 2))
            set_loss("depth_tv", loss_depth_tv)

        # Total variation regularization for normals
        if (
            out.__contains__("comp_normal")
            and self.cfg.loss["lambda_normal_tv"] > 0.0
        ):
            loss_normal_tv = tv_loss(out["comp_normal"].permute(0, 3, 1, 2))
            set_loss("normal_tv", loss_normal_tv)

        # Normal-depth consistency regularization
        if (
            out.__contains__("comp_normal_from_dist")
            and out.__contains__("comp_normal")
            and self.C(self.cfg.loss.lambda_normal_depth_consistency) > 0
        ):
            raw_normal = out["comp_normal"] * 2 - 1
            raw_normal_from_dist = out["comp_normal_from_dist"] * 2 - 1
            loss_normal_depth_consistency = (1 - raw_normal.unsqueeze(-2) @ raw_normal_from_dist.unsqueeze(-1)).mean()
            set_loss("normal_depth_consistency", loss_normal_depth_consistency)

        # Reference XYZ regularization
        if self.stage != "static" and guidance == "ref" and self.C(self.cfg.loss.lambda_ref_xyz) > 0:
            xyz_f0 = self.geometry.get_timed_vertex_xyz(torch.as_tensor(0, dtype=torch.float32, device=self.device))
            loss_ref_xyz = torch.abs(xyz_f0 - self.geometry.get_xyz_verts).mean()
            set_loss("ref_xyz", loss_ref_xyz)

        # Object centric regularization
        if self.C(self.cfg.loss.lambda_obj_centric) > 0:
            vert_timed_xyz = torch.stack(
                [value for value in self.geometry._deformed_vert_positions.values()],
                dim=0
            )
            loss_obj_centric = (
                torch.abs(vert_timed_xyz[..., 0].mean())
                + torch.abs(vert_timed_xyz[..., 1].mean())
            )
            set_loss("obj_centric", loss_obj_centric)
        
        if self.C(self.cfg.loss.lambda_wks_consistency) > 0:
            base_wks = self.geometry.base_mesh_wks
            timed_wks_vectors = self.geometry.get_timed_mesh_wks(tgt_timestamp, tgt_frame_idx)
            
            wks_loss = torch.abs(base_wks - timed_wks_vectors).mean()
            set_loss("wks_consistency", wks_loss)
            
        if self.C(self.cfg.loss.lambda_mesh_volume_loss_l1) > 0:
            base_mesh_volume = self.geometry.base_mesh_volume
            timed_mesh_volumes = self.geometry.get_timed_mesh_volume(tgt_timestamp, tgt_frame_idx)
            
            volume_loss_l1 = torch.abs(base_mesh_volume - timed_mesh_volumes).mean()
            set_loss("mesh_volume_loss_l1", volume_loss_l1)
        
        if self.C(self.cfg.loss.lambda_mesh_volume_loss_l2) > 0:
            base_mesh_volume = self.geometry.base_mesh_volume
            timed_mesh_volumes = self.geometry.get_timed_mesh_volume(tgt_timestamp, tgt_frame_idx)
            
            volume_loss_l2 = torch.square(base_mesh_volume - timed_mesh_volumes).mean()
            set_loss("mesh_volume_loss_l2", volume_loss_l2)
        
    def on_train_batch_start(self, batch, batch_idx, unused=0):
        super().on_train_batch_start(batch, batch_idx, unused)

        if self.global_step == self.cfg.freq.milestone_arap_reg and self.arap_coach is None:
            self.arap_coach = ARAPCoach(
                self.geometry.get_xyz_verts,
                self.geometry.get_faces.cpu().numpy(),
                self.device
            )
    def training_step(self, batch, batch_idx):
        # opt = self.optimizers()

        total_loss = 0.0
        
        # batch_orig = deepcopy(batch)
        
        self.log(
            "gauss_num",
            int(self.geometry.get_xyz.shape[0]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        if self.enable_vid and self.stage == "refine":
            out_video_diffusion = self.training_substep(
                batch, batch_idx, guidance="video_diffusion")
            total_loss += out_video_diffusion["loss"]
        
        if self.stage == "motion":
            if self.cfg.loss.lambda_sds_zero123 > 0 or self.cfg.loss.lambda_sds_mesh_zero123 > 0:
                out_zero123 = self.training_substep(
                    batch, batch_idx, guidance="zero123")
                total_loss += out_zero123["loss"]
            
            out_ref = self.training_substep(
                batch, batch_idx, guidance="ref")
            total_loss += out_ref["loss"]
            
            if self.cfg.loss.lambda_novel_rgb > 0 or self.cfg.loss.lambda_novel_mask > 0:
                out_multiview = self.training_substep(
                    batch, batch_idx, guidance="multiview")
                total_loss += out_multiview["loss"]
            
        if self.cfg.loss.lambda_clip_loss > 0:
            out_clip = self.training_substep(
                batch, batch_idx, guidance="clip")
            total_loss += out_clip["loss"]
        
        if self.cfg.freq.inter_frame_reg > 0 and self.global_step % self.cfg.freq.inter_frame_reg == 0:
            total_loss += self.training_substep_inter_frames(batch, batch_idx)

        self.log("train/loss", total_loss, prog_bar=True)
        
        # if not self.automatic_optimization:
        #     total_loss.backward()
        #     opt.step()
        #     opt.zero_grad(set_to_none=True)

        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        def save_out_to_image_grid(filename, out):
            self.save_image_grid(
                filename,
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["mesh_comp_rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "mesh_comp_rgb" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "comp_rgb" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal_from_dist"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal_from_dist" in out
                    else []
                ) 
                + (
                    [
                        {
                            "type": "grayscale",
                            "img": out["mesh_comp_mask"][0, ..., 0],
                            "kwargs": {"cmap": None, "data_range": (0, 1)},
                        },
                    ] 
                    if "mesh_comp_mask" in out
                    else []
                )
                ,
                # claforte: TODO: don't hardcode the frame numbers to record... read them from cfg instead.
                name=None,
                step=self.true_global_step,
            )

        timestamps = batch["timestamps"][0]
        frame_indices = batch["frame_indices"][0]
        video_length = batch["video_length"][0]
        azimuth = int(batch['azimuth'][0].item())
        for i in range(video_length):
            batch.update(
                {
                    "timestamp": timestamps[i:i+1],
                    "frame_indices": frame_indices[i:i+1],
                    "width": int(batch["width"]),
                    "height": int(batch["height"]),
                }
            )
            out = self(batch)
            save_out_to_image_grid(f"it{self.true_global_step}-val/vid-azi{azimuth}/{i}.png", out)


        filestem = f"it{self.true_global_step}-val/vid-azi{azimuth}"
        self.save_img_sequence(
            filestem,
            filestem,
            "(\d+)\.png",
            save_format="mp4",
            fps=24,
            step=self.true_global_step
        )


    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        def save_out_to_image_grid(filename, out):
            self.save_image_grid(
                filename,
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ] 
                    if "comp_rgb" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal_from_dist"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal_from_dist" in out
                    else []
                ) 
                + (
                    [
                        {
                            "type": "grayscale",
                            "img": out["mesh_comp_mask"][0, ..., 0],
                            "kwargs": {"cmap": None, "data_range": (0, 1)},
                        }
                    ] if "mesh_comp_mask" in out
                    else []
                )
                ,
                name=None,
                step=self.true_global_step,
            )

        timestamps = batch["timestamps"][0]
        frame_indices = batch["frame_indices"][0]
        video_length = batch["video_length"][0]
        azimuth = int(batch['azimuth'][0].item())
        for i in range(video_length):
            batch.update(
                {
                    "timestamp": timestamps[i:i+1],
                    "frame_indices": frame_indices[i:i+1],
                }
            )

            batch["width"] = int(batch["width"])
            batch["height"] = int(batch["height"])
            out = self(batch)
            save_out_to_image_grid(f"it{self.true_global_step}-test/vid-azi{azimuth}/{i}.png", out)


        filestem = f"it{self.true_global_step}-test/vid-azi{azimuth}"
        self.save_img_sequence(
            filestem,
            filestem,
            "(\d+)\.png",
            save_format="mp4",
            fps=8,
            step=self.true_global_step
        )

    def on_predict_epoch_end(self) -> None:
        self.texture_img = self.texture_img / self.texture_counter.clamp(min=1)
        
        video_length = self.geometry.num_frames
        timestamps = torch.as_tensor(
            np.linspace(0, 1, video_length+2, endpoint=True), dtype=torch.float32
        )[1:-1].to(self.device)

        textures_uv = self.geometry.torch3d_mesh.textures

        mesh_save_dir = os.path.join(self.get_save_dir(), f"extracted_textured_meshes")
        os.makedirs(mesh_save_dir, exist_ok=True)
        for i, t in enumerate(timestamps):

            timed_surface_mesh = self.geometry.get_timed_surface_mesh(timestamps[i:i+1])
            verts = timed_surface_mesh.verts_list()[0]
            faces = timed_surface_mesh.faces_list()[0]
            threestudio.info(f"timed_surface_mesh verts: {verts.shape}, faces: {faces.shape}")
            textured_mesh = Meshes(
                verts=[verts],
                faces=[faces],
                textures=textures_uv
            )
            
            mesh_save_path = os.path.join(
                mesh_save_dir, f"{i}.obj"
            )
            with torch.no_grad():
                save_obj(  
                    mesh_save_path,
                    verts=textured_mesh.verts_list()[0],
                    faces=textured_mesh.faces_list()[0],
                    verts_uvs=textured_mesh.textures.verts_uvs_list()[0],
                    faces_uvs=textured_mesh.textures.faces_uvs_list()[0],
                    texture_map=textured_mesh.textures.maps_padded()[0].clamp(0., 1.),
                )
            threestudio.info(f"Textured mesh saved to {mesh_save_path}")