import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from .uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank
from threestudio.utils.typing import *
from threestudio.utils.misc import load_split_data
 
@dataclass
class SingleImageDataModuleConfig:
    # height and width should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 256
    width: Any = 256
    default_elevation_deg: float = 0.0
    default_azimuth_deg: float = 0.0
    default_camera_distance: float = 2.7
    default_fovy_deg: float = 60.0
    train_data_path: str = ""
    val_data_path: str = ""
    test_data_path: str = ""
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    rays_noise_scale: float = 2e-3
    batch_size: int = 1
    requires_depth: bool = False
    requires_normal: bool = False
    mesh_name: str = ""

    rays_d_normalize: bool = True


class SingleImageDataBase:
    def setup(self, cfg, split, image_paths, cam2world_dict, world2cam_dict, full_proj_transforms, camera_centers, intrinsics):
        self.split = split
        self.rank = get_rank()
        self.cfg: SingleImageDataModuleConfig = cfg
        self.image_paths = image_paths
        self.cam2world_dict = cam2world_dict
        self.intrinsics = intrinsics
        self.world2cam_dict = world2cam_dict
        self.full_proj_transforms_dict = full_proj_transforms
        self.camera_centers_dict = camera_centers
                     
        self.images = []
        self.masks = []
        self.camera_positions = []
        self.light_positions = []
        self.c2ws = []
        self.c2w4x4s = []
        self.elevations = []
        self.azimuths = []
        self.camera_distances = []
        self.fovys = []
        self.depths = []
        self.normals = []
        self.height = self.cfg.height
        self.width = self.cfg.width
        self.fovy = 2 * torch.atan(self.height / (2 * self.intrinsics[1, 1])) # In radians
        self.w2cs = []
        self.full_proj_transforms = []
        self.camera_centers = []

        self.load_images()
        self.num_images = len(self.images)
        self.set_rays()

    def get_ray_directions_K(self, H, W, K):
        """
        Get ray directions for all pixels in camera coordinate, using intrinsics.
        """
        i_coords = torch.arange(H, dtype=torch.float32, device=self.rank).view(-1, 1).repeat(1, W)
        j_coords = torch.arange(W, dtype=torch.float32, device=self.rank).view(1, -1).repeat(H, 1)
        ones = torch.ones_like(i_coords, device=self.rank)
        pixel_coords = torch.stack([j_coords, i_coords, ones], dim=-1)  # Shape (H, W, 3)

        # Invert the intrinsics
        K_inv = torch.inverse(K)

        # Reshape to (H*W, 3)
        pixel_coords_flat = pixel_coords.view(-1, 3)

        # Compute directions in camera coordinates
        directions_flat = pixel_coords_flat @ K_inv.T  # Shape (H*W, 3)

        # Reshape back to (H, W, 3)
        directions = directions_flat.view(H, W, 3)

        return directions

    def set_rays(self):
        # get directions by dividing directions_unit_focal by focal length
        self.rays_o_list = []
        self.rays_d_list = []
        self.mvp_mtx_list = []
        
        directions = self.get_ray_directions_K(self.height, self.width, self.intrinsics).to(self.rank)
        
        for i in range(self.num_images):        
            c2w = self.c2ws[i] # [3, 4]

            # rays_d: apply rotation (ignore translation)
            rays_d = directions @ c2w[:, :3].T  # Shape (H, W, 3)
            rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            
            # rays_o: camera position (translation component)
            rays_o = c2w[:, 3].unsqueeze(0).unsqueeze(0).expand(self.height, self.width, 3)
            
            rays_o = rays_o.unsqueeze(0)
            rays_d = rays_d.unsqueeze(0)
            
            self.rays_o_list.append(rays_o)
            self.rays_d_list.append(rays_d)


    def load_images(self):
        for image_path in self.image_paths:
            # load image
            assert os.path.exists(
                image_path
            ), f"Could not find image {image_path}!"
            rgba = cv2.cvtColor(
                cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
            )
            rgba = (
                cv2.resize(
                    rgba, (self.width, self.height), interpolation=cv2.INTER_AREA
                ).astype(np.float32)
                / 255.0
            )
            rgb = rgba[..., :3]
            rgb: Float[Tensor, "1 H W 3"] = (
                torch.from_numpy(rgb).unsqueeze(0).contiguous().to(self.rank)
            )

            # mask_path = image_path.replace("rgb", "masks")
            # assert os.path.exists(mask_path)
            # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        
            # mask: Float[Tensor, "1 H W 1"] = (
            #     torch.from_numpy((mask.astype(np.float32) / 255.0) > 0)[:, :, None]
            #     .unsqueeze(0)
            #     .to(self.rank)
            # )
            mask: Float[Tensor, "1 H W 1"] = (
                torch.from_numpy(rgba[..., 3:] > 0.5).unsqueeze(0).to(self.rank)
            )

            
            self.images.append(rgb)
            self.masks.append(mask)
            
            # Load camera parameters for this image
            cam2world, world2cam, full_proj_transform, cam_center = self.load_cam_for_image(image_path)
            self.c2w4x4s.append(cam2world)
            c2w = cam2world[:3, :] # [3, 4]
            self.c2ws.append(c2w)
            self.w2cs.append(world2cam)
            self.full_proj_transforms.append(full_proj_transform)
            self.camera_centers.append(cam_center)
            
            camera_position: Float[Tensor, "1 3"] = c2w[:, 3] # [1, 3]
            self.camera_positions.append(camera_position)
            self.light_positions.append(camera_position)
            
            self.camera_positions.append(camera_position)
            self.light_positions.append(camera_position)
            
            # load depth
            if self.cfg.requires_depth:
                depth_path = image_path.replace("rgb", "depths")
                assert os.path.exists(depth_path)
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                depth = cv2.resize(
                    depth, (self.width, self.height), interpolation=cv2.INTER_AREA
                )
                depth: Float[Tensor, "1 H W 1"] = (
                    torch.from_numpy(depth.astype(np.float32) / 255.0)
                    .unsqueeze(0)
                    .to(self.rank)
                )
                self.depths.append(depth)
            else:
                self.depths.append(None)

            # load normal
            if self.cfg.requires_normal:
                normal_path = image_path.replace("rgb", "normals")
                assert os.path.exists(normal_path)
                normal = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
                normal = cv2.resize(
                    normal, (self.width, self.height), interpolation=cv2.INTER_AREA
                )
                normal = normal[..., :3]
                normal: Float[Tensor, "1 H W 3"] = (
                    torch.from_numpy(normal.astype(np.float32) / 255.0)
                    .unsqueeze(0)
                    .to(self.rank)
                )
                self.normals.append(normal)
            else:
                self.normals.append(None)

                
    def load_cam_for_image(self, image_path):
        image_name = os.path.basename(image_path)
        cam2world = torch.FloatTensor(self.cam2world_dict[image_name]).to(self.rank)
        world2cam = torch.FloatTensor(self.world2cam_dict[image_name]).to(self.rank)
        full_proj_transform = torch.FloatTensor(self.full_proj_transforms_dict[image_name]).to(self.rank)
        camera_centers = torch.FloatTensor(self.camera_centers_dict[image_name]).to(self.rank)
        
        return cam2world, world2cam, full_proj_transform, camera_centers
         
    def get_all_images(self):
        return self.rgb

    def update_step_(self, epoch: int, global_step: int, on_load_weights: bool = False):
        pass # No updates since we are using static images
    
    def get_batch(self, index):
        image_name = self.image_paths[index]
        rgb = self.images[index]
        mask = self.masks[index]
        depth = self.depths[index] 
        normal = self.normals[index]
        camera_position = self.camera_positions[index]
        light_position = self.light_positions[index]
        c2w4x4 = self.c2w4x4s[index]
        rays_o = self.rays_o_list[index]
        rays_d = self.rays_d_list[index]
        w2c = self.w2cs[index]
        full_proj_transform = self.full_proj_transforms[index]
        camera_center = self.camera_centers[index]
        
        # Make sure each tensor has the 1 batch dimension
        c2w4x4 = c2w4x4.unsqueeze(0)
        camera_position = camera_position.unsqueeze(0)
        fovy = self.fovy.unsqueeze(0)
        w2c = w2c.unsqueeze(0)
        full_proj_transform = full_proj_transform.unsqueeze(0)
        camera_center = camera_center.unsqueeze(0)
        
        batch = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "camera_positions": camera_position,
            "rgb": rgb,
            "ref_depth": depth,
            "ref_normal": normal,
            "mask": mask,
            "height": self.height,
            "width": self.width,
            "c2w": c2w4x4,
            "fovy": fovy,
            "image_name": image_name,
            "w2c": w2c,
            "full_proj_transform": full_proj_transform,
            "camera_center": camera_center
        }
        return batch
        
class SingleImageIterableDataset(IterableDataset, SingleImageDataBase, Updateable):
    def __init__(self, cfg: Any, split: str, image_paths, cam2world_dict,  world2cam_dict, full_proj_transforms, camera_centers, intrinsics) -> None:
        super().__init__()
        self.setup(cfg, split, image_paths, cam2world_dict, world2cam_dict, full_proj_transforms, camera_centers, intrinsics)

    def collate(self, batch) -> Dict[str, Any]:
        return batch[0]

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.update_step_(epoch, global_step, on_load_weights)

    def __iter__(self):
        while True:
            index = np.random.randint(0, self.num_images)
            yield self.get_batch(index)


class SingleImageDataset(Dataset, SingleImageDataBase):
    def __init__(self, cfg: Any, split: str, image_paths, cam2world_dict, world2cam_dict, full_proj_transforms, camera_centers, intrinsics) -> None:
        super().__init__()
        self.setup(cfg, split, image_paths, cam2world_dict, world2cam_dict, full_proj_transforms, camera_centers, intrinsics)

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        return self.get_batch(index)
        # if index == 0:
        #     return {
        #         'rays_o': self.rays_o[0],
        #         'rays_d': self.rays_d[0],
        #         'mvp_mtx': self.mvp_mtx[0],
        #         'camera_positions': self.camera_position[0],
        #         'light_positions': self.light_position[0],
        #         'elevation': self.elevation_deg[0],
        #         'azimuth': self.azimuth_deg[0],
        #         'camera_distances': self.camera_distance[0],
        #         'rgb': self.rgb[0],
        #         'depth': self.depth[0],
        #         'mask': self.mask[0]
        #     }
        # else:
        #     return self.random_pose_generator[index - 1]
    def collate(self, batch) -> Dict[str, Any]:
        return batch[0]


@register("single-image-gs-datamodule")
class SingleImageDataModule(pl.LightningDataModule):
    cfg: SingleImageDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(SingleImageDataModuleConfig, cfg)
        self.rank = get_rank()

    def setup(self, stage=None) -> None:
        # Load training data:
        self.train_data = load_split_data(self.cfg, 'train', self.rank)
        self.val_data = load_split_data(self.cfg, 'val', self.rank)
        self.test_data = load_split_data(self.cfg, 'test', self.rank)
        
        if stage in [None, "fit"]:
            self.train_dataset = SingleImageIterableDataset(self.cfg, "train", 
                                                            image_paths=self.train_data['image_paths'], 
                                                            cam2world_dict=self.train_data['cam_2_world_poses'],
                                                            world2cam_dict=self.train_data['world_2_cam_poses'],
                                                            full_proj_transforms=self.train_data['full_proj_transforms'],
                                                            camera_centers=self.train_data['camera_centers'], 
                                                            intrinsics=self.train_data['intrinsics'])
        if stage in [None, "fit", "validate"]:
            self.val_dataset = SingleImageDataset(
                self.cfg, "val", 
                image_paths=self.val_data['image_paths'], 
                cam2world_dict=self.val_data['cam_2_world_poses'],
                world2cam_dict=self.val_data['world_2_cam_poses'],
                full_proj_transforms=self.val_data['full_proj_transforms'],
                camera_centers=self.val_data['camera_centers'],  
                intrinsics=self.val_data['intrinsics']
            )
        if stage in [None, "test", "predict"]:
            self.test_dataset = SingleImageDataset(
                self.cfg, "test", 
                image_paths=self.test_data['image_paths'], 
                cam2world_dict=self.test_data['cam_2_world_poses'],
                world2cam_dict=self.test_data['world_2_cam_poses'],
                full_proj_transforms=self.test_data['full_proj_transforms'],
                camera_centers=self.test_data['camera_centers'],  
                intrinsics=self.test_data['intrinsics']
            )

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=0,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=1,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset,
            batch_size=1,
            collate_fn=self.val_dataset.collate,
        )

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset,
            batch_size=1,
            collate_fn=self.test_dataset.collate,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset,
            batch_size=1,
            collate_fn=self.test_dataset.collate,
        )
