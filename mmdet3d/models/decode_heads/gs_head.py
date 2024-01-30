'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-01-26 11:11:03
Email: haimingzhang@link.cuhk.edu.cn
Description: The 3D Gaussian Splatting rendering head.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import os.path as osp
from einops import rearrange, repeat
from mmdet3d.models.builder import HEADS

from mmdet3d.models.decode_heads.nerf_head import NeRFDecoderHead
from .common.gaussians import build_covariance
from .common.cuda_splatting import render_cuda, render_depth_cuda, render_depth_cuda2


class Gaussians:
    means: torch.FloatTensor
    covariances: torch.FloatTensor
    scales: torch.FloatTensor
    rotations: torch.FloatTensor
    harmonics: torch.FloatTensor
    opacities: torch.FloatTensor


@HEADS.register_module()
class GaussianSplattingDecoder(NeRFDecoderHead):
    def __init__(self,
                 volume_size=(200, 200, 16),
                 **kwargs):
        super().__init__(**kwargs)

        xs = torch.arange(
            self.xyz_min[0], self.xyz_max[0],
            (self.xyz_max[0] - self.xyz_min[0]) / volume_size[0])
        ys = torch.arange(
            self.xyz_min[1], self.xyz_max[1],
            (self.xyz_max[1] - self.xyz_min[1]) / volume_size[1])
        zs = torch.arange(
            self.xyz_min[2], self.xyz_max[2],
            (self.xyz_max[2] - self.xyz_min[2]) / volume_size[2])
        W, H, D = len(xs), len(ys), len(zs)

        xyzs = torch.stack([
            xs[None, :, None].expand(H, W, D),
            ys[:, None, None].expand(H, W, D),
            zs[None, None, :].expand(H, W, D)
        ], dim=-1)  # (200, 200, 16, 3)

        # the volume grid coordinates in ego frame
        self.register_buffer(
            "volume_xyz",
            torch.tensor(xyzs, dtype=torch.float32),
            persistent=False,
        )

        self.NUSCENSE_LIDARSEG_PALETTE = torch.Tensor([
            (0, 0, 0),  # noise
            (112, 128, 144),  # barrier
            (220, 20, 60),  # bicycle
            (255, 127, 80),  # bus
            (255, 158, 0),  # car
            (233, 150, 70),  # construction_vehicle
            (255, 61, 99),  # motorcycle
            (0, 0, 230),  # pedestrian
            (47, 79, 79),  # traffic_cone
            (255, 140, 0),  # trailer
            (255, 99, 71),  # Tomato
            (0, 207, 191),  # nuTonomy green
            (175, 0, 75),
            (75, 0, 75),
            (112, 180, 60),
            (222, 184, 135),  # Burlywood
            (0, 175, 0),
            (0, 0, 0), # ignore
        ])

        self.OCC3D_PALETTE = torch.Tensor([
            [0, 0, 0],
            [255, 120, 50],  # barrier              orangey
            [255, 192, 203],  # bicycle              pink
            [255, 255, 0],  # bus                  yellow
            [0, 150, 245],  # car                  blue
            [0, 255, 255],  # construction_vehicle cyan
            [200, 180, 0],  # motorcycle           dark orange
            [255, 0, 0],  # pedestrian           red
            [255, 240, 150],  # traffic_cone         light yellow
            [135, 60, 0],  # trailer              brown
            [160, 32, 240],  # truck                purple
            [255, 0, 255],  # driveable_surface    dark pink
            [139, 137, 137], # other_flat           dark grey
            [75, 0, 75],  # sidewalk             dard purple
            [150, 240, 80],  # terrain              light green
            [230, 230, 250],  # manmade              white
            [0, 175, 0],  # vegetation           green
            [0, 255, 127],  # ego car              dark cyan
            [255, 99, 71],
            [0, 191, 255],
            [125, 125, 125]
        ])

    def forward(self, 
                density_prob, 
                rgb_recon,
                occ_semantic, 
                intricics, 
                pose_spatial, 
                is_train,
                render_mask=None,
                **kwargs):
        # print('density_prob', density_prob.shape)
        # print('rgb_recon', rgb_recon.shape)
        # print('occ_semantic', occ_semantic.shape) # (1, 1, 200, 200, 16)
        semantic_pred, render_depth = self.train_gaussian_rasterization(
            density_prob,
            rgb_recon,  # pseudo color
            occ_semantic,
            intricics,
            pose_spatial,
        )
        render_depth = render_depth.clamp(self.min_depth, self.max_depth)
        return render_depth, semantic_pred, None

    def train_gaussian_rasterization(self, 
                                     density_prob, 
                                     rgb_recon, 
                                     semantic_pred, 
                                     intrinsics, 
                                     extrinsics, 
                                     render_mask=None):
        b, v = intrinsics.shape[:2]
        device = density_prob.device
        
        near = torch.ones(b, v).to(device) * self.min_depth
        far = torch.ones(b, v).to(device) * self.max_depth
        background_color = torch.zeros((3), dtype=torch.float32).to(device)
        
        intrinsics = intrinsics[..., :3, :3]
        # normalize the intrinsics
        intrinsics[..., 0, :] /= self.render_w
        intrinsics[..., 1, :] /= self.render_h

        transform = torch.Tensor([[0, 1, 0, 0],
                                  [1, 0, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]]).to(device)
        extrinsics = transform.unsqueeze(0).unsqueeze(0) @ extrinsics
        
        bs = density_prob.shape[0]

        xyzs = repeat(self.volume_xyz, 'h w d dim3 -> bs h w d dim3', bs=bs)
        xyzs = rearrange(xyzs, 'b h w d dim3 -> b (h w d) dim3') # (bs, num, 3)

        density_prob = rearrange(density_prob, 'b dim1 h w d -> (b dim1) (h w d)')
        
        if self.semantic_head:
            harmonics = self.OCC3D_PALETTE[semantic_pred.long().flatten()].to(device)
        else:
            harmonics = self.OCC3D_PALETTE[torch.argmax(rgb_recon, dim=1).long()].to(device)
        harmonics = rearrange(harmonics, 'b h w d dim3 -> b (h w d) dim3 ()')

        g = xyzs.shape[1]

        gaussians = Gaussians
        gaussians.means = xyzs  ######## Gaussian center ########
        gaussians.opacities = torch.sigmoid(density_prob) ######## Gaussian opacities ########

        scales = torch.ones(3).unsqueeze(0).to(device) * 0.05
        rotations = torch.Tensor([1, 0, 0, 0]).unsqueeze(0).to(device)

        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)
        gaussians.covariances = covariances ######## Gaussian covariances ########

        gaussians.harmonics = harmonics ######## Gaussian harmonics ########

        color, depth = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            (self.render_h, self.render_w),
            repeat(background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b v i j -> (b v) g i j", g=g),
            repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            scale_invariant=False,
            use_sh=False,
        )
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)
        depth = rearrange(depth, "(b v) c h w -> b v c h w", b=b, v=v).squeeze(2)

        return color, depth
    
    def gaussian_rasterization(self, 
                               density_prob, 
                               rgb_recon, 
                               semantic_pred, 
                               intrinsics, 
                               extrinsics, 
                               render_mask=None):
        b, v = intrinsics.shape[:2]
        
        near = torch.ones(b, v).to(density_prob.device) * 1
        far = torch.ones(b, v).to(density_prob.device) * 100
        background_color = torch.zeros((3), dtype=torch.float32).to(density_prob.device)
        
        intrinsics = intrinsics[..., :3, :3]
        # normalize the intrinsics
        intrinsics[..., 0, :] /= self.render_w
        intrinsics[..., 1, :] /= self.render_h

        transform = torch.Tensor([[0, 1, 0, 0],
                                  [1, 0, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]]).to(density_prob.device)
        extrinsics = transform.unsqueeze(0).unsqueeze(0) @ extrinsics

        device = density_prob.device
        xs = torch.arange(
            self.xyz_min[0], self.xyz_max[0],
            (self.xyz_max[0] - self.xyz_min[0]) / density_prob.shape[2], device=device)
        ys = torch.arange(
            self.xyz_min[1], self.xyz_max[1],
            (self.xyz_max[1] - self.xyz_min[1]) / density_prob.shape[3], device=device)
        zs = torch.arange(
            self.xyz_min[2], self.xyz_max[2],
            (self.xyz_max[2] - self.xyz_min[2]) / density_prob.shape[4], device=device)
        W, H, D = len(xs), len(ys), len(zs)
        
        bs = density_prob.shape[0]
        xyzs = torch.stack([
            xs[None, :, None].expand(H, W, D),
            ys[:, None, None].expand(H, W, D),
            zs[None, None, :].expand(H, W, D)
        ], dim=-1)[None].expand(bs, H, W, D, 3).flatten(0, 3)
        density_prob = density_prob.flatten()

        mask = (density_prob > 0) #& (semantic_pred.flatten()==3)
        xyzs = xyzs[mask]

        harmonics = self.OCC3D_PALETTE[semantic_pred.long().flatten()].to(device)
        harmonics = harmonics[mask]

        density_prob = density_prob[mask]
        density_prob = density_prob.unsqueeze(0)
        xyzs = xyzs.unsqueeze(0)

        g = xyzs.shape[1]

        gaussians = Gaussians
        gaussians.means = xyzs  ######## Gaussian center ########
        gaussians.opacities = torch.where(density_prob>0, 1., 0.) ######## Gaussian opacities ########

        scales = torch.ones(3).unsqueeze(0).to(device) * 0.05
        rotations = torch.Tensor([1, 0, 0, 0]).unsqueeze(0).to(device)

        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)
        gaussians.covariances = covariances ######## Gaussian covariances ########

        harmonics = harmonics.unsqueeze(-1).unsqueeze(0)
        # harmonics = torch.ones_like(xyzs).unsqueeze(-1)
        gaussians.harmonics = harmonics ######## Gaussian harmonics ########

        color, depth = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            (self.render_h, self.render_w),
            repeat(background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b v i j -> (b v) g i j", g=g),
            repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            scale_invariant=False,
            use_sh=False,
        )

        return color, depth.squeeze(1)