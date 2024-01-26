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
from .common.cuda_splatting import render_cuda, render_depth_cuda


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
                 **kwargs):
        super().__init__(**kwargs)

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

    def forward(self, 
                density_prob, 
                rgb_recon,
                occ_semantic, 
                intricics, 
                pose_spatial, 
                is_train,
                render_mask=None,
                **kwargs):
        print('density_prob', density_prob.shape)
        print('rgb_recon', rgb_recon.shape)
        print('occ_semantic', occ_semantic.shape)
        semantic_pred, render_depth = self.gaussian_rasterization(
            density_prob,
            rgb_recon,  # pseudo color
            occ_semantic,
            intricics,
            pose_spatial,
        )
        render_depth = render_depth.unsqueeze(0)
        semantic_pred = semantic_pred.unsqueeze(0).clamp(0, 1)
        return render_depth, semantic_pred, None

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
        intrinsics[..., 0, :] /= 100
        intrinsics[...,1, :] /= 100

        transform = torch.Tensor(np.diag([1, 1, 1, 1])).to(density_prob.device)
        extrinsics = transform.unsqueeze(0).unsqueeze(0) @ extrinsics

        device = density_prob.device
        xs = torch.range(
            self.xyz_min[0], self.xyz_max[0],
            (self.xyz_max[0] - self.xyz_min[0]) / density_prob.shape[2], device=device)[:-1]
        ys = torch.range(
            self.xyz_min[1], self.xyz_max[1],
            (self.xyz_max[1] - self.xyz_min[1]) / density_prob.shape[3], device=device)[:-1]
        zs = torch.range(
            self.xyz_min[2], self.xyz_max[2],
            (self.xyz_max[2] - self.xyz_min[2]) / density_prob.shape[4], device=device)[:-1]
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

        harmonics = self.NUSCENSE_LIDARSEG_PALETTE[semantic_pred.long().flatten()].to(device)
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

        color = render_cuda(
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

        depth = render_depth_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            (self.render_h, self.render_w),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            mode="disparity",
            scale_invariant=False,

        )
        return color, depth