'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-10-31 21:05:56
Email: haimingzhang@link.cuhk.edu.cn
Description: Here we use the occupancy prediction head like RenderOcc.
'''

# Copyright (c) Phigent Robotics. All rights reserved.
from .bevdet_occ import BEVStereo4DOCC
import torch.nn.functional as F
import torch
import cv2
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from torch import nn
import numpy as np
from .. import builder
import time


# occ3d-nuscenes
nusc_class_frequencies = np.array([1163161, 2309034, 188743, 2997643, 20317180, 852476, 243808, 2457947, 
            497017, 2731022, 7224789, 214411435, 5565043, 63191967, 76098082, 128860031, 
            141625221, 2307405309])

@DETECTORS.register_module()
class BEVDetRenderOcc(BEVStereo4DOCC):
    def __init__(self,
                 out_dim=32,
                 num_classes=18,
                 test_threshold=8.5,
                 use_lss_depth_loss=True,
                 use_3d_loss=True,
                 balance_cls_weight=True,
                 final_softplus=False,
                 use_mask=True,
                 use_origin_testing=True,
                 **kwargs):
        super(BEVDetRenderOcc, self).__init__(use_predicter=False, **kwargs)
        self.out_dim = out_dim
        self.use_3d_loss = use_3d_loss
        self.test_threshold = test_threshold
        self.use_lss_depth_loss = use_lss_depth_loss
        self.balance_cls_weight = balance_cls_weight
        self.final_softplus = final_softplus
        self.use_mask = use_mask
        self.use_origin_testing = use_origin_testing

        if self.balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:17] + 0.001)).float()
            self.semantic_loss = nn.CrossEntropyLoss(
                    weight=self.class_weights, reduction="mean"
                )
        else:
            self.semantic_loss = nn.CrossEntropyLoss(reduction="mean")

        self.final_conv = ConvModule(
                        self.img_view_transformer.out_channels,
                        self.out_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        conv_cfg=dict(type='Conv3d'))

        if self.final_softplus:
            self.density_mlp = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, 2),
                nn.Softplus(),
            )
        else:
            self.density_mlp = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, 2),
            )

        self.semantic_mlp = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim*2),
            nn.Softplus(),
            nn.Linear(self.out_dim*2, num_classes-1),
        )


    def loss_3d(self, voxel_semantics, mask_camera, density_prob, semantic):
        voxel_semantics=voxel_semantics.long()
     
        voxel_semantics=voxel_semantics.reshape(-1)
        density_prob=density_prob.reshape(-1, 2)
        semantic = semantic.reshape(-1, self.num_classes-1)
        density_target = (voxel_semantics==17).long()
        semantic_mask = voxel_semantics!=17

        # compute loss
        if self.use_mask:
            mask_camera = mask_camera.to(torch.bool)
            mask_camera = mask_camera.reshape(-1)
            
            density_prob = density_prob[mask_camera]
            density_target = density_target[mask_camera]
            loss_geo = self.loss_occ(density_prob, density_target)

            semantic_mask = semantic_mask[mask_camera]
            loss_sem = self.semantic_loss(semantic[semantic_mask], 
                                          voxel_semantics[semantic_mask].long())
        else:
            loss_geo = self.loss_occ(density_prob, density_target)
            loss_sem = self.semantic_loss(semantic[semantic_mask], 
                                          voxel_semantics[semantic_mask].long())

        loss_ = dict()
        loss_['loss_3d_geo'] = loss_geo
        loss_['loss_3d_sem'] = loss_sem
        return loss_


    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        # extract volumn feature
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        voxel_feats = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        
        # predict SDF
        density_prob = self.density_mlp(voxel_feats)
        density = density_prob[...,0]
        semantic = self.semantic_mlp(voxel_feats)

        # SDF --> Occupancy
        if self.use_origin_testing:
            no_empty_mask = density > self.test_threshold
        else:
            density_score = density_prob.softmax(-1)
            no_empty_mask = density_score[..., 0] > density_score[..., 1]
        
        semantic_res = semantic.argmax(-1)

        B, H, W, Z, C = voxel_feats.shape
        occ = torch.ones((B,H,W,Z), dtype=semantic_res.dtype).to(semantic_res.device)
        occ = occ * (self.num_classes-1)
        occ[no_empty_mask] = semantic_res[no_empty_mask]

        occ = occ.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [occ]

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # extract volumn feature
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        voxel_feats = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc

        # predict SDF
        density_prob = self.density_mlp(voxel_feats)
        semantic = self.semantic_mlp(voxel_feats)

        # compute loss
        losses = dict()
        if self.use_3d_loss:      # 3D loss
            voxel_semantics = kwargs['voxel_semantics']
            mask_camera = kwargs['mask_camera']
            assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
            loss_occ = self.loss_3d(voxel_semantics, mask_camera, density_prob, semantic)
            losses.update(loss_occ)
        if self.use_lss_depth_loss: # lss-depth loss (BEVStereo's feature)
            loss_depth = self.img_view_transformer.get_depth_loss(kwargs['gt_depth'], depth)
            losses['loss_lss_depth'] = loss_depth
        return losses

