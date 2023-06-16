'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-06-07 19:52:14
Email: haimingzhang@link.cuhk.edu.cn
Description: Use the lidar point cloud to do the occupancy prediction.
'''

import torch
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from torch import nn
import numpy as np
from einops import repeat, rearrange
from .. import builder
from .centerpoint import CenterPoint

@DETECTORS.register_module()
class BEVLidarOCC(CenterPoint):
    def __init__(self, 
                 pts_bev_encoder_backbone,
                 pts_bev_encoder_neck,
                 loss_occ=None,
                 lidar_out_dim=32,
                 out_dim=32,
                 use_mask=True,
                 num_classes=18,
                 use_predicter=True, 
                 **kwargs):
        super(BEVLidarOCC, self).__init__(**kwargs)
        
        self.pts_bev_encoder_backbone = \
            builder.build_backbone(pts_bev_encoder_backbone)
        self.pts_bev_encoder_neck = builder.build_neck(pts_bev_encoder_neck)

        self.out_dim = out_dim
        out_channels = out_dim if use_predicter else num_classes
        self.final_conv = ConvModule(lidar_out_dim,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     bias=True,
                                     conv_cfg=dict(type='Conv3d'))
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, num_classes),
            )
        self.pts_bbox_head = None
        self.use_mask = use_mask
        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)

    def loss_single(self, voxel_semantics, mask_camera, preds):
        loss_ = dict()
        voxel_semantics=voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)
            voxel_semantics=voxel_semantics.reshape(-1)
            preds=preds.reshape(-1,self.num_classes)
            mask_camera = mask_camera.reshape(-1)
            num_total_samples=mask_camera.sum()
            loss_occ=self.loss_occ(preds,voxel_semantics,mask_camera, avg_factor=num_total_samples)
            loss_['loss_occ'] = loss_occ
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics,)
            loss_['loss_occ'] = loss_occ
        return loss_
    
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        """Forward function for BEV Lidar OCC.
        """
        img_feats, pts_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        
        losses = dict()
        occ_pred = self.occ_head(pts_feats)
        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
        losses.update(loss_occ)
        return losses
    
    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        _, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        occ_pred = self.final_conv(pts_feats[0]).permute(0, 4, 3, 2, 1)
        # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        occ_score=occ_pred.softmax(-1)
        occ_res=occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [occ_res]
    
    def get_intermediate_features(self, points, img, img_metas, **kwargs):
        """Obtain the features for cross-modality distillation when as a teacher
        model.
        """
        voxels, num_points, coors = self.voxelize(points)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        low_feats = self.pts_middle_encoder(voxel_features, coors, batch_size)

        x = self.pts_bev_encoder_backbone(low_feats)
        high_feats = self.pts_bev_encoder_neck(x)

        prob_feats = self.final_conv(high_feats).permute(0, 4, 3, 2, 1)
        return low_feats, high_feats, prob_feats  
    
    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract the needed features"""
        img_feats = None
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        if not isinstance(pts_feats, list):
            pts_feats = [pts_feats]
        return img_feats, pts_feats
    
    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)

        ## reshape the x to the shape of (batch_size, C, D, H, W)
        x = self.pts_bev_encoder_backbone(x)
        x = self.pts_bev_encoder_neck(x)
        return x
    
    def occ_head(self, x):
        if isinstance(x, list):
            feats = x[0]
        
        if feats.dim() == 4:
            feats = rearrange(
                feats, 'b (c Dimz) DimH DimW -> b c Dimz DimH DimW', Dimz=16)
        
        occ_pred = self.final_conv(feats).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        return occ_pred