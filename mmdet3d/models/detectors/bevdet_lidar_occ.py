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
    """Deprecated BEVLidarOCC class since it is not compatiable with
    the original mmdetection3d classes, please use the below `LidarOCC` class.

    Args:
        CenterPoint (_type_): _description_
    """
    def __init__(self, 
                 pts_bev_encoder_backbone=None,
                 pts_bev_encoder_neck=None,
                 loss_occ=None,
                 lidar_out_dim=32,
                 out_dim=32,
                 use_mask=True,
                 num_classes=18,
                 use_predicter=True, 
                 use_free_occ_token=False,
                 **kwargs):
        super(BEVLidarOCC, self).__init__(**kwargs)
        
        if pts_bev_encoder_backbone:
            self.pts_bev_encoder_backbone = builder.build_backbone(
                pts_bev_encoder_backbone)
        if pts_bev_encoder_neck:
            self.pts_bev_encoder_neck = builder.build_neck(
                pts_bev_encoder_neck)

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
                nn.Linear(self.out_dim*2, num_classes))
        
        if use_free_occ_token:
            self.free_occ_token = nn.Parameter(
                torch.randn(1, out_channels, 1, 1, 1))
        
        self.use_free_occ_token = use_free_occ_token
        
        self.use_mask = use_mask
        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)

    def loss_single(self, voxel_semantics, mask_camera, preds):
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17

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
        
        if self.use_free_occ_token:
            free_voxels = kwargs['mask_camera_free']  # the flag for voxels that are free
            free_voxels = rearrange(free_voxels, 'b h w d -> b () d h w')
            free_voxels = free_voxels.to(torch.float32)
            # free_voxels_token = rearrange(self.free_occ_token, 'c -> () c () () ()')
            pts_feats = free_voxels * self.free_occ_token + (1 - free_voxels) * pts_feats
        
        occ_pred = self.occ_head(pts_feats)
        occ_score=occ_pred.softmax(-1)
        occ_res=occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [occ_res]
    
    def get_intermediate_features(self, 
                                  points, 
                                  img=None, 
                                  img_metas=None, 
                                  return_loss=False,
                                  logits_as_prob_feat=False,
                                  **kwargs):       
        """Obtain the features for cross-modality distillation when as a teacher
        model.
        """
        voxels, num_points, coors = self.voxelize(points)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        low_feats = self.pts_middle_encoder(voxel_features, coors, batch_size)

        x = self.pts_bev_encoder_backbone(low_feats)
        high_feats = self.pts_bev_encoder_neck(x)  # to (b, c, d, h, w)
        
        if self.use_free_occ_token:
            free_voxels = kwargs['mask_camera_free']  # the flag for voxels that are free
            free_voxels = rearrange(free_voxels, 'b h w d -> b () d h w')
            free_voxels = free_voxels.to(torch.float32)
            # free_voxels_token = rearrange(self.free_occ_token, 'c -> () c () () ()')
            high_feats = free_voxels * self.free_occ_token + (1 - free_voxels) * high_feats
        
        ## get the prob_feat with shape (B, C, D, H, W)
        prob_feats = self.final_conv(high_feats).permute(0, 4, 3, 2, 1)

        if self.use_predicter:
            occ_pred = self.predicter(prob_feats)
        else:
            occ_pred = prob_feats

        if logits_as_prob_feat:
            prob_feats = occ_pred

        if return_loss:
            losses = dict()
            voxel_semantics = kwargs['voxel_semantics']
            mask_camera = kwargs['mask_camera']
            
            loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
            losses['teacher_loss_occ'] = loss_occ['loss_occ']

            return (None, high_feats, prob_feats), losses
        else:
            return (low_feats, high_feats, prob_feats)
    
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
    


@DETECTORS.register_module()
class MyBEVLidarOCCNeRF(CenterPoint):
    """

    Args:
        CenterPoint (_type_): _description_
    """
    def __init__(self, 
                 loss_occ=None,
                 nerf_head=None,
                 lidar_out_dim=32,
                 out_dim=32,
                 use_mask=True,
                 num_classes=18,
                 use_predicter=True, 
                 refine_conv=False,
                 **kwargs):
        super(MyBEVLidarOCCNeRF, self).__init__(**kwargs)
        
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
                nn.Linear(self.out_dim*2, num_classes))
        
        self.use_mask = use_mask
        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)

    def loss_single(self, voxel_semantics, mask_camera, preds):
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17

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
        
        occ_pred = self.occ_head(pts_feats)
        occ_score=occ_pred.softmax(-1)
        occ_res=occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [occ_res]
    
    def get_intermediate_features(self, 
                                  points, 
                                  img=None, 
                                  img_metas=None, 
                                  return_loss=False,
                                  logits_as_prob_feat=False,
                                  **kwargs):       
        """Obtain the features for cross-modality distillation when as a teacher
        model.
        """
        voxels, num_points, coors = self.voxelize(points)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        low_feats = self.pts_middle_encoder(voxel_features, coors, batch_size)

        x = self.pts_bev_encoder_backbone(low_feats)
        high_feats = self.pts_bev_encoder_neck(x)  # to (b, c, d, h, w)
        
        if self.use_free_occ_token:
            free_voxels = kwargs['mask_camera_free']  # the flag for voxels that are free
            free_voxels = rearrange(free_voxels, 'b h w d -> b () d h w')
            free_voxels = free_voxels.to(torch.float32)
            # free_voxels_token = rearrange(self.free_occ_token, 'c -> () c () () ()')
            high_feats = free_voxels * self.free_occ_token + (1 - free_voxels) * high_feats
        
        ## get the prob_feat with shape (B, C, D, H, W)
        prob_feats = self.final_conv(high_feats).permute(0, 4, 3, 2, 1)

        if self.use_predicter:
            occ_pred = self.predicter(prob_feats)
        else:
            occ_pred = prob_feats

        if logits_as_prob_feat:
            prob_feats = occ_pred

        if return_loss:
            losses = dict()
            voxel_semantics = kwargs['voxel_semantics']
            mask_camera = kwargs['mask_camera']
            
            loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
            losses['teacher_loss_occ'] = loss_occ['loss_occ']

            return (None, high_feats, prob_feats), losses
        else:
            return (low_feats, high_feats, prob_feats)
    
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
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
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
    

@DETECTORS.register_module()
class LidarOCC(CenterPoint):

    def __init__(self, 
                 occ_head=None,
                 use_free_occ_token=False,
                 use_binary_occupacy=False,
                 **kwargs):
        super(LidarOCC, self).__init__(**kwargs)
        
        if occ_head:
            self.occ_head = builder.build_head(occ_head)

        if use_free_occ_token:
            self.free_occ_token = nn.Parameter(
                torch.randn(1, 32, 1, 1, 1))
        
        self.use_free_occ_token = use_free_occ_token
        self.use_binary_occupacy = use_binary_occupacy

    @property
    def with_occ_head(self):
        """bool: Whether the detector has a 3D box head."""
        return hasattr(self,
                       'occ_head') and self.occ_head is not None
    
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        """Forward function for BEV Lidar OCC.
        """
        _, pts_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        
        if self.use_free_occ_token:
            free_voxels = kwargs['mask_camera_free']  # the flag for voxels that are free
            free_voxels = rearrange(free_voxels, 'b h w d -> b () d h w')
            free_voxels = free_voxels.to(torch.float32)
            # free_voxels_token = rearrange(self.free_occ_token, 'c -> () c () () ()')
            pts_feats = free_voxels * self.free_occ_token + (1 - free_voxels) * pts_feats

        loss_occ = self.occ_head.forward_train(pts_feats, 
                                               use_binary_occupacy=self.use_binary_occupacy,
                                               **kwargs)
        return loss_occ
    
    def extract_pts_feat(self, pts, img_feats, img_metas):
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x
    
    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        _, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        
        if self.use_free_occ_token:
            free_voxels = kwargs['mask_camera_free']  # the flag for voxels that are free
            free_voxels = rearrange(free_voxels, 'b h w d -> b () d h w')
            free_voxels = free_voxels.to(torch.float32)
            pts_feats = free_voxels * self.free_occ_token + (1 - free_voxels) * pts_feats
        
        if not self.with_occ_head:
            return pts_feats
        
        occ_pred = self.occ_head(pts_feats)
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [occ_res]
    
    def get_intermediate_features(self, 
                                  points, 
                                  return_loss=False,
                                  logits_as_prob_feat=False,
                                  **kwargs):       
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
        if self.use_predicter:
            occ_pred = self.predicter(prob_feats)
        else:
            occ_pred = prob_feats

        if logits_as_prob_feat:
            prob_feats = occ_pred

        if return_loss:
            losses = dict()
            voxel_semantics = kwargs['voxel_semantics']
            mask_camera = kwargs['mask_camera']
            
            loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
            losses['teacher_loss_occ'] = loss_occ['loss_occ']

            return (None, high_feats, prob_feats), losses
        else:
            return (low_feats, high_feats, prob_feats)