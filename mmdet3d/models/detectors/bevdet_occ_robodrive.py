'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-03-21 20:09:23
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
# Copyright (c) Phigent Robotics. All rights reserved.
from .bevdet import BEVStereo4D

import torch
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from mmcv.runner import force_fp32

from torch import nn
import numpy as np
from torch.nn import functional as F
from einops import repeat, rearrange
from mmdet.core import multi_apply
from mmdet3d.datasets.evaluation_metrics import evaluation_semantic
from mmdet3d.models.losses.lovasz_loss import Lovasz_loss
from .. import builder
from .bevdet_occ import BEVStereo4DOCC
from ..losses.loss_utils import sem_scal_loss, geo_scal_loss
from .bevdet_occ_ssc import SSCNet
from .bevdet_occ import SE_Block


@DETECTORS.register_module()
class BEVStereo4DOCCRoboDrive(BEVStereo4DOCC):

    def __init__(self,
                 use_surroundocc_loss=False,
                 use_lovasz_loss=False,
                 **kwargs):
        super(BEVStereo4DOCCRoboDrive, self).__init__(**kwargs)

        assert not self.use_mask, 'mask is not supported for robodrive'

        self.use_surroundocc_loss = use_surroundocc_loss
        
        self.use_lovasz_loss = use_lovasz_loss
        if use_lovasz_loss:
            self.loss_lovasz = Lovasz_loss(255)

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)

        ## start evaluate
        voxel_semantics = kwargs.get('voxel_semantics', None)
        pred_occ = occ_res
        class_num = self.num_classes
        if voxel_semantics is not None:
            eval_results = evaluation_semantic(pred_occ, 
                                               voxel_semantics[0], 
                                               img_metas[0], 
                                               class_num)
        else:
            eval_results = None
        return {'evaluation': eval_results, 'prediction': pred_occ}

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        """Forward training function.
        Returns:
            dict: Losses of different branches.
        """
        img_feats, _, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth

        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        
        # compute losses
        voxel_semantics = kwargs['voxel_semantics']
        loss_occ = self.loss_single(voxel_semantics, occ_pred)
        losses.update(loss_occ)
        return losses
    
    def loss_single(self, voxel_semantics, preds):
        loss_ = dict()

        if self.use_surroundocc_loss:
            preds_tmp = rearrange(preds, 'b h w d c -> b c h w d')
            loss_['loss_sem_scal'] = sem_scal_loss(preds_tmp, voxel_semantics.long())
            loss_['loss_geo_scal'] = geo_scal_loss(preds_tmp, voxel_semantics.long())

        voxel_semantics = voxel_semantics.long()
        voxel_semantics = voxel_semantics.reshape(-1)
        preds = preds.reshape(-1, self.num_classes)
        loss_occ = self.loss_occ(preds, voxel_semantics)
        loss_['loss_occ'] = loss_occ

        if self.use_lovasz_loss:
            loss_lovasz = self.loss_lovasz(F.softmax(preds, dim=1), 
                                        voxel_semantics)
            
            loss_['loss_lovasz'] = loss_lovasz

        return loss_
    

@DETECTORS.register_module()
class BEVStereo4DSSCOCCRoboDrive(BEVStereo4DOCCRoboDrive):

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

        self.final_conv = SSCNet(
            self.img_view_transformer.out_channels,
            [32, 32, 32, 32, 32],
            self.out_dim)
        

@DETECTORS.register_module()
class BEVFusionStereo4DOCCRoboDrive(BEVStereo4DOCCRoboDrive):
    """Fuse the lidar and camera images for occupancy prediction.

    Args:
        BEVStereo4DOCC (_type_): _description_
    """

    def __init__(self,
                 se=True,
                 lic=384,
                 fuse_3d_bev=False,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.fuse_3d_bev = fuse_3d_bev
        self.se = se
        
        if fuse_3d_bev:
            if se:
                self.seblock = SE_Block(32, mode='3d')
            self.reduc_conv = ConvModule(
                lic + 32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                conv_cfg=dict(type='Conv3d'))
        else:
            if se:
                self.seblock = SE_Block(32*16)
            self.reduc_conv = ConvModule(
                lic + 32*16,
                32*16,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_backbone:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(
            voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x
    
    def apply_lc_fusion(self, img_feats, pts_feats):
        if self.fuse_3d_bev:
            assert img_feats.ndim == pts_feats.ndim == 5

            pts_feats = [self.reduc_conv(torch.cat([img_feats, pts_feats], dim=1))]
            if self.se:
                pts_feats = self.seblock(pts_feats[0])
        else:
            if img_feats.ndim == 5:
                Dimz = img_feats.shape[2]
                img_bev_feat = rearrange(img_feats, 'b c Dimz DimH DimW -> b (c Dimz) DimH DimW')
            else:
                img_bev_feat = img_feats
            
            pts_feats = [self.reduc_conv(torch.cat([img_bev_feat, pts_feats[0]], dim=1))]
            if self.se:
                pts_feats = self.seblock(pts_feats[0])
            
            if img_feats.ndim == 5:
                pts_feats = rearrange(pts_feats, 'b (c Dimz) DimH DimW -> b c Dimz DimH DimW', Dimz=Dimz)
        return [pts_feats]
        
    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats, depth = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)  # [(b,384,200,200)]

        ## apply the lidar camera fusion
        fusion_feats = self.apply_lc_fusion(img_feats[0], pts_feats)
        return (fusion_feats, pts_feats, depth)

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        occ_score=occ_pred.softmax(-1)
        occ_res=occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [occ_res]
    
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        """Forward training function.
        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth

        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        voxel_semantics = kwargs['voxel_semantics']
        loss_occ = self.loss_single(voxel_semantics, occ_pred)
        losses.update(loss_occ)
        return losses
    
    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(points), len(img_metas)))

        if num_augs == 1:
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            return self.simple_test(points[0], img_metas[0], 
                                    img_inputs[0], **kwargs)
        else:
            return self.aug_test(points, img_metas, img_inputs, **kwargs)