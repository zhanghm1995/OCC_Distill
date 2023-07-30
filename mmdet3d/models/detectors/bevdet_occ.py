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

from .. import builder


def merge_aug_occs(aug_occ_list):
    """Merge the augmented testing occupacy prediction results

    Args:
        aug_occ_list (List): each element with shape (200, 200, 16, 18)
    """
    merged_occs = torch.stack(aug_occ_list, dim=-1)  # to (200, 200, 16, 18, L)
    merged_occs = torch.mean(merged_occs, dim=-1)  # to (200, 200, 16, 18)
    occ_res = merged_occs.argmax(-1)
    occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
    return occ_res


@DETECTORS.register_module()
class BEVStereo4DOCC(BEVStereo4D):

    def __init__(self,
                 loss_occ=None,
                 out_dim=32,
                 use_mask=False,
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,
                 **kwargs):
        super(BEVStereo4DOCC, self).__init__(**kwargs)
        self.out_dim = out_dim
        out_channels = out_dim if use_predicter else num_classes
        self.final_conv = ConvModule(
                        self.img_view_transformer.out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        conv_cfg=dict(type='Conv3d'))
        self.use_predicter =use_predicter
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
        self.class_wise = class_wise
        self.align_after_view_transfromation = False

    def loss_single(self,voxel_semantics,mask_camera,preds):
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
        # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        occ_score=occ_pred.softmax(-1)
        occ_res=occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [occ_res]

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

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
        mask_camera = kwargs['mask_camera']
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
        losses.update(loss_occ)
        return losses
    
    def get_intermediate_features(self, points, img_inputs, img_metas, 
                                  return_loss=False, 
                                  logits_as_prob_feat=False,
                                  **kwargs):
        """Obtain the intermediate features in the forwarding process.
        """
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        losses = dict()
        gt_depth = kwargs['gt_depth']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth

        prob_feats = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(prob_feats)
        else:
            occ_pred = prob_feats

        if logits_as_prob_feat:
            prob_feats = occ_pred
            
        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
        losses.update(loss_occ)

        if return_loss:
            return (None, img_feats[0], prob_feats), losses
        else:
            return (None, img_feats[0], prob_feats)


@DETECTORS.register_module()
class MultiScaleBEVStereo4DOCC(BEVStereo4DOCC):

    def __init__(self,
                 use_ms_loss=True,
                 occ_ms_head1=None,
                 occ_ms_head2=None,
                 occ_ms_head3=None,
                 **kwargs):
        super(MultiScaleBEVStereo4DOCC, self).__init__(**kwargs)

        self.use_ms_loss = use_ms_loss
        if use_ms_loss:
            ms_occ_head1 = builder.build_head(occ_ms_head1)
            ms_occ_head2 = builder.build_head(occ_ms_head2)
            ms_occ_head3 = builder.build_head(occ_ms_head3)
            blocks = [ms_occ_head1, ms_occ_head2, ms_occ_head3]
            self.ms_head_blocks = nn.ModuleList(blocks)

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
        # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        occ_score=occ_pred.softmax(-1)
        occ_res=occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [occ_res]

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, depth, ms_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, 
            return_ms_feats=True, **kwargs)
        
        gt_depth = kwargs['gt_depth']
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth

        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
        losses.update(loss_occ)

        ## add the multi-scale losses
        if self.use_ms_loss:
            ms_losses = self.compute_ms_occ_losses(ms_feats, **kwargs)
            losses.update(ms_losses)
        return losses
    
    def extract_feat(self, points, img, img_metas, 
                     return_ms_feats=False,
                     **kwargs):
        """Extract features from images and points."""
        pts_feats = None

        if return_ms_feats:
            img_feats, depth, ms_feats = self.extract_img_feat(
                img, 
                img_metas, 
                return_ms_feats=True,
                **kwargs)
            return img_feats, pts_feats, depth, ms_feats
        else:
            img_feats, depth = self.extract_img_feat(
                img, 
                img_metas, 
                return_ms_feats=False,
                **kwargs)
            return img_feats, pts_feats, depth
    
    def compute_ms_occ_losses(self, ms_feats, **kwargs):
        losses = dict()
        for i, feat in enumerate(ms_feats):
            loss = self.ms_head_blocks[i].forward_train(feat, **kwargs)
            loss_name = f'ms_occ_loss{i}'
            losses[loss_name] = loss['loss_occ']
        return losses
    

class SE_Block(nn.Module):
    def __init__(self, c, mode='2d'):
        super().__init__()
        if mode == '2d':
            self.att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c, c, kernel_size=1, stride=1),
                nn.Sigmoid())
        elif mode == '3d':
            self.att = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(c, c, kernel_size=1, stride=1),
                nn.Sigmoid())
        else:
            raise ValueError(
                'Expected mode should be 2d or 3d, but got {}.'.format(mode))
    def forward(self, x):
        return x * self.att(x)


@DETECTORS.register_module()
class BEVFusionStereo4DOCC(BEVStereo4DOCC):
    """Fuse the lidar and camera images for occupancy prediction.

    Args:
        BEVStereo4DOCC (_type_): _description_
    """

    def __init__(self,
                 se=True,
                 lic=384,
                 fuse_3d_bev=False,
                 **kwargs):
        super(BEVFusionStereo4DOCC, self).__init__(**kwargs)
        
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

    def loss_single(self,voxel_semantics,mask_camera,preds):
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
        # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        occ_score=occ_pred.softmax(-1)
        occ_res=occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [occ_res]
    
    def aug_test(self, 
                 points,
                 img_metas,
                 img=None,
                 rescale=False,
                 **kwargs):
        """Test function with augmentaiton."""
        img_feats, pts_feats, _ = multi_apply(self.extract_feat, 
                                              points, 
                                              img,
                                              img_metas, 
                                              **kwargs)
        
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        
        ## NOTE: here we assume only use the flip augmentation
        occ_res = self.aug_test_imgs(img_feats, img_metas, rescale)
        return [occ_res]

    def aug_test_imgs(self,
                      feats,
                      img_metas,
                      rescale=False):
        """Test function of point cloud branch with augmentaiton."""
        # only support aug_test for one sample
        aug_occs = []
        for x, img_meta in zip(feats, img_metas):
            occ_pred = self.final_conv(x[0]).permute(0, 4, 3, 2, 1)
            if self.use_predicter:
                occ_pred = self.predicter(occ_pred)
            occ_score = occ_pred.softmax(-1)
            aug_occs.append(occ_score)

        merged_occs = merge_aug_occs(aug_occs)
        return merged_occs

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

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
        mask_camera = kwargs['mask_camera']
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
        losses.update(loss_occ)
        return losses



@DETECTORS.register_module()
class BEVFusionStereo4DOCCDistill(BEVFusionStereo4DOCC):

    def __init__(self,
                 camera_occ_head=None,
                 use_high_feat_align=False,
                 loss_high_feat=dict(
                    type='L1Loss',
                    loss_weight=1.0),
                 **kwargs):
        super(BEVFusionStereo4DOCCDistill, self).__init__(**kwargs)
        
        self.camera_occ_head = builder.build_head(camera_occ_head)

        self.use_high_feat_align = use_high_feat_align
        
        if use_high_feat_align:
            self.loss_high_feat = build_loss(loss_high_feat)

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats, depth = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)  # [(b,384,200,200)]

        return (img_feats, pts_feats, depth)

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        
        occ_pred = self.camera_occ_head(img_feats[0])
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [occ_res]

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        """Forward training function.
        """
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        
        ## apply the lidar-camera fusion
        fusion_feats = self.apply_lc_fusion(img_feats[0], pts_feats)

        ## Compute the teacher losses
        gt_depth = kwargs['gt_depth']
        teacher_losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        teacher_losses['teacher_loss_depth'] = loss_depth

        occ_pred = self.final_conv(fusion_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        
        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']
        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
        teacher_losses['teacher_loss_occ'] = loss_occ['loss_occ']

        ## Compute the student losses
        student_losses = dict()
        student_loss_occ, student_occ_logits = self.camera_occ_head.forward_train(
            img_feats[0], return_logits=True, **kwargs)
        student_losses.update(student_loss_occ)

        ## Compute the distillation losses
        if self.use_high_feat_align:
            high_feat_align_loss = self.compute_feat_align_loss(
                mask_camera, img_feats[0], fusion_feats[0])

        teacher_occ_logits = occ_pred

        kl_loss = self.compute_kl_loss(
            mask_camera, student_occ_logits, teacher_occ_logits,
            detach_target=True)

        losses = dict()
        losses.update(teacher_losses)
        losses.update(student_losses)
        losses['kl_loss'] = kl_loss

        if self.use_high_feat_align:
            losses.update(high_feat_align_loss)
        return losses
    
    def compute_feat_align_loss(self, mask, student_feats, teacher_feats):
        losses = dict()

        mask1 = repeat(mask, 'b h w d -> b c d h w', 
                       c=student_feats.shape[1])
        mask1 = mask1.to(torch.float32)
        num_total_samples1 = mask1.sum()
        high_feat_loss = self.loss_high_feat(
            student_feats, teacher_feats, 
            mask1, avg_factor=num_total_samples1)
        
        losses['high_feat_loss'] = high_feat_loss
        return losses

    def compute_kl_loss(self, mask, student_logits, teacher_logits, 
                        detach_target=False):
        mask = mask.to(torch.bool)
        student_logits = student_logits[mask]
        teacher_logits = teacher_logits[mask]
        student_logits = F.log_softmax(student_logits, dim=1)
        teacher_logits = F.softmax(teacher_logits, dim=1)

        if detach_target:
            teacher_logits = teacher_logits.detach()
        
        kl_loss = F.kl_div(student_logits, teacher_logits, reduction='batchmean')
        return kl_loss