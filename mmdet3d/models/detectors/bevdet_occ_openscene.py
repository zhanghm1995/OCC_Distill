'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-05-08 16:01:11
Email: haimingzhang@link.cuhk.edu.cn
Description: For OpenScene dataset.
'''
import torch
from mmdet.models import DETECTORS
from mmcv.cnn.bricks.conv_module import ConvModule
from mmcv.runner import force_fp32

from torch import nn
import numpy as np
from torch.nn import functional as F
from einops import repeat, rearrange

from .bevdet_occ import BEVFusionStereo4DOCC


@DETECTORS.register_module()
class BEVFusionStereo4DOCCOpenScene(BEVFusionStereo4DOCC):
    """Fuse the lidar and camera images for occupancy prediction.

    Args:
        BEVFusionStereo4DOCCOpenScene (_type_): _description_
    """

    def __init__(self,
                 **kwargs):
        super(BEVFusionStereo4DOCCOpenScene, self).__init__(**kwargs)

        assert not self.use_mask, 'visibility mask is not supported for OpenScene dataset'
        
    def loss_single(self, voxel_semantics, preds):
        loss_ = dict()
        voxel_semantics=voxel_semantics.long()
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
        