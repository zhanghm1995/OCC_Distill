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
from .. import builder
from .bevdet_occ import BEVStereo4DOCC
from ..losses.loss_utils import sem_scal_loss, geo_scal_loss


@DETECTORS.register_module()
class BEVStereo4DOCCRoboDrive(BEVStereo4DOCC):

    def __init__(self,
                 use_surroundocc_loss=False,
                 **kwargs):
        super(BEVStereo4DOCCRoboDrive, self).__init__(**kwargs)

        assert not self.use_mask, 'mask is not supported for robodrive'

        self.use_surroundocc_loss = use_surroundocc_loss

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

        return loss_
    