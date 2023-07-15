'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-07-15 21:19:28
Email: haimingzhang@link.cuhk.edu.cn
Description: Pretrain the model with binary occupancy predictions.
'''

import torch

from mmdet.models import DETECTORS

from .bevdet_occ import BEVStereo4DOCC


@DETECTORS.register_module()
class BEVStereo4DOCCPretrain(BEVStereo4DOCC):

    def __init__(self,
                 **kwargs):
        super(BEVStereo4DOCCPretrain, self).__init__(**kwargs)
    
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
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth

        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        
        ## Start compute the losses
        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']
        ## Here we change the voxel_semantics to binary occupancy predictions
        voxel_semantics = (voxel_semantics < 17).float()

        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
        losses.update(loss_occ)
        return losses
    


