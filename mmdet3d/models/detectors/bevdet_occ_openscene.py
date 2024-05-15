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
from mmdet3d.models.losses.lovasz_loss import Lovasz_loss
from .bevdet_occ import BEVStereo4DOCC, BEVFusionStereo4DOCC
from .bevdet_occ_ssc import SSCNet

# openscene-nuscenes
nusc_class_frequencies = np.array([124531392, 0, 0, 0, 68091, 271946, 19732177, 18440931, 
    2131710, 941461, 1774002271, 24881000021])

@DETECTORS.register_module()
class BEVStereo4DOCCOpenScene(BEVStereo4DOCC):
    def __init__(self,
                 pred_binary_occ=False,
                 use_lovasz_loss=False,
                 balance_cls_weight=False,
                 use_sscnet=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.pred_binary_occ = pred_binary_occ

        if use_sscnet:
            assert self.use_predicter
            
            self.final_conv = SSCNet(
                self.img_view_transformer.out_channels,
                [128, 128, 128, 128, 128],
                self.out_dim)

        self.use_lovasz_loss = use_lovasz_loss 
        if use_lovasz_loss:
            self.loss_lovasz = Lovasz_loss(255)

        assert not self.use_mask, 'visibility mask is not supported for OpenScene dataset'

        if balance_cls_weight:
            class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:12] + 0.001)).float()
            self.loss_occ = nn.CrossEntropyLoss(
                    weight=class_weights, reduction="mean"
                )

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
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
       
    def loss_single(self, voxel_semantics, preds):
        loss_ = dict()
        voxel_semantics = voxel_semantics.long()
        voxel_semantics = voxel_semantics.reshape(-1)
        preds = preds.reshape(-1, self.num_classes)

        if not self.pred_binary_occ:
            loss_occ = self.loss_occ(preds, voxel_semantics)
        else:
            # predict binary occupancy, 1 is occupied, 0 is free
            density_target = (voxel_semantics != 11).long()
            loss_occ = self.loss_occ(preds, density_target)
        
        if self.use_lovasz_loss:
            loss_lovasz = self.loss_lovasz(F.softmax(preds, dim=1), 
                                           voxel_semantics)
            
            loss_['loss_lovasz'] = loss_lovasz
        
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
        
        if not self.pred_binary_occ:
            occ_score = occ_pred.softmax(-1)
            occ_res = occ_score.argmax(-1)
            occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        else:
            # density_score = occ_pred.softmax(-1)
            # no_empty_mask = density_score[..., 0] > density_score[..., 1]

            no_empty_mask = occ_pred.argmax(-1)
            occ_res = no_empty_mask.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [occ_res]
    


@DETECTORS.register_module()
class BEVFusionStereo4DOCCOpenScene(BEVFusionStereo4DOCC):
    """Fuse the lidar and camera images for occupancy prediction.

    Args:
        BEVFusionStereo4DOCCOpenScene (_type_): _description_
    """

    def __init__(self,
                 pred_binary_occ=False,
                 **kwargs):
        super(BEVFusionStereo4DOCCOpenScene, self).__init__(**kwargs)

        self.pred_binary_occ = pred_binary_occ

        assert not self.use_mask, 'visibility mask is not supported for OpenScene dataset'
        
    def loss_single(self, voxel_semantics, preds):
        loss_ = dict()
        voxel_semantics = voxel_semantics.long()
        voxel_semantics = voxel_semantics.reshape(-1)
        preds = preds.reshape(-1, self.num_classes)

        if not self.pred_binary_occ:
            loss_occ = self.loss_occ(preds, voxel_semantics)
        else:
            # predict binary occupancy, 1 is occupied, 0 is free
            density_target = (voxel_semantics != 11).long()
            loss_occ = self.loss_occ(preds, density_target)
        
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
        
        if not self.pred_binary_occ:
            occ_score = occ_pred.softmax(-1)
            occ_res = occ_score.argmax(-1)
            occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        else:
            # density_score = occ_pred.softmax(-1)
            # no_empty_mask = density_score[..., 0] > density_score[..., 1]

            no_empty_mask = occ_pred.argmax(-1)
            occ_res = no_empty_mask.squeeze(dim=0).cpu().numpy().astype(np.uint8)
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
    