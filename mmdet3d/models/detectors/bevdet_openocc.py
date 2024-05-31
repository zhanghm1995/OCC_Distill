'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-05-31 01:21:20
Email: haimingzhang@link.cuhk.edu.cn
Description: Adapt for the OpenOcc dataset.
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
from mmdet3d.models.losses.dice_loss import CustomDiceLoss
from mmdet.models.builder import build_loss
from .bevdet_occ import BEVStereo4DOCC, BEVFusionStereo4DOCC
from .bevdet import BEVDepth4D
from .bevdet_occ_ssc import SSCNet
from .. import builder

openocc_class_frequencies = np.array([
    75432995,
    25240097,
    12970402,
    10834080,
    4911253,
    313485,
    425131,
    4285461,
    701594,
    3848670,
    384886109,
    10222682,
    123337011,
    171792855,
    315173113,
    372498065,
    16486326997
 ])
    

@DETECTORS.register_module()
class BEVStereo4DOpenOcc(BEVStereo4DOCC):
    """BEVDet for occupancy prediction task in OpenOcc dataset.

    Args:
        BEVStereo4DOCC (_type_): _description_
    """
    def __init__(self,
                 pred_binary_occ=False,
                 use_lovasz_loss=False,
                 use_dice_loss=False,
                 balance_cls_weight=False,
                 loss_occ_weight=1.0,
                 use_sscnet=False,
                 use_sdb=False,
                 final_conv_cfg=None,
                 use_loss_norm=False,
                 pred_flow=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.pred_binary_occ = pred_binary_occ
        self.use_loss_norm = use_loss_norm

        assert not self.use_mask, 'visibility mask is not supported for OpenOcc dataset'

        if use_sscnet:
            assert self.use_predicter
            
            self.final_conv = SSCNet(
                self.img_view_transformer.out_channels,
                [128, 128, 128, 128, 128],
                self.out_dim)
        elif use_sdb:
            self.final_conv = builder.build_middle_encoder(final_conv_cfg)

        self.use_lovasz_loss = use_lovasz_loss 
        if use_lovasz_loss:
            self.loss_lovasz = Lovasz_loss(255)
        if use_dice_loss:
            assert pred_binary_occ, 'Now Dice loss is only used for binary occupancy prediction'
            self.locc_cc = CustomDiceLoss(mode='multiclass')

        # we use this weight when we set balance_cls_weight to true
        self.loss_occ_weight = loss_occ_weight  if balance_cls_weight else 1.0

        if balance_cls_weight:
            class_weights = torch.from_numpy(1 / np.log(openocc_class_frequencies[:18] + 0.001)).float()
            self.loss_occ = nn.CrossEntropyLoss(
                    weight=class_weights, reduction="mean"
                )
        
        self.pred_flow = pred_flow
        if pred_flow:
            self.flow_predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, 2),
            )
            self.flow_loss = builder.build_loss(dict(type='L1Loss', loss_weight=0.25))

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        """Forward training function.
        """
        img_feats, _, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth

        volume_feat = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc

        ## predict the occupancy
        occ_pred = self.predicter(volume_feat)
        voxel_semantics = kwargs['voxel_semantics']
        loss_occ = self.loss_single(voxel_semantics, occ_pred)
        losses.update(loss_occ)

        ## predict the flow
        if self.pred_flow:
            flow_pred = self.flow_predicter(volume_feat)
            flow_gt = kwargs['voxel_flow']
            loss_flow = self.flow_loss(flow_pred, flow_gt)
            losses['loss_flow'] = loss_flow

        # use the loss norm trick
        if self.use_loss_norm:
            for key, value in losses.items():
                losses[key] = value / (value.detach() + 1e-5)
        
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

        loss_['loss_occ'] = self.loss_occ_weight * loss_occ
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
        volume_feat = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)

        ## predict the occupancy
        occ_pred = self.predicter(volume_feat)
        
        if not self.pred_binary_occ:
            occ_score = occ_pred.softmax(-1)
            occ_res = occ_score.argmax(-1)
            occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        else:
            density_score = occ_pred.softmax(-1)
            no_empty_mask = density_score[..., 0] < density_score[..., 1] + 0.4

            # no_empty_mask = occ_pred.argmax(-1)
            occ_res = no_empty_mask.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        
        ## predict the flow
        if self.pred_flow:
            flow_pred = self.flow_predicter(volume_feat)

        result_dict = dict()
        result_dict['occ_pred'] = occ_res
        if self.pred_flow:
            result_dict['flow_pred'] = flow_pred
        return [result_dict]