'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-07-07 09:48:05
Email: haimingzhang@link.cuhk.edu.cn
Description: Use the SSCNet in the BEVDetOCC.
'''

from .bevdet import BEVStereo4D

import torch
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from torch import nn
import numpy as np
import torch.nn.functional as F
from mmdet3d.models.losses.lovasz_loss import Lovasz_loss


class SSCNet(nn.Module):
    def __init__(self, input_dim, nPlanes, classes):
        super().__init__()
        # Block 1
        self.b1_conv1 = nn.Sequential(nn.Conv3d(input_dim, 16, 3, 1, padding=1), nn.BatchNorm3d(16), nn.ReLU())
        self.b1_conv2 = nn.Sequential(nn.Conv3d(16, nPlanes[0], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[0]), nn.ReLU())
        self.b1_conv3 = nn.Sequential(nn.Conv3d(nPlanes[0], nPlanes[0], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[0]),
                                      nn.ReLU())
        self.b1_res = nn.Sequential(nn.Conv3d(16, nPlanes[0], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[0]), nn.ReLU())
        self.pool1 = nn.Sequential(nn.MaxPool3d(2, 2))

        # Block 2
        self.b2_conv1 = nn.Sequential(nn.Conv3d(nPlanes[0], nPlanes[1], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[1]),
                                      nn.ReLU())
        self.b2_conv2 = nn.Sequential(nn.Conv3d(nPlanes[1], nPlanes[1], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[1]),
                                      nn.ReLU())
        self.b2_res = nn.Sequential(nn.Conv3d(nPlanes[0], nPlanes[1], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[1]),
                                    nn.ReLU())

        # Block 3
        self.b3_conv1 = nn.Sequential(nn.Conv3d(nPlanes[1], nPlanes[2], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[2]),
                                      nn.ReLU())
        self.b3_conv2 = nn.Sequential(nn.Conv3d(nPlanes[2], nPlanes[2], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[2]),
                                      nn.ReLU())

        # Block 4
        self.b4_conv1 = nn.Sequential(nn.Conv3d(nPlanes[2], nPlanes[3], 3, 1, dilation=2, padding=2),
                                      nn.BatchNorm3d(nPlanes[3]), nn.ReLU())
        self.b4_conv2 = nn.Sequential(nn.Conv3d(nPlanes[3], nPlanes[3], 3, 1, dilation=2, padding=2),
                                      nn.BatchNorm3d(nPlanes[3]), nn.ReLU())

        # Block 5
        self.b5_conv1 = nn.Sequential(nn.Conv3d(nPlanes[3], nPlanes[4], 3, 1, dilation=2, padding=2),
                                      nn.BatchNorm3d(nPlanes[4]), nn.ReLU())
        self.b5_conv2 = nn.Sequential(nn.Conv3d(nPlanes[4], nPlanes[4], 3, 1, dilation=2, padding=2),
                                      nn.BatchNorm3d(nPlanes[4]), nn.ReLU())

        # Prediction
        self.pre_conv1 = nn.Sequential(
            nn.Conv3d(nPlanes[2] + nPlanes[3] + nPlanes[4], int((nPlanes[2] + nPlanes[3] + nPlanes[4]) / 3 * 2), 1, 1), \
            nn.BatchNorm3d(int((nPlanes[2] + nPlanes[3] + nPlanes[4]) / 3 * 2)), nn.ReLU())
        self.pre_conv2 = nn.Sequential(nn.Conv3d(int((nPlanes[2] + nPlanes[3] + nPlanes[4]) / 3 * 2), classes, 1, 1))


    def forward(self, x):
        # Block 1
        x = self.b1_conv1(x)
        res_x = self.b1_res(x)
        x = self.b1_conv2(x)
        x = self.b1_conv3(x)
        x = x + res_x

        # Block 2
        res_x = self.b2_res(x)
        x = self.b2_conv1(x)
        x = self.b2_conv2(x)
        x = x + res_x

        # Block 3
        b3_x1 = self.b3_conv1(x)
        b3_x2 = self.b3_conv2(b3_x1)
        b3_x = b3_x1 + b3_x2

        # Block 4
        b4_x1 = self.b4_conv1(b3_x)
        b4_x2 = self.b4_conv2(b4_x1)
        b4_x = b4_x1 + b4_x2

        # Block 5
        b5_x1 = self.b5_conv1(b4_x)
        b5_x2 = self.b5_conv2(b5_x1)
        b5_x = b5_x1 + b5_x2

        # Concat b3,b4,b5
        x = torch.cat((b3_x, b4_x, b5_x), dim=1)

        # Prediction
        x = self.pre_conv1(x)
        x = self.pre_conv2(x)

        return x

@DETECTORS.register_module()
class BEVStereo4DSSCOCC(BEVStereo4D):

    def __init__(self,
                 loss_occ=None,
                 out_dim=32,
                 use_mask=False,
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,
                 use_lovasz=True,
                 **kwargs):
        
        super(BEVStereo4DSSCOCC, self).__init__(**kwargs)

        self.out_dim = out_dim
        out_channels = out_dim if use_predicter else num_classes
        self.final_conv = SSCNet(
            self.img_view_transformer.out_channels,
            [128, 128, 128, 128, 128],
            out_channels)
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes),
            )
        self.pts_bbox_head = None
        self.use_lovasz = use_lovasz
        self.use_mask = use_mask
        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)
        self.class_wise = class_wise
        self.align_after_view_transfromation = False
        self.loss_lovasz = Lovasz_loss(255)

    def loss_single(self, voxel_semantics, mask_camera, preds):
        loss_ = dict()
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            mask_camera = mask_camera.reshape(-1)
            num_total_samples = mask_camera.sum()
            loss_occ = self.loss_occ(preds, voxel_semantics, mask_camera, avg_factor=num_total_samples)
            if self.use_lovasz:
                voxel_semantics_mask = torch.ones_like(voxel_semantics) * 255
                voxel_semantics_mask[mask_camera == 1] = voxel_semantics[mask_camera == 1]
                loss_occ += self.loss_lovasz(F.softmax(preds, dim=1), voxel_semantics_mask)
            loss_['loss_occ'] = loss_occ
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics, )
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
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
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
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth

        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)  # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
        losses.update(loss_occ)
        return losses

