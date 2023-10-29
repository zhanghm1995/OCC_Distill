'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-08-09 15:10:05
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import numpy as np
import torch
import torch.nn.functional as F

from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from torch import nn
from .. import builder
from .bevdet import BEVStereo4D


@DETECTORS.register_module()
class BEVStereo4DOCCNeRFDepth(BEVStereo4D):

    def __init__(self,
                 nerf_head=None,
                 loss_occ=None,
                 out_dim=32,
                 use_mask=False,
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,
                 scene_filter_index=-1,
                 refine_conv=False,
                 max_ray_number=80000,
                 **kwargs):
        super(BEVStereo4DOCCNeRFDepth, self).__init__(**kwargs)
        self.out_dim = out_dim
        self.scene_filter_index = scene_filter_index
        out_channels = out_dim if use_predicter else num_classes
        self.NeRFDecoder = builder.build_backbone(nerf_head)
        self.num_classes = num_classes
        self.lovasz = self.NeRFDecoder.lovasz
        if self.NeRFDecoder.img_recon_head:
            self.rgb_conv = nn.Sequential(
                *[
                    ConvModule(
                    self.img_view_transformer.out_channels,
                    3,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                    conv_cfg=dict(type='Conv3d')),
                ])

        self.final_conv = ConvModule(
            self.img_view_transformer.out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv3d'))
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes),
            )

        self.pts_bbox_head = None
        self.use_mask = use_mask
        self.max_ray_number = max_ray_number
        self.loss_occ = build_loss(loss_occ)
        self.class_wise = class_wise
        self.align_after_view_transfromation = False

    def loss_single(self, voxel_semantics, mask_camera, preds, semi_mask):
        loss_ = dict()
        mask_camera *= semi_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            mask_camera = mask_camera.reshape(-1)
            num_total_samples = mask_camera.sum()
            loss_occ = self.loss_occ(preds, voxel_semantics, mask_camera, avg_factor=num_total_samples)
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
        intricics = kwargs['intricics']
        pose_spatial = kwargs['pose_spatial']

        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)

        # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)

        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        occ_pred = occ_pred.permute(0, 4, 1, 2, 3)

        # semantic
        if self.NeRFDecoder.semantic_head:
            semantic = occ_pred[:, :-1]
        else:
            semantic = torch.zeros_like(occ_pred[:, :-1])

        # density
        density_prob = -occ_pred[:, self.NeRFDecoder.semantic_dim: self.NeRFDecoder.semantic_dim+1, ...]

        # image reconstruction
        if self.NeRFDecoder.img_recon_head:
            rgb_recons = self.rgb_conv(img_feats[0])
        else:
            rgb_recons = torch.zeros_like(occ_pred[:, 1:4, ...])

        render_gt_depth = kwargs['render_gt_depth'][0]
        img_semantic = kwargs['img_semantic'][0]
        render_mask = render_gt_depth > 0 if self.NeRFDecoder.mask_render else None

        ### DEBUG ####
        pose = pose_spatial[0]
        for i in np.linspace(-0.5, 0.5, 10):
            pose_temp = pose.clone()
            pose_temp[:, :, :3, -1] += i
            render_depth, rgb_pred, semantic_pred = self.NeRFDecoder(
                density_prob, rgb_recons, semantic, intricics[0], pose_temp, is_train=False, render_mask=render_mask)

            if self.NeRFDecoder.mask_render:
                render_gt_depth = render_gt_depth[render_mask]

            if True:
                render_img_gt = kwargs['render_gt_img'][0]
                current_frame_img = render_img_gt.view(6, self.num_frame, -1, render_img_gt.shape[-2], render_img_gt.shape[-1])[:, 0].cpu().numpy()
                sem = semantic_pred[0].argmax(1)
                NUSCENSE_LIDARSEG_PALETTE = torch.Tensor([
                    (0, 0, 0),  # noise
                    (112, 128, 144),  # barrier
                    (220, 20, 60),  # bicycle
                    (255, 127, 80),  # bus
                    (255, 158, 0),  # car
                    (233, 150, 70),  # construction_vehicle
                    (255, 61, 99),  # motorcycle
                    (0, 0, 230),  # pedestrian
                    (47, 79, 79),  # traffic_cone
                    (255, 140, 0),  # trailer
                    (255, 99, 71),  # Tomato
                    (0, 207, 191),  # nuTonomy green
                    (175, 0, 75),
                    (75, 0, 75),
                    (112, 180, 60),
                    (222, 184, 135),  # Burlywood
                    (0, 175, 0)
                ])
                sem_color = NUSCENSE_LIDARSEG_PALETTE[sem]
                self.NeRFDecoder.visualize_image_semantic_depth_pair(
                    current_frame_img,
                    sem_color,
                    render_depth[0]
                )

                render_gt_depth[render_gt_depth == 0] = 100
                # self.NeRFDecoder.visualize_image_lidar_semantic_depth_pair(
                #     current_frame_img,
                #     render_gt_depth[0],
                #     img_semantic[0],
                #     NUSCENSE_LIDARSEG_PALETTE.numpy()
                # )
            # exit()
        return [[occ_res, render_depth, render_gt_depth]]

    @staticmethod
    def inverse_flip_aug(feat, flip_dx, flip_dy):
        batch_size = feat.shape[0]
        feat_flip = []
        for b in range(batch_size):
            flip_flag_x = flip_dx[b]
            flip_flag_y = flip_dy[b]
            tmp = feat[b]
            if flip_flag_x:
                tmp = tmp.flip(1)
            if flip_flag_y:
                tmp = tmp.flip(2)
            feat_flip.append(tmp)
        feat_flip = torch.stack(feat_flip)
        return feat_flip

    @staticmethod
    def limit_true_count(tensor, max_true_count):
        flattened = tensor.view(-1)
        true_indices = torch.nonzero(flattened).squeeze()
        if true_indices.numel() <= max_true_count:
            return tensor
        selected_indices = true_indices[torch.randperm(true_indices.numel())[:max_true_count]]
        flattened.zero_()
        flattened[selected_indices] = 1
        return tensor.view(tensor.size())

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
        semi_mask = kwargs['scene_number'] < self.scene_filter_index
        intricics = kwargs['intricics']
        pose_spatial = kwargs['pose_spatial']
        batch_size = len(semi_mask)
        flip_dx, flip_dy = kwargs['flip_dx'], kwargs['flip_dy']
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(
            gt_depth.view(batch_size, 6, -1, *gt_depth.shape[2:])[:, :, 0],
            depth, semi_mask)
        losses['loss_depth'] = loss_depth
        render_img_gt = kwargs['render_gt_img']
        render_gt_depth = kwargs['render_gt_depth']

        # occupancy prediction
        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)  # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17

        # occupancy losses
        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred, semi_mask)
        losses.update(loss_occ)

        # NeRF loss
        if False:  # DEBUG ONLY!
            density_prob = kwargs['voxel_semantics'].unsqueeze(1)
            # img_semantic = kwargs['img_semantic']
            density_prob = density_prob != 17
            density_prob = density_prob.float()
            density_prob[density_prob == 0] = -10  # scaling to avoid 0 in alphas
            density_prob[density_prob == 1] = 10
            batch_size = density_prob.shape[0]
            density_prob_flip = self.inverse_flip_aug(density_prob, flip_dx, flip_dy)
            print(density_prob_flip.shape)
            print(intricics.shape)

            gt_depth = gt_depth.view(batch_size, -1, 1, *gt_depth.shape[2:])
            intricics = intricics.view(batch_size, -1, 1, *intricics.shape[2:])
            pose_spatial = pose_spatial.view(batch_size, -1, 1, *pose_spatial.shape[2:])
            render_img_gt = render_img_gt.view(batch_size, 6, self.num_frame, -1, render_img_gt.shape[-2], render_img_gt.shape[-1])
            render_gt_depth = render_gt_depth.view(batch_size, -1, 1, *render_gt_depth.shape[2:])

            seq = 0
            # nerf decoder
            render_depth, _, _ = self.NeRFDecoder(
                density_prob_flip,
                density_prob_flip.tile(1, 3, 1, 1, 1),
                density_prob_flip.tile(1, self.NeRFDecoder.semantic_dim, 1, 1, 1),
                intricics[:, :, seq], pose_spatial[:, :, seq], True
            )
            print('density_prob', density_prob.shape)  # [1, 1, 200, 200, 16]
            print('render_depth', render_depth.shape, render_depth.max(), render_depth.min())  # [1, 6, 224, 352]
            print('render_gt_depth', render_gt_depth.shape, render_gt_depth.max(), render_gt_depth.min())  # [1, 6, 224, 352]
            print('gt_depth', gt_depth.shape, gt_depth.max(), gt_depth.min())  # [1, 6, 384, 704]
            print(render_img_gt.shape)
            self.NeRFDecoder.visualize_image_depth_pair(render_img_gt[0, :, seq].cpu().numpy(), render_gt_depth[0, :, seq], render_depth[0])
            exit()

        else:
            occ_pred = occ_pred.permute(0, 4, 1, 2, 3)

            if self.NeRFDecoder.semantic_head:
                semantic = occ_pred[:, :-1]
            else:
                semantic = torch.zeros_like(occ_pred[:, :-1])

            # density
            density_prob = -occ_pred[:, self.NeRFDecoder.semantic_dim: self.NeRFDecoder.semantic_dim + 1, ...]

            # image reconstruction
            if self.NeRFDecoder.img_recon_head:
                rgb_recons = self.rgb_conv(img_feats[0])
            else:
                rgb_recons = torch.zeros_like(occ_pred[:, 1:4, ...])

            # cancel the effect of flip augmentation
            rgb_flip = self.inverse_flip_aug(rgb_recons, flip_dx, flip_dy)
            semantic_flip = self.inverse_flip_aug(semantic, flip_dx, flip_dy)
            density_prob_flip = self.inverse_flip_aug(density_prob, flip_dx, flip_dy)

            # random view selection
            if self.NeRFDecoder.mask_render:
                render_mask = render_gt_depth > 0.0
                render_mask = self.limit_true_count(render_mask, self.max_ray_number)
                render_depth, rgb_pred, semantic_pred = self.NeRFDecoder(density_prob_flip, rgb_flip, semantic_flip, intricics, pose_spatial, True, render_mask)

                # loss calculation
                render_gt_depth = render_gt_depth[render_mask]
                loss_nerf = self.NeRFDecoder.compute_depth_loss(render_depth, render_gt_depth, torch.ones_like(render_depth).bool())
                if torch.isnan(loss_nerf):
                    print('NaN in DepthNeRF loss!')
                    loss_nerf = loss_depth
                losses['loss_nerf'] = loss_nerf

                if self.NeRFDecoder.semantic_head:
                    img_semantic = kwargs['img_semantic']
                    img_semantic = img_semantic[render_mask]
                    loss_nerf_sem = self.NeRFDecoder.compute_semantic_loss_flatten(semantic_pred, img_semantic, lovasz=self.lovasz)
                    if torch.isnan(loss_nerf):
                        print('NaN in SemNeRF loss!')
                        loss_nerf_sem = loss_depth
                    losses['loss_nerf_sem'] = loss_nerf_sem

                if self.NeRFDecoder.img_recon_head:
                    batch_size, num_camera = intricics.shape[:2]
                    img_gts = render_img_gt.view(batch_size, num_camera, self.num_frame, -1, render_img_gt.shape[-2], render_img_gt.shape[-1])[:, :, 0]
                    img_gts = img_gts.permute(0, 1, 3, 4, 2)[render_mask]
                    loss_nerf_img = self.NeRFDecoder.compute_image_loss(rgb_pred, img_gts)
                    if torch.isnan(loss_nerf):
                        print('NaN in ImgNeRF loss!')
                        loss_nerf_img = loss_depth
                    losses['loss_nerf_img'] = loss_nerf_img

            else:
                render_depth, rgb_pred, semantic_pred = self.NeRFDecoder(density_prob_flip, rgb_flip, semantic_flip, intricics, pose_spatial, True)
                batch_size, num_camera, _, H, W = rgb_pred.shape

                # nerf loss calculation
                loss_nerf = self.NeRFDecoder.compute_depth_loss(render_depth, render_gt_depth, render_gt_depth > 0.0)

                if True:
                    sem = semantic_pred[0].argmax(1)
                    NUSCENSE_LIDARSEG_PALETTE = torch.Tensor([
                        (0, 0, 0),  # noise
                        (112, 128, 144),  # barrier
                        (220, 20, 60),  # bicycle
                        (255, 127, 80),  # bus
                        (255, 158, 0),  # car
                        (233, 150, 70),  # construction_vehicle
                        (255, 61, 99),  # motorcycle
                        (0, 0, 230),  # pedestrian
                        (47, 79, 79),  # traffic_cone
                        (255, 140, 0),  # trailer
                        (255, 99, 71),  # Tomato
                        (0, 207, 191),  # nuTonomy green
                        (175, 0, 75),
                        (75, 0, 75),
                        (112, 180, 60),
                        (222, 184, 135),  # Burlywood
                        (0, 175, 0)
                    ])
                    sem_color = NUSCENSE_LIDARSEG_PALETTE[sem]
                    img_gts = render_img_gt.view(batch_size, num_camera, self.num_frame, -1, render_img_gt.shape[-2],
                                                 render_img_gt.shape[-1])[:, :, 0]

                    self.NeRFDecoder.visualize_image_semantic_depth_pair(
                        # rgb_pred[0].cpu().numpy(),
                        img_gts[0].cpu().numpy(),
                        sem_color.detach(),
                        render_depth[0].detach()
                    )

                if torch.isnan(loss_nerf):
                    print('NaN in DepthNeRF loss!')
                    loss_nerf = loss_depth
                losses['loss_nerf'] = loss_nerf

                if self.NeRFDecoder.semantic_head:
                    img_semantic = kwargs['img_semantic']
                    loss_nerf_sem = self.NeRFDecoder.compute_semantic_loss(semantic_pred, img_semantic)
                    if torch.isnan(loss_nerf):
                        print('NaN in SemNeRF loss!')
                        loss_nerf_sem = loss_depth
                    losses['loss_nerf_sem'] = loss_nerf_sem

                if self.NeRFDecoder.img_recon_head:
                    img_gts = render_img_gt.view(batch_size, num_camera, self.num_frame, -1, render_img_gt.shape[-2], render_img_gt.shape[-1])[:, :, 0]
                    loss_nerf_img = self.NeRFDecoder.compute_image_loss(rgb_pred, img_gts)
                    if torch.isnan(loss_nerf):
                        print('NaN in ImgNeRF loss!')
                        loss_nerf_img = loss_depth
                    losses['loss_nerf_img'] = loss_nerf_img

        return losses
