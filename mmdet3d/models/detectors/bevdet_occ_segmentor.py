'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-07-31 23:09:39
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
from mmdet3d.models.losses.lovasz_loss import Lovasz_loss
from .. import builder
from .bevdet import BEVStereo4D
from mmdet3d.models.detectors.bevdet_occ_ssc import SSCNet


def sparse_quantize(pc, coors_range, spatial_shape):
    idx = spatial_shape * (pc - coors_range[0]) / (coors_range[1] - coors_range[0])
    return idx.long()

def gen_point_range_mask(pts, coors_range_xyz=[[-40, 40], [-40, 40], [-1, 5.4]]):
    mask11 = pts[:, 0] < coors_range_xyz[0][1]
    mask12 = pts[:, 0] > coors_range_xyz[0][0]
    mask21 = pts[:, 1] < coors_range_xyz[1][1]
    mask22 = pts[:, 1] > coors_range_xyz[1][0]
    mask31 = pts[:, 2] < coors_range_xyz[2][1]
    mask32 = pts[:, 2] > coors_range_xyz[2][0]

    return mask11 & mask12 & mask21 & mask22 & mask31 & mask32


def voxelization(point, batch_idx, coors_range_xyz=[[-40, 40], [-40, 40], [-1, 5.4]], spatial_shape=[200, 200, 16]):
    xidx = sparse_quantize(point[:, 0], coors_range_xyz[0], spatial_shape[0])
    yidx = sparse_quantize(point[:, 1], coors_range_xyz[1], spatial_shape[1])
    zidx = sparse_quantize(point[:, 2], coors_range_xyz[2], spatial_shape[2])

    bxyz_indx = torch.stack([batch_idx, xidx, yidx, zidx], dim=-1).long()
    unq, unq_inv, unq_cnt = torch.unique(bxyz_indx, return_inverse=True, return_counts=True, dim=0)
    return unq, unq_inv


@DETECTORS.register_module()
class BEVStereo4DOCCSegmentor(BEVStereo4D):

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
                 coors_range_xyz=None,
                 spatial_shape=None,
                 max_ray_number=80000,
                 **kwargs):
        super(BEVStereo4DOCCSegmentor, self).__init__(**kwargs)
        self.out_dim = out_dim
        self.scene_filter_index = scene_filter_index
        out_channels = out_dim if use_predicter else num_classes
        self.num_classes = num_classes
        self.coors_range_xyz = coors_range_xyz
        self.spatial_shape = spatial_shape

        if nerf_head is not None:
            self.NeRFDecoder = builder.build_backbone(nerf_head)
            self.lovasz = self.NeRFDecoder.lovasz
            if self.NeRFDecoder.img_recon_head:
                self.rgb_conv = nn.Sequential(
                    *[
                        ConvModule(
                        self.img_view_transformer.out_channels,
                        16,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        conv_cfg=dict(type='Conv3d')),
                        ConvModule(
                            16,
                            3,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                            conv_cfg=dict(type='Conv3d'),
                            act_cfg=dict(type='Sigmoid')
                        ),
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
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)

        # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)

        # map voxel prediction onto point cloud
        pc = points[0][:, :3] # (N, 3)
        pc_seg = torch.zeros(pc.shape[0], occ_pred.shape[-1]).to(pc.device) # (N, C)
        mask = gen_point_range_mask(pc, coors_range_xyz=self.coors_range_xyz)
        unq, unq_inv = voxelization(
            point=pc[mask],
            batch_idx=torch.zeros(pc[mask].shape[0]).to(pc.device),
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape,
        )
        point_pred = occ_pred[unq[:, 0], unq[:, 1], unq[:, 2], unq[:, 3]][unq_inv]
        pc_seg[mask] = point_pred

        # ignore free category and then conduct softmax
        occ_score = pc_seg.softmax(-1)
        occ_res = occ_score[:, :-1].argmax(-1)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)

        # output lidarseg label
        lidarseg = kwargs['lidarseg_pad'][0][0][:occ_res.shape[0]]
        lidarseg = lidarseg.cpu().numpy()
        assert len(occ_res) == len(lidarseg)

        occ_res = occ_res[mask.cpu().numpy()]
        lidarseg = lidarseg[mask.cpu().numpy()]
        return [[occ_res, lidarseg]]

        # ### DEBUG ###
        # occ_score = occ_pred.softmax(-1)
        # occ_res = occ_score.argmax(-1)
        # occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        # return [occ_res]

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
            depth)
        losses['loss_depth'] = loss_depth
        render_img_gt = kwargs['render_gt_img']
        render_gt_depth = kwargs['render_gt_depth']

        # occupancy prediction
        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)  # bncdhw->bnwhdc

        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)

        # map point cloud label to voxel
        lidarseg = kwargs['lidarseg_pad']
        points_list = []
        batch_list = []
        lidarseg_label = []
        for idx, pc in enumerate(points):
            pc = pc[:, :3]
            mask = gen_point_range_mask(pc, coors_range_xyz=self.coors_range_xyz)
            label = lidarseg[idx][:pc.shape[0]]
            pc = pc[mask]
            points_list.append(pc)
            batch_list.append(torch.ones_like(pc[:, 0]) * idx)
            lidarseg_label.append(label[mask])
        points_list = torch.cat(points_list, 0)
        batch_list = torch.cat(batch_list, 0)
        lidarseg_label = torch.cat(lidarseg_label, 0)
        unq, unq_inv = voxelization(
            point=points_list,
            batch_idx=batch_list,
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape,
        )

        # scatter label
        M = unq_inv.max() + 1
        voxel_lidarseg_label = torch.zeros(M, device=lidarseg_label.device)
        voxel_lidarseg_label = voxel_lidarseg_label.scatter(0, unq_inv, lidarseg_label)

        # generate voxel label and mask
        voxel_semantics = torch.zeros_like(occ_pred.argmax(-1)).detach()
        mask_camera = torch.zeros_like(occ_pred.argmax(-1)).detach()
        voxel_semantics[unq[:, 0], unq[:, 1], unq[:, 2], unq[:, 3]] = voxel_lidarseg_label.long()
        mask_camera[unq[:, 0], unq[:, 1], unq[:, 2], unq[:, 3]] = 1
        mask_camera[voxel_semantics == 0] = 0

        # occupancy losses
        occ_pred = self.inverse_flip_aug(occ_pred, flip_dx, flip_dy)
        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred, semi_mask)
        losses.update(loss_occ)

        # NeRF loss
        use_NeRF_loss = False
        if use_NeRF_loss:
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
                    semantic = occ_pred
                else:
                    semantic = torch.zeros_like(occ_pred)

                # density
                density_prob = kwargs['voxel_semantics'].unsqueeze(1)
                density_prob = density_prob != 17
                density_prob = density_prob.float()
                density_prob[density_prob == 0] = -10  # scaling to avoid 0 in alphas
                density_prob[density_prob == 1] = 10

                # image reconstruction
                if self.NeRFDecoder.img_recon_head:
                    rgb_recons = self.rgb_conv(img_feats[0])
                else:
                    rgb_recons = torch.zeros_like(occ_pred[:, 1:4, ...])

                # cancel the effect of flip augmentation
                rgb_flip = self.inverse_flip_aug(rgb_recons, flip_dx, flip_dy)

                # random view selection
                if self.NeRFDecoder.mask_render:
                    render_mask = render_gt_depth > 0.0
                    render_mask = self.limit_true_count(render_mask, self.max_ray_number)
                    render_depth, rgb_pred, semantic_pred = self.NeRFDecoder(density_prob, rgb_flip, semantic, intricics, pose_spatial, True, render_mask)

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
                        img_mean = img_gts.new_tensor([0.485, 0.456, 0.406])[None, None, :, None, None]
                        img_std = img_gts.new_tensor([0.229, 0.224, 0.225])[None, None, :, None, None]
                        img_gts = img_gts * img_std + img_mean
                        img_gts = img_gts.permute(0, 1, 3, 4, 2)[render_mask]
                        loss_nerf_img = self.NeRFDecoder.compute_image_loss(rgb_pred, img_gts)
                        if torch.isnan(loss_nerf):
                            print('NaN in ImgNeRF loss!')
                            loss_nerf_img = loss_depth
                        losses['loss_nerf_img'] = loss_nerf_img

                else:
                    render_depth, rgb_pred, semantic_pred = self.NeRFDecoder(density_prob, rgb_flip, semantic, intricics, pose_spatial, True)
                    batch_size, num_camera, _, H, W = rgb_pred.shape

                    # nerf loss calculation
                    loss_nerf = self.NeRFDecoder.compute_depth_loss(render_depth, render_gt_depth, render_gt_depth > 0.0)
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
                        img_mean = img_gts.new_tensor([0.485, 0.456, 0.406])[None, None, :, None, None]
                        img_std = img_gts.new_tensor([0.229, 0.224, 0.225])[None, None, :, None, None]
                        img_gts = img_gts * img_std + img_mean
                        loss_nerf_img = self.NeRFDecoder.compute_image_loss(rgb_pred, img_gts)
                        if torch.isnan(loss_nerf):
                            print('NaN in ImgNeRF loss!')
                            loss_nerf_img = loss_depth
                        losses['loss_nerf_img'] = loss_nerf_img

        return losses


@DETECTORS.register_module()
class BEVStereo4DSSCOCCSegmentor(BEVStereo4D):
    """The lidar segmentation task using SSCNet.

    Args:
        BEVStereo4D (_type_): _description_
    """

    def __init__(self,
                 loss_occ=None,
                 out_dim=32,
                 use_mask=False,
                 num_classes=17,
                 use_predicter=True,
                 class_wise=False,
                 use_lovasz=True,
                 coors_range_xyz=None,
                 spatial_shape=None,
                 **kwargs):
        
        super(BEVStereo4DSSCOCCSegmentor, self).__init__(**kwargs)

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
        self.coors_range_xyz = coors_range_xyz
        self.spatial_shape = spatial_shape

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
        
        # map voxel prediction onto point cloud
        pc = points[0][:, :3] # (N, 3)
        pc_seg = torch.zeros(pc.shape[0], occ_pred.shape[-1]).to(pc.device) # (N, C)
        mask = gen_point_range_mask(pc, coors_range_xyz=self.coors_range_xyz)
        unq, unq_inv = voxelization(
            point=pc[mask],
            batch_idx=torch.zeros(pc[mask].shape[0]).to(pc.device),
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape,
        )
        point_pred = occ_pred[unq[:, 0], unq[:, 1], unq[:, 2], unq[:, 3]][unq_inv]
        pc_seg[mask] = point_pred

        # ignore free category and then conduct softmax
        occ_score = pc_seg.softmax(-1)
        occ_res = occ_score[:, :-1].argmax(-1)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)

        # output lidarseg label
        lidarseg = kwargs['lidarseg_pad'][0][0][:occ_res.shape[0]]
        lidarseg = lidarseg.cpu().numpy()
        assert len(occ_res) == len(lidarseg)

        occ_res = occ_res[mask.cpu().numpy()]
        lidarseg = lidarseg[mask.cpu().numpy()]
        return [[occ_res, lidarseg]]

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
        flip_dx, flip_dy = kwargs['flip_dx'], kwargs['flip_dy']

        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth

        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)  # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        
        # map point cloud label to voxel
        lidarseg = kwargs['lidarseg_pad']
        points_list = []
        batch_list = []
        lidarseg_label = []
        for idx, pc in enumerate(points):
            pc = pc[:, :3]
            mask = gen_point_range_mask(pc, coors_range_xyz=self.coors_range_xyz)
            label = lidarseg[idx][:pc.shape[0]]
            pc = pc[mask]
            points_list.append(pc)
            batch_list.append(torch.ones_like(pc[:, 0]) * idx)
            lidarseg_label.append(label[mask])
        points_list = torch.cat(points_list, 0)
        batch_list = torch.cat(batch_list, 0)
        lidarseg_label = torch.cat(lidarseg_label, 0)
        unq, unq_inv = voxelization(
            point=points_list,
            batch_idx=batch_list,
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape,
        )

        # scatter label
        M = unq_inv.max() + 1
        voxel_lidarseg_label = torch.zeros(M, device=lidarseg_label.device)
        voxel_lidarseg_label = voxel_lidarseg_label.scatter(0, unq_inv, lidarseg_label)

        # generate voxel label and mask
        voxel_semantics = torch.zeros_like(occ_pred.argmax(-1)).detach()
        mask_camera = torch.zeros_like(occ_pred.argmax(-1)).detach()
        voxel_semantics[unq[:, 0], unq[:, 1], unq[:, 2], unq[:, 3]] = voxel_lidarseg_label.long()
        mask_camera[unq[:, 0], unq[:, 1], unq[:, 2], unq[:, 3]] = 1
        mask_camera[voxel_semantics == 0] = 0

        # occupancy losses
        occ_pred = self.inverse_flip_aug(occ_pred, flip_dx, flip_dy)
        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
        losses.update(loss_occ)
        return losses
    
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
    

@DETECTORS.register_module()
class BEVStereo4DOCCSegmentorDense(BEVStereo4D):

    def __init__(self,
                 nerf_head=None,
                 loss_occ=None,
                 out_dim=32,
                 use_mask=False,
                 num_classes=17,
                 use_predicter=True,
                 class_wise=False,
                 scene_filter_index=-1,
                 refine_conv=False,
                 coors_range_xyz=None,
                 spatial_shape=None,
                 max_ray_number=80000,
                 **kwargs):
        super(BEVStereo4DOCCSegmentorDense, self).__init__(**kwargs)
        self.out_dim = out_dim
        self.scene_filter_index = scene_filter_index
        out_channels = out_dim if use_predicter else num_classes
        self.num_classes = num_classes
        self.coors_range_xyz = coors_range_xyz
        self.spatial_shape = spatial_shape

        if nerf_head is not None:
            self.NeRFDecoder = builder.build_backbone(nerf_head)
            self.lovasz = self.NeRFDecoder.lovasz
            if self.NeRFDecoder.img_recon_head:
                self.rgb_conv = nn.Sequential(
                    *[
                        ConvModule(
                        self.img_view_transformer.out_channels,
                        16,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        conv_cfg=dict(type='Conv3d')),
                        ConvModule(
                            16,
                            3,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                            conv_cfg=dict(type='Conv3d'),
                            act_cfg=dict(type='Sigmoid')
                        ),
                    ])

        self.final_conv = ConvModule(
            self.img_view_transformer.out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv3d'))

        self.up_conv = nn.Sequential(
            nn.ConvTranspose3d(out_channels, out_channels//2, kernel_size=2, stride=2, padding=0),
            ConvModule(
                out_channels//2,
                out_channels//2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                conv_cfg=dict(type='Conv3d'))
        )

        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(out_channels//2, self.out_dim),
                nn.Softplus(),
                nn.Linear(self.out_dim, num_classes),
            )

        self.pts_bbox_head = None
        self.use_mask = use_mask
        self.max_ray_number = max_ray_number
        self.loss_occ = build_loss(loss_occ)
        self.class_wise = class_wise
        self.align_after_view_transfromation = False

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
        occ_pred = self.final_conv(img_feats[0]).permute(0, 1, 4, 3, 2)
        occ_pred = self.up_conv(occ_pred).permute(0, 2, 3, 4, 1)

        # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)

        # map voxel prediction onto point cloud
        pc = points[0][:, :3] # (N, 3)
        pc_seg = torch.zeros(pc.shape[0], occ_pred.shape[-1]).to(pc.device) # (N, C)
        mask = gen_point_range_mask(pc, coors_range_xyz=self.coors_range_xyz)
        unq, unq_inv = voxelization(
            point=pc[mask],
            batch_idx=torch.zeros(pc[mask].shape[0]).to(pc.device),
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape,
        )
        point_pred = occ_pred[unq[:, 0], unq[:, 1], unq[:, 2], unq[:, 3]][unq_inv]
        pc_seg[mask] = point_pred

        # ignore free category and then conduct softmax
        occ_score = pc_seg.softmax(-1)
        occ_res = occ_score.argmax(-1)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)

        # output lidarseg label
        lidarseg = kwargs['lidarseg_pad'][0][0][:occ_res.shape[0]]
        lidarseg = lidarseg.cpu().numpy()
        assert len(occ_res) == len(lidarseg)

        occ_res = occ_res[mask.cpu().numpy()]
        lidarseg = lidarseg[mask.cpu().numpy()]
        return [[occ_res, lidarseg]]

        # ### DEBUG ###
        # occ_score = occ_pred.softmax(-1)
        # occ_res = occ_score.argmax(-1)
        # occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        # return [occ_res]

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
                      img_inputs=None,
                      **kwargs):
        """Forward training function.

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
            depth)
        losses['loss_depth'] = loss_depth

        # occupancy prediction
        occ_pred = self.final_conv(img_feats[0]).permute(0, 1, 4, 3, 2)
        occ_pred = self.up_conv(occ_pred).permute(0, 2, 3, 4, 1)

        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)

        # map point cloud label to voxel
        lidarseg = kwargs['lidarseg_pad']
        points_list = []
        batch_list = []
        lidarseg_label = []
        for idx, pc in enumerate(points):
            pc = pc[:, :3]
            mask = gen_point_range_mask(pc, coors_range_xyz=self.coors_range_xyz)
            label = lidarseg[idx][:pc.shape[0]]
            pc = pc[mask]
            points_list.append(pc)
            batch_list.append(torch.ones_like(pc[:, 0]) * idx)
            lidarseg_label.append(label[mask])
        points_list = torch.cat(points_list, 0)
        batch_list = torch.cat(batch_list, 0)
        lidarseg_label = torch.cat(lidarseg_label, 0)
        unq, unq_inv = voxelization(
            point=points_list,
            batch_idx=batch_list,
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape,
        )

        # scatter label
        M = unq_inv.max() + 1
        voxel_lidarseg_label = torch.zeros(M, device=lidarseg_label.device)
        voxel_lidarseg_label = voxel_lidarseg_label.scatter(0, unq_inv, lidarseg_label)

        # generate voxel label and mask
        voxel_semantics = torch.zeros_like(occ_pred.argmax(-1)).detach()
        mask_camera = torch.zeros_like(occ_pred.argmax(-1)).detach()
        voxel_semantics[unq[:, 0], unq[:, 1], unq[:, 2], unq[:, 3]] = voxel_lidarseg_label.long()
        mask_camera[unq[:, 0], unq[:, 1], unq[:, 2], unq[:, 3]] = 1
        mask_camera[voxel_semantics == 0] = 0

        # occupancy losses
        occ_pred = self.inverse_flip_aug(occ_pred, flip_dx, flip_dy)
        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
        losses.update(loss_occ)

        return losses
