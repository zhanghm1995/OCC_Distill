'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-07-15 21:19:28
Email: haimingzhang@link.cuhk.edu.cn
Description: Pretrain the model with binary occupancy predictions.
'''

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet.models import DETECTORS
from .. import builder
from mmdet.models.builder import build_loss
from .bevdet_occ import BEVStereo4DOCC
from .bevdet import BEVStereo4D
from .bevdet_occ_nerf_S import NUSCENSE_LIDARSEG_PALETTE


@DETECTORS.register_module()
class BEVStereo4DOCCPretrain(BEVStereo4DOCC):
    """Pretrain the BEVDetOcc with binary occupancy predictions.

    Args:
        BEVStereo4DOCC (_type_): _description_
    """

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
    


@DETECTORS.register_module()
class BEVStereo4DOCCNeRFRretrain(BEVStereo4D):

    def __init__(self,
                 nerf_head=None,
                 out_dim=32,
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,
                 refine_conv=False,
                 return_weights=False,
                 **kwargs):
        
        super(BEVStereo4DOCCNeRFRretrain, self).__init__(**kwargs)

        self.out_dim = out_dim
        out_channels = out_dim if use_predicter else num_classes
        self.NeRFDecoder = builder.build_backbone(nerf_head)
        self.num_classes = num_classes
        self.return_weights = return_weights
        self.lovasz = self.NeRFDecoder.lovasz

        if self.NeRFDecoder.img_recon_head:
            num_classes += 3

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

        self.refine_conv = refine_conv
        if self.refine_conv:
            self.img_refine = nn.Sequential(
                nn.Conv2d(3, 16, 3, 1, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(16, 3, 3, 1, 1, bias=False),
            )
            self.sem_refine = nn.Sequential(
                nn.Conv2d(self.NeRFDecoder.semantic_dim, 16, 3, 1, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(16, self.NeRFDecoder.semantic_dim, 3, 1, 1, bias=False),
            )

        self.pts_bbox_head = None
        self.num_random_view = self.NeRFDecoder.num_random_view
        self.class_wise = class_wise
        self.align_after_view_transfromation = False

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        occ_pred_ori = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)

        if self.use_predicter:
            # to (b, 200, 200, 16, c)
            occ_pred_ori = self.predicter(occ_pred_ori)
        
        occ_pred = occ_pred_ori[..., :-3] \
            if self.NeRFDecoder.img_recon_head else occ_pred_ori

        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)

        ## nerf
        VISUALIZE = False
        if VISUALIZE:
            # to (b, c, 200, 200, 16)
            use_gt_occ = False

            occ_pred = occ_pred_ori.permute(0, 4, 1, 2, 3)

            # semantic
            if self.NeRFDecoder.semantic_head:
                # (b, 17, 200, 200, 16)
                semantic = occ_pred[:, :self.NeRFDecoder.semantic_dim, ...]
            else:
                semantic = torch.zeros_like(occ_pred[:, :self.NeRFDecoder.semantic_dim, ...])
            # density
            if use_gt_occ:
                # (b, 200, 200, 16)
                density_prob = kwargs['voxel_semantics'][0].unsqueeze(1)
                density_prob = density_prob != 17
                density_prob = density_prob.float()
                density_prob[density_prob == 0] = -10  # scaling to avoid 0 in alphas
                density_prob[density_prob == 1] = 10
            else:
                density_prob = -occ_pred[:, self.NeRFDecoder.semantic_dim: self.NeRFDecoder.semantic_dim+1]

            intricics = kwargs['intricics']
            pose_spatial = kwargs['pose_spatial']
            rgb_recons = occ_pred[:, -3:]

            ## firstly we need to find the semantic class for each voxel
            semantic = semantic.argmax(1)
            # mapping the color
            semantic = NUSCENSE_LIDARSEG_PALETTE[semantic].to(occ_pred)  # to (6, h, w, 3)
            semantic = semantic.permute(0, 4, 1, 2, 3)

            render_depth, rgb_pred, semantic_pred = self.NeRFDecoder(
                density_prob, rgb_recons, semantic, 
                intricics[0], pose_spatial[0], 
                is_train=False, render_mask=None)
            
            render_img_gt = kwargs['render_gt_img'][0]
            current_frame_img = render_img_gt.view(
                6, self.num_frame, -1, 
                render_img_gt.shape[-2], 
                render_img_gt.shape[-1])[:, 0].cpu().numpy()
            # sem = semantic_pred[0].argmax(1)  # to (6, H, W)
            # sem_color = NUSCENSE_LIDARSEG_PALETTE[sem]  # to (6, h, w, 3)
            self.NeRFDecoder.visualize_image_semantic_depth_pair(
                current_frame_img,
                semantic_pred[0].permute(0, 2, 3, 1),
                render_depth[0],
                save=True
            )
        return [occ_res]

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
        intricics = kwargs['intricics']
        pose_spatial = kwargs['pose_spatial']
        flip_dx, flip_dy = kwargs['flip_dx'], kwargs['flip_dy']
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth
        render_img_gt = kwargs['render_gt_img']
        render_gt_depth = kwargs['render_gt_depth']

        # occupancy prediction
        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)  # to (b, 200, 200, 16, c)
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)

        # occupancy losses
        if self.NeRFDecoder.img_recon_head:
            occ_pred = occ_pred[..., :-3]

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

            # nerf decoder
            render_depth, _, _ = self.NeRFDecoder(
                density_prob_flip,
                density_prob_flip.tile(1, 3, 1, 1, 1),
                density_prob_flip.tile(1, self.NeRFDecoder.semantic_dim, 1, 1, 1),
                intricics, pose_spatial, True
            )
            print('density_prob', density_prob.shape)  # [1, 1, 200, 200, 16]
            print('render_depth', render_depth.shape, render_depth.max(), render_depth.min())  # [1, 6, 224, 352]
            print('gt_depth', gt_depth.shape, gt_depth.max(), gt_depth.min())  # [1, 6, 384, 704]
            current_frame_img = render_img_gt.view(batch_size, 6, self.num_frame, -1, render_img_gt.shape[-2], render_img_gt.shape[-1])[0, :, 0].cpu().numpy()
            print(current_frame_img.shape)
            self.NeRFDecoder.visualize_image_depth_pair(current_frame_img, render_gt_depth[0], render_depth[0])
            exit()

        else:
            occ_pred = occ_pred.permute(0, 4, 1, 2, 3)  # to (B, c, 200, 200, 16)

            # semantic
            if self.NeRFDecoder.semantic_head:
                semantic = occ_pred[:, :self.NeRFDecoder.semantic_dim, ...]
            else:
                semantic = torch.zeros_like(occ_pred)

            # density
            density_prob = -occ_pred[:, self.NeRFDecoder.semantic_dim: self.NeRFDecoder.semantic_dim+1, ...]

            # image reconstruction
            if self.NeRFDecoder.img_recon_head:
                rgb_recons = occ_pred[:, -4:-1, ...]
            else:
                rgb_recons = torch.zeros_like(density_prob)

            # cancel the effect of flip augmentation
            rgb_flip = self.inverse_flip_aug(rgb_recons, flip_dx, flip_dy)
            semantic_flip = self.inverse_flip_aug(semantic, flip_dx, flip_dy)
            density_prob_flip = self.inverse_flip_aug(density_prob, flip_dx, flip_dy)

            # random view selection
            if self.num_random_view != -1:
                rand_ind = torch.multinomial(torch.tensor([1/self.num_random_view]*self.num_random_view), self.num_random_view, replacement=False)
                intricics = intricics[:, rand_ind]
                pose_spatial = pose_spatial[:, rand_ind]
                render_gt_depth = render_gt_depth[:, rand_ind]
                render_img_gt = render_img_gt[:, rand_ind]

            # rendering
            # import time
            # torch.cuda.synchronize()
            # start = time.time()
            if self.NeRFDecoder.mask_render:
                render_mask = render_gt_depth > 0.0
                render_depth, rgb_pred, semantic_pred = self.NeRFDecoder(
                    density_prob_flip, rgb_flip, semantic_flip, 
                    intricics, pose_spatial, True, render_mask)
                torch.cuda.synchronize()
                # end = time.time()
                # print("inference time:", end - start)

                render_gt_depth = render_gt_depth[render_mask]
                loss_nerf = self.NeRFDecoder.compute_depth_loss(
                    render_depth, render_gt_depth, render_gt_depth > 0.0)
                if torch.isnan(loss_nerf):
                    print('NaN in DepthNeRF loss!')
                    loss_nerf = loss_depth
                losses['loss_nerf'] = loss_nerf

                if self.NeRFDecoder.semantic_head:
                    img_semantic = kwargs['img_semantic']
                    img_semantic = img_semantic[render_mask]
                    loss_nerf_sem = self.NeRFDecoder.compute_semantic_loss_flatten(
                        semantic_pred, img_semantic, lovasz=self.lovasz)
                    if torch.isnan(loss_nerf):
                        print('NaN in SemNeRF loss!')
                        loss_nerf_sem = loss_depth
                    losses['loss_nerf_sem'] = loss_nerf_sem

                if self.NeRFDecoder.img_recon_head:
                    batch_size, num_camera = intricics.shape[:2]
                    img_gts = render_img_gt.view(
                        batch_size, num_camera, self.num_frame, -1, 
                        render_img_gt.shape[-2], render_img_gt.shape[-1])[:, :, 0]
                    img_gts = img_gts.permute(0, 1, 3, 4, 2)[render_mask]
                    loss_nerf_img = self.NeRFDecoder.compute_image_loss(
                        rgb_pred, img_gts)
                    if torch.isnan(loss_nerf):
                        print('NaN in ImgNeRF loss!')
                        loss_nerf_img = loss_depth
                    losses['loss_nerf_img'] = loss_nerf_img

            else:
                render_depth, rgb_pred, semantic_pred = self.NeRFDecoder(
                    density_prob_flip, rgb_flip, 
                    semantic_flip, intricics, pose_spatial, True)
                # torch.cuda.synchronize()
                # end = time.time()
                # print("inference time:", end - start)

                # Upsample
                batch_size, num_camera, _, H, W = rgb_pred.shape
                depth_pred = F.interpolate(render_depth, 
                                           size=[gt_depth.shape[-2], 
                                                 gt_depth.shape[-1]], 
                                           mode="bilinear", 
                                           align_corners=False)
                rgb_pred = F.upsample_bilinear(rgb_pred.view(batch_size * num_camera, -1, H, W), 
                                               size=[render_img_gt.shape[-2], 
                                                     render_img_gt.shape[-1]]).view(batch_size, num_camera, -1, render_img_gt.shape[-2], render_img_gt.shape[-1])
                semantic_pred = F.upsample_bilinear(
                    semantic_pred.view(batch_size * num_camera, -1, H, W), 
                    size=[gt_depth.shape[-2], 
                          gt_depth.shape[-1]]).view(batch_size, 
                                                    num_camera, 
                                                    -1, 
                                                    gt_depth.shape[-2], 
                                                    gt_depth.shape[-1])

                # Refine prediction
                if self.refine_conv:
                    # depth_pred = depth_pred.view(batch_size * num_camera, -1, gt_depth.shape[-2], gt_depth.shape[-1])
                    rgb_pred = rgb_pred.view(batch_size * num_camera, -1, render_img_gt.shape[-2], render_img_gt.shape[-1])
                    semantic_pred = semantic_pred.view(batch_size * num_camera, -1, gt_depth.shape[-2], gt_depth.shape[-1])

                    # depth_pred = self.depth_refine(depth_pred)
                    if self.NeRFDecoder.img_recon_head:
                        rgb_pred = self.img_refine(rgb_pred)
                    if self.NeRFDecoder.semantic_head:
                        semantic_pred = self.sem_refine(semantic_pred)

                    # depth_pred = depth_pred.view(batch_size, num_camera, gt_depth.shape[-2], gt_depth.shape[-1])
                    rgb_pred = rgb_pred.view(batch_size, num_camera, -1, render_img_gt.shape[-2], render_img_gt.shape[-1])
                    semantic_pred = semantic_pred.view(batch_size, num_camera, -1, gt_depth.shape[-2], gt_depth.shape[-1])

                # nerf loss calculation
                loss_nerf = self.NeRFDecoder.compute_depth_loss(depth_pred, render_gt_depth, render_gt_depth > 0.0)
                if torch.isnan(loss_nerf):
                    print('NaN in DepthNeRF loss!')
                    loss_nerf = loss_depth
                losses['loss_nerf'] = loss_nerf

                if self.NeRFDecoder.semantic_head:
                    img_semantic = kwargs['img_semantic']
                    if self.num_random_view != -1:
                        img_semantic = img_semantic[:, rand_ind]
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
    

@DETECTORS.register_module()
class BEVStereo4DOCCTemporalNeRFPretrain(BEVStereo4DOCCNeRFRretrain):

    def __init__(self,
                 loss_compare=None,
                 pretrain_head=None,
                 **kwargs):
        
        super(BEVStereo4DOCCTemporalNeRFPretrain, self).__init__(**kwargs)

        self.loss_compare = build_loss(loss_compare) if loss_compare else None
        self.pretrain_head = builder.build_head(pretrain_head) if pretrain_head else None

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        occ_pred_ori = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)

        if self.use_predicter:
            # to (b, 200, 200, 16, c)
            occ_pred_ori = self.predicter(occ_pred_ori)
        
        occ_pred = occ_pred_ori[..., :-3] \
            if self.NeRFDecoder.img_recon_head else occ_pred_ori

        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)

        ## nerf
        VISUALIZE = False
        if VISUALIZE:
            # to (b, c, 200, 200, 16)
            use_gt_occ = False

            occ_pred = occ_pred_ori.permute(0, 4, 1, 2, 3)

            # semantic
            if self.NeRFDecoder.semantic_head:
                # (b, 17, 200, 200, 16)
                semantic = occ_pred[:, :self.NeRFDecoder.semantic_dim, ...]
            else:
                semantic = torch.zeros_like(occ_pred[:, :self.NeRFDecoder.semantic_dim, ...])
            # density
            if use_gt_occ:
                # (b, 200, 200, 16)
                density_prob = kwargs['voxel_semantics'][0].unsqueeze(1)
                density_prob = density_prob != 17
                density_prob = density_prob.float()
                density_prob[density_prob == 0] = -10  # scaling to avoid 0 in alphas
                density_prob[density_prob == 1] = 10
            else:
                density_prob = -occ_pred[:, self.NeRFDecoder.semantic_dim: self.NeRFDecoder.semantic_dim+1]

            intricics = kwargs['intricics']
            pose_spatial = kwargs['pose_spatial']
            rgb_recons = occ_pred[:, -3:]

            ## firstly we need to find the semantic class for each voxel
            semantic = semantic.argmax(1)
            # mapping the color
            semantic = NUSCENSE_LIDARSEG_PALETTE[semantic].to(occ_pred)  # to (6, h, w, 3)
            semantic = semantic.permute(0, 4, 1, 2, 3)

            render_depth, rgb_pred, semantic_pred = self.NeRFDecoder(
                density_prob, rgb_recons, semantic, 
                intricics[0], pose_spatial[0], 
                is_train=False, render_mask=None)
            
            render_img_gt = kwargs['render_gt_img'][0]
            current_frame_img = render_img_gt.view(
                6, self.num_frame, -1, 
                render_img_gt.shape[-2], 
                render_img_gt.shape[-1])[:, 0].cpu().numpy()
            # sem = semantic_pred[0].argmax(1)  # to (6, H, W)
            # sem_color = NUSCENSE_LIDARSEG_PALETTE[sem]  # to (6, h, w, 3)
            self.NeRFDecoder.visualize_image_semantic_depth_pair(
                current_frame_img,
                semantic_pred[0].permute(0, 2, 3, 1),
                render_depth[0],
                save=True
            )
        return [occ_res]

    def reshape_kwargs(self, batch):
        output = {}
        for key, value in batch.items():
            if key == 'instance_selected_points':
                output[key] = value
                continue
            if len(value.shape) == 2:
                output[key] = value.view(-1)
            else:
                output[key] = value.view(-1, *value.shape[2:])
        return output

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        """Forward training function.

        Returns:
            dict: Losses of different branches.
        """
        if img_inputs[0].dim() == 6:
            img_inputs_new = []
            for entry in img_inputs:
                entry = entry.view(-1, *entry.shape[2:])
                img_inputs_new.append(entry)
            img_inputs = img_inputs_new

            kwargs = self.reshape_kwargs(kwargs)
        
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        
        # occupancy prediction
        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)  # to (b, 200, 200, 16, c)
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)

        gt_depth = kwargs['gt_depth']
        intricics = kwargs['intricics']
        pose_spatial = kwargs['pose_spatial']
        flip_dx, flip_dy = kwargs['flip_dx'], kwargs['flip_dy']
        losses = dict()
        loss_lss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_lss_depth'] = loss_lss_depth
        render_img_gt = kwargs['render_gt_img']
        render_gt_depth = kwargs['render_gt_depth']  # [B, num_cam, H, W]

        # occupancy losses
        if self.NeRFDecoder.img_recon_head:
            occ_pred = occ_pred[..., :-3]

        # NeRF loss
        VISUALIZE = True
        if VISUALIZE:  # DEBUG ONLY!
            from .bevdet_occ_nerf_S_visualizer import NeRFVisualizer
            visualizer = NeRFVisualizer()
            visualizer.visualize_gt_occ(self.NeRFDecoder, 
                                        self.num_frame,
                                        kwargs['voxel_semantics'],
                                        render_img_gt, 
                                        flip_dx, flip_dy,
                                        intricics, pose_spatial,
                                        save_dir="./results/cvpr_visualization/dvgo_render")
            exit()

        occ_pred = occ_pred.permute(0, 4, 1, 2, 3)  # to (B, c, 200, 200, 16)

        # semantic
        if self.NeRFDecoder.semantic_head:
            semantic = occ_pred[:, :self.NeRFDecoder.semantic_dim, ...]
        else:
            semantic = torch.zeros_like(occ_pred)

        # density
        density_prob = -occ_pred[:, self.NeRFDecoder.semantic_dim: self.NeRFDecoder.semantic_dim+1, ...]

        # image reconstruction
        if self.NeRFDecoder.img_recon_head:
            rgb_recons = occ_pred[:, -4:-1, ...]
        else:
            rgb_recons = torch.zeros_like(density_prob)

        # cancel the effect of flip augmentation
        rgb_flip = self.inverse_flip_aug(rgb_recons, flip_dx, flip_dy)
        semantic_flip = self.inverse_flip_aug(semantic, flip_dx, flip_dy)
        density_prob_flip = self.inverse_flip_aug(density_prob, flip_dx, flip_dy)

        # random view selection
        if self.num_random_view != -1:
            rand_ind = torch.multinomial(
                torch.tensor([1 / self.num_random_view] * 6), 
                self.num_random_view, 
                replacement=False)
            intricics = intricics[:, rand_ind]
            pose_spatial = pose_spatial[:, rand_ind]
            render_gt_depth = render_gt_depth[:, rand_ind]
            render_img_gt = render_img_gt[:, rand_ind]

        # rendering
        # import time
        # torch.cuda.synchronize()
        # start = time.time()
        if self.NeRFDecoder.mask_render:
            render_mask = render_gt_depth > 0.0
            render_depth, rgb_pred, semantic_pred = self.NeRFDecoder(
                density_prob_flip, rgb_flip, semantic_flip, 
                intricics, pose_spatial, True, render_mask)
            torch.cuda.synchronize()
            # end = time.time()
            # print("inference time:", end - start)

            render_gt_depth = render_gt_depth[render_mask]
            loss_nerf = self.NeRFDecoder.compute_depth_loss(
                render_depth, render_gt_depth, render_gt_depth > 0.0)
            if torch.isnan(loss_nerf):
                print('NaN in DepthNeRF loss!')
                loss_nerf = loss_lss_depth
            losses['loss_nerf'] = loss_nerf

            if self.NeRFDecoder.semantic_head:
                img_semantic = kwargs['img_semantic']
                img_semantic = img_semantic[render_mask]
                loss_nerf_sem = self.NeRFDecoder.compute_semantic_loss_flatten(
                    semantic_pred, img_semantic, lovasz=self.lovasz)
                if torch.isnan(loss_nerf):
                    print('NaN in SemNeRF loss!')
                    loss_nerf_sem = loss_lss_depth
                losses['loss_nerf_sem'] = loss_nerf_sem

            if self.NeRFDecoder.img_recon_head:
                batch_size, num_camera = intricics.shape[:2]
                img_gts = render_img_gt.view(
                    batch_size, num_camera, self.num_frame, -1, 
                    render_img_gt.shape[-2], render_img_gt.shape[-1])[:, :, 0]
                img_gts = img_gts.permute(0, 1, 3, 4, 2)[render_mask]
                loss_nerf_img = self.NeRFDecoder.compute_image_loss(
                    rgb_pred, img_gts)
                if torch.isnan(loss_nerf):
                    print('NaN in ImgNeRF loss!')
                    loss_nerf_img = loss_lss_depth
                losses['loss_nerf_img'] = loss_nerf_img

        else:  # rendering the whole images
            render_depth, rgb_pred, semantic_pred = self.NeRFDecoder(
                density_prob_flip, rgb_flip, 
                semantic_flip, intricics, pose_spatial, True)
            # torch.cuda.synchronize()
            # end = time.time()
            # print("inference time:", end - start)

            # nerf loss calculation
            # loss_nerf = self.NeRFDecoder.compute_depth_loss(
            #     depth_pred, render_gt_depth, render_gt_depth > 0.0)
            # if torch.isnan(loss_nerf):
            #     print('NaN in DepthNeRF loss!')
            #     loss_nerf = loss_depth
            # losses['loss_nerf'] = loss_nerf

            feats_dict = dict()
            if self.NeRFDecoder.semantic_head:
                feats_dict['render_semantic'] = semantic_pred
                ## compare the temporal rendering semantic map
                # semantic_pred = rearrange(
                #     semantic_pred, 
                #     '(b seq) num_cam c h w -> seq b num_cam c h w', b=bs_ori)
                # loss_nerf_sem = self.loss_compare(
                #     semantic_pred[0], semantic_pred[1])
                # if torch.isnan(loss_nerf_sem):
                #     print('NaN in SemNeRF loss!')
                # losses['loss_nerf_sem'] = loss_nerf_sem


            losses = self.pretrain_head.loss(feats_dict, 
                                             rand_view_idx=rand_ind, 
                                             **kwargs)

        return losses
    

@DETECTORS.register_module()
class BEVStereo4DOCCTemporalNeRFPretrainV2(BEVStereo4DOCCNeRFRretrain):
    """Using the temporal alignment strategy to pretrain the Occupancy models.
    In this version, we use the seperate the density prediction network.

    Args:
        BEVStereo4DOCCNeRFRretrain (_type_): _description_
    """

    def __init__(self,
                 pretrain_head=None,
                 final_softplus=True,
                 use_render_depth_loss=True,
                 use_lss_depth_loss=True,
                 use_temporal_align_loss=True,
                 **kwargs):
        
        super(BEVStereo4DOCCTemporalNeRFPretrainV2, self).__init__(
            use_predicter=False, **kwargs)

        self.use_lss_depth_loss = use_lss_depth_loss
        self.use_render_depth_loss = use_render_depth_loss
        self.use_temporal_align_loss = use_temporal_align_loss

        if pretrain_head is not None:
            self.pretrain_head = builder.build_head(pretrain_head)

        self.final_conv = ConvModule(
            self.img_view_transformer.out_channels,
            self.out_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv3d'))
    
        if final_softplus:
            self.density_mlp = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, 1),
                nn.Softplus(),
            )
        else:
            self.density_mlp = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, 1),
            )

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        occ_pred_ori = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)

        if self.use_predicter:
            # to (b, 200, 200, 16, c)
            occ_pred_ori = self.predicter(occ_pred_ori)
        
        occ_pred = occ_pred_ori[..., :-3] \
            if self.NeRFDecoder.img_recon_head else occ_pred_ori

        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)

        ## nerf
        VISUALIZE = False
        if VISUALIZE:
            # to (b, c, 200, 200, 16)
            use_gt_occ = False

            occ_pred = occ_pred_ori.permute(0, 4, 1, 2, 3)

            # semantic
            if self.NeRFDecoder.semantic_head:
                # (b, 17, 200, 200, 16)
                semantic = occ_pred[:, :self.NeRFDecoder.semantic_dim, ...]
            else:
                semantic = torch.zeros_like(occ_pred[:, :self.NeRFDecoder.semantic_dim, ...])
            # density
            if use_gt_occ:
                # (b, 200, 200, 16)
                density_prob = kwargs['voxel_semantics'][0].unsqueeze(1)
                density_prob = density_prob != 17
                density_prob = density_prob.float()
                density_prob[density_prob == 0] = -10  # scaling to avoid 0 in alphas
                density_prob[density_prob == 1] = 10
            else:
                density_prob = -occ_pred[:, self.NeRFDecoder.semantic_dim: self.NeRFDecoder.semantic_dim+1]

            intricics = kwargs['intricics']
            pose_spatial = kwargs['pose_spatial']
            rgb_recons = occ_pred[:, -3:]

            ## firstly we need to find the semantic class for each voxel
            semantic = semantic.argmax(1)
            # mapping the color
            semantic = NUSCENSE_LIDARSEG_PALETTE[semantic].to(occ_pred)  # to (6, h, w, 3)
            semantic = semantic.permute(0, 4, 1, 2, 3)

            render_depth, rgb_pred, semantic_pred = self.NeRFDecoder(
                density_prob, rgb_recons, semantic, 
                intricics[0], pose_spatial[0], 
                is_train=False, render_mask=None)
            
            render_img_gt = kwargs['render_gt_img'][0]
            current_frame_img = render_img_gt.view(
                6, self.num_frame, -1, 
                render_img_gt.shape[-2], 
                render_img_gt.shape[-1])[:, 0].cpu().numpy()
            # sem = semantic_pred[0].argmax(1)  # to (6, H, W)
            # sem_color = NUSCENSE_LIDARSEG_PALETTE[sem]  # to (6, h, w, 3)
            self.NeRFDecoder.visualize_image_semantic_depth_pair(
                current_frame_img,
                semantic_pred[0].permute(0, 2, 3, 1),
                render_depth[0],
                save=True
            )
        return [occ_res]

    def reshape_kwargs(self, 
                       batch, 
                       keep_keys=['instance_selected_points']):
        output = {}
        for key, value in batch.items():
            if key in keep_keys:
                output[key] = value
                continue
            
            if len(value.shape) == 2:
                output[key] = value.view(-1)
            else:
                output[key] = value.view(-1, *value.shape[2:])
        return output

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        """Forward training function.

        Returns:
            dict: Losses of different branches.
        """
        if img_inputs[0].dim() == 6:
            img_inputs_new = []
            for entry in img_inputs:
                entry = entry.view(-1, *entry.shape[2:])
                img_inputs_new.append(entry)
            img_inputs = img_inputs_new

            kwargs = self.reshape_kwargs(kwargs)
        
        import time
        torch.cuda.synchronize()
        start = time.time()
        img_feats, _, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        
        # occupancy prediction
        voxel_feats = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)  # to (b, 200, 200, 16, c)
        end = time.time()
        print("Forward time:", end - start)
        
        gt_depth = kwargs['gt_depth']
        intricics = kwargs['intricics']
        pose_spatial = kwargs['pose_spatial']
        flip_dx, flip_dy = kwargs['flip_dx'], kwargs['flip_dy']

        losses = dict()
        if self.use_lss_depth_loss:
            loss_lss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
            losses['loss_lss_depth'] = loss_lss_depth
        
        render_gt_depth = kwargs['render_gt_depth']  # [B, num_cam, H, W]

        # predict the density
        density_prob = self.density_mlp(voxel_feats)

        density_prob = rearrange(density_prob, 'b x y z c -> b c x y z')
        voxel_feats = rearrange(voxel_feats, 'b x y z c -> b c x y z')

        if self.NeRFDecoder.semantic_head:
            semantic = voxel_feats
        else:
            semantic = torch.zeros_like(voxel_feats)

        # image reconstruction
        if self.NeRFDecoder.img_recon_head:
            rgb_recons = voxel_feats[:, -4:-1, ...]
        else:
            rgb_recons = torch.zeros_like(density_prob)

        # cancel the effect of flip augmentation
        rgb_flip = self.inverse_flip_aug(rgb_recons, flip_dx, flip_dy)
        semantic_flip = self.inverse_flip_aug(semantic, flip_dx, flip_dy)
        density_prob_flip = self.inverse_flip_aug(density_prob, flip_dx, flip_dy)

        # random view selection
        rand_ind = None
        if self.num_random_view != -1:
            rand_ind = torch.multinomial(
                torch.tensor([1 / self.num_random_view] * 6), 
                self.num_random_view, 
                replacement=False)
            intricics = intricics[:, rand_ind]
            pose_spatial = pose_spatial[:, rand_ind]
            render_gt_depth = render_gt_depth[:, rand_ind]

        # rendering
        if self.NeRFDecoder.mask_render:
            import time
            torch.cuda.synchronize()
            start = time.time()

            render_mask = render_gt_depth > 0.0  # (b, num_cam, h, w)
            render_depth, rgb_pred, semantic_pred = self.NeRFDecoder(
                density_prob_flip, rgb_flip, semantic_flip, 
                intricics, pose_spatial, True, render_mask)
            
            end = time.time()
            print("Rendering time:", end - start)

            if self.use_render_depth_loss:
                render_gt_depth = render_gt_depth[render_mask]
                loss_nerf = self.NeRFDecoder.compute_depth_loss(
                    render_depth, render_gt_depth, render_gt_depth > 0.0)
                if torch.isnan(loss_nerf):
                    print('NaN in DepthNeRF loss!')
                    loss_nerf = loss_lss_depth
                losses['loss_nerf'] = loss_nerf
            
        else:  # rendering the whole images
            ## Rendering
            import time
            torch.cuda.synchronize()
            start = time.time()
            render_depth, rgb_pred, semantic_pred = self.NeRFDecoder(
                density_prob_flip, rgb_flip, semantic_flip, 
                intricics, pose_spatial, True)
            end = time.time()
            print("Rendering time:", end - start)

            ## Compute nerf losses
            if self.use_render_depth_loss:
                loss_render_depth = self.NeRFDecoder.compute_depth_loss(
                    render_depth, render_gt_depth, render_gt_depth > 0.0)
                if torch.isnan(loss_render_depth):
                    print('NaN in DepthNeRF loss!')
                    loss_render_depth = loss_lss_depth
                losses['loss_render_depth'] = loss_render_depth

            if self.use_temporal_align_loss:
                feats_dict = dict()
                feats_dict['render_semantic'] = semantic_pred

                start = time.time()
                loss_contrast_dict = self.pretrain_head.loss(
                    feats_dict, rand_view_idx=rand_ind, **kwargs)
                losses.update(loss_contrast_dict)
                end = time.time()
                print("Computing loss time:", end - start)

        return losses
    


@DETECTORS.register_module()
class BEVStereo4DOCCTemporalNeRFPretrainV3(BEVStereo4DOCCNeRFRretrain):
    """Using the temporal alignment strategy to pretrain the Occupancy models.
    In this version, we use the seperate the density prediction network, and we
    only render the sampled lidar points for reducing the memory cost.

    Args:
        BEVStereo4DOCCNeRFRretrain (_type_): _description_
    """

    def __init__(self,
                 pretrain_head=None,
                 final_softplus=True,
                 use_render_depth_loss=True,
                 use_lss_depth_loss=True,
                 use_temporal_align_loss=True,
                 use_loss_norm=False,
                 **kwargs):
        
        super(BEVStereo4DOCCTemporalNeRFPretrainV3, self).__init__(
            use_predicter=False, **kwargs)

        self.use_lss_depth_loss = use_lss_depth_loss
        self.use_render_depth_loss = use_render_depth_loss
        self.use_temporal_align_loss = use_temporal_align_loss
        self.use_loss_norm = use_loss_norm

        if pretrain_head is not None:
            self.pretrain_head = builder.build_head(pretrain_head)

        self.final_conv = ConvModule(
            self.img_view_transformer.out_channels,
            self.out_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv3d'))
    
        if final_softplus:
            self.density_mlp = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, 1),
                nn.Softplus(),
            )
        else:
            self.density_mlp = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, 1),
            )

    def reshape_kwargs(self, 
                       batch, 
                       keep_keys=['instance_selected_points']):
        output = {}
        for key, value in batch.items():
            if key in keep_keys:
                output[key] = value
                continue
            # to (seq_len * b, ...)
            if len(value.shape) == 2:
                output[key] = value.transpose(0, 1).reshape(-1)
            else:
                output[key] = value.transpose(0, 1).reshape(-1, *value.shape[2:])
        return output

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        """Forward training function.

        Returns:
            dict: Losses of different branches.
        """
        # import time
        # torch.cuda.synchronize()
        # start = time.time()
        
        if img_inputs[0].dim() == 6:
            img_inputs_new = []
            for entry in img_inputs:
                entry = entry.transpose(0, 1).reshape(-1, *entry.shape[2:])
                img_inputs_new.append(entry)
            img_inputs = img_inputs_new

            kwargs = self.reshape_kwargs(kwargs)
        
        img_feats, _, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        
        # occupancy prediction
        voxel_feats = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)  # to (b, 200, 200, 16, c)

        # end = time.time()
        # print("Forward time:", end - start)
        
        losses = dict()
        if self.use_lss_depth_loss:
            gt_depth = kwargs['gt_depth']
            loss_lss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
            losses['loss_lss_depth'] = loss_lss_depth
        
        intricics = kwargs['intricics']
        pose_spatial = kwargs['pose_spatial']
        flip_dx, flip_dy = kwargs['flip_dx'], kwargs['flip_dy']
        render_gt_depth = kwargs['render_gt_depth']  # [B, num_cam, H, W]
        sample_pts_pad = kwargs['sample_pts_pad']  # (b, num_cam, N, 2)

        # predict the density
        density_prob = self.density_mlp(voxel_feats)

        density_prob = rearrange(density_prob, 'b x y z c -> b c x y z')
        voxel_feats = rearrange(voxel_feats, 'b x y z c -> b c x y z')

        # image reconstruction
        rgb_recons = torch.zeros_like(density_prob)

        # cancel the effect of flip augmentation
        rgb_flip = self.inverse_flip_aug(rgb_recons, flip_dx, flip_dy)
        semantic_flip = self.inverse_flip_aug(voxel_feats, flip_dx, flip_dy)
        density_prob_flip = self.inverse_flip_aug(density_prob, flip_dx, flip_dy)

        # random view selection
        rand_ind = None
        if self.num_random_view != -1:
            rand_ind = torch.multinomial(
                torch.tensor([1 / self.num_random_view] * 6), 
                self.num_random_view, 
                replacement=False)
            intricics = intricics[:, rand_ind]
            pose_spatial = pose_spatial[:, rand_ind]
            render_gt_depth = render_gt_depth[:, rand_ind]

        # rendering
        if self.NeRFDecoder.mask_render:
            # (seq_len*b, num_cam, h, w)
            render_mask = self.prepare_render_mask(sample_pts_pad, render_gt_depth)

            render_depth, rgb_pred, semantic_pred = self.NeRFDecoder(
                density_prob_flip, rgb_flip, semantic_flip, 
                intricics, pose_spatial, True, render_mask)
            
            if self.use_render_depth_loss:
                render_gt_depth = render_gt_depth[render_mask]
                loss_render_depth = self.NeRFDecoder.compute_depth_loss(
                    render_depth, render_gt_depth, render_gt_depth > 0.0)
                if torch.isnan(loss_render_depth):
                    print('NaN in DepthNeRF loss!')
                    loss_render_depth = loss_lss_depth
                losses['loss_render_depth'] = loss_render_depth

            feats_dict = dict()
            feats_dict['render_semantic'] = semantic_pred
            feats_dict['render_mask'] = render_mask

            loss_contrast_dict = self.pretrain_head.loss_with_render_mask(
                feats_dict, rand_view_idx=rand_ind, **kwargs)
            losses.update(loss_contrast_dict)
            
        else:  # rendering the whole images
            ## Rendering
            import time
            torch.cuda.synchronize()
            start = time.time()
            render_depth, rgb_pred, semantic_pred = self.NeRFDecoder(
                density_prob_flip, rgb_flip, semantic_flip, 
                intricics, pose_spatial, True)
            end = time.time()
            print("Rendering time:", end - start)

            ## Compute nerf losses
            if self.use_render_depth_loss:
                loss_render_depth = self.NeRFDecoder.compute_depth_loss(
                    render_depth, render_gt_depth, render_gt_depth > 0.0)
                if torch.isnan(loss_render_depth):
                    print('NaN in DepthNeRF loss!')
                    loss_render_depth = loss_lss_depth
                losses['loss_render_depth'] = loss_render_depth

            if self.use_temporal_align_loss:
                feats_dict = dict()
                feats_dict['render_semantic'] = semantic_pred

                start = time.time()
                loss_contrast_dict = self.pretrain_head.loss(
                    feats_dict, rand_view_idx=rand_ind, **kwargs)
                losses.update(loss_contrast_dict)
                end = time.time()
                print("Computing loss time:", end - start)

        if self.use_loss_norm:
            for key, value in losses.items():
                losses[key] = value / (value.detach() + 1e-5)

        return losses
    
    def prepare_render_mask(self, sample_pts_pad, render_gt_depth):
        """_summary_

        Args:
            sample_pts_pad (_type_): (seq_len*b, num_cam, N, 2)
            render_gt_depth (_type_): (seq_len*b, num_cam, h, w)

        Returns:
            _type_: (seq_len*b, num_cam, h, w)
        """
        bs, num_cam = sample_pts_pad.shape[:2]
        render_mask = torch.zeros_like(render_gt_depth)

        for i in range(bs):
            for j in range(num_cam):
                num_valid_pts = sample_pts_pad[i, j, -1, 0]
                curr_sample_pts = sample_pts_pad[i, j, :num_valid_pts]
                # print(i, j, num_valid_pts, curr_sample_pts[:, 0].min(), curr_sample_pts[:, 0].max())
                # print(i, j, num_valid_pts, curr_sample_pts[:, 1].min(), curr_sample_pts[:, 1].max())
                
                render_mask[i, j][curr_sample_pts[:, 1], curr_sample_pts[:, 0]] = 1

        return render_mask.to(torch.bool)