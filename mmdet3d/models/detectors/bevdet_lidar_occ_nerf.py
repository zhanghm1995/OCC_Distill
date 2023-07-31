'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-07-27 17:29:07
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import torch
import torch.nn.functional as F
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from torch import nn
import numpy as np
from einops import repeat, rearrange
from .. import builder
from .centerpoint import CenterPoint
    

@DETECTORS.register_module()
class MyBEVLidarOCCNeRF(CenterPoint):
    """The LidarOCC with NeRF rendering supervisions.

    """
    def __init__(self, 
                 loss_occ=None,
                 nerf_head=None,
                 lidar_out_dim=32,
                 out_dim=32,
                 use_mask=True,
                 num_classes=18,
                 use_predicter=True, 
                 refine_conv=False,
                 return_weights=False,
                 **kwargs):
        super(MyBEVLidarOCCNeRF, self).__init__(**kwargs)
        
        self.return_weights = return_weights
        self.out_dim = out_dim
        out_channels = out_dim if use_predicter else num_classes

        self.num_frame = 3
        ## NeRF related
        self.NeRFDecoder = builder.build_backbone(nerf_head)
        self.num_classes = num_classes
        self.lovasz = self.NeRFDecoder.lovasz

        if self.NeRFDecoder.img_recon_head:
            num_classes += 3

        self.final_conv = ConvModule(lidar_out_dim,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     bias=True,
                                     conv_cfg=dict(type='Conv3d'))
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, num_classes))
        
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
        
        self.use_mask = use_mask
        self.num_classes = num_classes

        self.num_random_view = self.NeRFDecoder.num_random_view
        self.loss_occ = build_loss(loss_occ)

    def loss_single(self, voxel_semantics, mask_camera, preds):
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
    
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        """Forward function for BEV Lidar OCC.
        """
        img_feats, pts_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        
        losses = dict()
        occ_pred = self.occ_head(pts_feats)
        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']

        gt_depth = kwargs['gt_depth']
        intricics = kwargs['intricics']
        pose_spatial = kwargs['pose_spatial']
        flip_dx, flip_dy = kwargs['flip_dx'], kwargs['flip_dy']
        render_img_gt = kwargs['render_gt_img']
        render_gt_depth = kwargs['render_gt_depth']

        # occupancy losses
        if self.NeRFDecoder.img_recon_head:
            occ_pred = occ_pred[..., :-3]
        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
        losses.update(loss_occ)

        ## NOTE: here we assume loss_depth is loss_occ, used below
        loss_depth = loss_occ['loss_occ']

        if torch.isnan(loss_depth):
            print('NaN in loss_occ!')

        # NeRF loss
        if False:  # DEBUG ONLY!
            density_prob = kwargs['voxel_semantics'].unsqueeze(1)
            print("dendensity_prob.unique():", density_prob.unique())
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
            # get the current_frame_img with (6, 3, H, W) shape
            current_frame_img = render_img_gt.view(batch_size, 
                                                   6, 
                                                   self.num_frame, 
                                                   -1, 
                                                   render_img_gt.shape[-2], 
                                                   render_img_gt.shape[-1])[0, :, 0].cpu().numpy()
            print(current_frame_img.shape)
            self.NeRFDecoder.visualize_image_depth_pair(current_frame_img, 
                                                        render_gt_depth[0], 
                                                        render_depth[0])
            exit()

        else:
            occ_pred = occ_pred.permute(0, 4, 1, 2, 3)

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
                    img_gts = render_img_gt.view(batch_size, 
                                                 num_camera, 
                                                 self.num_frame, -1, 
                                                 render_img_gt.shape[-2], 
                                                 render_img_gt.shape[-1])[:, :, 0]
                    img_gts = img_gts.permute(0, 1, 3, 4, 2)[render_mask]
                    loss_nerf_img = self.NeRFDecoder.compute_image_loss(rgb_pred, img_gts)
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
                depth_pred = F.interpolate(render_depth, size=[gt_depth.shape[-2], gt_depth.shape[-1]], mode="bilinear", align_corners=False)
                rgb_pred = F.upsample_bilinear(rgb_pred.view(batch_size * num_camera, -1, H, W), size=[render_img_gt.shape[-2], render_img_gt.shape[-1]]).view(batch_size, num_camera, -1, render_img_gt.shape[-2], render_img_gt.shape[-1])
                semantic_pred = F.upsample_bilinear(semantic_pred.view(batch_size * num_camera, -1, H, W), size=[gt_depth.shape[-2], gt_depth.shape[-1]]).view(batch_size, num_camera, -1, gt_depth.shape[-2], gt_depth.shape[-1])

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
    
    def vis_voxels_rendering(self, 
                             occ_voxel_semantics, 
                             intricics, 
                             pose_spatial):
        density_prob = occ_voxel_semantics.clone()
        density_prob = density_prob != 17
        density_prob = density_prob.float()
        density_prob[density_prob == 0] = -10  # scaling to avoid 0 in alphas
        density_prob[density_prob == 1] = 10
        density_prob_flip = self.inverse_flip_aug(density_prob, [False], [False])
        print('density_prob', density_prob.shape)  # [1, 1, 200, 200, 16]

        # nerf decoder
        render_depth, _, _ = self.NeRFDecoder(
            density_prob_flip,
            density_prob_flip.tile(1, 3, 1, 1, 1),
            density_prob_flip.tile(1, self.NeRFDecoder.semantic_dim, 1, 1, 1),
            intricics, pose_spatial, True)
        return render_depth
        
    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        _, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        
        occ_pred = self.occ_head(pts_feats)

        if self.NeRFDecoder.img_recon_head:
            occ_pred = occ_pred[..., :-3]
        
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)

        ## NOTE: for debug only!!
        visualize_only = False
        if visualize_only:
            intricics = kwargs['intricics'][0]
            pose_spatial = kwargs['pose_spatial'][0]
            render_img_gt = kwargs['render_gt_img'][0]
            render_gt_depth = kwargs['render_gt_depth'][0]

            voxel_semantics = torch.from_numpy(occ_res).to(intricics.device)
            density_prob = voxel_semantics.unsqueeze(0).unsqueeze(0)  # to (1, 1, 200, 200, 16)
            batch_size = density_prob.shape[0]
            
            # neural rendering
            render_depth = self.vis_voxels_rendering(
                density_prob, intricics, pose_spatial)
            print('render_depth', render_depth.shape, 
                  render_depth.max(), render_depth.min())  # [1, 6, 224, 352]

            # neural rendering the ground truth
            occ_voxels_gt = kwargs['voxel_semantics'][0].unsqueeze(1)
            occ_gt_render_depth = self.vis_voxels_rendering(
                occ_voxels_gt, intricics, pose_spatial)
            print('occ_gt_render_depth', occ_gt_render_depth.shape, 
                  occ_gt_render_depth.max(), occ_gt_render_depth.min())
            
            # get the current_frame_img with (6, 3, H, W) shape
            current_frame_img = render_img_gt.view(batch_size, 
                                                   6, 
                                                   self.num_frame, 
                                                   -1, 
                                                   render_img_gt.shape[-2], 
                                                   render_img_gt.shape[-1])[0, :, 0].cpu().numpy()
            print(current_frame_img.shape)
            self.NeRFDecoder.visualize_image_and_render_depth_pair(
                current_frame_img, 
                occ_gt_render_depth[0], 
                render_depth[0])
            exit()
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
    
    def get_intermediate_features(self, 
                                  points, 
                                  img=None, 
                                  img_metas=None, 
                                  return_loss=False,
                                  logits_as_prob_feat=False,
                                  **kwargs):       
        _, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        
        losses = dict()
        occ_pred = self.occ_head(pts_feats)
        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']

        gt_depth = kwargs['gt_depth']
        intricics = kwargs['intricics']
        pose_spatial = kwargs['pose_spatial']
        flip_dx, flip_dy = kwargs['flip_dx'], kwargs['flip_dy']
        render_img_gt = kwargs['render_gt_img']
        render_gt_depth = kwargs['render_gt_depth']

        ## we use the internal_feats_dict to store the intermediate
        # features used in computing distillation losses
        internal_feats_dict = dict()

        # occupancy losses
        if self.NeRFDecoder.img_recon_head:
            occ_pred = occ_pred[..., :-3]
        
        # loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
        internal_feats_dict['occ_logits'] = occ_pred
        
        # ------ Compute neural rendering losses ------
        occ_pred = occ_pred.permute(0, 4, 1, 2, 3)
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
            rendering_results = self.NeRFDecoder(
                density_prob_flip, rgb_flip, semantic_flip, 
                intricics, pose_spatial, True, render_mask, self.return_weights)
            torch.cuda.synchronize()

            render_depth, rgb_pred, semantic_pred = rendering_results[:3]
            
            internal_feats_dict['render_depth'] = render_depth
            internal_feats_dict['render_semantic'] = semantic_pred
            if self.return_weights:
                internal_feats_dict['render_weights'] = rendering_results[3]
            # end = time.time()
            # print("inference time:", end - start)

            # render_gt_depth = render_gt_depth[render_mask]
            
            # loss_nerf = self.NeRFDecoder.compute_depth_loss(render_depth, render_gt_depth, render_gt_depth > 0.0)
            # if torch.isnan(loss_nerf):
            #     print('NaN in DepthNeRF loss!')
            #     loss_nerf = loss_depth
            # losses['loss_nerf'] = loss_nerf

        else:
            render_depth, rgb_pred, semantic_pred = self.NeRFDecoder(
                density_prob_flip, rgb_flip, semantic_flip, 
                intricics, pose_spatial, True)
            # torch.cuda.synchronize()
            # end = time.time()
            # print("inference time:", end - start)

            # Upsample
            batch_size, num_camera, _, H, W = rgb_pred.shape
            depth_pred = F.interpolate(render_depth, size=[gt_depth.shape[-2], gt_depth.shape[-1]], mode="bilinear", align_corners=False)
            rgb_pred = F.upsample_bilinear(rgb_pred.view(batch_size * num_camera, -1, H, W), size=[render_img_gt.shape[-2], render_img_gt.shape[-1]]).view(batch_size, num_camera, -1, render_img_gt.shape[-2], render_img_gt.shape[-1])
            semantic_pred = F.upsample_bilinear(
                semantic_pred.view(batch_size * num_camera, -1, H, W), 
                size=[gt_depth.shape[-2], gt_depth.shape[-1]]).view(batch_size, num_camera, -1, gt_depth.shape[-2], gt_depth.shape[-1])

            # Refine prediction
            if self.refine_conv:
                # depth_pred = depth_pred.view(batch_size * num_camera, -1, gt_depth.shape[-2], gt_depth.shape[-1])
                rgb_pred = rgb_pred.view(batch_size * num_camera, -1, render_img_gt.shape[-2], render_img_gt.shape[-1])
                semantic_pred = semantic_pred.view(
                    batch_size * num_camera, 
                    -1, 
                    gt_depth.shape[-2], 
                    gt_depth.shape[-1])

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

        return internal_feats_dict
    
    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract the needed features"""
        img_feats = None
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        if not isinstance(pts_feats, list):
            pts_feats = [pts_feats]
        return img_feats, pts_feats
    
    def extract_pts_feat(self, pts, img_feats, img_metas):
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x
    
    def occ_head(self, x):
        if isinstance(x, list):
            feats = x[0]
        
        if feats.dim() == 4:
            feats = rearrange(
                feats, 'b (c Dimz) DimH DimW -> b c Dimz DimH DimW', Dimz=16)
        
        occ_pred = self.final_conv(feats).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        
        return occ_pred
