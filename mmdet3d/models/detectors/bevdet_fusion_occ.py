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

from .. import builder
from .bevdet_occ import BEVFusionStereo4DOCC


def merge_aug_occs(aug_occ_list):
    """Merge the augmented testing occupacy prediction results

    Args:
        aug_occ_list (List): each element with shape (200, 200, 16, 18)
    """
    merged_occs = torch.stack(aug_occ_list, dim=-1)  # to (200, 200, 16, 18, L)
    merged_occs = torch.mean(merged_occs, dim=-1)  # to (200, 200, 16, 18)
    occ_res = merged_occs.argmax(-1)
    occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
    return occ_res



@DETECTORS.register_module()
class BEVFusionStereo4DOCCNeRF(BEVFusionStereo4DOCC):
    """Fuse the lidar and camera images for occupancy prediction with the 
    nerf head for volume rendering.

    Args:
        BEVStereo4DOCC (_type_): _description_
    """

    def __init__(self,
                 nerf_head=None,
                 return_weights=False,
                 **kwargs):
        super(BEVFusionStereo4DOCCNeRF, self).__init__(**kwargs)
        
        self.return_weights = return_weights
        
        self.NeRFDecoder = builder.build_backbone(nerf_head)
        self.num_random_view = self.NeRFDecoder.num_random_view

        ## NOTE: here we assume do not rendering the image
        assert not self.NeRFDecoder.img_recon_head
        
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
    
    def aug_test_v1(self, 
                 points,
                 img_metas,
                 img=None,
                 rescale=False,
                 **kwargs):
        """Test function with augmentaiton in batched manner."""
        points = [item for sublist in points for item in sublist]
        img_metas = [item for sublist in img_metas for item in sublist]
        img = [item for sublist in img for item in sublist]

        new_kwargs = dict()
        for key, value in kwargs.items():
            new_kwargs[key] = [item for sublist in value for item in sublist]
        
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **new_kwargs)

        ## NOTE: here we assume only use the flip augmentation
        # occ_res = self.aug_test_imgs(img_feats, 
        #                              img_metas, 
        #                              rescale,
        #                              **kwargs)
        
    def aug_test(self, 
                 points,
                 img_metas,
                 img=None,
                 rescale=False,
                 **kwargs):
        """Test function with augmentaiton."""
        img_feats, _, _ = multi_apply(self.extract_feat, 
                                      points, 
                                      img,
                                      img_metas, 
                                      **kwargs)
        
        ## NOTE: here we assume only use the flip augmentation
        occ_res = self.aug_test_imgs(img_feats, img_metas, 
                                     rescale,
                                     **kwargs)
        return [occ_res]

    def aug_test_imgs(self,
                      feats,
                      img_metas,
                      rescale=False,
                      **kwargs):
        """Test function of image branch with augmentaiton."""
        # only support aug_test for one sample
        aug_occs = []
        idx = -1
        for x, img_meta in zip(feats, img_metas):
            idx += 1
            occ_pred = self.final_conv(x[0]).permute(0, 4, 3, 2, 1)
            if self.use_predicter:
                occ_pred = self.predicter(occ_pred)
            occ_score = occ_pred.softmax(-1)  # to (1, 200, 200, 16, 18)
            if kwargs['flip_dx'][idx][0]:
                occ_score = occ_score.flip(dims=[1])
            if kwargs['flip_dy'][idx][0]:
                occ_score = occ_score.flip(dims=[2])

            aug_occs.append(occ_score)

        merged_occs = merge_aug_occs(aug_occs)
        return merged_occs

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
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth

        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
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
        """Obtain the intermediate features for distillation.
        """   
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        
        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        
        intricics = kwargs['intricics']
        pose_spatial = kwargs['pose_spatial']
        flip_dx, flip_dy = kwargs['flip_dx'], kwargs['flip_dy']
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

        # rendering
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
        