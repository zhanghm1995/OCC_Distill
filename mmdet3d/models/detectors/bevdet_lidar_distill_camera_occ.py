'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-06-10 08:38:46
Email: haimingzhang@link.cuhk.edu.cn
Description: Using the lidar as teacher model to distill the
camera student model.
'''
import torch
import torch.nn.functional as F
from mmcv.runner import _load_checkpoint
from mmdet.models.losses import L1Loss
from einops import repeat

from .. import builder
from ..builder import DETECTORS
from .base import Base3DDetector
from ...utils import get_root_logger


@DETECTORS.register_module()
class BEVLidarDistillCameraOCC(Base3DDetector):

    def __init__(self,
                 teacher_model=None,
                 student_model=None,
                 teacher_model_checkpoint=None,
                 freeze_teacher_branch=True,
                 init_cfg=None,
                 use_distill_mask=False,
                 occ_distill_head=None,
                 **kwargs):
        
        super(BEVLidarDistillCameraOCC, self).__init__(init_cfg=init_cfg)

        self.use_distill_mask = use_distill_mask
        self.freeze_teacher_branch = freeze_teacher_branch

        self.teacher_model = builder.build_detector(teacher_model)
        self.student_model = builder.build_detector(student_model)
        
        if freeze_teacher_branch:
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()
            # NOTE: we must set this flag to True, 
            # otherwise the teacher model will be re-initialized.
            # self.teacher_model._is_init = True

        self.loss_distill = L1Loss(loss_weight=1.0)

        self.occ_distill_head = builder.build_head(occ_distill_head)
        
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        losses = dict()

        if self.freeze_teacher_branch:
            with torch.no_grad():
                self.teacher_model.eval()
                teacher_feats_list = self.teacher_model.get_intermediate_features(
                    points, img_inputs, img_metas, **kwargs)
        else:
            teacher_feats_list, loss_teacher_occ = \
                self.teacher_model.get_intermediate_features(points, 
                                                             img_inputs, 
                                                             img_metas, 
                                                             return_loss=True, 
                                                             **kwargs)
            losses.update(loss_teacher_occ)
        
        ## Forward the student model and get the loss
        student_feats_list, loss_student_occ = \
            self.student_model.get_intermediate_features(
                points, img_inputs, img_metas, return_loss=True, **kwargs)
        
        ## Compute the distillation losses
        assert len(teacher_feats_list) == len(student_feats_list)

        # distill_loss_dict = self.compute_distill_loss(
        #     teacher_feats_list, student_feats_list,
        #     self.use_distill_mask, mask=kwargs['mask_camera'])
        distill_loss_dict = self.occ_distill_head(
            teacher_feats_list, student_feats_list,
            self.use_distill_mask, mask=kwargs['mask_camera'])
        
        losses.update(distill_loss_dict)
        losses.update(loss_student_occ)
        return losses

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function using the student model and without augmentaiton."""
        ## Forward the student model and get the loss
        results = self.student_model.simple_test(
            points, img_metas, img, rescale=rescale, **kwargs)
        return results
    
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
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)
    
    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images."""
        pass

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        pass

    def compute_distill_loss(self, 
                             teacher_feats_list, 
                             student_feats_list,
                             use_distill_mask=False,
                             mask=None):
        losses = dict()

        if use_distill_mask:
            assert mask is not None
            mask1 = repeat(mask, 'b h w d -> b c d h w', c=student_feats_list[1].shape[1])
            mask1 = mask1.to(torch.float32)
            num_total_samples1 = mask1.sum()
            high_feat_loss = self.loss_distill(
                teacher_feats_list[1], student_feats_list[1], 
                mask1, avg_factor=num_total_samples1)

            mask2 = repeat(mask, 'b h w d -> b h w d c', c=student_feats_list[2].shape[4])
            mask2 = mask2.to(torch.float32)
            num_total_samples2 = mask2.sum()
            prob_feat_loss = self.loss_distill(
                teacher_feats_list[2], student_feats_list[2],
                mask2, avg_factor=num_total_samples2)
        else:
            high_feat_loss = F.l1_loss(teacher_feats_list[1], 
                                       student_feats_list[1])
            prob_feat_loss = F.l1_loss(teacher_feats_list[2], 
                                       student_feats_list[2])
        
        losses['high_feat_loss'] = high_feat_loss
        losses['prob_feat_loss'] = prob_feat_loss
        return losses
        
