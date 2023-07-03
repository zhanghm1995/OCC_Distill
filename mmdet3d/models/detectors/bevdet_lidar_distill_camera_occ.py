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
                 logits_as_prob_feat=False,
                 freeze_teacher_branch=True,
                 use_distill_mask=False,
                 occ_distill_head=None,
                 use_cross_kd=False,
                 init_cfg=None,
                 **kwargs):
        
        super(BEVLidarDistillCameraOCC, self).__init__(init_cfg=init_cfg)

        self.logits_as_prob_feat = logits_as_prob_feat
        self.use_distill_mask = use_distill_mask
        self.freeze_teacher_branch = freeze_teacher_branch
        self.use_cross_kd = use_cross_kd

        self.teacher_model = builder.build_detector(teacher_model)
        self.student_model = builder.build_detector(student_model)
        
        if freeze_teacher_branch:
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()
            # NOTE: we must set this flag to True, 
            # otherwise the teacher model will be re-initialized.
            # self.teacher_model._is_init = True

        ## Define the distillation head
        if occ_distill_head:
            self.occ_distill_head = builder.build_head(occ_distill_head)
        else:
            occ_distill_head = dict(type='OccDistillHead')
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
                    points, img_inputs, img_metas, 
                    logits_as_prob_feat=self.logits_as_prob_feat,
                    **kwargs)
        else:
            teacher_feats_list, loss_teacher_occ = \
                self.teacher_model.get_intermediate_features(
                    points, 
                    return_loss=True,
                    logits_as_prob_feat=self.logits_as_prob_feat, 
                    **kwargs)
            losses.update(loss_teacher_occ)
        
        ## Forward the student model and get the loss
        student_feats_list, loss_student_occ = \
            self.student_model.get_intermediate_features(
                points, img_inputs, img_metas, 
                return_loss=True, 
                logits_as_prob_feat=self.logits_as_prob_feat,
                **kwargs)
        
        ## Compute the distillation losses
        assert len(teacher_feats_list) == len(student_feats_list)
        
        if self.use_cross_kd:
            student_high_feat = student_feats_list[1]
            
        distill_loss_dict = self.occ_distill_head.loss(
            teacher_feats_list, student_feats_list,
            self.use_distill_mask, 
            mask=kwargs['mask_camera'],
            **kwargs)
        
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
        results = self.teacher_model.simple_test(
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
        
