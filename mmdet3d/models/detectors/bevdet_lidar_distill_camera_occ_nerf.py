'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-07-27 17:27:57
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import torch
import torch.nn.functional as F
from mmdet.models.losses import L1Loss
from einops import repeat

from .. import builder
from ..builder import DETECTORS
from .base import Base3DDetector
from ...utils import get_root_logger


@DETECTORS.register_module()
class MyBEVLidarDistillCameraOCCNeRF(Base3DDetector):

    def __init__(self,
                 teacher_model=None,
                 student_model=None,
                 logits_as_prob_feat=False,
                 freeze_teacher_branch=True,
                 use_distill_mask=False,
                 occ_distill_head=None,
                 init_cfg=None,
                 **kwargs):
        
        super(MyBEVLidarDistillCameraOCCNeRF, self).__init__(init_cfg=init_cfg)

        self.logits_as_prob_feat = logits_as_prob_feat
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

        ## Define the distillation head
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
                teacher_internal_feats = self.teacher_model.get_intermediate_features(
                    points, img_inputs, img_metas, 
                    logits_as_prob_feat=self.logits_as_prob_feat,
                    **kwargs)
        else:
            teacher_internal_feats, loss_teacher_occ = \
                self.teacher_model.get_intermediate_features(
                    points, 
                    return_loss=True,
                    logits_as_prob_feat=self.logits_as_prob_feat, 
                    **kwargs)
            losses.update(loss_teacher_occ)
        
        ## Forward the student model and get the loss
        student_internal_feats, loss_student_occ = \
            self.student_model.get_intermediate_features(
                points, img_inputs, img_metas, 
                return_loss=True, 
                logits_as_prob_feat=self.logits_as_prob_feat,
                **kwargs)
        
        ## Compute the distillation losses
        distill_loss_dict = self.occ_distill_head.loss(
            student_internal_feats,
            teacher_internal_feats, 
            depth_mask=student_internal_feats['depth_mask'],
            render_mask=student_internal_feats['render_mask'],
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
        results = self.student_model.simple_test(
            points, img_metas, img, rescale=rescale, **kwargs)
        return results
    
    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
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
        
