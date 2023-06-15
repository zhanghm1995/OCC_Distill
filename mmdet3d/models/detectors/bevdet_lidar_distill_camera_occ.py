'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-06-10 08:38:46
Email: haimingzhang@link.cuhk.edu.cn
Description: Using the lidar as teacher model to distill the
camera student model.
'''
import torch.nn.functional as F
from mmcv.runner import _load_checkpoint

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
                 **kwargs):
        
        super(BEVLidarDistillCameraOCC, self).__init__(init_cfg=init_cfg)

        self.teacher_model = builder.build_detector(teacher_model)
        self.student_model = builder.build_detector(student_model)
        
        if teacher_model_checkpoint is not None:
            logger = get_root_logger()
            ckpt = _load_checkpoint(
                teacher_model_checkpoint, logger=logger, map_location='cpu')
            self.teacher_model.load_state_dict(ckpt['state_dict'], True)
        
        if freeze_teacher_branch:
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()
        
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        teacher_feats_list = self.teacher_model.get_intermediate_features(
            points, img_inputs, img_metas, **kwargs)
        
        ## Forward the student model and get the loss
        student_feats_list, loss_occ = self.student_model.get_intermediate_features(
            points, img_inputs, img_metas, return_loss=True, **kwargs)
        
        ## Compute the distillation losses
        assert len(teacher_feats_list) == len(student_feats_list)

        # low_feat_loss = F.l1_loss(teacher_feats_list[0], student_feats_list[0])
        high_feat_loss = F.l1_loss(teacher_feats_list[1], student_feats_list[1])
        prob_feat_loss = F.l1_loss(teacher_feats_list[2], student_feats_list[2])
        
        distill_loss_dict = dict(
            high_feat_loss=high_feat_loss,
            prob_feat_loss=prob_feat_loss)
        
        losses = dict()
        losses.update(distill_loss_dict)
        losses.update(loss_occ)
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
