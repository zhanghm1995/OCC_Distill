'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-07-27 17:10:22
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''


import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from typing import Dict
from einops import rearrange, repeat
from mmcv.runner import BaseModule
from mmdet.models.builder import build_loss
from mmdet3d.models.losses.lovasz_loss import lovasz_softmax

from ..builder import HEADS


@HEADS.register_module()
class NeRFOccDistillSimpleHead(BaseModule):
    """Simple version of the head to do the knowledge distillation
      by using the neural rendering.

    Args:
        BaseModule (_type_): _description_
    """

    def __init__(self,
                 depth_range,
                 use_kl_loss=False,
                 use_depth_align=False,
                 use_semantic_align=False,
                 depth_loss_type='sml1',
                 variance_focus=0.85,
                 detach_target=False,
                 loss_depth_align_weight=1.0,
                 loss_semantic_align_weight=1.0,
                 loss_kl_align_weight=1.0,
                 init_cfg=None,
                 **kwargs):
        
        super(NeRFOccDistillSimpleHead, self).__init__(init_cfg=init_cfg)

        self.use_kl_loss = use_kl_loss
        self.use_depth_align = use_depth_align
        self.use_semantic_align = use_semantic_align
        self.depth_loss_type = depth_loss_type
        self.variance_focus = variance_focus
        self.min_depth, self.max_depth = depth_range
        self.loss_depth_align_weight = loss_depth_align_weight
        self.loss_semantic_align_weight = loss_semantic_align_weight
        self.detach_target = detach_target
        
    def compute_depth_loss(self, depth_est, depth_gt, mask, loss_weight=1.0):
        '''
        Args:
            mask: depth_gt > 0
        '''
        if self.detach_target:
            depth_gt = depth_gt.detach()
        
        if self.depth_loss_type == 'silog':
            d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
            loss = torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2))
        elif self.depth_loss_type == 'l1':
            loss = F.l1_loss(depth_est[mask], depth_gt[mask], size_average=True)
        elif self.depth_loss_type == 'rl1':
            depth_est = (1 / depth_est) * self.max_depth
            depth_gt = (1 / depth_gt) * self.max_depth
            loss = F.l1_loss(depth_est[mask], depth_gt[mask], size_average=True)

        elif self.depth_loss_type == 'sml1':
            loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True)
        else:
            raise NotImplementedError()

        return loss_weight * loss
    
    def compute_semantic_loss_flatten(self, sem_est, sem_gt, 
                                      loss_weight=1.0, lovasz=False):
        '''
        Args:
            sem_est: N, C
            sem_gt: N
        '''
        loss = F.cross_entropy(sem_est, sem_gt.long(), ignore_index=255)
        if lovasz:
            loss += lovasz_softmax(sem_est, sem_gt.long(), ignore=255)
        return loss_weight * loss
    
    def loss(self, 
             student_feats: Dict,
             teacher_feats: Dict, 
             visibility_mask=None,
             depth_mask=None,
             **kwargs):
        
        losses = dict()

        if self.use_depth_align:
            assert depth_mask is not None
            render_depth_stu = student_feats['render_depth']
            render_depth_tea = teacher_feats['render_depth']
            loss_depth_align = self.compute_depth_loss(
                render_depth_stu, render_depth_tea, depth_mask,
                loss_weight=self.loss_depth_align_weight)
            losses['loss_depth_align'] = loss_depth_align
        
        if self.use_semantic_align:
            render_semantic_stu = student_feats['render_semantic']
            render_semantic_tea = teacher_feats['render_semantic']
            loss_semantic_align = self.compute_semantic_loss_flatten(
                render_semantic_stu, render_semantic_tea,
                loss_weight=self.loss_semantic_align_weight)
            losses['loss_semantic_align'] = loss_semantic_align

        return losses
    

@HEADS.register_module()
class NeRFOccDistillHead(BaseModule):
    """Here we align the depth distribution and the semantic contrastive loss

    Args:
        BaseModule (_type_): _description_
    """

    def __init__(self,
                 use_depth_align=False,
                 use_semantic_align=False,
                 depth_align_loss=dict(
                    type='KnowledgeDistillationKLDivLoss', 
                    loss_weight=1.0),
                 detach_target=False,
                 loss_semantic_align_weight=1.0,
                 init_cfg=None,
                 **kwargs):
        
        super(NeRFOccDistillHead, self).__init__(init_cfg=init_cfg)

        self.use_depth_align = use_depth_align
        self.use_semantic_align = use_semantic_align
        self.loss_semantic_align_weight = loss_semantic_align_weight
        self.detach_target = detach_target

        self.compute_depth_align_loss = build_loss(depth_align_loss)
        
    def loss(self, 
             student_feats: Dict,
             teacher_feats: Dict, 
             visibility_mask=None,
             **kwargs):
        
        losses = dict()

        if self.use_depth_align:
            render_weights_stu = student_feats['render_weights']
            render_weights_tea = teacher_feats['render_weights']
            loss_depth_align = self.compute_depth_align_loss(
                render_weights_stu, render_weights_tea)
            losses['loss_depth_align'] = loss_depth_align
        
        if self.use_semantic_align:
            render_semantic_stu = student_feats['render_semantic']
            render_semantic_tea = teacher_feats['render_semantic']
            loss_semantic_align = self.compute_semantic_loss_flatten(
                render_semantic_stu, render_semantic_tea,
                loss_weight=self.loss_semantic_align_weight)
            losses['loss_semantic_align'] = loss_semantic_align

        return losses


@HEADS.register_module()
class NeRFOccDistillHeadV1(BaseModule):
    """The head to do the knowledge distillation by using the neural rendering.

    Args:
        BaseModule (_type_): _description_
    """

    def __init__(self,
                 use_kl_loss=False,
                 use_affinity_loss=False,
                 detach_target=False,
                 loss_high_feat=dict(
                    type='L1Loss',
                    loss_weight=1.0),
                 loss_prob_feat=dict(
                    type='L1Loss', 
                    loss_weight=1.0),
                 init_cfg=None,
                 **kwargs):
        
        super(NeRFOccDistillHead, self).__init__(init_cfg=init_cfg)
        self.use_kl_loss = use_kl_loss
        self.use_affinity_loss = use_affinity_loss
        self.detach_target = detach_target
        
        if loss_high_feat is not None:
            self.loss_high_feat = build_loss(loss_high_feat)
        
        self.loss_prob_feat = build_loss(loss_prob_feat)
    
    @property
    def with_high_feat_loss(self):
        """bool: Whether calculate the high-level feature alignment loss."""
        return hasattr(self,
                       'loss_high_feat') and self.loss_high_feat is not None

    def loss(self, 
             teacher_feats_list, 
             student_feats_list,
             use_distill_mask=False,
             mask=None,
             **kwargs):
        
        losses = dict()

        if use_distill_mask:
            assert mask is not None

            if self.use_affinity_loss:
                high_feat_loss = self.compute_inter_affinity_loss(
                    student_feats_list[1],
                    teacher_feats_list[1],
                    mask)
                losses['high_feat_loss'] = high_feat_loss
            elif self.with_high_feat_loss:
                mask1 = repeat(mask, 'b h w d -> b c d h w', 
                               c=student_feats_list[1].shape[1])
                mask1 = mask1.to(torch.float32)
                num_total_samples1 = mask1.sum()
                high_feat_loss = self.loss_high_feat(
                    student_feats_list[1], teacher_feats_list[1], 
                    mask1, avg_factor=num_total_samples1)
                
                losses['high_feat_loss'] = high_feat_loss

            if self.use_kl_loss:
                mask2 = mask.reshape(-1)
                num_total_samples2 = mask2.sum()

                num_classes = student_feats_list[2].shape[4]

                pred = student_feats_list[2].reshape(-1, num_classes)
                target = teacher_feats_list[2].reshape(-1, num_classes)

                prob_feat_loss = self.loss_prob_feat(
                    pred, target,
                    mask2, avg_factor=num_total_samples2)
            else:
                mask2 = repeat(mask, 'b h w d -> b h w d c', 
                               c=student_feats_list[2].shape[4])
                mask2 = mask2.to(torch.float32)
                num_total_samples2 = mask2.sum()
                prob_feat_loss = self.loss_prob_feat(
                    student_feats_list[2], teacher_feats_list[2],
                    mask2, avg_factor=num_total_samples2)
            
        else:
            high_feat_loss = self.loss_high_feat(student_feats_list[1],
                                                 teacher_feats_list[1])
            prob_feat_loss = self.loss_prob_feat(student_feats_list[2],
                                                 teacher_feats_list[2])
            
            losses['high_feat_loss'] = high_feat_loss
        
        losses['prob_feat_loss'] = prob_feat_loss
        return losses
    
    def compute_inter_affinity_loss(self,
                                    student_feats,
                                    teacher_feats,
                                    mask):
        bs = mask.shape[0]
        student_feats = rearrange(student_feats, 'b c d h w -> b h w d c')
        teacher_feats = rearrange(teacher_feats, 'b c d h w -> b h w d c')

        all_batch_affinity_loss = []
        for i in range(bs):
            curr_mask = mask[i].to(torch.bool)
            curr_student_feat = student_feats[i][curr_mask]  # to (N, C)
            curr_teacher_feat = teacher_feats[i][curr_mask]

            curr_student_feat = curr_student_feat.matmul(curr_student_feat.T)  # to (N, N)
            curr_teacher_feat = curr_teacher_feat.matmul(curr_teacher_feat.T)

            curr_student_feat = F.normalize(curr_student_feat, dim=1)
            curr_teacher_feat = F.normalize(curr_teacher_feat, dim=1)

            if self.detach_target:
                curr_teacher_feat = curr_teacher_feat.detach()
            
            affinity_loss = F.mse_loss(curr_student_feat, 
                                       curr_teacher_feat)
            all_batch_affinity_loss.append(affinity_loss)
        
        all_batch_affinity_loss = torch.stack(all_batch_affinity_loss)
        return all_batch_affinity_loss.mean()


# def cross_modality_contrastive_loss(image_features, 
#                                     point_cloud_features, 
#                                     temperature=0.07):
#     # Normalize the features
#     image_features = F.normalize(image_features, dim=-1)
#     point_cloud_features = F.normalize(point_cloud_features, dim=-1)

#     # Compute the scalar product between the image features and the point cloud features
#     logits = torch.matmul(image_features, point_cloud_features.t()) / temperature

#     # Compute the labels
#     labels = torch.arange(image_features.size(0)).to(image_features.device)

#     # Compute the contrastive loss
#     loss = F.cross_entropy(logits, labels)

#     return loss


# @HEADS.register_module()
# class NeRFDecoderHead(nn.Module):

#     def __init__(self):
#         pass
    

#     def loss(self,
#              student_feats,
#              teacher_feats,
#              instance_mask=None):
        
#         losses = dict()
        
#         if self.use_semantic_align:
#             render_semantic_stu = student_feats['render_semantic']
#             render_semantic_tea = teacher_feats['render_semantic']

#             ## scatter the features according to the instance mask
#             grouped_semantic_stu = scatter_mean(render_semantic_stu,
#                                                 instance_mask,
#                                                 dim=1)
#             grouped_semantic_tea = scatter_mean(render_semantic_tea,
#                                                 instance_mask,
#                                                 dim=1)
#             loss_semantic_align = cross_modality_contrastive_loss(
#                 grouped_semantic_stu, grouped_semantic_tea)
#             losses['loss_semantic_align']
        
#         return losses


