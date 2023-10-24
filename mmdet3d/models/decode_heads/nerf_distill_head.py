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


def cross_modality_contrastive_loss(image_features, 
                                    point_cloud_features, 
                                    temperature=0.07):
    # Normalize the features
    image_features = F.normalize(image_features, dim=-1)
    point_cloud_features = F.normalize(point_cloud_features, dim=-1)

    # Compute the scalar product between the image features and the point cloud features
    logits = torch.matmul(image_features, point_cloud_features.t()) / temperature

    # Compute the labels
    labels = torch.arange(image_features.size(0)).to(image_features.device)

    # Compute the contrastive loss
    loss = F.cross_entropy(logits, labels)

    return loss


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
    
    def compute_semantic_loss_flatten(self, 
                                      sem_est, 
                                      sem_gt, 
                                      loss_weight=1.0):
        '''
        Args:
            sem_est: N, C
            sem_gt: N, C
        '''
        kl_loss = F.kl_div(
            F.log_softmax(sem_est, dim=1),
            F.softmax(sem_gt.detach(), dim=1))
        return loss_weight * kl_loss
    
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
                 use_occ_logits_align=False,
                 depth_align_loss=dict(
                    type='KnowledgeDistillationKLDivLoss', 
                    loss_weight=1.0),
                 loss_semantic_align_weight=1.0,
                 semantic_align_type='kl', # kl | contrastive
                 occ_logits_align_loss=None,
                 detach_target=False,
                 init_cfg=None,
                 **kwargs):
        
        super(NeRFOccDistillHead, self).__init__(init_cfg=init_cfg)

        self.use_depth_align = use_depth_align
        self.use_semantic_align = use_semantic_align
        self.loss_semantic_align_weight = loss_semantic_align_weight
        self.detach_target = detach_target
        self.semantic_align_type = semantic_align_type

        self.compute_depth_align_loss = build_loss(depth_align_loss)

        self.use_occ_logits_align = use_occ_logits_align
        if use_occ_logits_align:
            assert occ_logits_align_loss is not None
            self.compute_occ_logits_align_loss = build_loss(
                occ_logits_align_loss)
        
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
            
            if self.semantic_align_type == 'kl':
                loss_semantic_align = self.compute_semantic_kl_loss(
                    render_semantic_stu, render_semantic_tea,
                    loss_weight=self.loss_semantic_align_weight)
            elif self.semantic_align_type == 'contrastive':
                assert 'instance_mask' in kwargs

                instance_mask = kwargs['instance_mask']
                render_mask = kwargs['render_mask']
                instance_mask = instance_mask[render_mask]

                loss_semantic_align = self.compute_semantic_contrastive_loss(
                    render_semantic_stu, render_semantic_tea,
                    instance_mask=instance_mask,
                    loss_weight=self.loss_semantic_align_weight)
            else:
                raise NotImplementedError()
            
            losses['loss_semantic_align'] = loss_semantic_align
        
        if self.use_occ_logits_align:
            occ_logits_stu = student_feats['occ_logits']
            occ_logits_tea = teacher_feats['occ_logits']
            loss_occ_logits_align = self.compute_occ_logits_align_loss(
                occ_logits_stu, occ_logits_tea)
            losses['loss_occ_logits_align'] = loss_occ_logits_align

        return losses
    
    def compute_semantic_kl_loss(self, 
                                 sem_est, 
                                 sem_gt, 
                                 loss_weight=1.0):
        '''
        Args:
            sem_est: N, C
            sem_gt: N, C
        '''
        kl_loss = F.kl_div(
            F.log_softmax(sem_est, dim=1),
            F.softmax(sem_gt.detach(), dim=1))
        return loss_weight * kl_loss
    
    def compute_semantic_contrastive_loss(self,
                                          render_semantic_stu,
                                          render_semantic_tea,
                                          instance_mask,
                                          loss_weight=1.0):
        ## scatter the features according to the instance mask
        grouped_semantic_stu = scatter_mean(render_semantic_stu,
                                            instance_mask,
                                            dim=1)
        grouped_semantic_tea = scatter_mean(render_semantic_tea,
                                            instance_mask,
                                            dim=1)
        loss_semantic_align = cross_modality_contrastive_loss(
            grouped_semantic_stu, grouped_semantic_tea)
        return loss_semantic_align * loss_weight
    


@HEADS.register_module()
class NeRFOccPretrainHead(BaseModule):
    """Pretrain the Occupancy network with the rendering losses.

    Args:
        BaseModule (_type_): _description_
    """

    def __init__(self,
                 use_depth_align=False,
                 use_semantic_align=True,
                 use_occ_logits_align=False,
                 depth_align_loss=dict(
                    type='KnowledgeDistillationKLDivLoss', 
                    loss_weight=1.0),
                 loss_semantic_align_weight=1.0,
                 semantic_align_type='kl', # kl | contrastive
                 occ_logits_align_loss=None,
                 detach_target=False,
                 init_cfg=None,
                 **kwargs):
        
        super(NeRFOccPretrainHead, self).__init__(init_cfg=init_cfg)

        self.use_depth_align = use_depth_align
        self.use_semantic_align = use_semantic_align
        self.loss_semantic_align_weight = loss_semantic_align_weight
        self.detach_target = detach_target
        self.semantic_align_type = semantic_align_type

        self.compute_depth_align_loss = build_loss(depth_align_loss)

        self.use_occ_logits_align = use_occ_logits_align
        if use_occ_logits_align:
            assert occ_logits_align_loss is not None
            self.compute_occ_logits_align_loss = build_loss(
                occ_logits_align_loss)
        
    def loss(self, 
             input: Dict,
             seq_len=2,
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
            ## compare the temporal rendering semantic map
            semantic_pred = rearrange(
                semantic_pred, 
                '(b seq) num_cam c h w -> seq b num_cam c h w', seq=seq_len)

            if self.semantic_align_type == 'kl':
                loss_semantic_align = self.compute_semantic_kl_loss(
                    semantic_pred[0], semantic_pred[1],
                    loss_weight=self.loss_semantic_align_weight)
            elif self.semantic_align_type == 'contrastive':
                assert 'instance_mask' in kwargs

                instance_mask = kwargs['instance_mask']  # (b, seq, num_cam, h, w)
                instance_mask = rearrange(instance_mask,
                                          'b seq num_cam h w -> seq b num_cam h w')

                loss_semantic_align = self.compute_semantic_contrastive_loss(
                    semantic_pred[0], semantic_pred[1],
                    instance_mask[0], instance_mask[1],
                    loss_weight=self.loss_semantic_align_weight)
            else:
                raise NotImplementedError()
            
            losses['loss_semantic_align'] = loss_semantic_align
        
        return losses
    
    def compute_semantic_kl_loss(self, 
                                 sem_est, 
                                 sem_gt, 
                                 loss_weight=1.0):
        '''
        Args:
            sem_est: N, C
            sem_gt: N, C
        '''
        kl_loss = F.kl_div(
            F.log_softmax(sem_est, dim=1),
            F.softmax(sem_gt.detach(), dim=1))
        return loss_weight * kl_loss
    
    def compute_semantic_contrastive_loss(self,
                                          render_semantic_1,
                                          render_semantic_2,
                                          instance_mask_1,
                                          instance_mask_2,
                                          loss_weight=1.0):
        ## scatter the features according to the instance mask
        grouped_semantic_stu = scatter_mean(render_semantic_stu,
                                            instance_mask,
                                            dim=1)
        grouped_semantic_tea = scatter_mean(render_semantic_tea,
                                            instance_mask,
                                            dim=1)
        loss_semantic_align = cross_modality_contrastive_loss(
            grouped_semantic_stu, grouped_semantic_tea)
        return loss_semantic_align * loss_weight

    def select_correspondences(self, 
                               instance_map1,
                               instance_map2):
        """_summary_

        Args:
            instance_map1 (_type_): (b, num_cam, h, w)
            instance_map2 (_type_): (b, num_cam, h, w)
        """

        num_selected_pixels = 1  # the number of selected pixels for each instance
        unique_instance_ids1 = torch.unique(instance_map1)
        unique_instance_ids2 = torch.unique(instance_map2)

        mask1 = (instance_map1.unsqueeze(2) == unique_instance_ids1)

        all_selected_pixels = []
        for i in range(len(unique_instance_ids1)):
            valid_indices = torch.nonzero(mask1[..., i])
            # randomly select the pixels
            selected_indices = torch.randperm(valid_indices.size(0))[:num_selected_pixels]
            selected_indices = valid_indices[selected_indices]
            all_selected_pixels.append(selected_indices)
        
        all_selected_pixels = torch.cat(all_selected_pixels, dim=0)




        
    



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


