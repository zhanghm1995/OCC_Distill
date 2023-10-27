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


def cross_modality_contrastive_loss(feat1, 
                                    feat2, 
                                    temperature=0.07):
    """Compute the contrastive loss between the two features.

    Args:
        feat1 (Tensor): (B, C)
        feat2 (Tensor): (B, C)
        temperature (float, optional): _description_. Defaults to 0.07.

    Returns:
        _type_: _description_
    """
    # Normalize the features
    feat1 = F.normalize(feat1, dim=-1)
    feat2 = F.normalize(feat2, dim=-1)

    # Compute the scalar product between the image features and the point cloud features
    logits = torch.matmul(feat1, feat2.t()) / temperature

    # Compute the labels
    labels = torch.arange(feat1.size(0)).to(feat1.device)

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
                 semantic_align_type='query_fixed', # kl | contrastive
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
             input_dict: Dict,
             seq_len=2,
             rand_view_idx=None,
             **kwargs):
        
        losses = dict()

        if self.use_depth_align:
            render_weights_stu = input_dict['render_weights']
            render_weights_tea = input_dict['render_weights']
            loss_depth_align = self.compute_depth_align_loss(
                render_weights_stu, render_weights_tea)
            losses['loss_depth_align'] = loss_depth_align
        
        if self.use_semantic_align:
            assert 'render_semantic' in input_dict
            semantic_pred = input_dict['render_semantic']  # (b*seq, num_cam, c, h, w)
            
            ## compare the temporal rendering semantic map
            semantic_pred = rearrange(
                semantic_pred, 
                '(b seq) num_cam c h w -> seq b num_cam h w c', seq=seq_len)

            if self.semantic_align_type == 'kl':
                loss_semantic_align = self.compute_semantic_kl_loss(
                    semantic_pred[0], semantic_pred[1],
                    loss_weight=self.loss_semantic_align_weight)
            elif self.semantic_align_type == 'query_fixed':
                ## Here we fixed the query points in different frames, and
                # align the corresponding instance semantics features.
                assert 'instance_masks' in kwargs

                instance_mask = kwargs['instance_masks']  # (b*seq, num_cam, h, w)
                if rand_view_idx is not None:
                    instance_mask = instance_mask[:, rand_view_idx]
                
                instance_mask = rearrange(instance_mask,
                                          '(b seq) num_cam h w -> seq b num_cam h w',
                                          seq=seq_len)

                loss_semantic_align = self.compute_semantic_contrastive_loss(
                    semantic_pred[0], semantic_pred[1],
                    instance_mask[0], instance_mask[1],
                    loss_weight=self.loss_semantic_align_weight)
            elif self.semantic_align_type == 'query_flow_background':
                ## Here we adapt the background flow info to align the temporal
                # instance masks
                assert 'instance_masks' in kwargs
                assert 'sample_pts_pad' in kwargs

                instance_mask = kwargs['instance_masks']  # (b*seq, num_cam, h, w)
                sample_pts = kwargs['sample_pts_pad']  # (b*seq, num_cam, N, 2)
                if rand_view_idx is not None:
                    instance_mask = instance_mask[:, rand_view_idx]
                    sample_pts = sample_pts[:, rand_view_idx]
                
                instance_mask = rearrange(instance_mask,
                                          '(b seq) num_cam h w -> seq b num_cam h w',
                                          seq=seq_len)
                sample_pts = rearrange(sample_pts,
                                       '(b seq) num_cam N uv -> seq b num_cam N uv',
                                       seq=seq_len)
                
                loss_semantic_align = self.compute_pointwise_loss_with_flow(
                    semantic_pred[0], semantic_pred[1],
                    instance_mask[0], instance_mask[1],
                    sample_pts[0], sample_pts[1],
                    loss_weight=self.loss_semantic_align_weight)
                
                loss_semantic_align_contrastive = self.compute_semantic_contrastive_loss_with_flow(
                    semantic_pred[0], semantic_pred[1],
                    instance_mask[0], instance_mask[1],
                    sample_pts[0], sample_pts[1],
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
        """_summary_

        Args:
            render_semantic_1 (_type_): (b, num_cam, c, h, w)
            render_semantic_2 (_type_): (b, num_cam, c, h, w)
            instance_mask_1 (_type_): (b, num_cam, h, w)
            instance_mask_2 (_type_): (b, num_cam, h, w)
            loss_weight (float, optional): _description_. Defaults to 1.0.

        Returns:
            _type_: _description_
        """
        bs, num_cam, h, w = instance_mask_1.shape

        render_semantic_1 = rearrange(render_semantic_1, 'b n c h w -> b n (h w) c')
        render_semantic_2 = rearrange(render_semantic_2, 'b n c h w -> b n (h w) c')
        instance_mask_1 = rearrange(instance_mask_1, 'b n h w -> b n (h w)')
        instance_mask_2 = rearrange(instance_mask_2, 'b n h w -> b n (h w)')

        ## scatter the features according to the instance mask
        grouped_semantic_1 = scatter_mean(render_semantic_1,
                                          instance_mask_1,
                                          dim=2)
        grouped_semantic_2 = scatter_mean(render_semantic_2,
                                          instance_mask_2,
                                          dim=2)
        loss_semantic_temporal_align = 0.0
        for i in range(bs):
            for j in range(num_cam):
                curr_instance_map_1 = instance_mask_1[i, j].reshape(h, w)
                # NOTE: here we only select the pixels from the first frame
                selected_points = self.sample_correspondence_points(curr_instance_map_1)

                curr_instance_map_2 = instance_mask_2[i, j].reshape(h, w)

                # fetch the corresponding instance mask ids
                instance_id_list_1 = curr_instance_map_1[selected_points[:, 0], selected_points[:, 1]]
                instance_id_list_2 = curr_instance_map_2[selected_points[:, 0], selected_points[:, 1]]

                # obtain the grouped semantic features according to the instance ids
                curr_grouped_feats_1 = grouped_semantic_1[i, j, instance_id_list_1]
                curr_grouped_feats_2 = grouped_semantic_2[i, j, instance_id_list_2]
                loss_semantic_align = cross_modality_contrastive_loss(
                    curr_grouped_feats_1, curr_grouped_feats_2)
                loss_semantic_temporal_align += loss_semantic_align
        
        return loss_semantic_temporal_align * loss_weight
    
    def compute_pointwise_loss_with_flow(self,
                                         render_semantic1,
                                         render_semantic2,
                                         instance_mask1=None,
                                         instance_mask2=None,
                                         sample_pts1=None,
                                         sample_pts2=None,
                                         loss_weight=1.0):
        """Compute the rendered semantics pointwise loss with the corresponding
        sample points coordinates.

        Args:
            render_semantic1 (Tensor): (b, num_cam, h, w, c)
            render_semantic2 (Tensor): same as above
            instance_mask1 (Tensor, optional): (b, num_cam, h, w). Defaults to None.
            instance_mask2 (Tensor, optional): same as above. Defaults to None.
            sample_pts1 (Tensor, optional): (b, num_cam, N, 2). N is the number of
                sampled points, 2 is the coordinates in (u, v) format. Defaults to None.
            sample_pts2 (_type_, optional): _description_. Defaults to None.
            loss_weight (float, optional): _description_. Defaults to 1.0.

        Returns:
            _type_: _description_
        """
        bs, num_cam, h, w, c = render_semantic1.shape

        all_points_feats1 = []
        all_points_feats2 = []
        for i in range(bs):
            for j in range(num_cam):
                num_valid_pts = sample_pts1[i, j, -1, 0]

                curr_sample_pts1 = sample_pts1[i, j, :num_valid_pts]
                curr_sample_pts2 = sample_pts2[i, j, :num_valid_pts]

                curr_render_semantic1 = render_semantic1[i, j][curr_sample_pts1[:, 1], curr_sample_pts1[:, 0]]
                curr_render_semantic2 = render_semantic2[i, j][curr_sample_pts2[:, 1], curr_sample_pts2[:, 0]]
                all_points_feats1.append(curr_render_semantic1)
                all_points_feats2.append(curr_render_semantic2)
        
        all_points_feats1 = torch.cat(all_points_feats1, dim=0)
        all_points_feats2 = torch.cat(all_points_feats2, dim=0)

        ## compute the loss
        loss_semantic_align_pointwise = F.mse_loss(
            all_points_feats1, all_points_feats2)

        return loss_semantic_align_pointwise * loss_weight

    def compute_semantic_contrastive_loss_with_flow(self,
                                                    render_semantic_1,
                                                    render_semantic_2,
                                                    instance_mask_1,
                                                    instance_mask_2,
                                                    sample_pts_1,
                                                    sample_pts_2,
                                                    loss_weight=1.0):
        """_summary_

        Args:
            render_semantic_1 (_type_): (b, num_cam, h, w, c)
            render_semantic_2 (_type_): (b, num_cam, c, h, w)
            instance_mask_1 (_type_): (b, num_cam, h, w)
            instance_mask_2 (_type_): (b, num_cam, h, w)
            loss_weight (float, optional): _description_. Defaults to 1.0.

        Returns:
            _type_: _description_
        """
        bs, num_cam, h, w = instance_mask_1.shape

        render_semantic_1 = rearrange(render_semantic_1, 'b n h w c-> b n (h w) c')
        render_semantic_2 = rearrange(render_semantic_2, 'b n h w c -> b n (h w) c')
        instance_mask_1 = rearrange(instance_mask_1, 'b n h w -> b n (h w)')
        instance_mask_2 = rearrange(instance_mask_2, 'b n h w -> b n (h w)')

        ## scatter the features according to the instance mask
        ## NOTE: the results of scatter_mean will be sorted according to the instance ids
        grouped_semantic_1 = scatter_mean(render_semantic_1,
                                          instance_mask_1,
                                          dim=2)
        grouped_semantic_2 = scatter_mean(render_semantic_2,
                                          instance_mask_2,
                                          dim=2)
        loss_semantic_temporal_align = 0.0
        for i in range(bs):
            for j in range(num_cam):
                curr_instance_map_1 = instance_mask_1[i, j].reshape(h, w)
                # NOTE: here we only select the pixels from the first frame
                selected_points_indices = self.sample_points(
                    curr_instance_map_1, sample_pts_1[i, j])

                curr_instance_map_2 = instance_mask_2[i, j].reshape(h, w)

                selected_points1 = sample_pts_1[i, j][selected_points_indices]
                selected_points2 = sample_pts_2[i, j][selected_points_indices]

                # fetch the corresponding instance mask ids
                instance_id_list_1 = curr_instance_map_1[selected_points1[:, 1], selected_points1[:, 0]]
                instance_id_list_2 = curr_instance_map_2[selected_points2[:, 1], selected_points2[:, 0]]

                # obtain the grouped semantic features according to the instance ids
                curr_grouped_feats_1 = grouped_semantic_1[i, j, instance_id_list_1]
                curr_grouped_feats_2 = grouped_semantic_2[i, j, instance_id_list_2]
                loss_semantic_align = cross_modality_contrastive_loss(
                    curr_grouped_feats_1, curr_grouped_feats_2)
                loss_semantic_temporal_align += loss_semantic_align
        
        return loss_semantic_temporal_align * loss_weight

    def select_correspondences(self, 
                               instance_map1,
                               instance_map2):
        """_summary_

        Args:
            instance_map1 (_type_): (b, num_cam, h, w)
            instance_map2 (_type_): (b, num_cam, h, w)
        """
        bs, num_cam, h, w = instance_map1.shape

        for i in range(bs):
            for j in range(num_cam):
                curr_instance_map_1 = instance_map1[i, j]
                # NOTE: here we only select the pixels from the first frame
                selected_points = self.sample_correspondence_points(curr_instance_map_1)

                curr_instance_map_2 = instance_map2[i, j]

                # fetch the corresponding instance mask ids
                instance_id_list_1 = curr_instance_map_1[selected_points]
                instance_id_list_2 = curr_instance_map_2[selected_points]

                
    def sample_correspondence_points(self, instance_mask):
        num_selected_pixels = 1  # the number of selected pixels for each instance
        unique_instance_ids = torch.unique(instance_mask)

        mask = (instance_mask.unsqueeze(-1) == unique_instance_ids)

        all_selected_pixels = []
        for i in range(len(unique_instance_ids)):
            valid_indices = torch.nonzero(mask[..., i])
            # randomly select the pixels
            selected_indices = torch.randperm(valid_indices.size(0))[:num_selected_pixels]
            selected_indices_coord = valid_indices[selected_indices]
            all_selected_pixels.extend(selected_indices_coord)
        
        all_selected_pixels = torch.stack(all_selected_pixels, dim=0)
        return all_selected_pixels

    def sample_points(self, instance_mask, sample_pts):
        num_selected_pixels = 1  # the number of selected pixels for each instance

        num_valid_pts = sample_pts[-1, 0]
        sample_pts = sample_pts[:num_valid_pts]
        
        # obtain the valid points
        sampled_instance_mask = instance_mask[sample_pts[:, 1], sample_pts[:, 0]]  # (N,)
        unique_sampled_inst_ids = torch.unique(sampled_instance_mask)
        sampled_mask = (sampled_instance_mask.unsqueeze(-1) == unique_sampled_inst_ids)

        all_selected_pixels = []
        for i in range(len(unique_sampled_inst_ids)):
            valid_indices = torch.nonzero(sampled_mask[..., i])
            # randomly select the points
            selected_indices = torch.randperm(valid_indices.size(0))[:num_selected_pixels]
            selected_indices_raw = valid_indices[selected_indices].squeeze()
            all_selected_pixels.extend(selected_indices_raw)
        
        all_selected_pixels = torch.stack(all_selected_pixels, dim=0)
        return all_selected_pixels

