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


def get_inter_channel_loss(feat1, feat2):
    """Compute the inter channel loss.

    Args:
        feat1 (Tensor): (B, N, C)
        feat2 (Tensor): (B, N, C)

    Returns:
        _type_: _description_
    """
    # get the inter instance loss
    feat1 = feat1.permute(0, 2, 1).matmul(feat1) # (B, C, C)
    feat2 = feat2.permute(0, 2, 1).matmul(feat2)

    feat1 = F.normalize(feat1, dim=-1)
    feat2 = F.normalize(feat2, dim=-1)
    
    loss_inter_channel = F.mse_loss(feat1, feat2, reduction='none')
    loss_inter_channel = loss_inter_channel.sum(-1)
    loss_inter_channel = loss_inter_channel.mean()
    return loss_inter_channel


def get_inter_instance_loss(feat1, feat2):
    feat1 = feat1.matmul(feat1.permute(0, 2, 1)) # B, N, N 
    feat2 = feat2.matmul(feat2.permute(0, 2, 1)) # B, N, N 
    
    feat1 = F.normalize(feat1, dim=-1)
    feat2 = F.normalize(feat2, dim=-1)
    
    loss_inter_instance = F.mse_loss(feat1, feat2, reduction='none')
    loss_inter_instance = loss_inter_instance.sum(-1)
    loss_inter_instance = loss_inter_instance.mean()
    return loss_inter_instance


def compute_tig_feat_align_loss(feat1, feat2,
                                use_inter_instance_loss=True,
                                use_inter_channel_loss=True,
                                loss_inter_instance_weight=1.0,
                                loss_inter_channel_weight=1.0):
    """Calculate the TiG-BEV feature alignment loss.

    Args:
        feat1 (Tensor): (B, N, C)
        feat2 (Tensor): (B, N, C)
    """
    loss_dict = dict()
    
    if use_inter_instance_loss:
        loss_inter_instance = get_inter_instance_loss(feat1, feat2)
        loss_dict['loss_inter_instance'] = loss_inter_instance * loss_inter_instance_weight

    if use_inter_channel_loss:
        loss_inter_channel = get_inter_channel_loss(feat1, feat2)
        loss_dict['loss_inter_channel'] = loss_inter_channel * loss_inter_channel_weight

    return loss_dict


def compute_batched_loss_dict_list(loss_dict_list):
    """Compute the batched loss dict.

    Args:
        loss_dict (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert isinstance(loss_dict_list, list)
    assert len(loss_dict_list) > 0

    # compute the batched loss dict
    batched_loss_dict = dict()
    for k in loss_dict_list[0].keys():
        loss = torch.stack([loss_dict[k] for loss_dict in loss_dict_list], 
                            dim=0).mean()
        batched_loss_dict[k] = loss
    return batched_loss_dict


def compute_pointwise_similarity_loss(feat1, feat2, instance_idx):
    # 1) Compute the similarity of pointwise features
    feat1 = F.normalize(feat1, dim=-1)
    feat2 = F.normalize(feat2, dim=-1)

    # Compute the cosine similarity
    similarity = nn.CosineSimilarity(dim=-1)(feat1, feat2)

    # Compute the average similarity of each instance
    instance_similarity = scatter_mean(similarity, instance_idx, dim=0)



    


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
                 use_pointwise_align=True,
                 pointwise_align_type='mse',
                 loss_pointwise_align_weight=1.0,
                 loss_semantic_align_weight=1.0,
                 loss_inter_instance_weight=1.0,
                 loss_inter_channel_weight=1.0,
                 semantic_align_type='query_flow_all', # kl | contrastive
                 occ_logits_align_loss=None,
                 detach_target=False,
                 init_cfg=None,
                 **kwargs):
        
        super(NeRFOccPretrainHead, self).__init__(init_cfg=init_cfg)

        self.use_depth_align = use_depth_align

        self.use_pointwise_align = use_pointwise_align
        self.loss_pointwise_align_weight = loss_pointwise_align_weight
        self.pointwise_align_type = pointwise_align_type

        self.use_semantic_align = use_semantic_align
        self.loss_semantic_align_weight = loss_semantic_align_weight
        self.loss_inter_instance_weight = loss_inter_instance_weight
        self.loss_inter_channel_weight = loss_inter_channel_weight
        self.detach_target = detach_target
        self.semantic_align_type = semantic_align_type

    def loss_with_render_mask(self,
                              input_dict: Dict,
                              seq_len=2,
                              rand_view_idx=None,
                              **kwargs):
        """Compute the loss with the render mask.
        """
        losses = dict()

        assert 'render_semantic' in input_dict
        assert 'sample_pts_pad' in kwargs

        semantic_pred = input_dict['render_semantic']  # (N, C)
        render_mask = input_dict['render_mask']  # (seq*b num_cam h w)

        seq = 2
        render_mask = rearrange(render_mask,
                                '(seq b) num_cam h w -> seq b num_cam h w',
                                seq=seq)
        
        render_mask1 = render_mask[0]  # (b, num_cam, h, w)
        render_mask2 = render_mask[1]  # (b, num_cam, h, w)
        assert (render_mask1).sum() == (render_mask2).sum()

        num_frame_pixels = render_mask1.sum()  # the number of rendered rays in the frame

        semantic_pred1 = semantic_pred[:num_frame_pixels]
        semantic_pred2 = semantic_pred[num_frame_pixels:]

        if self.use_pointwise_align and self.pointwise_align_type != 'similarity':
            ## compute the temporal pointwise align loss
            if self.pointwise_align_type == 'mse':
                loss_semantic_align_pointwise = F.mse_loss(
                    semantic_pred1, semantic_pred2)
            elif self.pointwise_align_type == 'contrastive':
                bs, num_cam, h, w = render_mask1.shape
                start, end = 0, 0

                batched_loss = []
                for i in range(bs):
                    all_camera_contrast_loss = []
                    for j in range(num_cam):
                        curr_render_mask1 = render_mask1[i, j]  # (h, w)
                        curr_render_mask2 = render_mask2[i, j]

                        num_valid_pts = int(curr_render_mask1.sum())
                        if num_valid_pts == 0:
                            continue

                        end += num_valid_pts

                        curr_semantic_pred1 = semantic_pred1[start:end]  # (num_sample_points, C)
                        curr_semantic_pred2 = semantic_pred2[start:end]  # (num_sample_points, C)

                        loss_curr_pw_align = cross_modality_contrastive_loss(
                            curr_semantic_pred1, curr_semantic_pred2
                        )
                        if torch.isnan(loss_curr_pw_align):
                            print("NaN loss_curr_pw_align")
                            loss_curr_pw_align = torch.nan_to_num(loss_curr_pw_align)

                        all_camera_contrast_loss.append(loss_curr_pw_align)

                        start = end
                    
                    batched_loss.append(torch.stack(all_camera_contrast_loss, dim=0).mean())
                
                loss_semantic_align_pointwise = torch.stack(batched_loss, dim=0).mean()
            else:
                raise NotImplementedError()
            
            losses['loss_pointwise_align'] = \
                loss_semantic_align_pointwise * self.loss_pointwise_align_weight
            
        if self.use_semantic_align:
            assert 'instance_masks' in kwargs

            instance_mask = kwargs['instance_masks']  # (seq*b, num_cam, h, w)
            sample_pts = kwargs['sample_pts_pad']  # (seq*b, num_cam, N, 2)
            if rand_view_idx is not None:
                instance_mask = instance_mask[:, rand_view_idx]
                sample_pts = sample_pts[:, rand_view_idx]
            
            instance_mask = rearrange(instance_mask,
                                      '(seq b) num_cam h w -> seq b num_cam h w',
                                      seq=seq_len)
            sample_pts = rearrange(sample_pts,
                                    '(seq b) num_cam N uv -> seq b num_cam N uv',
                                    seq=seq_len)
            
            instance_mask1 = instance_mask[0]  # (b, num_cam, h, w)
            instance_mask2 = instance_mask[1]  # (b, num_cam, h, w)

            sample_pts1 = sample_pts[0]  # (b, num_cam, N, 2)
            sample_pts2 = sample_pts[1]  # (b, num_cam, N, 2)

            bs, num_cam, h, w = instance_mask1.shape
            start, end = 0, 0

            batched_loss = []
            for i in range(bs):
                all_camera_grouped_feats_1 = []
                all_camera_grouped_feats_2 = []
                for j in range(num_cam):
                    curr_instance_map_1 = instance_mask1[i, j]  # (h, w)
                    curr_instance_map_2 = instance_mask2[i, j]

                    curr_sample_pts_1 = sample_pts1[i, j]  # (N, 2)
                    curr_sample_pts_2 = sample_pts2[i, j]

                    num_valid_pts = int(curr_sample_pts_1[-1, 0])
                    end += num_valid_pts

                    curr_sample_pts_1 = curr_sample_pts_1[:num_valid_pts]
                    curr_sample_pts_2 = curr_sample_pts_2[:num_valid_pts]

                    # fetch the corresponding instance mask ids
                    valid_inst_ids1, valid_inst_ids2, selected_pts_indices, final_inst_ids1, final_inst_ids2 = \
                        self.sample_both_valid_instance(
                            curr_instance_map_1, curr_sample_pts_1,
                            curr_instance_map_2, curr_sample_pts_2)

                    curr_render_semantic1 = semantic_pred1[start:end][selected_pts_indices]  # (num_sample_points, C)
                    curr_render_semantic2 = semantic_pred2[start:end][selected_pts_indices]  # (num_sample_points, C)

                    ## compute the pointwise similarity loss by using the instance ids
                    # compute_pointwise_similarity_loss(curr_render_semantic1, 
                    #                                   curr_render_semantic2, 
                    #                                   final_inst_ids1,
                    #                                   final_inst_ids2)

                    grouped_semantic1 = scatter_mean(curr_render_semantic1,
                                                     valid_inst_ids1,
                                                     dim=0)
                    grouped_semantic2 = scatter_mean(curr_render_semantic2,
                                                     valid_inst_ids2,
                                                     dim=0)
                    curr_grouped_feats1 = grouped_semantic1[final_inst_ids1]
                    curr_grouped_feats2 = grouped_semantic2[final_inst_ids2]

                    all_camera_grouped_feats_1.append(curr_grouped_feats1)
                    all_camera_grouped_feats_2.append(curr_grouped_feats2)
                    
                    start = end
                
                all_camera_grouped_feats_1 = torch.concat(all_camera_grouped_feats_1, dim=0)[None]
                all_camera_grouped_feats_2 = torch.concat(all_camera_grouped_feats_2, dim=0)[None]
                curr_batch_loss = compute_tig_feat_align_loss(
                    all_camera_grouped_feats_1, all_camera_grouped_feats_2,
                    loss_inter_instance_weight=self.loss_inter_instance_weight,
                    loss_inter_channel_weight=self.loss_inter_channel_weight
                    )

                batched_loss.append(curr_batch_loss)
            
            loss_semantic_temporal_align = compute_batched_loss_dict_list(batched_loss)
            losses.update(loss_semantic_temporal_align)
        
        return losses
    
    def sample_both_valid_instance(self, 
                                   instance_mask1, sample_pts1, 
                                   instance_mask2, sample_pts2):
        """_summary_

        Args:
            instance_mask1 (Tensor): (h, w)
            sample_pts1 (Tensor): (N, 2)
            instance_mask2 (Tensor): (h, w)
            sample_pts2 (Tensor): (N, 2)

        Returns:
            _type_: _description_
        """
        # obtain the sampled instance ids according to the sample pts coordinates
        sampled_instance_mask1 = instance_mask1[sample_pts1[:, 1], sample_pts1[:, 0]]  # (N,)
        sampled_instance_mask2 = instance_mask2[sample_pts2[:, 1], sample_pts2[:, 0]]  # (N,)

        valid_sampled_instance_mask1 = sampled_instance_mask1 != -1
        valid_sampled_instance_mask2 = sampled_instance_mask2 != -1
        both_valid_instace_mask = valid_sampled_instance_mask1 & valid_sampled_instance_mask2

        selected_instance_id1 = sampled_instance_mask1[both_valid_instace_mask]  # (M,)
        selected_instance_id2 = sampled_instance_mask2[both_valid_instace_mask]  # (M,)

        all_selected_points_indices = torch.nonzero(both_valid_instace_mask).squeeze(1)

        # use the fewer points to sample the instance ids
        unique_sampled_inst_ids1 = torch.unique(selected_instance_id1)
        unique_sampled_inst_ids2 = torch.unique(selected_instance_id2)

        use_first_frame = True
        if True:  # len(unique_sampled_inst_ids1) <= len(unique_sampled_inst_ids2):
            unique_sampled_inst_ids = unique_sampled_inst_ids1
            sampled_mask = (sampled_instance_mask1.unsqueeze(-1) == unique_sampled_inst_ids)
        else:
            unique_sampled_inst_ids = unique_sampled_inst_ids2
            sampled_mask = (sampled_instance_mask2.unsqueeze(-1) == unique_sampled_inst_ids)
            use_first_frame = False

        final_inst_ids1, final_inst_ids2 = [], []
        for i in range(len(unique_sampled_inst_ids)):
            valid_indices = torch.nonzero(sampled_mask[:, i])  # (M, 1)

            selected_indix = torch.randperm(valid_indices.size(0))[:1]
            selected_indix = valid_indices[selected_indix].squeeze(1)
            if use_first_frame:
                selected_pt2 = sample_pts2[selected_indix]  # (1, 2)
                final_inst_id2 = instance_mask2[selected_pt2[:, 1], selected_pt2[:, 0]]
                
                final_inst_ids1.append(unique_sampled_inst_ids[i].item())
                final_inst_ids2.append(final_inst_id2.item())
            else:
                selected_pt1 = sample_pts1[selected_indix]
                final_inst_id1 = instance_mask1[selected_pt1[:, 1], selected_pt1[:, 0]]
                
                final_inst_ids1.append(final_inst_id1.item())
                final_inst_ids2.append(unique_sampled_inst_ids[i].item())

        return (selected_instance_id1, 
                selected_instance_id2, 
                all_selected_points_indices,
                final_inst_ids1,
                final_inst_ids2)
        
    def loss(self, 
             input_dict: Dict,
             seq_len=2,
             rand_view_idx=None,
             **kwargs):
        
        losses = dict()

        assert 'render_semantic' in input_dict
        assert 'sample_pts_pad' in kwargs

        semantic_pred = input_dict['render_semantic']  # (seq*b, num_cam, c, h, w)
        semantic_pred = rearrange(
            semantic_pred, 
            '(seq b) num_cam c h w -> seq b num_cam h w c', seq=seq_len)
        
        sample_pts = kwargs['sample_pts_pad']  # (seq*b, num_cam, N, 2)
        if rand_view_idx is not None:
                sample_pts = sample_pts[:, rand_view_idx]
            
        sample_pts = rearrange(sample_pts,
                               '(seq b) num_cam N uv -> seq b num_cam N uv',
                               seq=seq_len)
    
        if self.use_pointwise_align:
            loss_pointwise_align = self.compute_pointwise_loss_with_flow(
                semantic_pred[0], semantic_pred[1],
                sample_pts[0], sample_pts[1],
                loss_weight=self.loss_pointwise_align_weight)
            losses['loss_pointwise_align'] = loss_pointwise_align
        
        if self.use_semantic_align:
            ## compare the temporal rendering semantic map
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
            elif self.semantic_align_type == 'query_flow_all':
                ## Here we adapt the background flow info to align the temporal
                # instance masks
                assert 'instance_masks' in kwargs
                instance_mask = kwargs['instance_masks']  # (seq*b, num_cam, h, w)
                if rand_view_idx is not None:
                    instance_mask = instance_mask[:, rand_view_idx]
                
                instance_mask = rearrange(instance_mask,
                                          '(seq b) num_cam h w -> seq b num_cam h w',
                                          seq=seq_len)
                loss_semantic_contrast_dict = self.compute_semantic_contrast_loss_with_flow(
                    semantic_pred[0], semantic_pred[1],
                    instance_mask[0], instance_mask[1],
                    sample_pts[0], sample_pts[1],
                    loss_weight=self.loss_semantic_align_weight,
                    **kwargs)
            else:
                raise NotImplementedError()
            
            losses.update(loss_semantic_contrast_dict)
        
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
                                         sample_pts1,
                                         sample_pts2,
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

        batched_loss = []

        for i in range(bs):
            all_cams_feats1 = []
            all_cams_feats2 = []

            for j in range(num_cam):
                num_valid_pts = int(sample_pts1[i, j, -1, 0])

                curr_sample_pts1 = sample_pts1[i, j, :num_valid_pts]
                curr_sample_pts2 = sample_pts2[i, j, :num_valid_pts]

                curr_render_semantic1 = render_semantic1[i, j][curr_sample_pts1[:, 1], curr_sample_pts1[:, 0]]
                curr_render_semantic2 = render_semantic2[i, j][curr_sample_pts2[:, 1], curr_sample_pts2[:, 0]]
                all_cams_feats1.append(curr_render_semantic1)
                all_cams_feats2.append(curr_render_semantic2)
        
            all_cams_feats1 = torch.cat(all_cams_feats1, dim=0)  # (N, c)
            all_cams_feats2 = torch.cat(all_cams_feats2, dim=0)  # (N, c)
            loss_pointwise_all_cams = F.mse_loss(
                all_cams_feats1, all_cams_feats2, reduction='mean')

            batched_loss.append(loss_pointwise_all_cams)
        
        ## compute the loss
        loss_semantic_align_pointwise = torch.stack(batched_loss).mean()

        return loss_semantic_align_pointwise * loss_weight

    def compute_semantic_contrast_loss_with_flow(self,
                                                 render_semantic_1,
                                                 render_semantic_2,
                                                 instance_mask_1,
                                                 instance_mask_2,
                                                 sample_pts_1,
                                                 sample_pts_2,
                                                 loss_weight=1.0,
                                                 **kwargs):
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
        instance_mask1_valid = instance_mask_1.clone()
        instance_mask1_valid[instance_mask1_valid == -1] = instance_mask1_valid.max() + 1
        grouped_semantic_1 = scatter_mean(render_semantic_1,
                                          instance_mask1_valid,
                                          dim=2)
        
        instance_mask2_valid = instance_mask_2.clone()
        instance_mask2_valid[instance_mask2_valid == -1] = instance_mask2_valid.max() + 1
        grouped_semantic_2 = scatter_mean(render_semantic_2,
                                          instance_mask2_valid,
                                          dim=2)
        
        VISUALIZE = False

        batched_loss = []
        for i in range(bs):
            all_camera_grouped_feats_1 = []
            all_camera_grouped_feats_2 = []
            for j in range(num_cam):
                curr_instance_map_1 = instance_mask_1[i, j].reshape(h, w)
                curr_instance_map_2 = instance_mask_2[i, j].reshape(h, w)

                curr_sample_pts_1 = sample_pts_1[i, j]
                curr_sample_pts_2 = sample_pts_2[i, j]

                # NOTE: here we only select the pixels from the first frame
                selected_points_indices = self.sample_both_valid_points(
                    curr_instance_map_1, curr_sample_pts_1,
                    curr_instance_map_2, curr_sample_pts_2)

                if selected_points_indices is None:
                    continue

                selected_points1 = curr_sample_pts_1[selected_points_indices]
                selected_points2 = curr_sample_pts_2[selected_points_indices]

                # fetch the corresponding instance mask ids
                instance_id_list_1 = curr_instance_map_1[selected_points1[:, 1], selected_points1[:, 0]]
                instance_id_list_2 = curr_instance_map_2[selected_points2[:, 1], selected_points2[:, 0]]

                if VISUALIZE:  # DEBUG ONLY
                    if j != 1:
                        continue
                    import matplotlib.pyplot as plt
                    import numpy as np
                    import cv2
                    from mmcv.image.photometric import imdenormalize

                    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
                    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

                    render_gt_img = kwargs['render_gt_img']  # (b*seq_len, num_cam, 3, h, w)
                    render_gt_img = rearrange(
                        render_gt_img, 
                        '(b seq_len) (n_cam n_frame) c h w -> seq_len b n_frame n_cam h w c', 
                        seq_len=2, n_frame=3)[:, :, 0]

                    render_gt_img1 = render_gt_img[0][i, j].cpu().numpy()
                    render_gt_img2 = render_gt_img[1][i, j].cpu().numpy()

                    render_gt_img1 = imdenormalize(render_gt_img1, mean, std, to_bgr=False)
                    render_gt_img2 = imdenormalize(render_gt_img2, mean, std, to_bgr=False)

                    selected_points1 = selected_points1.cpu().numpy()
                    selected_points2 = selected_points2.cpu().numpy()

                    img1 = np.zeros((h, w, 3))
                    img2 = np.zeros((h, w, 3))

                    # draw circles
                    center = np.median(selected_points1, axis=0)
                    set_max = range(128)
                    colors = {m: i for i, m in enumerate(set_max)}
                    colors = {m: (255 * np.array(plt.cm.hsv(i/float(len(colors))))[:3][::-1]).astype(np.int32)
                                for m, i in colors.items()}
                    for k in range(selected_points1.shape[0]):
                        pt1 = (int(selected_points1[k, 0]), int(selected_points1[k, 1]))
                        pt2 = (int(selected_points2[k, 0]), int(selected_points2[k, 1]))

                        coord_angle = np.arctan2(pt1[1] - center[1], pt1[0] - center[0])
                        corr_color = np.int32(64 * coord_angle / np.pi) % 128
                        color = tuple(colors[corr_color].tolist())

                        render_gt_img1 = cv2.circle(render_gt_img1, pt1, 1, color, -1)
                        render_gt_img2 = cv2.circle(render_gt_img2, pt2, 1, color, -1)
                        
                        img1 = cv2.circle(img1, pt1, 2, color, -1)
                        img2 = cv2.circle(img2, pt2, 2, color, -1)
                    
                    # concatenate the images
                    img_bar = np.ones((h, 10, 3)) * 255

                    render_gt_img_concat = np.concatenate(
                        [render_gt_img1, img_bar, render_gt_img2], axis=1)
                    
                    img_concat = np.concatenate([img1, img_bar, img2], axis=1)

                    # vertical concatenate
                    img_save = np.concatenate([render_gt_img_concat, img_concat], axis=0)
                    cv2.imwrite(f'debug_nerf_head_{i}_{j}.png', img_save.astype(np.uint8))
                    exit()

                # obtain the grouped semantic features according to the instance ids
                curr_grouped_feats_1 = grouped_semantic_1[i, j, instance_id_list_1]  # (N, C)
                curr_grouped_feats_2 = grouped_semantic_2[i, j, instance_id_list_2]
                
                all_camera_grouped_feats_1.append(curr_grouped_feats_1)
                all_camera_grouped_feats_2.append(curr_grouped_feats_2)
            
            all_camera_grouped_feats_1 = torch.concat(all_camera_grouped_feats_1, dim=0)[None]
            all_camera_grouped_feats_2 = torch.concat(all_camera_grouped_feats_2, dim=0)[None]
            curr_batch_loss = compute_tig_feat_align_loss(
                all_camera_grouped_feats_1, all_camera_grouped_feats_2,
                loss_inter_instance_weight=self.loss_inter_instance_weight,
                loss_inter_channel_weight=self.loss_inter_channel_weight
                )
            batched_loss.append(curr_batch_loss)

        loss_semantic_temporal_align = compute_batched_loss_dict_list(batched_loss)
        
        return loss_semantic_temporal_align

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
            if valid_indices.size(0) <= num_selected_pixels:
                continue

            # randomly select the points
            selected_indices = torch.randperm(valid_indices.size(0))[:num_selected_pixels]
            selected_indices_raw = valid_indices[selected_indices].squeeze()
            all_selected_pixels.extend(selected_indices_raw)
        
        all_selected_pixels = torch.stack(all_selected_pixels, dim=0)
        return all_selected_pixels
    
    def sample_both_valid_points(self, 
                                 instance_mask1, sample_pts1,
                                 instance_mask2, sample_pts2):
        num_selected_pixels = 1  # the number of selected pixels for each instance

        num_valid_pts = sample_pts1[-1, 0]
        sample_pts1 = sample_pts1[:num_valid_pts]
        
        # obtain the valid points
        sampled_instance_mask = instance_mask1[sample_pts1[:, 1], sample_pts1[:, 0]]  # (N,)
        unique_sampled_inst_ids = torch.unique(sampled_instance_mask)
        sampled_mask = (sampled_instance_mask.unsqueeze(-1) == unique_sampled_inst_ids)

        all_selected_points_indices = []
        for i in range(len(unique_sampled_inst_ids)):
            if unique_sampled_inst_ids[i] == -1:
                continue

            valid_indices = torch.nonzero(sampled_mask[..., i])
            if valid_indices.size(0) < 3:
                continue
            
            # randomly select the points
            selected_indices = torch.randperm(valid_indices.size(0))[:num_selected_pixels]
            selected_indices_raw = valid_indices[selected_indices].squeeze(1)

            # check whether this point is valid in the second frame
            instance_id2 = sample_pts2[selected_indices_raw]
            if torch.any(instance_mask2[instance_id2[:, 1], instance_id2[:, 0]] == -1):
                continue
            all_selected_points_indices.extend(selected_indices_raw)
        
        if len(all_selected_points_indices) == 0:
            return None
        all_selected_points_indices = torch.stack(all_selected_points_indices, dim=0)
        return all_selected_points_indices

