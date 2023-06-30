'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-06-20 19:47:30
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mmcv.runner import BaseModule
from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet.models.builder import build_loss
from ..builder import HEADS
from mmcv.runner import BaseModule


@HEADS.register_module()
class OccSimpleHead(BaseModule):

    def __init__(self,
                 num_classes,
                 use_mask=True,
                 mask_key="mask_camera",
                 in_channels=32,
                 out_channels=32,
                 loss_occ=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                 **kwargs):
        
        super(OccSimpleHead, self).__init__(**kwargs)
        
        _default_mask_keys = ('mask_camera', 'mask_lidar')
        
        if use_mask:
            assert mask_key in _default_mask_keys

        self.num_classes = num_classes
        self.use_mask = use_mask
        self.mask_key = mask_key

        self.final_conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv3d'))
        
        self.predicter = nn.Sequential(
            nn.Linear(out_channels, out_channels*2),
            nn.Softplus(),
            nn.Linear(out_channels*2, num_classes))
        
        self.loss_occ = build_loss(loss_occ)

    def loss_single(self, voxel_semantics, mask, preds):
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17

        loss_ = dict()
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            mask = mask.to(torch.int32)
            voxel_semantics=voxel_semantics.reshape(-1)
            preds=preds.reshape(-1,self.num_classes)
            mask = mask.reshape(-1)
            num_total_samples = mask.sum()
            loss_occ = self.loss_occ(preds,
                                     voxel_semantics, 
                                     mask, 
                                     avg_factor=num_total_samples)
            loss_['loss_occ'] = loss_occ
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics,)
            loss_['loss_occ'] = loss_occ
        return loss_
    
    def forward(self, x):
        if isinstance(x, list):
            feats = x[0]
        
        if feats.dim() == 4:
            feats = rearrange(
                feats, 'b (c Dimz) DimH DimW -> b c Dimz DimH DimW', Dimz=16)
        
        occ_pred = self.final_conv(feats).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        occ_pred = self.predicter(occ_pred)
        return occ_pred
    
    def forward_train(self, 
                      inputs, 
                      **kwargs):
        """Forward function for training.

        Args:
            inputs (list[torch.Tensor]): List of multi-level point features.
            img_metas (list[dict]): Meta information of each sample.
            voxel_semantics (torch.Tensor): The GT semantic voxesl.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        occ_logits = self.forward(inputs)
        
        mask = kwargs[self.mask_key] if self.use_mask else None
        voxel_semantics = kwargs['voxel_semantics']

        losses = dict()
        loss_occ = self.loss_single(voxel_semantics, mask, occ_logits)
        losses.update(loss_occ)
        return losses


@HEADS.register_module()
class OccDistillHead(BaseModule):
    """The head for calculating the distillation loss.

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
        
        super(OccDistillHead, self).__init__(init_cfg=init_cfg)
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
             mask=None):
        
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
    

@HEADS.register_module()
class OccDistillHeadV2(BaseModule):
    """The head for calculating the distillation loss.

    Args:
        BaseModule (_type_): _description_
    """

    def __init__(self,
                 detach_target=True,
                 loss_high_feat=dict(
                    type='L1Loss',
                    loss_weight=1.0),
                 **kwargs):
        
        super(OccDistillHeadV2, self).__init__()

        self.detach_target = detach_target
        
        if loss_high_feat is not None:
            self.loss_high_feat = build_loss(loss_high_feat)
        
    @property
    def with_high_feat_loss(self):
        """bool: Whether calculate the high-level feature alignment loss."""
        return hasattr(self,
                       'loss_high_feat') and self.loss_high_feat is not None

    def loss(self, 
             teacher_feats_list, 
             student_feats_list,
             use_distill_mask=False,
             mask=None):
        
        losses = dict()

        if use_distill_mask:
            assert mask is not None

            mask1 = repeat(mask, 'b h w d -> b c d h w', 
                            c=student_feats_list[1].shape[1])
            mask1 = mask1.to(torch.float32)
            num_total_samples1 = mask1.sum()

            if self.detach_target:
                teacher_feat = teacher_feats_list[1].detach()
            high_feat_loss = self.loss_high_feat(
                student_feats_list[1], teacher_feat, 
                mask1, avg_factor=num_total_samples1)
            
            losses['high_feat_loss'] = high_feat_loss
        else:
            high_feat_loss = self.loss_high_feat(student_feats_list[1],
                                                 teacher_feats_list[1])
            losses['high_feat_loss'] = high_feat_loss
        return losses
    