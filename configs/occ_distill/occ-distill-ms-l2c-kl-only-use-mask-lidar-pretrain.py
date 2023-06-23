'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-06-23 21:35:21
Email: haimingzhang@link.cuhk.edu.cn
Description: Only use the KL loss, and train the camera branch from scratch.
'''

_base_ = ['./bevdet-lidar-occ-voxel-multi-sweeps-lidar-distill-camera-24e.py']

model = dict(
    use_distill_mask=True,
    logits_as_prob_feat=True,

    occ_distill_head=dict(
        type='OccDistillHead',
        use_kl_loss=True,
        loss_high_feat=None,
        loss_prob_feat=dict(
            type='KnowledgeDistillationKLDivLoss', 
            loss_weight=1.0,
            T=1)),
)
