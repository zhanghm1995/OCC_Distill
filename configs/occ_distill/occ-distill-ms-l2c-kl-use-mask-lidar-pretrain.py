'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-06-19 22:08:45
Email: haimingzhang@link.cuhk.edu.cn
Description: Use the pretrained lidar and KL loss.
Information: 
    ms: multi-sweeps
    l2c: lidar to camera
'''

_base_ = ['./bevdet-lidar-occ-voxel-multi-sweeps-lidar-distill-camera-24e.py']

model = dict(
    use_distill_mask=True,
    logits_as_prob_feat=True,

    occ_distill_head=dict(
        type='OccDistillHead',
        use_kl_loss=True,
        loss_high_feat=dict(
            type='L1Loss',
            loss_weight=0.5),
        loss_prob_feat=dict(
            type='KnowledgeDistillationKLDivLoss', 
            loss_weight=1.0,
            T=1)),
)
