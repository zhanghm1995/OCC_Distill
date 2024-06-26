'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-06-28 21:18:45
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

_base_ = ['./lp-cf-ms-l2c-use-mask-quarter.py']

model = dict(
    freeze_teacher_branch=True,
    logits_as_prob_feat=True,
    use_cross_kd=True,

    occ_distill_head=dict(
        type='OccDistillHead',
        use_kl_loss=True,
        loss_high_feat=dict(
            type='L1Loss',
            loss_weight=0.0),
        loss_prob_feat=dict(
            type='KnowledgeDistillationKLDivLoss', 
            loss_weight=0.1,
            T=1)),
)

optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-2)

load_from = "bevdet-r50-4d-stereo-as-student-lidar-voxel-w-aug-as-teacher-quarter.pth"