'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-07-02 14:48:24
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

_base_ = ['./lp-cf-ms-l2c-use-mask-quarter.py']

model = dict(
    freeze_teacher_branch=True,
    logits_as_prob_feat=True,

    occ_distill_head=dict(
        type='OccDistillHeadV2',
        use_kl_loss=True,
        only_align_fg=True,
        loss_high_feat=dict(
            type='L1Loss',
            loss_weight=0.1),
        loss_prob_feat=dict(
            type='KnowledgeDistillationKLDivLoss', 
            loss_weight=0.1)),
)

load_from = "bevdet-r50-4d-stereo-as-student-lidar-voxel-w-aug-as-teacher-quarter.pth"