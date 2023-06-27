'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-06-27 10:35:03
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''


_base_ = ['./lp-cf-ms-l2c-use-mask-quarter.py']

model = dict(
    freeze_teacher_branch=False,
    logits_as_prob_feat=False,

    occ_distill_head=dict(
        type='OccDistillHead',
        loss_high_feat=dict(
            type='L1Loss',
            loss_weight=1.0),
        loss_prob_feat=dict(
            type='L1Loss', 
            loss_weight=0.1)),
)

load_from = "bevdet-occ-r50-4d-stereo-24e-student_lidar-occ-teacher-model-quarter.pth"