'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-06-30 00:17:08
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''


_base_ = ['./lp-cf-ms-l2c-use-mask-quarter.py']

model = dict(
    freeze_teacher_branch=False,
    logits_as_prob_feat=True,

    teacher_model=dict(
        use_free_occ_token=True),

    occ_distill_head=dict(
        type='OccDistillHeadV2',
        loss_high_feat=dict(
            type='L1Loss',
            loss_weight=0.1))
)

load_from = "bevdet-occ-r50-4d-stereo-24e-student_lidar-occ-teacher-model-quarter.pth"