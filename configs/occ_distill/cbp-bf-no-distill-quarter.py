'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-06-26 11:19:44
Email: haimingzhang@link.cuhk.edu.cn
Description: Camera branch load the pretrained BEVDet model, and train both the
camera and lidar branch without distill losses.
'''

_base_ = ['./lp-cf-ms-l2c-use-mask-quarter.py']

model = dict(
    freeze_teacher_branch=False,
    use_distill_mask=True,
    
    occ_distill_head=dict(
        type='OccDistillHead',
        loss_high_feat=dict(
            type='L1Loss',
            loss_weight=0.0),
        loss_prob_feat=dict(
            type='L1Loss', 
            loss_weight=0.0))
)

load_from = "bevdet-r50-4d-stereo-cbgs-as-student-model.pth"