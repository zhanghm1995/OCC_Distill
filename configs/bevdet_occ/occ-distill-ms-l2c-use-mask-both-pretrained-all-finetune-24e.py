'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-06-22 00:18:39
Email: haimingzhang@link.cuhk.edu.cn
Description: Both the teacher model and student model are pre-trained.
             And we fine-tune all models.
Information: 
    ms: multi-sweeps
    l2c: lidar to camera
'''

_base_ = ['./occ-distill-ms-l2c-use-mask-both-pretrained-24e.py']

model = dict(
    freeze_teacher_branch=False,
    
    occ_distill_head=dict(
        type='OccDistillHead',
        loss_high_feat=dict(
            type='L1Loss',
            loss_weight=1.0),
        loss_prob_feat=dict(
            type='L1Loss', 
            loss_weight=0.1)),
)