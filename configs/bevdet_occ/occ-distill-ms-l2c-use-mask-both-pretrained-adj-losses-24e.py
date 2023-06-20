'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-06-19 22:08:45
Email: haimingzhang@link.cuhk.edu.cn
Description: Both the teacher model and student model are pre-trained.
             Here we adjust different losses.
Information: 
    ms: multi-sweeps
    l2c: lidar to camera
'''

_base_ = ['./occ-distill-ms-l2c-use-mask-both-pretrained-24e.py']

model = dict(
    occ_distill_head=dict(
        type='OccDistillHead',
        loss_high_feat=dict(
            type='L1Loss',
            loss_weight=1.0),
        loss_prob_feat=dict(
            type='L1Loss', 
            loss_weight=0.1)),
)