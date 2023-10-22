'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-06-30 00:48:19
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''


_base_ = ['./lp-cf-ms-l2c-use-mask-quarter.py']

model = dict(
    freeze_teacher_branch=False,
    logits_as_prob_feat=True,
    
    occ_distill_head=dict(
        type='OccDistillHead',
        use_kl_loss=True,
        loss_high_feat=dict(
            type='L1Loss',
            loss_weight=0.0),
        loss_prob_feat=dict(
            type='KnowledgeDistillationKLDivLoss', 
            loss_weight=0.1,
            T=1, 
            detach_target=False)),
)

paramwise_cfg = dict(custom_keys={
    '.teacher_model': dict(lr_mult=0.01)})

load_from = "bevdet-r50-4d-stereo-cbgs-as-student-model.pth"