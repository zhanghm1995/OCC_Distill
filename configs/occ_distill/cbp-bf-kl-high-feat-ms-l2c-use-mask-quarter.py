'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-06-30 19:34:47
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
        use_kl_loss=True,
        loss_high_feat=dict(
            type='L1Loss',
            loss_weight=0.1),
        loss_prob_feat=dict(
            type='KnowledgeDistillationKLDivLoss', 
            loss_weight=0.1))
)

load_from = "bevdet-r50-4d-stereo-cbgs-as-student-model.pth"
