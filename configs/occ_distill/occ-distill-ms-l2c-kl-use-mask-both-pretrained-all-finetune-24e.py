'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-06-22 09:10:49
Email: haimingzhang@link.cuhk.edu.cn
Description: Here we use the KL loss to compute the prob feat
'''

_base_ = ['./occ-distill-ms-l2c-use-mask-both-pretrained-24e.py']

model = dict(
    freeze_teacher_branch=False,
    logits_as_prob_feat=True,

    occ_distill_head=dict(
        type='OccDistillHead',
        use_kl_loss=True,
        loss_high_feat=dict(
            type='L1Loss',
            loss_weight=0.5),
        loss_prob_feat=dict(
            type='KnowledgeDistillationKLDivLoss', 
            loss_weight=1.0)),
)