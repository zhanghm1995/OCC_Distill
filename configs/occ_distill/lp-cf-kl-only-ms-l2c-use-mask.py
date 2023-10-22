'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-06-26 19:51:19
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

_base_ = ['./lp-cf-ms-l2c-use-mask.py']

model = dict(
    logits_as_prob_feat=True,

    occ_distill_head=dict(
        type='OccDistillHead',
        use_kl_loss=True,
        loss_high_feat=dict(
            type='L1Loss',
            loss_weight=0.0),
        loss_prob_feat=dict(
            type='KnowledgeDistillationKLDivLoss', 
            loss_weight=0.005)),
)