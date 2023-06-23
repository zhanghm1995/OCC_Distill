'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-06-23 21:44:47
Email: haimingzhang@link.cuhk.edu.cn
Description: 
Information: 
    ms: multi-sweeps
    l2c: lidar to camera
    bp: both-pretrained
    bf: both-finetune
'''

_base_ = ['./occ-distill-ms-l2c-use-mask-both-pretrained-24e.py']

model = dict(
    freeze_teacher_branch=False,
    logits_as_prob_feat=True,

    occ_distill_head=dict(
        type='OccDistillHead',
        use_kl_loss=True,
        loss_high_feat=None,
        loss_prob_feat=dict(
            type='KnowledgeDistillationKLDivLoss', 
            loss_weight=1.0)),
)