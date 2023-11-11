'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-11-10 10:41:45
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

_base_ = ['./pretrain-dvgo-nerf-depth-pw-align-tarl-256img-12e.py']

model = dict(
    pretrain_head=dict(
        use_semantic_align=True, 
        use_pointwise_align=False,
        loss_semantic_align_weight=0.1
    ),
)
