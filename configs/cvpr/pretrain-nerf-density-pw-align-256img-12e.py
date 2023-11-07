'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-11-07 00:26:06
Email: haimingzhang@link.cuhk.edu.cn
Description: Pretrain the BEVDet with pointwise align loss only.
'''

_base_ = ['./pretrain-nerf-dvgo-pw-align-256img-12e.py']

model = dict(
    pretrain_head=dict(
        type='NeRFOccPretrainHead',
        use_semantic_align=False, 
        use_pointwise_align=True,
        loss_pointwise_align_weight=1.0
    ),

    ## the nerf decoder head
    nerf_head=dict(
        render_type='density',  # ['DVGO', 'prob', 'density']
    ),
)
