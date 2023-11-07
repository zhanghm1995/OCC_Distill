'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-11-07 09:41:33
Email: haimingzhang@link.cuhk.edu.cn
Description: Pretrain the BEVDet with pointwise align loss only by using the
contrastive loss.
'''

_base_ = ['./pretrain-nerf-dvgo-pw-align-256img-12e.py']

model = dict(
    pretrain_head=dict(
        type='NeRFOccPretrainHead',
        use_semantic_align=False, 
        use_pointwise_align=True,
        pointwise_align_type='contrastive',
        loss_pointwise_align_weight=0.1
    ),

    ## the nerf decoder head
    nerf_head=dict(
        render_type='density',  # ['DVGO', 'prob', 'density']
    ),
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
])
