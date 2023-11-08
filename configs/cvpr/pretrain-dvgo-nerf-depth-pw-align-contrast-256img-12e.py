'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-11-08 17:05:58
Email: haimingzhang@link.cuhk.edu.cn
Description: Pretrain the BEVDet with rendered depth loss + pointwise align loss by using the
contrastive loss.
'''

_base_ = ['./pretrain-dvgo-nerf-depth-pw-align-256img-12e.py']

model = dict(
    use_projector=True,
    
    pretrain_head=dict(
        type='NeRFOccPretrainHead',
        use_semantic_align=False, 
        use_pointwise_align=True,
        pointwise_align_type='contrastive',
        loss_pointwise_align_weight=0.1
    ),
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
])
