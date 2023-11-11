'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-11-11 17:09:41
Email: haimingzhang@link.cuhk.edu.cn
Description: Pretrain the BEVDet with rendered depth loss in the original 
BEVDet volume feature space.
'''

_base_ = ['./pretrain-dvgo-nerf-depth-256img-12e.py']

model = dict(
    type='BEVStereo4DOCCTemporalNeRFPretrainV2Origin',

    ## the nerf decoder head
    nerf_head=dict(
        render_type='density',  # ['prob', 'density', 'DVGO']
    ),

    ## not used but need to be set
    loss_occ=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0),
)