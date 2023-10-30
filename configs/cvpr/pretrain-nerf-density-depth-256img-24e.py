'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-10-27 15:02:30
Email: haimingzhang@link.cuhk.edu.cn
Description: Pretrain the BEVDet with rendered depth loss.
'''

_base_ = ['./pretrain-nerf-dvgo-depth-256img-24e.py']

model = dict(
    ## the nerf decoder head
    nerf_head=dict(
        render_type='density',  # ['prob', 'density', 'DVGO']
    )
)


data = dict(
    samples_per_gpu=8,
    workers_per_gpu=6,
)