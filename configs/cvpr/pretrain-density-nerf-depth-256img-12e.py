'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-10-27 15:02:30
Email: haimingzhang@link.cuhk.edu.cn
Description: Pretrain the BEVDet with rendered depth loss.
'''

_base_ = ['./pretrain-dvgo-nerf-depth-256img-12e.py']

model = dict(
    ## the nerf decoder head
    nerf_head=dict(
        render_type='density',  # ['prob', 'density', 'DVGO']
    )
)