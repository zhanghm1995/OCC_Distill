'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-10-27 14:32:54
Email: haimingzhang@link.cuhk.edu.cn
Description: Align all the instance mask from all of the objects, including
foreground and background objects.
'''

_base_ = ['pretrain-nerf-d-s-flow-inst-mask-256img-24e.py']

model = dict(
    pretrain_head=dict(
        semantic_align_type='query_flow_all',
    )
)
