'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-09-13 07:22:43
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

_base_ = ['./bevdet-lidar-occ-voxel-24e.py']

use_checkpoint = True
voxel_channels = [32, 32*2, 32*4]
voxel_out_indices = [0, 1, 2]

model = dict(
    pts_bev_encoder_backbone=dict(
        _delete_=True,
        type='CustomResNet3DFBOCC',
        depth=18,
        with_cp=use_checkpoint,
        block_strides=[1, 2, 2],
        n_input_channels=128,
        block_inplanes=voxel_channels,
        out_indices=voxel_out_indices,
        norm_cfg=dict(type='BN3d', requires_grad=True))
)

