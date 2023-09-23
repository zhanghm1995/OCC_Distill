'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-09-23 16:15:31
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

_base_ = ['bevdet-lidar-occ-voxel-24e.py']

voxel_size = [0.1, 0.1, 0.1]

model = dict(
    type='BEVLidarOCC',
    pts_voxel_layer=dict(
        max_num_points=10, 
        voxel_size=voxel_size, 
        max_voxels=(90000, 120000)),
    pts_middle_encoder=dict(
        type='SparseEncoderLidarOCC',
        in_channels=5,
        sparse_shape=[65, 800, 800],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
)

