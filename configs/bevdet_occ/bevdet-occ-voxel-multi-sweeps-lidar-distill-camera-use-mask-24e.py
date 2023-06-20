'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-06-16 21:41:17
Email: haimingzhang@link.cuhk.edu.cn
Description: Use the masked features to compute the distillation losses.
'''

_base_ = ['./bevdet-lidar-occ-voxel-multi-sweeps-lidar-distill-camera-24e.py']

model = dict(
    use_distill_mask=True,
    freeze_teacher_branch=True,
)
