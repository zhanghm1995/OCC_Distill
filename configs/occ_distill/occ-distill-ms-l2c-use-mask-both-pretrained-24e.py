'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-06-19 22:08:45
Email: haimingzhang@link.cuhk.edu.cn
Description: Both the teacher model and student model are pre-trained.
Information: 
    ms: multi-sweeps
    l2c: lidar to camera
'''

_base_ = ['./bevdet-lidar-occ-voxel-multi-sweeps-lidar-distill-camera-24e.py']

model = dict(
    use_distill_mask=True,
)

load_from = "bevdet-occ-r50-4d-stereo-24e-student_lidar-occ-teacher-model.pth"