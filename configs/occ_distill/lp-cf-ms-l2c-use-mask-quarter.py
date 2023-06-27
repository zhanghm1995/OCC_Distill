'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-06-27 10:21:03
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

_base_ = ['./lp-cf-ms-l2c-use-mask.py']

data_root = 'data/nuscenes/'
data = dict(
    train=dict(
        ann_file=data_root + 'quarter-bevdetv3-nuscenes_infos_train.pkl'
    )
)

load_from = "bevdet-lidar-occ-voxel-ms-w-aug-24e_teacher_model-quarter.pth"
