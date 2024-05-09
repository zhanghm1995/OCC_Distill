'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-05-09 17:03:24
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

_base_ = ['./openscene-fusion-occ-r50-4d-stereo-24e.py']

pred_binary_occ = True

model = dict(
    type='BEVFusionStereo4DOCCOpenScene',
    pred_binary_occ=pred_binary_occ,
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        pred_binary_occ=pred_binary_occ),
    val=dict(
        pred_binary_occ=pred_binary_occ),
    test=dict(
        pred_binary_occ=pred_binary_occ),
)


load_from = "ckpts/bevdet-r50-4d-stereo-cbgs.pth"