'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-05-12 00:22:58
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

_base_ = ['./openscene-occ-r50-4d-stereo-24e.py']

pred_binary_occ = True
is_private_test = False

model = dict(
    type='BEVStereo4DOCCOpenScene',
    num_classes=2,
    pred_binary_occ=pred_binary_occ,

    # loss_occ=dict(
    #     type='FocalLoss',
    #     use_sigmoid=True,
    #     gamma=2.0,
    #     alpha=0.25,
    #     loss_weight=1.0,
    #     reduction='none'),
    loss_occ=dict(
        type='CrossEntropyLoss',
        class_weight=[0.2, 0.8],
        use_sigmoid=False,
        loss_weight=1.0),
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        pred_binary_occ=pred_binary_occ),
    val=dict(
        pred_binary_occ=pred_binary_occ),
    test=dict(
        pred_binary_occ=pred_binary_occ,
        is_private_test=is_private_test),
)


# load_from = "ckpts/bevdet-r50-4d-stereo-cbgs.pth"
load_from = "work_dirs/openscene-occ-r50-4d-stereo-24e-binary/epoch_22_ema.pth"