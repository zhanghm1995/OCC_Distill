'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-05-23 11:46:54
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-05-09 17:03:24
Email: haimingzhang@link.cuhk.edu.cn
Description: Predict the binary occupancy.
'''

_base_ = ['./openscene-fusion-occ-r50-4d-stereo-24e-use-pred-pc.py']

pred_binary_occ = True
is_private_test = False

model = dict(
    type='BEVFusionStereo4DOCCOpenScene',
    num_classes=2,
    pred_binary_occ=pred_binary_occ,
    use_lovasz_loss=False,
    balance_cls_weight=False,

    loss_occ=dict(
        type='FocalLoss',
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=1.0,
        reduction='none'),
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        pred_binary_occ=pred_binary_occ),
    val=dict(
        pred_binary_occ=pred_binary_occ),
    test=dict(
        pred_binary_occ=pred_binary_occ,
        is_private_test=is_private_test),
)


load_from = "ckpts/bevdet-r50-4d-stereo-cbgs.pth"