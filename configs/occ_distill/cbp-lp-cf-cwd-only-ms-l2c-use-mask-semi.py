'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-08-14 11:12:46
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

_base_ = ['./lp-cf-ms-l2c-use-mask-quarter.py']

data_root = 'data/nuscenes/'
data = dict(
    train=dict(
        ann_file=data_root + 'bevdetv3-lidarseg-nuscenes-semi_infos_train.pkl'
    )
)

model = dict(
    freeze_teacher_branch=True,
    logits_as_prob_feat=True,
    use_cross_kd=False,

    occ_distill_head=dict(
        type='OccDistillHeadV1',
        use_cwd_loss=True,
        loss_cwd_criterion=dict(
            type='CriterionCWD',
            with_mask=True,
            norm_type='channel',
            divergence='kl',
            temperature=4,)),
)

optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-2)

load_from = "old-bevdet-r50-4d-stereo-as-student-lidar-voxel-ms-w-aug-new-as-teacher.pth"