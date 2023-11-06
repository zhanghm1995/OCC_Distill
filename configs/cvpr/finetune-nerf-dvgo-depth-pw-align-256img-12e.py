'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-11-04 20:24:27
Email: haimingzhang@link.cuhk.edu.cn
Description: Fintune
'''

_base_ = ['./fintune-nerf-dvgo-depth-256img-24e.py']

load_from = 'work_dirs/pretrain-nerf-dvgo-depth-pw-align-256img-12e-DVGO-weigth-10/latest.pth'
