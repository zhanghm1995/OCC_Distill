'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-10-17 17:45:17
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
# Copyright (c) OpenMMLab. All rights reserved.
import os
import mmcv
import torch
import cv2
import numpy as np
from tqdm import tqdm

from .builder import DATASETS
from .nuscenes_dataset_occ import NuScenesDatasetOccpancy
from .occ_metrics import Metric_mIoU, Metric_FScore, Metric_OccIoU


@DATASETS.register_module()
class NuScenesDatasetBinaryOccpancy(NuScenesDatasetOccpancy):
    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        self.occ_eval_metrics = Metric_OccIoU(
            num_classes=2,
            use_lidar_mask=False,
            use_image_mask=False)

        print('\nStarting Evaluation...')
        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]

            occ_gt = np.load(os.path.join(info['occ_path'], 'labels.npz'))
            gt_semantics = occ_gt['semantics']
            mask_lidar = occ_gt['mask_lidar'].astype(bool)
            mask_camera = occ_gt['mask_camera'].astype(bool)

            ## DEBUG ONLY
            gt_semantics = mask_lidar
            self.occ_eval_metrics.add_batch(occ_pred, gt_semantics, 
                                            mask_lidar, mask_camera, 
                                            already_is_binary_gt=True)

        return self.occ_eval_metrics.count_miou()

    def format_results(self, occ_results, submission_prefix, **kwargs):
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            sample_token = info['token']
            save_path = os.path.join(submission_prefix, '{}.npz'.format(sample_token))
            np.savez_compressed(save_path, occ_pred.astype(np.uint8))
        print('\nFinished.')