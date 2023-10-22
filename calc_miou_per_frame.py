'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-08-18 11:41:50
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import pickle

from mmdet3d.datasets.occ_metrics_new import Metric_mIoU


def compute_radocc_metrics():
    occ_eval_metrics = Metric_mIoU(
        num_classes=18,
        use_lidar_mask=False,
        use_image_mask=True)

    # load the pickle
    # read the validation occ token
    pickle_path = "data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_val.pkl"
    with open(pickle_path, 'rb') as f:
        val_infos = pickle.load(f)['infos']

    pred_occ_dir = "./occ_submission_RadOCC"

    print('\nStarting Evaluation...')
    sample_token_list, miou_list = [], []
    for index, info in tqdm(enumerate(val_infos)):
        occ_path = info['occ_path']
        occ_path = osp.join(occ_path, "labels.npz")
        occ_gt = np.load(occ_path)

        gt_semantics = occ_gt['semantics']
        mask_lidar = occ_gt['mask_lidar'].astype(bool)
        mask_camera = occ_gt['mask_camera'].astype(bool)

        sample_token = info['token']
        occ_pred_path = osp.join(pred_occ_dir, f"{sample_token}.npz")
        occ_pred = np.load(occ_pred_path)['arr_0'].astype(np.int32)

        # occ_pred = occ_pred
        miou = occ_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)

        sample_token_list.append(sample_token)
        miou_list.append(miou)

    occ_eval_metrics.count_miou()
    return sample_token_list, miou_list

def compute_baseline_metrics():
    occ_eval_metrics = Metric_mIoU(
        num_classes=18,
        use_lidar_mask=False,
        use_image_mask=True)

    # load the pickle
    # read the validation occ token
    pickle_path = "data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_val.pkl"
    with open(pickle_path, 'rb') as f:
        val_infos = pickle.load(f)['infos']

    pred_occ_dir = "./occ_submission_baseline"

    print('\nStarting Evaluation...')
    sample_token_list, miou_list = [], []
    for index, info in tqdm(enumerate(val_infos)):
        occ_path = info['occ_path']
        occ_path = osp.join(occ_path, "labels.npz")
        occ_gt = np.load(occ_path)

        gt_semantics = occ_gt['semantics']
        mask_lidar = occ_gt['mask_lidar'].astype(bool)
        mask_camera = occ_gt['mask_camera'].astype(bool)

        sample_token = info['token']
        occ_pred_path = osp.join(pred_occ_dir, f"{sample_token}.npz")
        occ_pred = np.load(occ_pred_path)['arr_0'].astype(np.int32)

        # occ_pred = occ_pred
        miou = occ_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)

        sample_token_list.append(sample_token)
        miou_list.append(miou)

    occ_eval_metrics.count_miou()
    return sample_token_list, miou_list


if __name__ == "__main__":
    radocc_sample_token_list, radocc_miou_list = compute_radocc_metrics()

    baseline_sample_token_list, baseline_miou_list = compute_baseline_metrics()

    # save the result
    # compute the difference of the miou
    diff_miou_list = []
    for radocc_miou, baseline_miou in zip(radocc_miou_list, baseline_miou_list):
        diff_miou = radocc_miou - baseline_miou
        diff_miou_list.append(diff_miou)
    
    # merge the token list with the diff miou
    merge_token_list = []
    for radocc_token, diff_miou in zip(radocc_sample_token_list, diff_miou_list):
        merge_token_list.append((radocc_token, diff_miou))

    # sort the diff_miou_list
    sorted_diff_miou_list = sorted(merge_token_list, key = lambda x : x[1], reverse=True)

    # save to the txt
    save_path = "./diff_miou.txt"
    with open(save_path, 'w') as f:
        for token, diff_miou in sorted_diff_miou_list:
            f.write(f"{token} {diff_miou}\n")
    