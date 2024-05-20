'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-05-20 10:00:31
Email: haimingzhang@link.cuhk.edu.cn
Description: Check the dataset.
'''
import os
import os.path as osp
import numpy as np
import pickle
import torch
import warnings
import mmengine
from tqdm import tqdm
from update_pickle import get_scene_list


def check_data_complete(data_name, split):
    pkl_fp = f"data/openscene-v1.1/openscene_{data_name}_{split}_v2.pkl"
    # pkl_fp = f"data/openscene-v1.1/openscene_{data_name}_{split}.pkl"
    print(pkl_fp)

    data_root = f"data/openscene-v1.1/sensor_blobs/{data_name}"

    meta = mmengine.load(pkl_fp)[:100]
    print(type(meta), len(meta))

    missed_scene_token = []
    for idx, info in tqdm(enumerate(meta)):
        if info['occ_gt_final_path'] is None:
            print(info['scene_token'], info['token'], idx)
            missed_scene_token.append(info['scene_token'])
            continue
        
        # check occupancy
        if not osp.exists(info['occ_gt_final_path']):
            print(f"Occupancy File not exists: {info['occ_gt_final_path']}")
            print(info['scene_token'], info['token'], idx)
            missed_scene_token.append(info['scene_token'])
            continue
        
        pts_filename = osp.join(data_root, info['lidar_path'])
        if not osp.exists(pts_filename):
            print(f"LiDAR File not exists: {pts_filename}")
            print(info['scene_token'], info['token'], idx)
            missed_scene_token.append(info['scene_token'])
            continue

    # convert to set
    missed_scene_token = set(missed_scene_token)
    print(f"Missed scene token: {missed_scene_token}, {len(missed_scene_token)}")

    scene_infos_list = get_scene_list(meta, scene_flag='scene_token')
    kept_data_infos = []
    for scene_info, data in scene_infos_list.items():
        if scene_info not in list(missed_scene_token):
            kept_data_infos.extend(data)

    print(len(meta), len(kept_data_infos))


if __name__ == "__main__":
    check_data_complete('mini', 'train')