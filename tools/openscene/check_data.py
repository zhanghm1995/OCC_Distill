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

    meta = mmengine.load(pkl_fp)
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

        # check camera image
        cam_invalid = False
        for cam_name, cam_info in info['cams'].items():
            if not osp.exists(cam_info['data_path']):
                print(f"Camera Image File not exists: {cam_info['data_path']}")
                print(info['scene_token'], info['token'], idx)
                missed_scene_token.append(info['scene_token'])
                cam_invalid = True
                break
        if cam_invalid:
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

    save_pkl_fp = f"data/openscene-v1.1/openscene_{data_name}_{split}_v2_valid.pkl"
    with open(save_pkl_fp, "wb") as f:
        pickle.dump(kept_data_infos, f, protocol=pickle.HIGHEST_PROTOCOL)


def check_data_length(data_part, split):
    pkl_fp = f"data/openscene-v1.1/openscene_{data_part}_{split}.pkl"
    print(pkl_fp)

    meta = mmengine.load(pkl_fp)
    print(type(meta), len(meta))

    scene_infos_list = get_scene_list(meta, scene_flag='scene_token')
    print(len(scene_infos_list))
    for scene_info, data in scene_infos_list.items():
        print(scene_info, len(data))

    ## only keep the scene with more than 10 data
    kept_data_infos = []
    for scene_info, data in scene_infos_list.items():
        if len(data) > 10:
            kept_data_infos.extend(data)
    
    print(len(kept_data_infos))
    save_pkl_fp = f"data/openscene-v1.1/openscene_{data_part}_{split}_valid.pkl"
    with open(save_pkl_fp, "wb") as f:
        pickle.dump(kept_data_infos, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    check_data_length('trainval', 'val')
    exit()
    # check_data_complete('mini', 'train')
    check_data_complete('trainval', 'train')