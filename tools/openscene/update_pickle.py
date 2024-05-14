'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-05-08 10:45:50
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import pickle
import mmengine
import os
import os.path as osp
import numpy as np
import torch
import os, sys
from tqdm import tqdm
from collections import defaultdict


def remove_data_wo_occ_path(split, 
                            need_update_occ_path=False,
                            need_absolute_cam_path=False):
    """Because the original data has some missing occ_gt_final_path, we need to remove them.
    """
    pkl_fp = f"data/openscene-v1.1/openscene_mini_{split}.pkl"
    meta = mmengine.load(pkl_fp)
    print("split:", split, type(meta), len(meta))
    
    if need_absolute_cam_path:
        # modify the camera path to make it with absolute file path
        data_root = "data/openscene-v1.1/sensor_blobs/mini"
        for info in meta:
            for cam_type, cam_info in info['cams'].items():
                cam_info['data_path'] = osp.join(data_root, cam_info['data_path'])

    print("Modify the camera path ok.")

    ## check the occ
    src_dir = "dataset"
    target_dir = "data"
    
    new_val_infos = []
    for info in meta:
        occ_gt_path = info['occ_gt_final_path']
        if occ_gt_path is None:
            continue
        if need_update_occ_path:
            info['occ_gt_final_path'] = occ_gt_path.replace(src_dir, target_dir)
        new_val_infos.append(info)
    print(len(new_val_infos))

    pkl_file_path = f"data/openscene-v1.1/openscene_mini_{split}_v2.pkl"
    with open(pkl_file_path, "wb") as f:
        pickle.dump(new_val_infos, f, protocol=pickle.HIGHEST_PROTOCOL)


def preprocess_private_wm():
    pkl_fp = "data/openscene-v1.1/meta_datas/private_test_wm/private_test_wm.pkl"
    data_infos = mmengine.load(pkl_fp)
    print(len(data_infos))

    # modify the camera path to make it with absolute file path
    data_root = "data/openscene-v1.1/sensor_blobs/private_test_wm"
    for info in tqdm(data_infos):
        for cam_type, cam_info in info['cams'].items():
            cam_info['data_path'] = osp.join(data_root, cam_info['data_path'])
        
        # give a occupancy path for offline saving
        log_name = info['log_name']
        frame_idx = info['frame_idx']
        occ_save_dir = osp.join("data/openscene-v1.0/", "occupancy/private", log_name)
        occ_save_path = osp.join(occ_save_dir, f"{frame_idx:03d}_occ_final.npy")
        info['occ_gt_final_path'] = occ_save_path
        
    print(len(data_infos))
    pkl_file_path = f"data/openscene-v1.1/meta_datas/private_test_wm/private_test_wm_v2.pkl"
    with open(pkl_file_path, "wb") as f:
        pickle.dump(data_infos, f, protocol=pickle.HIGHEST_PROTOCOL)
    

def create_partial_data():
    """Create a partial data for accelerating training process.
    """
    pkl_fp = "data/openscene-v1.1/openscene_mini_train_v2.pkl"

    data_infos = mmengine.load(pkl_fp)  # list type
    print(f"original data length: {len(data_infos)}")
    print(data_infos[0].keys())

    # convert the list of dict to dict
    data_infos_dict = defaultdict(list)
    for d in data_infos:
        data_infos_dict[d["scene_name"]].append(d)

    data_infos_dict = dict(data_infos_dict)
    print(f"scene numbers: {len(data_infos_dict)}")

    num_frames_each_scene = [len(_scene) for _scene in data_infos_dict.values()]
    print(min(num_frames_each_scene), max(num_frames_each_scene))

    filtered_scene_names = []
    for key, value in data_infos_dict.items():
        # filter the scenes with less than 8 frames
        if len(value) < 8:
            continue
        filtered_scene_names.append(key)

    # only keep the first 1/4 for acceleration
    partial_filtered_scene_names = filtered_scene_names[:len(filtered_scene_names) // 4]
    partial_data_infos = []
    for key in partial_filtered_scene_names:
        partial_data_infos.extend(data_infos_dict[key])
    print(f"partial data length: {len(partial_data_infos)}")

    pkl_file_path = f"data/openscene-v1.1/openscene_mini_train_v2_partial.pkl"
    with open(pkl_file_path, "wb") as f:
        pickle.dump(partial_data_infos, f, protocol=pickle.HIGHEST_PROTOCOL)
    

if __name__ == "__main__":
    create_partial_data()
    exit()
    remove_data_wo_occ_path('train', 
                            need_update_occ_path=True, 
                            need_absolute_cam_path=True)
    remove_data_wo_occ_path('val', 
                            need_update_occ_path=True, 
                            need_absolute_cam_path=True)