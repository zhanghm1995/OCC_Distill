'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-08-15 10:58:39
Email: haimingzhang@link.cuhk.edu.cn
Description: Create some information mappings in nuScenes dataset.
'''

import pickle
import os.path as osp
from tqdm import tqdm
from nuscenes import NuScenes


def frame_idx_to_scene_name(anno_file, 
                            save_root=None):
    with open(anno_file, "rb") as fp:
        dataset = pickle.load(fp)
    
    data_infos = dataset['infos']

    data_infos = list(sorted(data_infos, key=lambda e: e['timestamp']))

    version = 'v1.0-trainval'
    data_root = 'data/nuscenes'
    nusc = NuScenes(version, data_root)

    output_str_list = []
    for idx, info in enumerate(tqdm(data_infos)):
        scene_token = nusc.get('sample', info['token'])['scene_token']
        scene_meta = nusc.get('scene', scene_token)
        scene_name = scene_meta['name']

        output_str = f"{idx} {scene_name}"
        output_str_list.append(output_str)
    
    if save_root is not None:
        output_str = "\n".join(output_str_list)
        
        save_file = osp.join(save_root, "frame_idx_2_secene_name.txt")
        with open(save_file, "w") as fp:
            fp.write(output_str)


def frame_idx_to_sample_token(anno_file, save_root=None, split='val'):
    with open(anno_file, "rb") as fp:
        dataset = pickle.load(fp)
    
    data_infos = dataset['infos']
    data_infos = list(sorted(data_infos, key=lambda e: e['timestamp']))

    output_str_list = []
    for idx, info in enumerate(tqdm(data_infos)):
        sample_token = info['token']
        if 'lidar_token' in info.keys():
            lidar_token = info['lidar_token']
        lidarseg = info['lidarseg']

        output_str = f"{idx} {sample_token} {lidarseg}"
        output_str_list.append(output_str)
    
    if save_root is not None:
        output_str = "\n".join(output_str_list)
        
        save_file = osp.join(save_root, f"frame_idx_2_sample_token_lidarseg_{split}.txt")
        with open(save_file, "w") as fp:
            fp.write(output_str)


if __name__ == "__main__":
    pickle_path = "data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_val.pkl"
    pickle_path = "data/nuscenes/bevdetv3-lidartoken-nuscenes_infos_val.pkl"

    pickle_path = "data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_train.pkl"

    pickle_path = "/data1/zhanghm/Code/Occupancy/Occ_Challenge/data/nuscenes/bevdetv2-lidarseg-nuscenes_infos_trainvaltest.pkl"
    frame_idx_to_sample_token(pickle_path, save_root="./data/nuscenes", split='trainvaltest')