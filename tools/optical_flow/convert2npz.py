'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-11-02 22:04:10
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import numpy as np
import pickle
import os
import os.path as osp
import time
from tqdm import tqdm


def load_nuscene_scene_flow():
    nuscene_flow_dir = "data/nuscenes_scene_sequence"
    all_dirs = os.listdir(nuscene_flow_dir)

    print(len(all_dirs))
    
    all_files_length = []
    for dir in tqdm(all_dirs):
        dir_path = osp.join(nuscene_flow_dir, dir)
        all_files = os.listdir(dir_path)
        all_files_length.append(len(all_files))
    
    print(np.sum(all_files_length))

    ## read the pickle data
    all_data = []
    data_length_list = []
    start = time.time()
    for dir in tqdm(all_dirs):
        dir_path = osp.join(nuscene_flow_dir, dir)
        all_files = os.listdir(dir_path)
        for file in all_files:
            file_path = osp.join(dir_path, file)
            
            if not file_path.endswith('pkl'):
                continue
            
            with open(file_path, 'rb') as f:
                scene_flow_dict = pickle.load(f)
            all_data.append(scene_flow_dict)
            coord1 = scene_flow_dict['coord1']  # list of [N, 2]
            coord2 = scene_flow_dict['coord2']
            coord1_length = np.max([c.shape[0] for c in coord1])
            coord2_length = np.max([c.shape[0] for c in coord2])
            max_length = np.max([coord1_length, coord2_length])
            data_length_list.append(max_length)

    print(f"Read {len(all_data)} data, time cost: {time.time() - start:.2f}s")
    max_data_length = np.max(data_length_list)
    print(f"Max data length: {max_data_length}")  # 5416


def convert2npz():
    """Convert the original pickle data to npz format, and pad the 
    data to the same length.
    """
    save_root = "./data/nuscenes/nuscenes_scene_sequence_npz"

    nuscene_flow_dir = "data/nuscenes_scene_sequence"
    all_dirs = sorted(os.listdir(nuscene_flow_dir))
    print(len(all_dirs))
    
    ## read the pickle data
    all_data = []
    for dir in tqdm(all_dirs):
        save_dir = osp.join(save_root, dir)
        os.makedirs(save_dir, exist_ok=True)

        dir_path = osp.join(nuscene_flow_dir, dir)
        all_files = sorted(os.listdir(dir_path))
        for file in all_files:
            file_path = osp.join(dir_path, file)
            
            if not file_path.endswith('pkl'):
                continue
            
            with open(file_path, 'rb') as f:
                scene_flow_dict = pickle.load(f)
            all_data.append(scene_flow_dict)
            coord1 = scene_flow_dict['coord1']  # list of [N, 2]
            coord2 = scene_flow_dict['coord2']

            sample_pts_pad1, sample_pts_pad2 = unify2same_length(
                coord1, coord2, max_num_points=5500)
            
            save_path = osp.join(save_dir, file.replace('.pkl', '.npz'))
            np.savez_compressed(save_path, 
                                coord1=sample_pts_pad1, 
                                coord2=sample_pts_pad2)



def unify2same_length(coords1, coords2, max_num_points=5500):
    sample_pts_pad1 = np.zeros((len(coords1), max_num_points, 2), dtype=np.float32)
    sample_pts_pad2 = np.zeros((len(coords2), max_num_points, 2), dtype=np.float32)
    for idx, (coord1, coord2) in enumerate(zip(coords1, coords2)):
        assert coord1.shape[0] == coord2.shape[0]

        num_points = min(coord1.shape[0], max_num_points - 1)
        sample_pts_pad1[idx][:num_points] = coord1[:num_points]
        sample_pts_pad2[idx][:num_points] = coord2[:num_points]

        # NOTE: here we set the last dimension to be the number of points
        sample_pts_pad1[idx][max_num_points - 1] = num_points
        sample_pts_pad2[idx][max_num_points - 1] = num_points

    return sample_pts_pad1, sample_pts_pad2


if __name__ == "__main__":
    convert2npz()