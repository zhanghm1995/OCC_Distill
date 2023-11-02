'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-10-30 08:30:30
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import pickle
import time
import mmcv


def check_sam_mask_data():
    import pycocotools.mask as maskUtils

    sam_mask_root = "data/nuscenes/sam_mask_json/CAM_FRONT"
    sam_mask_files = os.listdir(sam_mask_root)
    
    num_instances_list = []
    for file in sam_mask_files:
        fp = osp.join(sam_mask_root, file)
        anns = mmcv.load(fp)

        num_instances_list.append(len(anns))
    
    print(np.mean(num_instances_list), np.max(num_instances_list), np.min(num_instances_list))


def check_nuscene_flow_data():
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
    start = time.time()
    for dir in tqdm(all_dirs):
        dir_path = osp.join(nuscene_flow_dir, dir)
        all_files = os.listdir(dir_path)
        for file in all_files:
            file_path = osp.join(dir_path, file)
            try:
                with open(file_path, 'rb') as f:
                    scene_flow_dict = pickle.load(f)
                all_data.append(scene_flow_dict)
            except:
                print(file_path)
    end = time.time()
    print("Data time:", end - start)

    save_path = osp.join('results/', "scene_data.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(all_data, f)


if __name__ == "__main__":
    check_nuscene_flow_data()