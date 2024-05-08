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
import tqdm


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

    
if __name__ == "__main__":
    remove_data_wo_occ_path('train', 
                            need_update_occ_path=True, 
                            need_absolute_cam_path=True)
    remove_data_wo_occ_path('val', 
                            need_update_occ_path=True, 
                            need_absolute_cam_path=True)