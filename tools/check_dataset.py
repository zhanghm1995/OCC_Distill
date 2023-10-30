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


if __name__ == "__main__":
    check_sam_mask_data()