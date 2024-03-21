'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-03-21 16:59:16
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import os
import os.path as osp
import pickle
import numpy as np
from tqdm import tqdm


def load_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    data_infos = data['infos']
    metadata = data['metadata']

    print(len(data_infos))

    return metadata, data_infos


def use_surroundocc_path(origin_pickle_fp,
                         robodrive_pickle_fp):
    metadata1, data_infos1 = load_pickle(origin_pickle_fp)
    metadata2, data_infos2 = load_pickle(robodrive_pickle_fp)

    for i, data_info in tqdm(enumerate(data_infos1)):
        data_info['occ_path'] = data_infos2[i]['occ_path']
    
    result_dict = dict()
    result_dict['metadata'] = metadata1
    result_dict['infos'] = data_infos1

    pickle_file = origin_pickle_fp.replace('.pkl', '_robodrive.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(result_dict, f)


if __name__ == "__main__":
    pickle1 = "data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_train.pkl"
    pickle2 = "/mnt/data2/zhanghm/RoboDrive/BEVFormer_RoboDrive/data/robodrive_nuscenes_infos_train.pkl"
    use_surroundocc_path(pickle1, pickle2)

    # validation
    pickle1 = "data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_val.pkl"
    pickle2 = "/mnt/data2/zhanghm/RoboDrive/BEVFormer_RoboDrive/data/robodrive_nuscenes_infos_val.pkl"
    use_surroundocc_path(pickle1, pickle2)