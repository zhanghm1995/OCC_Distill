'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-08-03 14:53:54
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import os
import os.path as osp
import pickle
from nuscenes import NuScenes


def add_liarseg_info():
    version = 'v1.0-trainval'
    data_root = 'data/nuscenes'
    nuscenes = NuScenes(version, data_root)

    pkl_file = "data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_val.pkl"
    with open(pkl_file, "rb") as fp:
        dataset = pickle.load(fp)

    infos = []
    for info in dataset['infos']:
        token = info['token']
        lidar_token = nuscenes.get('sample', token)['data']['LIDAR_TOP']

        lidarseg_label = os.path.join(nuscenes.dataroot, 
                                      nuscenes.get('lidarseg', lidar_token)['filename'])
        info['lidarseg'] = lidarseg_label
        infos.append(info)

    dataset['infos'] = infos
    print(len(dataset['infos']))

    # with open("data/nuscenes/bevdetv3-lidartoken-nuscenes_infos_val.pkl", "wb") as fp:
    #     pickle.dump(dataset, fp)


def prepare_test_pkl():
    version = 'v1.0-test'
    data_root = 'data/nuscenes'
    nuscenes = NuScenes(version, data_root)

    test_pkl_file = "data/nuscenes/bevdetv3-nuscenes_infos_test.pkl"
    with open(test_pkl_file, "rb") as fp:
        dataset = pickle.load(fp)

    infos = []
    for info in dataset['infos']:
        token = info['token']
        info['lidar_token'] = nuscenes.get('sample', token)['data']['LIDAR_TOP']
        infos.append(info)

    dataset['infos'] = infos
    print(len(dataset['infos']))

    with open("data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_test.pkl", "wb") as fp:
        pickle.dump(dataset, fp)


def prepare_trainval_pkl():
    version = 'v1.0-trainval'
    data_root = 'data/nuscenes'
    nuscenes = NuScenes(version, data_root)

    pkl_file = "data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_val.pkl"
    with open(pkl_file, "rb") as fp:
        dataset = pickle.load(fp)

    infos = []
    for info in dataset['infos']:
        token = info['token']
        info['lidar_token'] = nuscenes.get('sample', token)['data']['LIDAR_TOP']
        infos.append(info)

    dataset['infos'] = infos
    print(len(dataset['infos']))

    with open("data/nuscenes/bevdetv3-lidartoken-nuscenes_infos_val.pkl", "wb") as fp:
        pickle.dump(dataset, fp)


if __name__ == "__main__":
    add_liarseg_info()