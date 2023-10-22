'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-08-18 11:01:02
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import os
import os.path as osp
import numpy as np
import json
import pickle
from tqdm import tqdm
from pyquaternion import Quaternion
import torch


def collect_occ_gts():
    data_root = "./data/nuscenes/gts"

    all_scenes = os.listdir(data_root)

    save_root = "./data/gt_validation_occ"
    os.makedirs(save_root, exist_ok=True)

    # read the validation occ token
    pickle_path = "data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_val.pkl"
    with open(pickle_path, 'rb') as f:
        val_infos = pickle.load(f)['infos']

    print(len(val_infos))
    print(val_infos[0]['occ_path'])

    for info in val_infos:
        occ_path = info['occ_path']
        sample_name = osp.basename(occ_path)

        occ_path = osp.join(occ_path, "labels.npz")

        save_occ_path = osp.join(save_root, f"{sample_name}.npz")
        ## copy the occ file
        os.system(f"cp {occ_path} {save_occ_path}")


def collect_lidarseg_gts():
    save_root = "gt_lidarseg_validation"
    os.makedirs(save_root, exist_ok=True)
    # read the validation occ token
    pickle_path = "data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_val.pkl"
    with open(pickle_path, 'rb') as f:
        val_infos = pickle.load(f)['infos']

    print(len(val_infos))

    for info in val_infos:
        lidarseg_path = info['lidarseg']
        save_file_name = osp.basename(lidarseg_path)

        save_path = osp.join(save_root, save_file_name)
        os.system(f"cp {lidarseg_path} {save_path}")


def collect_point_cloud_samples():
    pickle_path = "data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_val.pkl"
    with open(pickle_path, 'rb') as f:
        val_infos = pickle.load(f)['infos']

    print(len(val_infos))

    save_root = "point_cloud_xyz_validation"
    os.makedirs(save_root, exist_ok=True)
    for info in tqdm(val_infos):
        points_path = info['lidar_path']
        save_file_name = osp.basename(points_path)

        save_path = osp.join(save_root, save_file_name)
        os.system(f"cp {points_path} {save_path}")


def points2ego(points, info):
    ## Obtain the transformation matrix from lidar to ego
    lidar2lidarego = np.eye(4, dtype=np.float32)
    lidar2lidarego[:3, :3] = Quaternion(
        info['lidar2ego_rotation']).rotation_matrix
    lidar2lidarego[:3, 3] = info['lidar2ego_translation']
    lidar2lidarego = torch.from_numpy(lidar2lidarego)

    points_lidar = torch.from_numpy(points)

    points_ego = points_lidar[:, :3].matmul(
            lidar2lidarego[:3, :3].T) + lidar2lidarego[:3, 3].unsqueeze(0)
    points_ego = torch.cat([points_ego, points_lidar[:, 3:]], dim=1)
    return points_ego.numpy()


def save_point_cloud_samples(to_ego=False):
    pickle_path = "data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_val.pkl"
    with open(pickle_path, 'rb') as f:
        val_infos = pickle.load(f)['infos']

    print(len(val_infos))

    save_root = "results/point_cloud_ego_xyz_validation"
    os.makedirs(save_root, exist_ok=True)

    for info in tqdm(val_infos):
        points_path = info['lidar_path']
        sample_token = info['token']
        points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 5)

        if to_ego:
            points = points2ego(points, info)

        save_path = osp.join(save_root, f"{sample_token}.xyz")
        np.savetxt(save_path, points[:, :3])


if __name__ == "__main__":
    save_point_cloud_samples(to_ego=True)