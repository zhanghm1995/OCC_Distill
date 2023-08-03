'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-08-03 14:14:27
Email: haimingzhang@link.cuhk.edu.cn
Description: Prepare the lidarseg testing data for the chanllenge submissions.
'''

import os
import os.path as osp
import numpy as np
import json
from copy import deepcopy
import torch
import pickle
from tqdm import tqdm
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes


def sparse_quantize(pc, coors_range, spatial_shape):
    idx = spatial_shape * (pc - coors_range[0]) / (coors_range[1] - coors_range[0])
    return idx.long()


def gen_point_range_mask(pts, coors_range_xyz=[[-40, 40], [-40, 40], [-1, 5.4]]):
    mask11 = pts[:, 0] < coors_range_xyz[0][1]
    mask12 = pts[:, 0] > coors_range_xyz[0][0]
    mask21 = pts[:, 1] < coors_range_xyz[1][1]
    mask22 = pts[:, 1] > coors_range_xyz[1][0]
    mask31 = pts[:, 2] < coors_range_xyz[2][1]
    mask32 = pts[:, 2] > coors_range_xyz[2][0]

    return mask11 & mask12 & mask21 & mask22 & mask31 & mask32


def voxelization(point, batch_idx, 
                 coors_range_xyz=[[-40, 40], [-40, 40], [-1, 5.4]], 
                 spatial_shape=[200, 200, 16]):
    xidx = sparse_quantize(point[:, 0], coors_range_xyz[0], spatial_shape[0])
    yidx = sparse_quantize(point[:, 1], coors_range_xyz[1], spatial_shape[1])
    zidx = sparse_quantize(point[:, 2], coors_range_xyz[2], spatial_shape[2])

    bxyz_indx = torch.stack([batch_idx, xidx, yidx, zidx], dim=-1).long()
    unq, unq_inv, unq_cnt = torch.unique(bxyz_indx, 
                                         return_inverse=True, 
                                         return_counts=True, 
                                         dim=0)
    return unq, unq_inv


def load_js3cnet_results(data_root):
    lidarseg_dir = osp.join(data_root, 'lidarseg/test')
    lidarseg_bin_files = sorted(os.listdir(lidarseg_dir))

    print(len(lidarseg_bin_files), lidarseg_bin_files[:3])

    lidar_seg_results = dict()
    for idx, fp in tqdm(enumerate(lidarseg_bin_files)):
        lidarseg_bin_file = osp.join(lidarseg_dir, fp)
        lidarseg_bin = np.fromfile(lidarseg_bin_file, dtype=np.uint8)
        # print(lidarseg_bin.shape, lidarseg_bin.dtype)
        # print(np.unique(lidarseg_bin), lidarseg_bin.max(), lidarseg_bin.min())

        token = fp.split('_')[0]
        lidar_seg_results[token] = lidarseg_bin
    return lidar_seg_results


def load_occ_submission_results(data_root):
    occ_dir = data_root
    occ_files = sorted(os.listdir(occ_dir))
    print(len(occ_files), occ_files[:3])

    occ_results = dict()
    for idx, fp in tqdm(enumerate(occ_files)):
        occ_npz_fp = osp.join(occ_dir, fp)
        occ_pred = np.load(occ_npz_fp)
        occ_pred = occ_pred['arr_0']

        token = fp.split('.')[0]
        occ_results[token] = occ_pred

    return occ_results


def point_to_ego(points_lidar, info):
    """Transform the points to ego coordinates.
    """

    ## Obtain the transformation matrix from lidar to ego
    lidar2lidarego = np.eye(4, dtype=np.float32)
    lidar2lidarego[:3, :3] = Quaternion(
        info['lidar2ego_rotation']).rotation_matrix
    lidar2lidarego[:3, 3] = info['lidar2ego_translation']
    lidar2lidarego = torch.from_numpy(lidar2lidarego)

    points_ego = points_lidar[:, :3].matmul(
            lidar2lidarego[:3, :3].T) + lidar2lidarego[:3, 3].unsqueeze(0)
    points_ego = torch.cat([points_ego, points_lidar[:, 3:]], dim=1)
    return points_ego

    

def merge_results(points, 
                  lidarseg_pred, 
                  occ_pred,
                  coors_range_xyz=[[-40, 40], [-40, 40], [-1, 5.4]],
                  spatial_shape=[200, 200, 16]):
    
    ## Transform the point cloud to ego coordinates

    pc = points[:, :3] # (N, 3)
    pc_seg = torch.zeros(pc.shape[0], occ_pred.shape[-1]).to(pc.device) # (N, C)
    lidarseg = torch.from_numpy(deepcopy(lidarseg_pred)) # (N, C)
    mask = gen_point_range_mask(pc, coors_range_xyz)
    unq, unq_inv = voxelization(
        point=pc[mask],
        batch_idx=torch.zeros(pc[mask].shape[0]).to(pc.device),
        coors_range_xyz=coors_range_xyz,
        spatial_shape=spatial_shape,
    )
    occ_pred = torch.from_numpy(occ_pred).unsqueeze(0).unsqueeze(-1)
    ## reset the category 17 to 0
    occ_pred[occ_pred == 17] = 0

    point_pred = occ_pred[unq[:, 0], unq[:, 1], unq[:, 2], unq[:, 3]][unq_inv]
    point_pred = point_pred.squeeze()

    # use the occ prediction to replace the lidarseg prediction
    lidarseg[mask] = point_pred

    lidarseg = lidarseg.numpy()

    # set the 0 in lidarseg to the value of original lidarseg_pred
    mask = lidarseg == 0
    lidarseg[mask] = lidarseg_pred[mask]
    
    return lidarseg.astype(np.uint8)


def main():
    test_pkl_file = "data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_test.pkl"
    with open(test_pkl_file, "rb") as fp:
        dataset = pickle.load(fp)

    js3cnet_data_root = "data/JS3C-Net/38b6f44c-6832-420b-a5ed-70138e87896b"
    js3cnet_lidarseg_results = load_js3cnet_results(js3cnet_data_root)

    occ_data_root = "data/occ_submission"
    occ_results = load_occ_submission_results(occ_data_root)
    
    data_infos = dataset['infos']

    save_dir = "./RedOCC_lidarseg/lidarseg/test"
    os.makedirs(save_dir, exist_ok=True)

    for idx, info in tqdm(enumerate(data_infos)):
        save_file_path = osp.join(save_dir, lidar_token + '_lidarseg.bin')
        if osp.exists(save_file_path):
            continue

        lidar_token = info['lidar_token']
        sample_token = info['token']
        lidar_path = info['lidar_path']

        curr_lidarseg = js3cnet_lidarseg_results[lidar_token]
        curr_point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
        curr_occ = occ_results[sample_token]

        curr_point_cloud = torch.from_numpy(curr_point_cloud)
        curr_point_cloud = point_to_ego(curr_point_cloud, info)

        occ_lidarseg = merge_results(curr_point_cloud, curr_lidarseg, curr_occ)
        
        # save the results
        occ_lidarseg.tofile(save_file_path)


if __name__ == "__main__":
    main()
