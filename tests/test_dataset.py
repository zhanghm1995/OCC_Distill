
import pickle
import numpy as np
import os
from pyquaternion import Quaternion
import torch


nuscenes_info_pickle = "data/nuscenes/bevdetv2-nuscenes_infos_train.pkl"

with open(nuscenes_info_pickle, "rb") as fp:
    nuscenes_infos = pickle.load(fp)

infos = nuscenes_infos["infos"]
print(len(infos))

results = infos[10]
print(results.keys())

## 1) Load the point cloud
lidar_path = results["lidar_path"]
points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]

## 2) Read the transformation matrix
lidar2lidarego = np.eye(4, dtype=np.float32)
lidar2lidarego[:3, :3] = Quaternion(
    results['lidar2ego_rotation']).rotation_matrix
lidar2lidarego[:3, 3] = results['lidar2ego_translation']
lidar2lidarego = torch.from_numpy(lidar2lidarego)
print(lidar2lidarego)

points = torch.from_numpy(points)

z = points[:, 2]
print(z.min(), z.max())

point_ego = lidar2lidarego[:3, :3] @ points.T + lidar2lidarego[:3, 3].unsqueeze(1)
point_ego = point_ego.T
print(point_ego.shape)

z = point_ego[:, 2]
print(z.min(), z.max())