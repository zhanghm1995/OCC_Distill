'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-09-21 11:39:25
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import torch
import numpy as np

from mmdet3d.models.model_utils.voxelization import in_range_3d
from mmdet3d.models.model_utils.voxelization import VoxelizationWithMapping


def main():
    voxel_layer_dict = dict(point_cloud_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
                            grid_shape=[200, 200, 16])
    voxelizer = VoxelizationWithMapping(**voxel_layer_dict)

    points_path = "results/point_cloud_ego_xyz_validation/0a0d6b8c2e884134a3b48df43d54c36a.xyz"
    points = np.loadtxt(points_path, dtype=np.float32)

    points = torch.from_numpy(points)
    mask = in_range_3d(points, voxel_layer_dict["point_cloud_range"])
    points = points[mask]
    print(points.shape)

    indices, inverse_indices = voxelizer([points, points])
    print(indices.shape, inverse_indices.shape)


if __name__ == "__main__":
    main()