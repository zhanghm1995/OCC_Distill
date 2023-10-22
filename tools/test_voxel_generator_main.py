'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-09-17 17:07:32
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''


import numpy as np

from mmdet3d.core.voxel.voxel_generator import VoxelGenerator


def test_voxel_generator():
    np.random.seed(0)
    voxel_size = [0.5, 0.5, 0.5]
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]
    max_num_points = 1000
    self = VoxelGenerator(voxel_size, point_cloud_range, max_num_points)
    points = np.random.rand(1000, 4)
    voxels = self.generate(points)
    voxels, coors, num_points_per_voxel = voxels
    expected_coors = np.array([[7, 81, 1], [6, 81, 0], [7, 80, 1], [6, 81, 1],
                               [7, 81, 0], [6, 80, 1], [7, 80, 0], [6, 80, 0]])
    expected_num_points_per_voxel = np.array(
        [120, 121, 127, 134, 115, 127, 125, 131])
    assert voxels.shape == (8, 1000, 4)
    assert np.all(coors == expected_coors)
    assert np.all(num_points_per_voxel == expected_num_points_per_voxel)


def test_voxel_generator2():
    np.random.seed(0)
    voxel_size = [0.4, 0.4, 0.4]
    point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
    max_num_points = 3
    self = VoxelGenerator(voxel_size, point_cloud_range, max_num_points)
    points = np.array([[0, 0, 5.4],
                       [-40.0, -40.0, -1.0],
                       [40.0, -40.0, -1.0],
                       [40.0, 40.0, 5.4]])
    print(points.shape)
    voxels = self.generate(points)
    voxels, coors, num_points_per_voxel = voxels
    print(voxels.shape, coors.shape, num_points_per_voxel.shape)

    print(coors)


if __name__ == "__main__":
    test_voxel_generator2()