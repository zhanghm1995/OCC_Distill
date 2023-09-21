'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-09-21 14:06:29
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Sequence, Tuple, Union


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


def voxelization(point, 
                 batch_idx, 
                 coors_range_xyz=[[-40, 40], [-40, 40], [-1, 5.4]], 
                 spatial_shape=[200, 200, 16]):
    xidx = sparse_quantize(point[:, 0], coors_range_xyz[0], spatial_shape[0])
    yidx = sparse_quantize(point[:, 1], coors_range_xyz[1], spatial_shape[1])
    zidx = sparse_quantize(point[:, 2], coors_range_xyz[2], spatial_shape[2])

    bxyz_indx = torch.stack([batch_idx, xidx, yidx, zidx], dim=-1).long()
    unq, unq_inv, unq_cnt = torch.unique(bxyz_indx, return_inverse=True, return_counts=True, dim=0)
    return unq, unq_inv


def in_range_3d(pts, point_range):
    """Check whether the points are in the given range.

    Args:
        point_range (list | torch.Tensor): The range of point
            (x_min, y_min, z_min, x_max, y_max, z_max)

    Note:
        In the original implementation of SECOND, checking whether
        a box in the range checks whether the points are in a convex
        polygon, we try to reduce the burden for simpler cases.

    Returns:
        torch.Tensor: A binary vector indicating whether each point is
            inside the reference range.
    """
    in_range_flags = ((pts[:, 0] > point_range[0])
                        & (pts[:, 1] > point_range[1])
                        & (pts[:, 2] > point_range[2])
                        & (pts[:, 0] < point_range[3])
                        & (pts[:, 1] < point_range[4])
                        & (pts[:, 2] < point_range[5]))
    return in_range_flags


class VoxelizationWithMapping(nn.Module):
    """_summary_

    Args:
        point_cloud_range (_type_): [x_min, y_min, z_min, x_max, y_max, z_max]]
        voxel_size (list, optional): _description_. Defaults to [].
        grid_shape (list, optional): _description_. Defaults to [].

    Raises:
        ValueError: _description_
        ValueError: _description_
    """
    
    def __init__(self, 
                 point_cloud_range, 
                 voxel_size=[],
                 grid_shape=[]):
        
        super().__init__()
        if voxel_size and grid_shape:
            raise ValueError('voxel_size is mutually exclusive grid_shape')
        
        self.coors_range_xyz = [[point_cloud_range[0], point_cloud_range[3]],
                                [point_cloud_range[1], point_cloud_range[4]],
                                [point_cloud_range[2], point_cloud_range[5]]]
        
        point_cloud_range = torch.tensor(
            point_cloud_range, dtype=torch.float32)
        self.point_cloud_range = point_cloud_range

        if voxel_size:
            self.voxel_size = voxel_size
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
            grid_shape = (point_cloud_range[3:] -
                          point_cloud_range[:3]) / voxel_size
            grid_shape = torch.round(grid_shape).long().tolist()
            self.grid_shape = grid_shape
        elif grid_shape:
            self.grid_shape = grid_shape
            grid_shape = torch.tensor(grid_shape, dtype=torch.float32)
            voxel_size = (point_cloud_range[3:] - point_cloud_range[:3]) / (
                grid_shape - 1)
            voxel_size = voxel_size.tolist()
            self.voxel_size = voxel_size
        else:
            raise ValueError('must assign a value to voxel_size or grid_shape')

    def forward(self, points: List[Tensor]):
        points_list = []
        batch_list = []
        for idx, pc in enumerate(points):
            pc = pc[:, :3]
            mask = in_range_3d(pc, self.point_cloud_range)
            
            pc = pc[mask]
            points_list.append(pc)
            batch_list.append(torch.ones_like(pc[:, 0]) * idx)
        
        points_list = torch.cat(points_list, 0)
        batch_list = torch.cat(batch_list, 0)

        indices, inverse_indices = voxelization(
            point=points_list,
            batch_idx=batch_list,
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.grid_shape
        )
        return indices, inverse_indices

    def voxelization(point: Tensor, 
                     batch_idx, 
                     coors_range_xyz=[[-40, 40], [-40, 40], [-1, 5.4]], 
                     spatial_shape=[200, 200, 16]):
        xidx = sparse_quantize(point[:, 0], coors_range_xyz[0], spatial_shape[0])
        yidx = sparse_quantize(point[:, 1], coors_range_xyz[1], spatial_shape[1])
        zidx = sparse_quantize(point[:, 2], coors_range_xyz[2], spatial_shape[2])

        bxyz_indx = torch.stack([batch_idx, xidx, yidx, zidx], dim=-1).long()
        unq, unq_inv, unq_cnt = torch.unique(
            bxyz_indx, return_inverse=True, return_counts=True, dim=0)
        return unq, unq_inv
    
    def sparse_quantize(pc, coors_range, spatial_shape):
        idx = spatial_shape * (pc - coors_range[0]) / (coors_range[1] - coors_range[0])
        return idx.long()

