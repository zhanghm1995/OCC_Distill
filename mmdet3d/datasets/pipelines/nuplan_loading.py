'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-05-08 14:05:44
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import torch
import numpy as np
from io import BytesIO
from typing import IO, Any, List, NamedTuple
import copy
from pyquaternion import Quaternion
import os
import mmcv
from mmdet3d.core.points import BasePoints, get_points_type
from ..builder import PIPELINES


class PointCloudHeader(NamedTuple):
    """Class for Point Cloud header."""

    version: str
    fields: List[str]
    size: List[int]
    type: List[str]
    count: List[int]  # type: ignore
    width: int
    height: int
    viewpoint: List[int]
    points: int
    data: str


class PointCloud:
    """
    Class for raw .pcd file.
    """

    def __init__(self, header, points) -> None:
        """
        PointCloud.
        :param header: Pointcloud header.
        :param points: <np.ndarray, X, N>. X columns, N points.
        """
        self._header = header
        self._points = points

    @property
    def header(self) -> PointCloudHeader:
        """
        Returns pointcloud header.
        :return: A PointCloudHeader instance.
        """
        return self._header

    @property
    def points(self):
        """
        Returns points.
        :return: <np.ndarray, X, N>. X columns, N points.
        """
        return self._points

    def save(self, file_path: str) -> None:
        """
        Saves to .pcd file.
        :param file_path: The path to the .pcd file.
        """
        with open(file_path, 'wb') as fp:
            fp.write('# .PCD v{} - Point Cloud Data file format\n'.format(self._header.version).encode('utf8'))
            for field in self._header._fields:
                value = getattr(self._header, field)
                if isinstance(value, list):
                    text = ' '.join(map(str, value))
                else:
                    text = str(value)
                fp.write('{} {}\n'.format(field.upper(), text).encode('utf8'))
            fp.write(self._points.tobytes())

    @classmethod
    def parse(cls, pcd_content: bytes):
        """
        Parses the pointcloud from byte stream.
        :param pcd_content: The byte stream that holds the pcd content.
        :return: A PointCloud object.
        """
        with BytesIO(pcd_content) as stream:
            header = cls.parse_header(stream)
            points = cls.parse_points(stream, header)
            return cls(header, points)

    @classmethod
    def parse_from_file(cls, pcd_file: str):
        """
        Parses the pointcloud from .pcd file on disk.
        :param pcd_file: The path to the .pcd file.
        :return: A PointCloud instance.
        """
        with open(pcd_file, 'rb') as stream:
            header = cls.parse_header(stream)
            points = cls.parse_points(stream, header)
            return cls(header, points)

    @staticmethod
    def parse_header(stream: IO[Any]) -> PointCloudHeader:
        """
        Parses the header of a pointcloud from byte IO stream.
        :param stream: Binary stream.
        :return: A PointCloudHeader instance.
        """
        headers_list = []
        while True:
            line = stream.readline().decode('utf8').strip()
            if line.startswith('#'):
                continue
            columns = line.split()
            key = columns[0].lower()
            val = columns[1:] if len(columns) > 2 else columns[1]
            headers_list.append((key, val))

            if key == 'data':
                break

        headers = dict(headers_list)
        headers['size'] = list(map(int, headers['size']))
        headers['count'] = list(map(int, headers['count']))
        headers['width'] = int(headers['width'])
        headers['height'] = int(headers['height'])
        headers['viewpoint'] = list(map(int, headers['viewpoint']))
        headers['points'] = int(headers['points'])
        header = PointCloudHeader(**headers)

        if any([c != 1 for c in header.count]):
            raise RuntimeError('"count" has to be 1')

        if not len(header.fields) == len(header.size) == len(header.type) == len(header.count):
            raise RuntimeError('fields/size/type/count field number are inconsistent')

        return header

    @staticmethod
    def parse_points(stream: IO[Any], header: PointCloudHeader):
        """
        Parses points from byte IO stream.
        :param stream: Byte stream that holds the points.
        :param header: <np.ndarray, X, N>. A numpy array that has X columns(features), N points.
        :return: Points of Point Cloud.
        """
        if header.data != 'binary':
            raise RuntimeError('Un-supported data foramt: {}. "binary" is expected.'.format(header.data))

        # There is garbage data at the end of the stream, usually all b'\x00'.
        row_type = PointCloud.np_type(header)
        length = row_type.itemsize * header.points
        buff = stream.read(length)
        if len(buff) != length:
            raise RuntimeError('Incomplete pointcloud stream: {} bytes expected, {} got'.format(length, len(buff)))

        points = np.frombuffer(buff, row_type)

        return points

    @staticmethod
    def np_type(header: PointCloudHeader) -> np.dtype:  # type: ignore
        """
        Helper function that translate column types in pointcloud to np types.
        :param header: A PointCloudHeader object.
        :return: np.dtype that holds the X features.
        """
        type_mapping = {'I': 'int', 'U': 'uint', 'F': 'float'}
        np_types = [type_mapping[t] + str(int(s) * 8) for t, s in zip(header.type, header.size)]

        return np.dtype([(f, getattr(np, nt)) for f, nt in zip(header.fields, np_types)])

    def to_pcd_bin(self):
        """
        Converts pointcloud to .pcd.bin format.
        :return: <np.float32, 5, N>, the point cloud in .pcd.bin format.
        """
        lidar_fields = ['x', 'y', 'z', 'intensity', 'ring']
        return np.array([np.array(self.points[f], dtype=np.float32) for f in lidar_fields])

    def to_pcd_bin2(self):
        """
        Converts pointcloud to .pcd.bin2 format.
        :return: <np.float32, 6, N>, the point cloud in .pcd.bin2 format.
        """
        lidar_fields = ['x', 'y', 'z', 'intensity', 'ring', 'lidar_info']
        return np.array([np.array(self.points[f], dtype=np.float32) for f in lidar_fields])


@PIPELINES.register_module()
class LoadNuPlanPointsFromFile(object):
    """Load NuPlan points from file.
    """
    def __init__(self,
                 coord_type,
                 use_dim=-1):
        assert coord_type in ['LIDAR']
        self.coord_type = coord_type
        self.use_dim = use_dim

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if pts_filename.endswith('.npz'):
            pc = np.load(pts_filename)['arr_0']
        elif pts_filename.endswith('.pcd'):
            pc = PointCloud.parse_from_file(pts_filename).to_pcd_bin2().T
        else:
            raise NotImplementedError
        return pc

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        # origin_data_root = "data/openscene-v1.1/sensor_blobs/mini"
        # pts_filename = pts_filename.replace(origin_data_root, "")
        # pts_filename = pts_filename[1:] if pts_filename.startswith("/") else pts_filename
        # pts_filename = os.path.join("/mnt/data2/zhanghm/Code/Occupancy/ViDAR/results/vidar_pred_pc", 
        #                             pts_filename + ".npz")

        points = self._load_points(pts_filename)
        if points.shape[1] <= 3:
            points = np.concatenate([points, np.zeros((points.shape[0], 6 - points.shape[1]))], axis=1)

        if self.use_dim != -1:
            assert isinstance(self.use_dim, list)
            points = points[:, self.use_dim]

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=None)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str
    

class LoadPointsFromMultiSweeps(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 random_select=True,
                 load_future=False,
                 test_mode=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

        self.random_select = random_select
        self.load_future = load_future

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def _select_index(self, sweeps, ts):
        if len(sweeps) <= self.sweeps_num:
            choices = np.arange(len(sweeps))
        elif self.test_mode:
            choices = np.arange(self.sweeps_num)
        elif self.random_select:
            choices = np.random.choice(
                len(sweeps), self.sweeps_num, replace=False)
        else:
            # sort those sweeps by their timestamps:
            #    from close to cur_frame to distant to cur_frame, and select top.
            sweep_ts = [sweep['timestamp'] for sweep in sweeps]
            sweep_ts = np.array(sweep_ts) / 1e6
            ts_interval = np.abs(sweep_ts - ts)
            choices = np.argsort(ts_interval)
            choices = choices[:self.sweeps_num]
        return choices

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results['points']
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            choices = self._select_index(results['sweeps'], ts)
            if self.load_future:
                future_choices = self._select_index(results['future_sweeps'], ts)
                future_choices = future_choices + len(results['sweeps'])
                results['sweeps'] = (copy.deepcopy(results['sweeps']) +
                                     copy.deepcopy(results['future_sweeps']))
                choices = np.concatenate([choices, future_choices])

            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class LoadNuPlanPointsFromMultiSweeps(LoadPointsFromMultiSweeps):
    def __init__(self,
                 ego_mask=None,
                 hard_sweeps_timestamp=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.ego_mask = ego_mask

        # if hard_sweeps_timestamp:
        #  set timestamps of all points to {hard_sweeps_timestamp}.
        self.hard_sweeps_timestamp = hard_sweeps_timestamp

    def _load_points(self, pts_filename):
        pc = PointCloud.parse_from_file(pts_filename).to_pcd_bin2().T
        return pc

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        points = super()._remove_close(points, radius)

        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        if self.ego_mask is not None:
            # remove points belonging to ego vehicle.
            ego_mask = np.logical_and(
                np.logical_and(self.ego_mask[0] <= points_numpy[:, 0],
                               self.ego_mask[2] >= points_numpy[:, 0]),
                np.logical_and(self.ego_mask[1] <= points_numpy[:, 1],
                               self.ego_mask[3] >= points_numpy[:, 1]),
            )
            not_ego = np.logical_not(ego_mask)
            points = points[not_ego]
        return points

    def __call__(self, results):
        results = super().__call__(results)

        if self.hard_sweeps_timestamp is not None:
            points = results['points']
            points.tensor[:, -1] = self.hard_sweeps_timestamp
            results['points'] = points
        return results
    

@PIPELINES.register_module()
class OpenScenePointToMultiViewDepth(object):

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda = results['img_inputs'][4:]
        depth_map_list = []
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]
            cam_info = results['curr']['cams'][cam_name]

            # obtain lidar to image transformation matrix
            cam2lidar_r = cam_info['sensor2lidar_rotation']
            cam2lidar_t = cam_info['sensor2lidar_translation']
            cam2lidar_rt = np.eye(4)
            cam2lidar_rt[:3, :3] = cam2lidar_r
            cam2lidar_rt[3, :3] = cam2lidar_t
            cam2lidar_rt = torch.Tensor(cam2lidar_rt)

            intrinsic = torch.from_numpy(cam_info['cam_intrinsic'])
            cam2img = np.eye(4, dtype=np.float32)
            cam2img = torch.from_numpy(cam2img)
            cam2img[:3, :3] = intrinsic

            lidar2img = cam2img.matmul(torch.inverse(cam2lidar_rt))

            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]
            depth_map = self.points2depthmap(points_img, imgs.shape[2],
                                             imgs.shape[3])
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        results['gt_depth'] = depth_map
        return results
    

def openscene_occ_to_voxel(occ_data,
                           point_cloud_range=[-50.0, -50.0, -4.0, 50.0, 50.0, 4.0], 
                           occupancy_size=[0.5, 0.5, 0.5],
                           occupancy_classes=11):
    if isinstance(occ_data, np.ndarray):
        occ_data = torch.from_numpy(occ_data)
    
    occ_data = occ_data.long()
    occ_xdim = int((point_cloud_range[3] - point_cloud_range[0]) / occupancy_size[0])
    occ_ydim = int((point_cloud_range[4] - point_cloud_range[1]) / occupancy_size[1])
    occ_zdim = int((point_cloud_range[5] - point_cloud_range[2]) / occupancy_size[2])

    voxel_num = occ_xdim * occ_ydim * occ_zdim

    gt_occupancy = (torch.ones(voxel_num, dtype=torch.long) * occupancy_classes).to(occ_data.device)
    gt_occupancy[occ_data[:, 0]] = occ_data[:, 1]

    gt_occupancy = gt_occupancy.reshape(occ_zdim, occ_ydim, occ_xdim)

    gt_occupancy = gt_occupancy.permute(2, 1, 0)
    return gt_occupancy


@PIPELINES.register_module()
class LoadOpenSceneOccupancy(object):
    """load occupancy GT data
       gt_type: index_class, store the occ index and occ class in one file with shape (n, 2)
    """
    def __call__(self, results):
        occ_gt_path = results['occ_gt_path']
        occ_gts = torch.from_numpy(np.load(occ_gt_path))  # (n, 2)

        results['voxel_semantics'] = openscene_occ_to_voxel(occ_gts).numpy()
        return results