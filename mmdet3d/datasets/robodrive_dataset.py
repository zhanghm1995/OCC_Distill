'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-03-21 23:51:48
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import copy

import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import mmcv
from os import path as osp
from mmdet.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from mmcv.parallel import DataContainer as DC
import warnings
import pickle
import pdb, os
from .nuscenes_dataset_occ import NuScenesDatasetOccpancy


@DATASETS.register_module()
class RoboDriveDataset(NuScenesDatasetOccpancy):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """
    CORRUPTION = [
        'clean', 'brightness', 'dark', 'fog', 'frost', 'snow', 'contrast',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'elastic_transform', 'color_quant',
        'gaussian_noise', 'impulse_noise', 'shot_noise', 'iso_noise', 'pixelate', 'jpeg_compression'
    ]

    def __init__(self, 
                 corruption=None,
                 corruption_root=None,
                 use_semantic=True,
                 use_clean_inputs=False,
                 *args, **kwargs):
        self.corruption_root = corruption_root
        if corruption is not None:
            assert corruption in self.CORRUPTION, f'Corruption {corruption} not in {self.CORRUPTION}'
        self.corruption = corruption
        self.sample_scenes_dict = self.load_sample_scenes()

        self.use_clean_inputs = use_clean_inputs  # just for debugging
        if use_clean_inputs:
            warnings.warn('Testing with the clean inputs.')

        super().__init__(*args, **kwargs)
        
        self.use_semantic = use_semantic
        
        self.class_names = ['barrier','bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
                            'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface',
                            'other_flat', 'sidewalk', 'terrain', 'manmade','vegetation']

    def load_sample_scenes(self):
        """Load scenes in each corruption.
        """
        with open(osp.join(self.corruption_root, 'sample_scenes.pkl'), 'rb') as f:
            sample_scenes_dict = pickle.load(f)
        return sample_scenes_dict

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file, file_format='pkl')
        
        # filter scenes
        data_infos = data['infos']
        sample_data_infos = []
        for data_info in data_infos:
            if self.corruption is not None:
                if data_info['scene_token'] in self.sample_scenes_dict[self.corruption]:
                    sample_data_infos.append(data_info)
            else:
                sample_data_infos.append(data_info)
        
        data_infos = list(sorted(sample_data_infos, key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']

        ## if test the corruption data, we need to change the camera image path
        data_root = self.data_root[:-1] if self.data_root.endswith('/') else self.data_root
        if not self.use_clean_inputs and self.corruption is not None:
            # modify the camera data path
            for data_info in data_infos:
                for cam in data_info['cams']:
                    data_info['cams'][cam]['data_path'] = \
                        data_info['cams'][cam]['data_path'].replace(
                            data_root, os.path.join(self.corruption_root, self.corruption))
        return data_infos
    
    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).
        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        return results, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        results, tmp_dir = self.format_results(results, jsonfile_prefix)
        results_dict = {}
        if self.use_semantic:
            class_names = {0: 'IoU'}
            class_num = len(self.class_names) + 1
            for i, name in enumerate(self.class_names):
                class_names[i + 1] = self.class_names[i]
            
            results = np.stack(results, axis=0).mean(0)
            mean_ious = []
            
            for i in range(class_num):
                tp = results[i, 0]
                p = results[i, 1]
                g = results[i, 2]
                union = p + g - tp
                mean_ious.append(tp / union)
            
            for i in range(class_num):
                results_dict[class_names[i]] = mean_ious[i]
            results_dict['mIoU'] = np.mean(np.array(mean_ious)[1:])


        else:
            results = np.stack(results, axis=0).mean(0)
            results_dict={'Acc':results[0],
                          'Comp':results[1],
                          'CD':results[2],
                          'Prec':results[3],
                          'Recall':results[4],
                          'F-score':results[5]}

        return results_dict
