'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-05-31 00:05:21
Email: haimingzhang@link.cuhk.edu.cn
Description: Load the OpenOcc dataset, and evaluate the Occupancy by using the RayIoU metrics.
Please note that the OpenOcc dataset is derived from nuScenes dataset.
'''

import os
import numpy as np
import os.path as osp
from torch.utils.data import DataLoader
from tqdm import tqdm
from .builder import DATASETS
from .nuscenes_dataset import NuScenesDataset
from .ego_pose_extractor import EgoPoseDataset
from .ray_metrics import main as ray_based_miou


@DATASETS.register_module()
class NuScenesOpenOccDataset(NuScenesDataset):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def evaluate_rayiou(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        # def sort_by_index(item):
        #     return item['index']

        # occ_results = sorted(occ_results, key=sort_by_index)

        occ_gts = []
        flow_gts = []
        occ_preds = []
        flow_preds = []
        lidar_origins = []

        print('\nStarting Evaluation...')

        if 'LightwheelOcc' in self.version:
            ego_pose_dataset = EgoPoseDataset(self.data_infos, dataset_type='lightwheelocc')
        else:
            ego_pose_dataset = EgoPoseDataset(self.data_infos, dataset_type='openocc_v2')

        data_loader_kwargs={
            "pin_memory": False,
            "shuffle": False,
            "batch_size": 1,
            "num_workers": 8,
        }

        data_loader = DataLoader(
            ego_pose_dataset,
            **data_loader_kwargs,
        )
        
        sample_tokens = [info['token'] for info in self.data_infos]

        for i, batch in tqdm(enumerate(data_loader), ncols=50):
            token = batch[0][0]
            output_origin = batch[1]
            
            data_id = sample_tokens.index(token)
            info = self.data_infos[data_id]

            occ_gt = np.load(info['occ_path'], allow_pickle=True)
            gt_semantics = occ_gt['semantics']
            gt_flow = occ_gt['flow']

            lidar_origins.append(output_origin)
            occ_gts.append(gt_semantics)
            flow_gts.append(gt_flow)
            occ_preds.append(occ_results[data_id]['occ_results'].cpu().numpy())
            flow_preds.append(occ_results[data_id]['flow_results'].cpu().numpy())
        
        ray_based_miou(occ_preds, occ_gts, flow_preds, flow_gts, lidar_origins)
    
    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        from .occ_metrics import Metric_mIoU

        occ_class_names = [
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
            'driveable_surface', 'other_flat', 'sidewalk',
            'terrain', 'manmade', 'vegetation', 'free'
        ]
        
        self.occ_eval_metrics = Metric_mIoU(
            num_classes=17,
            use_lidar_mask=False,
            use_image_mask=False,
            class_names=occ_class_names)
        
        print('\nStarting Evaluation...')
        for index, results in enumerate(tqdm(occ_results)):
            occ_pred = results['occ_results']

            info = self.data_infos[index]
            occupancy_file_path = info['occ_path']
            occ_gt = np.load(occupancy_file_path)
 
            gt_semantics = occ_gt['semantics']

            self.occ_eval_metrics.add_batch(occ_pred, gt_semantics)
   
        res = self.occ_eval_metrics.count_miou()
        return res 