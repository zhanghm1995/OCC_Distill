'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-10-23 14:26:08
Email: haimingzhang@link.cuhk.edu.cn
Description: The Pretraining nuscenes dataset, using the temporal contrastive learning.
'''

import torch
import numpy as np
from copy import deepcopy
import os.path as osp
import pickle
from mmcv.parallel import collate
from .builder import DATASETS
from .nuscenes_dataset_occ import NuScenesDatasetOccpancy


@DATASETS.register_module()
class NuScenesDatasetOccPretrain(NuScenesDatasetOccpancy):

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """

        idx = 50
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            output = []
            data = self.prepare_train_data(idx)
            output.append(data)

            # select the neiborhood frame
            temporal_interval = 1
            if np.random.choice([0, 1]):
                temporal_interval *= -1
            
            # select the neiborhood frame
            select_idx = idx + temporal_interval
            select_idx = np.clip(select_idx, 0, len(self) - 1)
            if self.data_infos[select_idx]['scene_token'] == self.data_infos[idx][
                'scene_token']:
                other_data = self.prepare_train_data(select_idx)
            else:
                other_data = deepcopy(data)
            
            output.append(other_data)

            if any([x is None for x in output]):
                idx = self._rand_another(idx)
                continue

            # load the scene flow TODO: when these two data are same, we need to generate a fake scene flow
            scene_token = self.data_infos[idx]['scene_token']
            sample_token1 = data['img_metas'].data['sample_idx']
            sample_token2 = other_data['img_metas'].data['sample_idx']

            scene_flow_fp = osp.join("./data/nuscenes_scene_sequence",
                                     scene_token, 
                                     sample_token1 + "_" + sample_token2 + ".pkl")
            with open(scene_flow_fp, 'rb') as f:
                scene_flow_dict = pickle.load(f)
            coord1 = scene_flow_dict['coord1']  # (u, v) coordinate
            coord2 = scene_flow_dict['coord2']

            render_size = data['render_gt_depth'].shape[-2:]
            sample_pts_pad1, sample_pts_pad2 = self.unify_sample_points(
                coord1, coord2, render_size)
            output[0]['sample_pts_pad'] = torch.from_numpy(sample_pts_pad1).to(torch.long)
            output[1]['sample_pts_pad'] = torch.from_numpy(sample_pts_pad2).to(torch.long)
            
            # collate these two frames together
            res = collate(output, samples_per_gpu=1)
            return res
        
    def unify_sample_points(self, coords1, coords2, render_size=(900, 1600)):
        # we generate these sample points in the 900x1600 image, in this case,
        # we need to rescale these points coordinate to the rendered image size
        # Here we also pad the coord1 and coord2 to have the same length
        origin_h, origin_w = 900, 1600
        height, width = render_size
        scale_h, scale_w = height / origin_h, width / origin_w

        max_num_sample_pts = 5000
        sample_pts_pad1 = np.zeros((len(coords1), max_num_sample_pts, 2), dtype=np.float32)
        sample_pts_pad2 = np.zeros((len(coords2), max_num_sample_pts, 2), dtype=np.float32)
        for idx, (coord1, coord2) in enumerate(zip(coords1, coords2)):
            assert coord1.shape[0] == coord2.shape[0]

            coord1 *= np.array([scale_w, scale_h])
            coord2 *= np.array([scale_w, scale_h])

            num_points = min(coord1.shape[0], max_num_sample_pts - 1)
            sample_pts_pad1[idx][:num_points] = coord1[:num_points]
            sample_pts_pad2[idx][:num_points] = coord2[:num_points]

            # NOTE: here we set the last point to be the number of points
            sample_pts_pad1[idx][max_num_sample_pts - 1] = num_points
            sample_pts_pad2[idx][max_num_sample_pts - 1] = num_points

        return sample_pts_pad1, sample_pts_pad2
