'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-10-23 14:26:08
Email: haimingzhang@link.cuhk.edu.cn
Description: The Pretraining nuscenes dataset, using the temporal contrastive learning.
'''
import numpy as np
from copy import deepcopy

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
                data = self.prepare_train_data(select_idx)
                output.append(data)
            else:
                output.append(deepcopy(data))

            if any(output) is None:
                idx = self._rand_another(idx)
                continue

            res = collate(output, samples_per_gpu=1)
            return res