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
from einops import rearrange, repeat
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

        idx = 10  # TODO: DEBUG ONLY
        if self.test_mode:
            return self.prepare_test_data(idx)
        
        while True:
            output = []
            data = self.prepare_train_data(idx)
            output.append(data)

            # select the neiborhood frame
            temporal_interval = 1
            # if np.random.choice([0, 1]):
            #     temporal_interval *= -1
            
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

            render_size = data['render_gt_depth'].shape[-2:]  # (h, w)
            sample_pts_pad1, sample_pts_pad2 = self.unify_sample_points(
                coord1, coord2, render_size)
            output[0]['sample_pts_pad'] = torch.from_numpy(sample_pts_pad1).to(torch.long)
            output[1]['sample_pts_pad'] = torch.from_numpy(sample_pts_pad2).to(torch.long)

            VISUALIZE = False
            if VISUALIZE:  # DEBUG ONLY
                import matplotlib.pyplot as plt
                import cv2
                from mmcv.image.photometric import imdenormalize

                mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
                std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

                render_gt_img1 = rearrange(
                    data['render_gt_img'], 
                    '(n_cam n_frame) c h w -> n_frame n_cam h w c', 
                    n_frame=3)[0]
                render_gt_img2 = rearrange(
                    other_data['render_gt_img'],
                    '(n_cam n_frame) c h w -> n_frame n_cam h w c',
                    n_frame=3)[0]
                
                # render_gt_img1 = rearrange(
                #     data['render_gt_img'], 
                #     'n_cam c h w -> n_cam h w c')
                # render_gt_img2 = rearrange(
                #     other_data['render_gt_img'],
                #     'n_cam c h w -> n_cam h w c')

                # we only visualize one camera
                cam_idx = 1
                render_gt_img1 = render_gt_img1[cam_idx].cpu().numpy()
                render_gt_img2 = render_gt_img2[cam_idx].cpu().numpy()

                render_gt_img1 = imdenormalize(render_gt_img1, mean, std, to_bgr=False)
                render_gt_img2 = imdenormalize(render_gt_img2, mean, std, to_bgr=False)

                num_valid_pts = int(sample_pts_pad1[cam_idx][-1, 0])
                selected_points1 = sample_pts_pad1[cam_idx][:num_valid_pts]
                selected_points2 = sample_pts_pad2[cam_idx][:num_valid_pts]

                # draw circles
                for k in range(selected_points1.shape[0]):
                    pt1 = (int(selected_points1[k, 0]), int(selected_points1[k, 1]))
                    pt2 = (int(selected_points2[k, 0]), int(selected_points2[k, 1]))

                    render_gt_img1 = cv2.circle(render_gt_img1, pt1, 1, (0, 0, 255), -1)
                    render_gt_img2 = cv2.circle(render_gt_img2, pt2, 1, (0, 0, 255), -1)
                    
                # concatenate the images
                h, w, c = render_gt_img1.shape
                img_bar = np.ones((h, 10, 3)) * 255

                render_gt_img_concat = np.concatenate(
                    [render_gt_img1, img_bar, render_gt_img2], axis=1)
                
                # vertical concatenate
                cv2.imwrite('debug_origin7.png', render_gt_img_concat.astype(np.uint8))
                exit()
            
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
