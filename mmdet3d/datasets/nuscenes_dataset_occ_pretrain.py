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


def visualize_instance_image(instance_mask):
    instance_img = np.zeros((*instance_mask.shape, 4), dtype=np.uint8)

    unique_id = np.unique(instance_mask)
    for id in unique_id:
        if id == -1:
            continue
        # color = np.random.random(3) * 255
        color = np.concatenate([np.random.random(3), [0.7]]) * 255
        instance_img[instance_mask == id] = color
    return instance_img


@DATASETS.register_module()
class NuScenesDatasetOccPretrain(NuScenesDatasetOccpancy):

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """

        # import time
        # start = time.time()

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
            
            select_idx = idx + temporal_interval
            select_idx = np.clip(select_idx, 0, len(self) - 1)
            if self.data_infos[select_idx]['token'] == self.data_infos[idx]['token'] or \
                self.data_infos[select_idx]['scene_token'] != self.data_infos[idx]['scene_token']:
                idx = self._rand_another(idx)
                continue

            other_data = self.prepare_train_data(select_idx)

            output.append(other_data)

            if any([x is None for x in output]):
                idx = self._rand_another(idx)
                continue

            # load the scene flow
            scene_token = self.data_infos[idx]['scene_token']
            sample_token1 = data['img_metas'].data['sample_idx']
            sample_token2 = other_data['img_metas'].data['sample_idx']

            scene_flow_fp = osp.join("./data/nuscenes/nuscenes_scene_sequence_npz",
                                     scene_token, 
                                     sample_token1 + "_" + sample_token2 + ".npz")
            
            if scene_flow_fp.endswith('pkl'):
                with open(scene_flow_fp, 'rb') as f:
                    scene_flow_dict = pickle.load(f)
                coord1 = scene_flow_dict['coord1']  # NOTE: (u, v) coordinate
                coord2 = scene_flow_dict['coord2']

                render_size = data['render_gt_depth'].shape[-2:]  # (h, w)
                sample_pts_pad1, sample_pts_pad2 = self.unify_sample_points(
                    coord1, coord2, render_size)
            elif scene_flow_fp.endswith('npz'):
                scene_flow_dict = np.load(scene_flow_fp)

                sample_pts_pad1 = scene_flow_dict['coord1']  # (n_cam, n_points, 2)
                sample_pts_pad2 = scene_flow_dict['coord2']

                render_size = data['render_gt_depth'].shape[-2:]  # (h, w)
                origin_h, origin_w = 900, 1600
                height, width = render_size

                # need add processing when not using (900, 1600) render size
                # and please note that the last dimension is the number of points
                if height != origin_h or width != origin_w:
                    scale_h, scale_w = height / origin_h, width / origin_w
                    
                    num_valid_pts_arr1 = sample_pts_pad1[:, -1, :].copy()
                    num_valid_pts_arr2 = sample_pts_pad2[:, -1, :].copy()

                    sample_pts_pad1 *= np.array([scale_w, scale_h])
                    sample_pts_pad2 *= np.array([scale_w, scale_h])

                    sample_pts_pad1[:, -1, :] = num_valid_pts_arr1
                    sample_pts_pad2[:, -1, :] = num_valid_pts_arr2
            else:
                raise NotImplementedError
            
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

                instance_mask1 = data['instance_masks'][cam_idx].cpu().numpy()
                instance_mask2 = other_data['instance_masks'][cam_idx].cpu().numpy()

                num_valid_pts = int(sample_pts_pad1[cam_idx][-1, 0])
                selected_points1 = sample_pts_pad1[cam_idx][:num_valid_pts]
                selected_points2 = sample_pts_pad2[cam_idx][:num_valid_pts]

                # draw circles
                center = np.median(selected_points1, axis=0)
                set_max = range(128)
                colors = {m: i for i, m in enumerate(set_max)}
                colors = {m: (255 * np.array(plt.cm.hsv(i/float(len(colors))))[:3][::-1]).astype(np.int32)
                            for m, i in colors.items()}
                
                instance_img1 = visualize_instance_image(instance_mask1)
                instance_img2 = visualize_instance_image(instance_mask2)

                for k in range(selected_points1.shape[0]):
                    pt1 = (int(selected_points1[k, 0]), int(selected_points1[k, 1]))
                    pt2 = (int(selected_points2[k, 0]), int(selected_points2[k, 1]))

                    coord_angle = np.arctan2(pt1[1] - center[1], pt1[0] - center[0])
                    corr_color = np.int32(64 * coord_angle / np.pi) % 128
                    color = tuple(colors[corr_color].tolist())

                    render_gt_img1 = cv2.circle(render_gt_img1, pt1, 1, color, -1, cv2.LINE_AA)
                    render_gt_img2 = cv2.circle(render_gt_img2, pt2, 1, color, -1, cv2.LINE_AA)

                    instance_img1 = cv2.circle(instance_img1, pt1, 1, color, -1, cv2.LINE_AA)
                    instance_img2 = cv2.circle(instance_img2, pt2, 1, color, -1, cv2.LINE_AA)
                
                # concatenate the images
                h, w, c = render_gt_img1.shape
                img_bar = np.ones((h, 10, 3)) * 255

                render_gt_img_concat = np.concatenate(
                    [render_gt_img1, img_bar, render_gt_img2], axis=1)
                cv2.imwrite(f'debug_origin_{idx}_{select_idx}.png', render_gt_img_concat.astype(np.uint8))

                img_bar = np.ones((h, 10, 4)) * 255
                instance_img_concat = np.concatenate(
                    [instance_img1, img_bar, instance_img2], axis=1)
                cv2.imwrite(f'debug_instance_{idx}_{select_idx}.png', instance_img_concat.astype(np.uint8))
                
                exit()
            
            # collate these two frames together
            res = collate(output, samples_per_gpu=1)

            # end = time.time()
            # print("Data time:", end - start)

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


@DATASETS.register_module()
class NuScenesDatasetOccPretrainV2(NuScenesDatasetOccpancy):

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        
        while True:
            idx1 = idx  # the first frame

            # select the neiborhood frame
            temporal_interval = 1
            if np.random.choice([0, 1]):
                temporal_interval *= -1

            idx2 = idx1 + temporal_interval
            idx2 = np.clip(idx2, 0, len(self) - 1)
            if self.data_infos[idx2]['token'] == self.data_infos[idx1]['token'] or \
                self.data_infos[idx2]['scene_token'] != self.data_infos[idx1]['scene_token']:
                idx = self._rand_another(idx)
                continue
            
            ## start loading data
            output = []
            data1 = self.prepare_train_data(idx1)
            output.append(data1)
            
            data2 = self.prepare_train_data(idx2)
            output.append(data2)

            if any([x is None for x in output]):
                idx = self._rand_another(idx)
                continue

            # load the scene flow
            scene_token = self.data_infos[idx]['scene_token']
            sample_token1 = data1['img_metas'].data['sample_idx']
            sample_token2 = data2['img_metas'].data['sample_idx']

            scene_flow_fp = osp.join("./data/nuscenes/nuscenes_scene_sequence_npz",
                                     scene_token, 
                                     sample_token1 + "_" + sample_token2 + ".npz")
            
            scene_flow_dict = np.load(scene_flow_fp)
            sample_pts_pad1 = scene_flow_dict['coord1']  # (n_cam, n_points, 2)
            sample_pts_pad2 = scene_flow_dict['coord2']

            render_size = data1['render_gt_depth'].shape[-2:]  # (h, w)
            origin_h, origin_w = 900, 1600
            height, width = render_size

            # need add processing when not using (900, 1600) render size
            # and please note that the last dimension is the number of points
            if height != origin_h or width != origin_w:
                scale_h, scale_w = height / origin_h, width / origin_w
                
                num_valid_pts_arr1 = sample_pts_pad1[:, -1, :].copy()
                num_valid_pts_arr2 = sample_pts_pad2[:, -1, :].copy()

                sample_pts_pad1 *= np.array([scale_w, scale_h])
                sample_pts_pad2 *= np.array([scale_w, scale_h])

                sample_pts_pad1[:, -1, :] = num_valid_pts_arr1
                sample_pts_pad2[:, -1, :] = num_valid_pts_arr2
            
            sample_pts_pad1, sample_pts_pad2 = self.assign_instance_id(
                data1, sample_pts_pad1, sample_pts_pad2)

            output[0]['sample_pts_pad'] = torch.from_numpy(sample_pts_pad1).to(torch.long)
            output[1]['sample_pts_pad'] = torch.from_numpy(sample_pts_pad2).to(torch.long)

            # collate these two frames together
            res = collate(output, samples_per_gpu=1)

            # end = time.time()
            # print("Data time:", end - start)

            return res

    def assign_instance_id(self, 
                           base_data, 
                           sample_pts_pad1, 
                           sample_pts_pad2):
        """Assign the instance id for the paired points.
        """

        if 'instance_masks' not in base_data:
            return sample_pts_pad1, sample_pts_pad2
        
        min_num_instance_pixels = 10
        
        instance_mask = base_data['instance_masks'].numpy()  # (n_cam, h, w)
        num_cam = instance_mask.shape[0]

        all_cams_sample_pts_list1 = []
        all_cams_sample_pts_list2 = []
        for i in range(num_cam):
            sample_pts1 = sample_pts_pad1[i]  # (n_points, 2)
            sample_pts2 = sample_pts_pad2[i]
            
            num_valid_pts = int(sample_pts1[-1, 0])
            curr_sample_pts1 = sample_pts1[:num_valid_pts]
            curr_sample_pts2 = sample_pts2[:num_valid_pts]

            curr_instance_mask = instance_mask[i]  # (h, w)

            # fiter the instances which have fewer pixels
            origin_unique_inst, origin_count = np.unique(curr_instance_mask, return_counts=True)
            origin_unique_inst = origin_unique_inst[origin_count > min_num_instance_pixels]
            
            sampled_instance_mask1 = curr_instance_mask[curr_sample_pts1[:, 1].astype(np.int64), 
                                                        curr_sample_pts1[:, 0].astype(np.int64)]  # (N,)
            sampled_unique_inst1 = np.unique(sampled_instance_mask1)
            valid_sampled_unique_inst1 = np.intersect1d(sampled_unique_inst1, origin_unique_inst)

            sampled_instance_mask1[np.isin(sampled_instance_mask1, 
                                           valid_sampled_unique_inst1, 
                                           invert=True)] = -1
            
            sample_pts_inst1 = np.concatenate([curr_sample_pts1, sampled_instance_mask1[:, None]], axis=1)
            sample_pts_inst2 = np.concatenate([curr_sample_pts2, sampled_instance_mask1[:, None]], axis=1)

            ## padding to the same length
            final_sample_pts1 = np.zeros((sample_pts1.shape[0], 3)).astype(sample_pts1.dtype)
            final_sample_pts2 = np.zeros((sample_pts2.shape[0], 3)).astype(sample_pts2.dtype)

            final_sample_pts1[:num_valid_pts] = sample_pts_inst1
            final_sample_pts2[:num_valid_pts] = sample_pts_inst2

            final_sample_pts1[-1, :] = num_valid_pts
            final_sample_pts2[-1, :] = num_valid_pts
            
            all_cams_sample_pts_list1.append(final_sample_pts1)
            all_cams_sample_pts_list2.append(final_sample_pts2)

        all_cams_sample_pts_inst1 = np.stack(all_cams_sample_pts_list1, axis=0)
        all_cams_sample_pts_inst2 = np.stack(all_cams_sample_pts_list2, axis=0)
        
        return all_cams_sample_pts_inst1, all_cams_sample_pts_inst2

