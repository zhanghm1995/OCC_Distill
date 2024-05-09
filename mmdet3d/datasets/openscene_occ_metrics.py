'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-05-09 14:59:40
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import os
import os.path as osp
import numpy as np
import pickle
import torch
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import pickle as pkl
import argparse
import time
import torch
import sys, platform
from sklearn.neighbors import KDTree
from termcolor import colored
from pathlib import Path
from copy import deepcopy
from functools import reduce

np.seterr(divide='ignore', invalid='ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def pcolor(string, color, on_color=None, attrs=None):
    """
    Produces a colored string for printing

    Parameters
    ----------
    string : str
        String that will be colored
    color : str
        Color to use
    on_color : str
        Background color to use
    attrs : list of str
        Different attributes for the string

    Returns
    -------
    string: str
        Colored string
    """
    return colored(string, color, on_color, attrs)


def getCellCoordinates(points, voxelSize):
    return (points / voxelSize).astype(np.int)


def getNumUniqueCells(cells):
    M = cells.max() + 1
    return np.unique(cells[:, 0] + M * cells[:, 1] + M ** 2 * cells[:, 2]).shape[0]


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


class Metric_mIoU():
    def __init__(self,
                 save_dir='.',
                 num_classes=12,
                 use_lidar_mask=False,
                 use_image_mask=False,
                 ):
        self.class_names = ['vehicle', 'place_holder1', 
                            'place_holder2', 'place_holder3', 
                            'czone_sign', 'bicycle', 
                            'generic_object', 'pedestrian', 
                            'traffic_cone', 'barrier',
                            'background', 'free']
        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes

        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.hist_occ = np.zeros((2, 2))
        self.cnt = 0

    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        # empty classes:0
        non-empty class: 0-16
        free voxel class: 17

        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label
            gt (1-d array): gt_occupancu_label

        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample)
        """
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def per_class_iu(self, hist):

        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def compute_mIoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        # for ind_class in range(n_classes):
        #     print(str(round(mIoUs[ind_class] * 100, 2)))
        # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        return round(np.nanmean(mIoUs) * 100, 2), hist


    def add_batch(self,semantics_pred,semantics_gt,mask_lidar,mask_camera):
        self.cnt += 1
        if self.use_image_mask:
            masked_semantics_gt = semantics_gt[mask_camera]
            masked_semantics_pred = semantics_pred[mask_camera]
        elif self.use_lidar_mask:
            masked_semantics_gt = semantics_gt[mask_lidar]
            masked_semantics_pred = semantics_pred[mask_lidar]
        else:
            masked_semantics_gt = semantics_gt
            masked_semantics_pred = semantics_pred

        _, _hist = self.compute_mIoU(masked_semantics_pred, masked_semantics_gt, self.num_classes)
        self.hist += _hist

        ## add below for binary occupancy evaluation
        masked_semantics_gt[masked_semantics_gt != 11] = 1
        masked_semantics_gt[masked_semantics_gt == 11] = 0
        masked_semantics_pred[masked_semantics_pred != 11] = 1
        masked_semantics_pred[masked_semantics_pred == 11] = 0
        _, _hist_occ = self.compute_mIoU(masked_semantics_pred, masked_semantics_gt, 2)
        self.hist_occ += _hist_occ

    def count_miou(self):
        mIoU = self.per_class_iu(self.hist)
        IoU = self.per_class_iu(self.hist_occ)
        # assert cnt == num_samples, 'some samples are not included in the miou calculation'
        print(f'===> per class IoU of {self.cnt} samples:')
        for ind_class in range(self.num_classes-1):
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU[ind_class] * 100, 2)))

        print(f'===> mIoU of {self.cnt} samples: ' + str(round(np.nanmean(mIoU[:self.num_classes-1]) * 100, 2)))
        print(f'===> IoU of {self.cnt} samples: ' + str(round(np.nanmean(IoU[1]) * 100, 2)))
        # print(f'===> sample-wise averaged mIoU of {cnt} samples: ' + str(round(np.nanmean(mIoU_avg), 2)))

        return self.class_names, mIoU, self.cnt


class Metric_OccIoU():
    def __init__(self,
                 save_dir='.',
                 use_lidar_mask=False,
                 use_image_mask=False,
                 **kwargs,
                 ):
        self.class_names = ['free', 'occupancy']
        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask

        assert not self.use_image_mask and not self.use_lidar_mask, \
            'mask is not supported for openscene occupancy evaluation'

        self.num_classes = 2
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0

    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        # empty classes:0
        non-empty class: 0-16
        free voxel class: 17

        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label
            gt (1-d array): gt_occupancu_label

        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample)
        """
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def per_class_iu(self, hist):

        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def compute_mIoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        # for ind_class in range(n_classes):
        #     print(str(round(mIoUs[ind_class] * 100, 2)))
        # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        return round(np.nanmean(mIoUs) * 100, 2), hist

    def add_batch(self, 
                  semantics_pred, 
                  semantics_gt, 
                  mask_lidar=None, 
                  mask_camera=None,
                  already_is_binary_gt=False):
        self.cnt += 1
        masked_semantics_gt = semantics_gt
        masked_semantics_pred = semantics_pred

        occupancy_gt = masked_semantics_gt.copy()
        if not already_is_binary_gt:
            occupancy_gt = np.zeros_like(masked_semantics_gt)
            occupancy_gt[masked_semantics_gt != 11] = 1
        _, _hist = self.compute_mIoU(masked_semantics_pred, occupancy_gt, self.num_classes)
        self.hist += _hist

    def count_miou(self):
        IoU = self.per_class_iu(self.hist)
        if np.isnan(IoU[1]):
            print(IoU, "!!!!")
            IoU[1] = 0
        
        print(f'===> per class IoU of {self.cnt} samples:')
        for ind_class in range(self.num_classes):
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(IoU[ind_class] * 100, 2)))

        print(f'===> OccIoU of {self.cnt} samples: ' + str(round(np.nanmean(IoU[1]) * 100, 2)))

        return {'IoU': round(IoU[1] * 100, 2)}


class Metric_Segmentation_mIoU():
    def __init__(self,
                 save_dir='.',
                 num_classes=17,
                 use_lidar_mask=False,
                 use_image_mask=False,
                 ):
        self.class_names = ['ignore', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                            'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation']
        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes
        self.hist_list = []
        self.cnt = 0

    def add_batch(self, semantics_pred, semantics_gt):
        self.hist_list.append(fast_hist(semantics_pred, semantics_gt, len(self.class_names)))
        self.cnt += 1

    def count_miou(self):
        iou = per_class_iu(sum(self.hist_list))
        # assert cnt == num_samples, 'some samples are not included in the miou calculation'
        print(f'===> per class IoU of {self.cnt} samples:')
        for ind_class in range(1, self.num_classes):
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(iou[ind_class] * 100, 2)))

        print(f'===> mIoU of {self.cnt} samples: ' + str(round(np.nanmean(iou[1:]) * 100, 2)))

        return {'mIoU': round(np.nanmean(iou[1:]) * 100, 2)}

class Metric_FScore():
    def __init__(self,

                 leaf_size=10,
                 threshold_acc=0.6,
                 threshold_complete=0.6,
                 voxel_size=[0.4, 0.4, 0.4],
                 range=[-40, -40, -1, 40, 40, 5.4],
                 void=[17, 255],
                 use_lidar_mask=False,
                 use_image_mask=False, ) -> None:

        self.leaf_size = leaf_size
        self.threshold_acc = threshold_acc
        self.threshold_complete = threshold_complete
        self.voxel_size = voxel_size
        self.range = range
        self.void = void
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.cnt=0
        self.tot_acc = 0.
        self.tot_cmpl = 0.
        self.tot_f1_mean = 0.
        self.eps = 1e-8



    def voxel2points(self, voxel):
        # occIdx = torch.where(torch.logical_and(voxel != FREE, voxel != NOT_OBSERVED))
        # if isinstance(voxel, np.ndarray): voxel = torch.from_numpy(voxel)
        mask = np.logical_not(reduce(np.logical_or, [voxel == self.void[i] for i in range(len(self.void))]))
        occIdx = np.where(mask)

        points = np.concatenate((occIdx[0][:, None] * self.voxel_size[0] + self.voxel_size[0] / 2 + self.range[0], \
                                 occIdx[1][:, None] * self.voxel_size[1] + self.voxel_size[1] / 2 + self.range[1], \
                                 occIdx[2][:, None] * self.voxel_size[2] + self.voxel_size[2] / 2 + self.range[2]),
                                axis=1)
        return points

    def add_batch(self,semantics_pred,semantics_gt,mask_lidar,mask_camera ):

        # for scene_token in tqdm(preds_dict.keys()):
        self.cnt += 1

        if self.use_image_mask:

            semantics_gt[mask_camera == False] = 255
            semantics_pred[mask_camera == False] = 255
        elif self.use_lidar_mask:
            semantics_gt[mask_lidar == False] = 255
            semantics_pred[mask_lidar == False] = 255
        else:
            pass

        ground_truth = self.voxel2points(semantics_gt)
        prediction = self.voxel2points(semantics_pred)
        if prediction.shape[0] == 0:
            accuracy=0
            completeness=0
            fmean=0

        else:
            prediction_tree = KDTree(prediction, leaf_size=self.leaf_size)
            ground_truth_tree = KDTree(ground_truth, leaf_size=self.leaf_size)
            complete_distance, _ = prediction_tree.query(ground_truth)
            complete_distance = complete_distance.flatten()

            accuracy_distance, _ = ground_truth_tree.query(prediction)
            accuracy_distance = accuracy_distance.flatten()

            # evaluate completeness
            complete_mask = complete_distance < self.threshold_complete
            completeness = complete_mask.mean()

            # evalute accuracy
            accuracy_mask = accuracy_distance < self.threshold_acc
            accuracy = accuracy_mask.mean()

            fmean = 2.0 / (1 / (accuracy+self.eps) + 1 / (completeness+self.eps))

        self.tot_acc += accuracy
        self.tot_cmpl += completeness
        self.tot_f1_mean += fmean

    def count_fscore(self,):
        base_color, attrs = 'red', ['bold', 'dark']
        print(pcolor('\n######## F score: {} #######'.format(self.tot_f1_mean / self.cnt), base_color, attrs=attrs))

