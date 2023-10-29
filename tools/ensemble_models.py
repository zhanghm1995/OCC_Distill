'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-10-29 06:21:54
Email: haimingzhang@link.cuhk.edu.cn
Description: Ensemble different models.
'''

import os
import os.path as osp
import numpy as np


def load_results(file_path):
    with open(file_path) as f:
        lines = [line.rstrip() for line in f.readlines()]

    # remove the string
    lines = [line.split(' ')[-1] for line in lines]

    # convert to float
    lines = [float(line) for line in lines]
    return lines


def load_results_list(fp_list):
    lines_list = []
    for fp in fp_list:
        lines = load_results(fp)
        lines_list.append(lines)
    return lines_list


file_path = "./results/results_5189.txt"
file_path2 = "./results/results_5284.txt"
all_files = [file_path, file_path2]

lines_list = load_results_list(all_files)

all_results = np.array(lines_list)
print(all_results.shape)

max_results = np.max(all_results, axis=0)
print(max_results)

mAP = np.mean(max_results)
print(mAP)