'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-07-08 03:20:13
Email: haimingzhang@link.cuhk.edu.cn
Description: Combine the train and val data of nuscenes dataset together.
'''

import pickle
import os
import numpy as np


def merge_trainval_data(extra_tag):
    dataroot = './data/nuscenes/'

    trainval_infos = []
    for set in ['train', 'val']:
        dataset = pickle.load(
            open('./data/nuscenes/%s_infos_%s.pkl' % (extra_tag, set), 'rb'))
        print(type(dataset), dataset.keys())
        print(len(dataset['infos']), len(dataset['metadata']))

        trainval_infos += dataset['infos']
    
    print(len(trainval_infos))
    trainval_data = {'infos': trainval_infos, 'metadata': dataset['metadata']}

    with open('./data/nuscenes/%s_infos_%s.pkl' % (extra_tag, "trainval"),
              'wb') as fid:
        pickle.dump(trainval_data, fid)


if __name__ == '__main__':
    extra_tag = 'bevdetv3-nuscenes'
    merge_trainval_data(extra_tag)