'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-10-26 21:06:33
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''


import os
import pickle
from tqdm import tqdm
from nuscenes import NuScenes


def create_instance_pickle():
    version = 'v1.0-trainval'
    dataroot = 'data/nuscenes/'
    nuscenes = NuScenes(version, dataroot)

    for set in ['train', 'val']:
        dataset = pickle.load(
            open('data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_%s.pkl' % set, 'rb'))
        
        infos = []
        for info in tqdm(dataset['infos']):
            token = info['token']
            curr_sample = nuscenes.get('sample', token)
            lidar_data = nuscenes.get('sample_data', curr_sample['data']['LIDAR_TOP'])
            lidar_path, boxes, _ = nuscenes.get_sample_data(lidar_data['token'])
            boxes_token = [box.token for box in boxes]
            instance_tokens = [nuscenes.get(
                'sample_annotation', box_token)['instance_token'] for box_token in boxes_token]
            info['instance_tokens'] = instance_tokens
            infos.append(info)
        
        dataset['infos'] = infos
        
        print(len(dataset['infos']))
        with open('data/nuscenes/bevdetv3-inst-nuscenes_infos_%s.pkl' % set, 'wb') as fid:
            pickle.dump(dataset, fid)


if __name__ == '__main__':
    create_instance_pickle()
