'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-10-26 21:06:33
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''


import os
import pickle
import numpy as np
from tqdm import tqdm
from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, view_points


def load_pickle():
    """Load the nuscenes dataset pickle file, and explore the
    infos contents.

    Args:
        pickle_file_path (_type_): _description_
    """
    pickle_file_path = "data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_train.pkl"

    with open(pickle_file_path, "rb") as fp:
        dataset = pickle.load(fp)
    
    print(type(dataset))
    print(dataset.keys())
    
    data_infos = dataset['infos']
    print(len(data_infos))

    idx = 100
    info = data_infos[idx]
    print(info.keys())


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


def create_instance_pickle_with_corners():
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

            # obtain the bboxes in camera from the nerfstudio
            for camera in info['cams']:
                camera_token = curr_sample['data'][camera]

                _, boxes, _ = nuscenes.get_sample_data(
                    camera_token, box_vis_level=BoxVisibility.ANY)

                all_box_corners = [box.corners() for box in boxes]
                if len(all_box_corners) > 0:
                    all_box_corners = np.stack(all_box_corners)
                info['cams'][camera]['box_corners'] = all_box_corners

            infos.append(info)
        
        dataset['infos'] = infos
        
        print(len(dataset['infos']))
        with open('data/nuscenes/bevdetv3-inst-nuscenes_infos_%s.pkl' % set, 'wb') as fid:
            pickle.dump(dataset, fid)


if __name__ == '__main__':
    load_pickle()
    exit()
    create_instance_pickle_with_corners()
