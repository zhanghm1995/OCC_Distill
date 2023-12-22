'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-06-25 19:41:35
Email: haimingzhang@link.cuhk.edu.cn
Description: Create the part of the dataset samples to accelerate the
training process.
'''

import pickle
from collections import defaultdict
from tqdm import tqdm
import os.path as osp
import numpy as np
import mmcv


def load_mmdetection3d_infos(nuscenes_info_pickle):
    with open(nuscenes_info_pickle, "rb") as fp:
        nuscenes_infos = pickle.load(fp)

    print(type(nuscenes_infos))
    print(nuscenes_infos.keys())

    return nuscenes_infos


def create_part_pickle_deprecated(pickle_path, ratio=0.5):
    nuscenes_infos = load_mmdetection3d_infos(pickle_path)

    metadata = nuscenes_infos['metadata']
    infos = nuscenes_infos['infos']

    infos = list(sorted(infos, key=lambda e: e['timestamp']))

    scenes_dict = defaultdict(list)

    scene_token_list = []
    for data in tqdm(infos):
        scene_token = data['scene_token']
        scenes_dict[scene_token].append(1)
        scene_token_list.append(scene_token)
    
    split_scenes = np.split(data, np.unique(data, return_index=True)[1][1:])
    print(len(split_scenes))

    
    scenes_len_dict = dict()
    for k, v in scenes_dict.items():
        scenes_len_dict[k] = int(len(v) * ratio)
    
    count = -1
    for k, v in scenes_len_dict.items():
        count += 1
        print(k, v)
        if count >= 10:
            break

    ## split the dataset according to the length of each scenes
    new_infos = []
    num_cnt = 0

    curr_seq = []
    flag = True
    for idx, data in tqdm(enumerate(infos)):
        scene_token = data['scene_token']
        next_idx = min(idx + 1, len(infos) - 1)
        next_scene_token = infos[next_idx]['scene_token']

        if next_scene_token != scene_token:
            if len(curr_seq) >= scenes_len_dict[scene_token]:
                new_infos.extend(curr_seq)
                curr_seq = []
            else:
                curr_seq.append(data)
        else:
            curr_seq.append(data)

        #     num_cnt = 0

        # if num_cnt < scenes_len_dict[scene_token]:
        #     new_infos.append(data)
        #     num_cnt += 1

    print(len(new_infos))
    data = dict(infos=new_infos, metadata=metadata)

    filename, ext = osp.splitext(osp.basename(pickle_path))
    root_path = "./data/nuscenes"
    new_filename = f"{filename}-quarter{ext}"
    info_path = osp.join(root_path, new_filename)
    print(f"The results would be saved into {info_path}")
    mmcv.dump(data, info_path)
    

    ### Check the same scenes are neighbors
    # all_tokens = []
    # for idx, data in tqdm(enumerate(new_infos)):
    #     pre_idx = max(idx - 1, 0)
    #     pre_scene_token = new_infos[pre_idx]['scene_token']
    #     curr_scene_token = data['scene_token']

    #     if pre_scene_token == curr_scene_token:
    #         continue
    #     else:
    #         all_tokens.append(curr_scene_token)
    # print(len(all_tokens))


def get_scene_sequence_data(data_infos):
    scene_name_list = []
    total_scene_seq = []
    curr_seq = []
    for idx, data in enumerate(data_infos):
        scene_token = data['scene_token']
        next_idx = min(idx + 1, len(data_infos) - 1)
        next_scene_token = data_infos[next_idx]['scene_token']

        curr_seq.append(data)

        if next_scene_token != scene_token:
            total_scene_seq.append(curr_seq)
            scene_name_list.append(scene_token)
            curr_seq = []

    total_scene_seq.append(curr_seq)
    scene_name_list.append(scene_token)
    return scene_name_list, total_scene_seq


def create_part_pickle(pickle_path, ratio=0.5):
    nuscenes_infos = load_mmdetection3d_infos(pickle_path)

    metadata = nuscenes_infos['metadata']
    infos = nuscenes_infos['infos']

    infos = list(sorted(infos, key=lambda e: e['timestamp']))

    scene_name_list, total_scene_seq = get_scene_sequence_data(infos)

    new_infos = []
    for idx, (scene_name, scene_seq) in tqdm(enumerate(zip(scene_name_list, total_scene_seq)),
                                             total=len(scene_name_list)):
        keep_length = int(len(scene_seq) * ratio)
        new_infos.extend(scene_seq[:keep_length])

    print(len(new_infos))
    data = dict(infos=new_infos, metadata=metadata)

    filename, ext = osp.splitext(osp.basename(pickle_path))
    root_path = "./data/nuscenes"
    new_filename = f"{filename}-quarter{ext}"
    info_path = osp.join(root_path, new_filename)
    print(f"The results would be saved into {info_path}")
    mmcv.dump(data, info_path)


if __name__ == "__main__":
    nuscenes_info_pickle = "data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_train.pkl"
    create_part_pickle(nuscenes_info_pickle, ratio=0.25)