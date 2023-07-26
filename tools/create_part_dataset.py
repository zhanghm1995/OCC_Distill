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
import mmcv


def load_mmdetection3d_infos():
    nuscenes_info_pickle = "/home/zhanghm/Research/Occupancy/BEVDet/data/nuscenes/bevdetv3-nuscenes_infos_train.pkl"

    with open(nuscenes_info_pickle, "rb") as fp:
        nuscenes_infos = pickle.load(fp)

    print(type(nuscenes_infos))
    print(nuscenes_infos.keys())

    return nuscenes_infos


def create_part_pickle(input, ratio=0.5):
    metadata = input['metadata']
    infos = input['infos']

    scenes_dict = defaultdict(list)

    for data in tqdm(infos):
        scene_token = data['scene_token']
        scenes_dict[scene_token].append(1)
    
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
    for idx, data in tqdm(enumerate(infos)):
        scene_token = data['scene_token']
        next_idx = min(idx + 1, len(infos) - 1)
        next_scene_token = infos[next_idx]['scene_token']

        if next_scene_token != scene_token:
            num_cnt = 0

        if num_cnt < scenes_len_dict[scene_token]:
            new_infos.append(data)
            num_cnt += 1

    print(len(new_infos))
    data = dict(infos=new_infos, metadata=metadata)
    root_path = "./data/nuscenes"
    filename = "quarter-bevdetv3-nuscenes_infos_train.pkl"
    info_path = osp.join(root_path, filename)
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


if __name__ == "__main__":
    data_infos = load_mmdetection3d_infos()
    
    create_part_pickle(data_infos, ratio=0.25)