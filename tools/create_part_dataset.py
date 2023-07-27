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


def load_mmdetection3d_infos(nuscenes_info_pickle):
    with open(nuscenes_info_pickle, "rb") as fp:
        nuscenes_infos = pickle.load(fp)

    print(type(nuscenes_infos))
    print(nuscenes_infos.keys())

    return nuscenes_infos


def create_part_pickle(pickle_path, ratio=0.5):
    nuscenes_infos = load_mmdetection3d_infos(pickle_path)

    metadata = nuscenes_infos['metadata']
    infos = nuscenes_infos['infos']

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


if __name__ == "__main__":
    nuscenes_info_pickle = "data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_train.pkl"
    create_part_pickle(nuscenes_info_pickle, ratio=0.25)