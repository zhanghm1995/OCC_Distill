'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-10-21 20:48:06
Email: haimingzhang@link.cuhk.edu.cn
Description: Create the symbolic link for the nuscene scene sequence data.
'''

import pickle
from PIL import Image
import os
import os.path as osp
import numpy as np
from tqdm import tqdm


def save_all_scene_sequence_images(anno_file):
    with open(anno_file, "rb") as fp:
        dataset = pickle.load(fp)
    
    data_infos = dataset['infos']
    data_infos = list(sorted(data_infos, key=lambda e: e['timestamp']))

    scene_name_list = []
    total_scene_seq = []
    curr_seq = []
    for idx, data in tqdm(enumerate(data_infos)):
        scene_token = data['scene_token']
        pre_idx = max(idx - 1, 0)
        pre_scene_token = data_infos[pre_idx]['scene_token']

        if scene_token != pre_scene_token:
            total_scene_seq.append(curr_seq)
            scene_name_list.append(scene_token)
            curr_seq = [data]
        else:
            curr_seq.append(data)

        if idx == len(data_infos) - 1:
            total_scene_seq.append(curr_seq)
            scene_name_list.append(scene_token)

    print(len(total_scene_seq), len(scene_name_list))

    cam_img_size = [800, 450]  # [w, h]
    save_dir = "./data/nuscenes_scene_sequence"
    for idx, (scene_name, scene_seq) in tqdm(enumerate(zip(scene_name_list, total_scene_seq)),
                                             total=len(scene_name_list)):
        # create the scene directory
        scene_dir = osp.join(save_dir, scene_name)
        # save the scene sequence
        for frame in scene_seq:
            cam_names = frame['cams']
            for cam_name in cam_names:
                cam_data = cam_names[cam_name]
                filename = cam_data['data_path']

                # create the camera image directory
                cam_dir = osp.join(scene_dir, cam_name, 'color')
                os.makedirs(cam_dir, exist_ok=True)

                # copy the camera image to the directory
                cam_img_resized = Image.open(filename).resize(
                    cam_img_size, Image.BILINEAR)
                
                dst_file_name = osp.basename(filename)
                cam_img_path = osp.join(cam_dir, dst_file_name)
                cam_img_resized.save(cam_img_path)


def link_all_scene_sequence_images(anno_file):
    with open(anno_file, "rb") as fp:
        dataset = pickle.load(fp)
    
    data_infos = dataset['infos']
    data_infos = list(sorted(data_infos, key=lambda e: e['timestamp']))

    scene_name_list = []
    total_scene_seq = []
    curr_seq = []
    for idx, data in tqdm(enumerate(data_infos)):
        scene_token = data['scene_token']
        next_idx = min(idx + 1, len(data_infos) - 1)
        next_scene_token = data_infos[next_idx]['scene_token']

        curr_seq.append(data)

        if next_scene_token != scene_token:
            total_scene_seq.append(curr_seq)
            scene_name_list.append(scene_token)
            curr_seq = []

        if idx == len(data_infos) - 1:
            total_scene_seq.append(curr_seq)
            scene_name_list.append(scene_token)

    print(len(total_scene_seq))
    scene_length = [len(seq) for seq in total_scene_seq]
    print(np.min(scene_length), np.max(scene_length))

    data_root_relative = "./data/nuscenes"
    data_root_abs = "/data2/dataset/nuscenes"

    save_dir = "/data1/zhanghm/Datasets/AD/nuscenes_scene_sequence"
    for idx, (scene_name, scene_seq) in tqdm(enumerate(zip(scene_name_list, total_scene_seq)),
                                             total=len(scene_name_list)):
        # create the scene directory
        scene_dir = osp.join(save_dir, scene_name)
        # save the scene sequence
        for frame in scene_seq:
            cam_names = frame['cams']
            for cam_name in cam_names:
                cam_data = cam_names[cam_name]
                filename = cam_data['data_path']

                # create the camera image directory
                cam_dir = osp.join(scene_dir, cam_name, 'color')
                os.makedirs(cam_dir, exist_ok=True)
                
                dst_file_name = osp.basename(filename)
                cam_img_path = osp.join(cam_dir, dst_file_name)
                tgt_abs_file_path = osp.abspath(cam_img_path)

                # create the symbolic link
                src_abs_file_path = filename.replace(data_root_relative, data_root_abs)
                # print(filename, src_abs_file_path, tgt_abs_file_path)
                if not os.path.exists(tgt_abs_file_path):
                    os.symlink(src_abs_file_path, tgt_abs_file_path)


if __name__ == "__main__":
    splits = ["train", "val"]
    for split in splits:
        pickle_path = f"data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_{split}.pkl"
        link_all_scene_sequence_images(pickle_path)
