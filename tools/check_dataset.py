'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-10-30 08:30:30
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import pickle
import time
import mmcv
import torch


def check_sam_mask_data():
    import pycocotools.mask as maskUtils

    sam_mask_root = "data/nuscenes/sam_mask_json/CAM_FRONT"
    sam_mask_files = os.listdir(sam_mask_root)
    
    num_instances_list = []
    for file in tqdm(sam_mask_files):
        fp = osp.join(sam_mask_root, file)
        anns = mmcv.load(fp)

        anns = sorted(anns, key=(lambda x: x['area']))

        all_valid_mask = []
        for idx, ann in enumerate(anns):
            valid_mask = torch.tensor(maskUtils.decode(ann['segmentation']))
            all_valid_mask.append(valid_mask)

        all_valid_mask = torch.stack(all_valid_mask)  # (N, H, W)

        num_instances_list.append(len(anns))
    
    print(np.mean(num_instances_list), np.max(num_instances_list), np.min(num_instances_list))


def convert_json_sam_mask_to_npz():
    import pycocotools.mask as maskUtils

    # anno_file = "data/nuscenes/bevdetv3-inst-nuscenes_infos_train.pkl"
    anno_file = "data/nuscenes/bevdetv3-inst-nuscenes_infos_val.pkl"
    # 1) load the dataset pickle file
    with open(anno_file, "rb") as fp:
        dataset = pickle.load(fp)
    
    data_infos = dataset['infos']
    data_infos = list(sorted(data_infos, key=lambda e: e['timestamp']))
    print(f"Total length of data infos: {len(data_infos)}")

    range1, range2 = 5000, 6020
    data_infos = data_infos[range1:range2]
    print(f"Convert the data from {range1} to {range2}")

    choose_cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 
                   'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    
    instance_mask_dir = 'data/nuscenes/sam_mask_json'

    save_root = "data/nuscenes/sam_mask_npz_val"
    os.makedirs(save_root, exist_ok=True)

    MAX_NUM_INSTANCES = 254
    for curr in tqdm(data_infos):
        instance_imgs = []
        save_path = osp.join(save_root, f"{curr['token']}.npz")
        if osp.exists(save_path):
            continue

        for cam_name in choose_cams:
            cam_data = curr['cams'][cam_name]
            file_name = osp.splitext(osp.basename(cam_data['data_path']))[0]
            instance_mask_path = osp.join(instance_mask_dir, 
                                          cam_name,
                                          f"{file_name}.json")
            
            anns = mmcv.load(instance_mask_path)
            anns = sorted(anns, key=(lambda x: x['area']))

            if len(anns) > MAX_NUM_INSTANCES:
                print(f"Warning: the number of instances {len(anns)} is larger than 254: {curr['token']}")
                anns = anns[len(anns) - MAX_NUM_INSTANCES:]

            all_valid_mask = []
            for idx, ann in enumerate(anns):
                valid_mask = maskUtils.decode(ann['segmentation'])
                all_valid_mask.append(valid_mask)

            all_valid_mask = np.stack(all_valid_mask)  # (N, H, W)
        
            N, H, W = all_valid_mask.shape
            indices = np.arange(1, N + 1)[:, None, None]
            transformed_mask = all_valid_mask * indices  # (N, H, W)
            mask = np.max(transformed_mask, axis=0)

            # get the unique continous mask
            unique_id, mapping = np.unique(mask, return_inverse=True)
            mask = mapping.reshape(mask.shape)
            # mask = self.transform(mask[None]).squeeze(0)  # resize
            # mask = mask - 1  # -1 means we donot detect any instances in these regions
            instance_imgs.append(mask)
        
        instance_imgs = np.stack(instance_imgs)
        instance_imgs = instance_imgs.astype(np.uint8)
        np.savez_compressed(save_path, instance_imgs)


def load_data():
    import time
    start = time.time()
    file_path = "data/nuscenes/sam_mask_npz/0a447061deec491692a90cfd61c59673.npz"
    data = np.load(file_path)
    print(data.files)
    data = data['arr_0'].astype(np.int64) - 1

    data = torch.from_numpy(data)
    print(data.dtype)
    # print(data.shape, data.dtype, np.unique(data))
    end = time.time()
    print("Data time:", end - start)


def check_nuscene_flow_data():
    nuscene_flow_dir = "data/nuscenes_scene_sequence"
    all_dirs = os.listdir(nuscene_flow_dir)

    print(len(all_dirs))
    
    all_files_length = []
    for dir in tqdm(all_dirs):
        dir_path = osp.join(nuscene_flow_dir, dir)
        all_files = os.listdir(dir_path)
        all_files_length.append(len(all_files))
    
    print(np.sum(all_files_length))

    ## read the pickle data
    all_data = []
    start = time.time()
    for dir in tqdm(all_dirs):
        dir_path = osp.join(nuscene_flow_dir, dir)
        all_files = os.listdir(dir_path)
        for file in all_files:
            file_path = osp.join(dir_path, file)
            try:
                with open(file_path, 'rb') as f:
                    scene_flow_dict = pickle.load(f)
                all_data.append(scene_flow_dict)
            except:
                print(file_path)
    end = time.time()
    print("Data time:", end - start)

    save_path = osp.join('results/', "scene_data.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(all_data, f)


def change_file_extensions():
    import glob
    all_files = sorted(glob.glob("data/nuscenes/nuscenes_scene_sequence_npz/*/*.pkl.npz"))
    print(len(all_files), all_files[:3])

    for file in tqdm(all_files):
        new_file = file.replace(".pkl.npz", ".npz")
        print(new_file)
        os.rename(file, new_file)
    


if __name__ == "__main__":
    change_file_extensions()
    exit()

    convert_json_sam_mask_to_npz()

    # load_data()