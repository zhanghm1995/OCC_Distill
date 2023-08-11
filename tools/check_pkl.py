'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-07-26 09:57:35
Email: haimingzhang@link.cuhk.edu.cn
Description: Check the .pkl dataset files.
'''

import pickle
from PIL import Image
import os
import os.path as osp
import numpy as np
from tqdm import tqdm


def load_pickle(pickle_file_path):
    """Load the nuscenes dataset pickle file, and explore the
    infos contents.

    Args:
        pickle_file_path (_type_): _description_
    """
    with open(pickle_file_path, "rb") as fp:
        dataset = pickle.load(fp)
    
    print(type(dataset))
    print(dataset.keys())

    data_infos = dataset['infos']
    print(len(data_infos))

    print(data_infos[0].keys())
    for idx, frame in enumerate(data_infos):
        print(f"============={idx}==============")
        for key, value in frame.items():
            if isinstance(value, str):
                print(key, value)
        
        if idx >= 3:
            break


def sample_camera_images(anno_file, 
                         sample_idx=None,
                         save_root=None,
                         need_sort_data=True):
    """Sample the camera images according to the sample_idx
    from the dataset and save them to the directory.

    Args:
        anno_file (_type_): The .pkl file path.
        save_root (str, optional): The save root. Defaults to None.
    """
    with open(anno_file, "rb") as fp:
        dataset = pickle.load(fp)
    
    data_infos = dataset['infos']

    if need_sort_data:
        data_infos = list(sorted(data_infos, key=lambda e: e['timestamp']))
    
    if sample_idx is None:
        sample_idx = np.random.randint(6019)
    info = data_infos[sample_idx]
    print(sample_idx)
    
    cam_names = info['cams']
    for cam_name in cam_names:
        cam_data = cam_names[cam_name]
        filename = cam_data['data_path']
        cam_token = cam_data['sample_data_token']
        # print(filename, cam_token)
        sam_mask_path = osp.join("./data/nuscenes/superpixels_sam", 
                                 f"{cam_token}.png")

        if save_root is not None:
            # create the camera image directory
            os.makedirs(save_root, exist_ok=True)
            cam_dir = osp.join(save_root, cam_name)
            os.makedirs(cam_dir, exist_ok=True)

            # copy the camera image to the directory
            dst_file_name = osp.splitext(osp.basename(filename))[0]
            cam_img_path = osp.join(cam_dir, f"{dst_file_name}_image_{sample_idx}.jpg")
            os.system(f"cp {filename} {cam_img_path}")

            # copy the semantic mask to the directory
            sam_mask_dst_path = osp.join(cam_dir, f"{dst_file_name}_mask.png")
            os.system(f"cp {sam_mask_path} {sam_mask_dst_path}")


def save_all_cam_images(anno_file, save_root=None):
    with open(anno_file, "rb") as fp:
        dataset = pickle.load(fp)
    
    data_infos = dataset['infos']

    data_infos = list(sorted(data_infos, key=lambda e: e['timestamp']))
    
    for idx, info in tqdm(enumerate(data_infos)):
        cam_names = info['cams']
        for cam_name in cam_names:
            if cam_name != 'CAM_FRONT':
                continue
            cam_data = cam_names[cam_name]
            filename = cam_data['data_path']
            cam_token = cam_data['sample_data_token']
            sam_mask_path = osp.join("./data/nuscenes/superpixels_sam", 
                                    f"{cam_token}.png")

            if save_root is not None:
                # create the camera image directory
                os.makedirs(save_root, exist_ok=True)
                cam_dir = osp.join(save_root, cam_name)
                os.makedirs(cam_dir, exist_ok=True)

                # copy the camera image to the directory
                dst_file_name = osp.splitext(osp.basename(filename))[0]
                cam_img_path = osp.join(cam_dir, f"{idx}_{dst_file_name}.jpg")
                os.system(f"cp {filename} {cam_img_path}")


def read_sam_mask():
    image_path = "debug/CAM_BACK/98f0569def5c4a5cb980642d2cff2f5b_mask.png"
    image_path = "debug/CAM_BACK_LEFT/0b04a20f4e2043b4be89673cc0787cd0_mask.png"
    img = Image.open(image_path)
    print(img.size, img.format, img.mode)

    img = np.array(img)
    print(img.shape, img.dtype, np.unique(img), img.min(), img.max())


def save_point_cloud(anno_file, sample_idx: int = None):
    with open(anno_file, "rb") as fp:
        dataset = pickle.load(fp)
    
    data_infos = dataset['infos']
    
    if sample_idx is None:
        sample_idx = np.random.randint(6019)
    info = data_infos[sample_idx]

    points_path = info['lidar_path']
    points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 5)
    print(points.shape)

    save_path = f"./pts_{sample_idx}.xyz"
    np.savetxt(save_path, points[:, :3])


if __name__ == "__main__":
    # pickle_path = "data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_train.pkl"
    pickle_path = "data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_val.pkl"
    sample_camera_images(pickle_path, sample_idx=4637, save_root="./AAAI_visualization")
    exit()

    save_all_cam_images(pickle_path, save_root="./aaai_all_validation")
    exit(0)
    save_point_cloud(pickle_path, sample_idx=3827)
    exit()
    
    
    
    
    read_sam_mask()
    exit()
    
    exit(0)
    load_pickle(pickle_path)

