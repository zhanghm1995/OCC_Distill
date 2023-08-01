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


def sample_camera_images(anno_file, save_root=None):
    """Sample the camera images from the dataset and save them to the directory.

    Args:
        anno_file (_type_): The .pkl file path.
        save_root (str, optional): The save root. Defaults to None.
    """
    with open(anno_file, "rb") as fp:
        dataset = pickle.load(fp)
    
    data_infos = dataset['infos']

    info = data_infos[10]
    cam_names = info['cams']
    for cam_name in cam_names:
        cam_data = cam_names[cam_name]
        print(cam_data)
        filename = cam_data['data_path']
        cam_token = cam_data['sample_data_token']
        print(filename, cam_token)
        sam_mask_path = osp.join("./data/nuscenes/superpixels_sam", 
                                 f"{cam_token}.png")

        if save_root is not None:
            # create the camera image directory
            os.makedirs(save_root, exist_ok=True)
            cam_dir = osp.join(save_root, cam_name)
            os.makedirs(cam_dir, exist_ok=True)

            # copy the camera image to the directory
            cam_img_path = osp.join(cam_dir, f"{cam_token}_image.jpg")
            os.system(f"cp {filename} {cam_img_path}")

            # copy the semantic mask to the directory
            sam_mask_dst_path = osp.join(cam_dir, f"{cam_token}_mask.png")
            os.system(f"cp {sam_mask_path} {sam_mask_dst_path}")


def read_sam_mask():
    image_path = "debug/CAM_BACK/98f0569def5c4a5cb980642d2cff2f5b_mask.png"
    image_path = "debug/CAM_BACK_LEFT/0b04a20f4e2043b4be89673cc0787cd0_mask.png"
    img = Image.open(image_path)
    print(img.size, img.format, img.mode)

    img = np.array(img)
    print(img.shape, img.dtype, np.unique(img), img.min(), img.max())


if __name__ == "__main__":
    read_sam_mask()
    exit()
    pickle_path = "data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_train.pkl"
    sample_camera_images(pickle_path, save_root="./debug")
    exit(0)
    load_pickle(pickle_path)

