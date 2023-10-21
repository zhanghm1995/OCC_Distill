'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-07-26 09:57:35
Email: haimingzhang@link.cuhk.edu.cn
Description: Save the data offline from the .pkl dataset files.
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
                         need_sort_data=True,
                         need_camera_directory=True,
                         save_mask_img=False):
    """Sample the camera images according to the sample_idx
    from the dataset and save them to the specified directory.

    Args:
        anno_file (_type_): The .pkl file path.
        save_root (str, optional): The save root. Defaults to None.
        need_camera_directory (bool): Whether save the images in the CAM_FRONT, CAM_BACK,
            etc. directories.
        need_sort_data (bool, optional): Whether to sort the data infos according
            to the timestamp. When training in the BEVDet, it sorts the data by
            default. Defaults to True.
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
        
        if save_root is not None:
            # create the camera image directory
            cam_dir = osp.join(save_root, cam_name) if need_camera_directory \
                else save_root
            # os.makedirs(save_root, exist_ok=True)
            # cam_dir = osp.join(save_root, cam_name)
            os.makedirs(cam_dir, exist_ok=True)

            # copy the camera image to the directory
            dst_file_name = osp.splitext(osp.basename(filename))[0]
            cam_img_path = osp.join(cam_dir, f"{dst_file_name}_image_{sample_idx}.jpg")
            os.system(f"cp {filename} {cam_img_path}")

            # copy the semantic mask to the directory
            if save_mask_img:
                sam_mask_path = osp.join("./data/nuscenes/superpixels_sam", 
                                         f"{cam_token}.png")
                sam_mask_dst_path = osp.join(cam_dir, f"{dst_file_name}_mask.png")
                os.system(f"cp {sam_mask_path} {sam_mask_dst_path}")


def sample_camera_images_from_token(anno_file, 
                                    sample_token,
                                    save_root=None,
                                    need_sort_data=True,
                                    save_mask_img=False):
    """Sample the camera images according to the sample_idx
    from the dataset and save them to the directory.

    Args:
        anno_file (_type_): The .pkl file path.
        save_root (str, optional): The save root. Defaults to None.
        need_sort_data (bool, optional): Whether to sort the data infos according
            to the timestamp. When training in the BEVDet, it sorts the data by
            default. Defaults to True.
    """
    with open(anno_file, "rb") as fp:
        dataset = pickle.load(fp)
    
    data_infos = dataset['infos']

    if need_sort_data:
        data_infos = list(sorted(data_infos, key=lambda e: e['timestamp']))
    
    for info in data_infos:
        if info['token'] == sample_token:
            break
    print(info['token'])

    cam_names = info['cams']
    for cam_name in cam_names:
        cam_data = cam_names[cam_name]
        filename = cam_data['data_path']
        cam_token = cam_data['sample_data_token']
        # print(filename, cam_token)
        
        if save_root is not None:
            # create the camera image directory
            os.makedirs(save_root, exist_ok=True)
            cam_dir = osp.join(save_root, cam_name)
            os.makedirs(cam_dir, exist_ok=True)

            # copy the camera image to the directory
            dst_file_name = osp.splitext(osp.basename(filename))[0]
            cam_img_path = osp.join(cam_dir, f"{dst_file_name}_image.jpg")
            os.system(f"cp {filename} {cam_img_path}")

            # copy the semantic mask to the directory
            if save_mask_img:
                sam_mask_path = osp.join("./data/nuscenes/superpixels_sam", 
                                         f"{cam_token}.png")
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

            if save_root is not None:
                # create the camera image directory
                os.makedirs(save_root, exist_ok=True)
                cam_dir = osp.join(save_root, cam_name)
                os.makedirs(cam_dir, exist_ok=True)

                # copy the camera image to the directory
                dst_file_name = osp.splitext(osp.basename(filename))[0]
                cam_img_path = osp.join(cam_dir, f"{idx}_{dst_file_name}.jpg")
                os.system(f"cp {filename} {cam_img_path}")


def save_all_cam_cat_images(anno_file, save_root=None):
    """Concatenate all the camera images into one image.

    Args:
        anno_file (_type_): _description_
        save_root (_type_, optional): _description_. Defaults to None.
    """
    with open(anno_file, "rb") as fp:
        dataset = pickle.load(fp)
    
    data_infos = dataset['infos']

    ## sort the data infos according to the timestamp to keep the same with BEVDet dataloader.
    data_infos = list(sorted(data_infos, key=lambda e: e['timestamp']))
    
    cam_img_size = [480, 270]  # [w, h]
    for idx, info in tqdm(enumerate(data_infos), total=len(data_infos)):
        cam_names = info['cams']

        cam_imgs = []
        for cam_name in cam_names:
            cam_data = cam_names[cam_name]
            filename = cam_data['data_path']
            cam_token = cam_data['sample_data_token']

            cam_img_resized = Image.open(filename).resize(
                cam_img_size, Image.BILINEAR)
            cam_imgs.append(cam_img_resized)
        
        spacing = 10
        cam_w, cam_h = cam_img_size
        result_w = cam_w * 3 + 2 * spacing
        result_h = cam_h * 2 + 1 * spacing
        result = Image.new(cam_imgs[0].mode, (result_w, result_h), (0, 0, 0))

        result.paste(cam_imgs[0], box=(1*cam_w+1*spacing, 0))
        result.paste(cam_imgs[1], box=(2*cam_w+2*spacing, 0))
        result.paste(cam_imgs[2], box=(0, 0))
        result.paste(cam_imgs[3], box=(1*cam_w+1*spacing, 1*cam_h+1*spacing))
        result.paste(cam_imgs[4], box=(0, 1*cam_h+1*spacing))
        result.paste(cam_imgs[5], box=(2*cam_w+2*spacing, 1*cam_h+1*spacing))

        if save_root is not None:
            # create the camera image directory
            os.makedirs(save_root, exist_ok=True)

            cam_img_path = osp.join(save_root, f"{idx:06d}.jpg")
            result.save(cam_img_path)
        


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


def save_scene_sequence_image(anno_file):
    with open(anno_file, "rb") as fp:
        dataset = pickle.load(fp)
    
    data_infos = dataset['infos']
    curr_info = data_infos[100]

    data_infos = list(sorted(data_infos, key=lambda e: e['timestamp']))
    print('Done')

    total_scene_seq = []
    curr_seq = []
    for idx, data in tqdm(enumerate(data_infos)):
        scene_token = data['scene_token']
        pre_idx = max(idx - 1, 0)
        pre_scene_token = data_infos[pre_idx]['scene_token']

        if scene_token != pre_scene_token:
            total_scene_seq.append(curr_seq)
            curr_seq = [data]
        else:
            curr_seq.append(data)

        if idx == len(data_infos) - 1:
            total_scene_seq.append(curr_seq)

    cam_img_size = [800, 450]  # [w, h]
    # save the first scene sequence
    curr_seq = total_scene_seq[0]
    for frame in curr_seq:
        cam_names = frame['cams']
        for cam_name in cam_names:
            cam_data = cam_names[cam_name]
            filename = cam_data['data_path']
            cam_token = cam_data['sample_data_token']

            # create the camera image directory
            cam_dir = osp.join("./results/visualization/valiation_scene_seq", 
                               cam_name, 'color')
            os.makedirs(cam_dir, exist_ok=True)

            # copy the camera image to the directory
            cam_img_resized = Image.open(filename).resize(
                cam_img_size, Image.BILINEAR)
            
            dst_file_name = osp.splitext(osp.basename(filename))[0]
            cam_img_path = osp.join(cam_dir, f"{dst_file_name}_image.jpg")
            cam_img_resized.save(cam_img_path)



if __name__ == "__main__":
    pickle_path = "data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_val.pkl"
    save_scene_sequence_image(pickle_path)
    exit(0)

    save_all_cam_cat_images(pickle_path, save_root="./aaai_all_validation_cat_debug")
    exit(0)

    sample_camera_images(pickle_path, 
                         sample_idx=4165, 
                         need_camera_directory=False,
                         save_root="./AAAI_visualization/4165")
    exit()

    pickle_path = "data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_train.pkl"
    sample_camera_images_from_token(pickle_path,
                                    '5abb17189643497f8600a187fb20aa55',
                                     save_root="./AAAI_visualization/sample")
    exit()

    save_all_cam_images(pickle_path, save_root="./aaai_all_validation")
    exit(0)
    save_point_cloud(pickle_path, sample_idx=3827)
    exit()
    
    
    
    
    read_sam_mask()
    exit()
    
    exit(0)
    load_pickle(pickle_path)

