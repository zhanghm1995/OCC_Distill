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
import matplotlib.pyplot as plt
from copy import deepcopy
import cv2


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


def get_real_sample_points(sample_pts_pad1, sample_pts_pad2, target_size=(900, 1600)):
    # we generate these sample points in the 900x1600 image, in this case,
    origin_h, origin_w = 900, 1600
    height, width = target_size
    scale_h, scale_w = height / origin_h, width / origin_w

    assert sample_pts_pad1.shape[0] == sample_pts_pad1.shape[0]
    num_cam = sample_pts_pad1.shape[0]

    all_cam_sample_pts1, all_cam_sample_pts2 = [], []
    for i in range(num_cam):
        num_valid_pts = int(sample_pts_pad1[i, -1, 0])
        sample_pts1 = sample_pts_pad1[i, :num_valid_pts]
        sample_pts2 = sample_pts_pad2[i, :num_valid_pts]

        sample_pts1 *= np.array([scale_w, scale_h])
        sample_pts2 *= np.array([scale_w, scale_h])
        
        sample_pts1 = sample_pts1.astype(np.int32)
        sample_pts2 = sample_pts2.astype(np.int32)

        all_cam_sample_pts1.append(sample_pts1)
        all_cam_sample_pts2.append(sample_pts2)
    return all_cam_sample_pts1, all_cam_sample_pts2


def load_images(cam_infos, choose_cam_names, img_size):
    cam_imgs = []
    for cam_name in choose_cam_names:
        cam_data = cam_infos[cam_name]
        filename = cam_data['data_path']
        cam_img = cv2.imread(filename)
        cam_img_resized = cv2.resize(cam_img, img_size)
        cam_imgs.append(cam_img_resized)
    return cam_imgs


def combine_multi_cam_images(cam_imgs, cam_img_size):
    """_summary_

    Args:
        cam_imgs (List): each image is PIL.Image
        cam_img_size (_type_): (w, h)

    Returns:
        _type_: _description_
    """
    spacing = 10
    cam_w, cam_h = cam_img_size
    result_w = cam_w * 3 + 2 * spacing
    result_h = cam_h * 2 + 1 * spacing
    result = Image.new(cam_imgs[0].mode, (result_w, result_h), (0, 0, 0))

    result.paste(cam_imgs[1], box=(1*cam_w+1*spacing, 0))
    result.paste(cam_imgs[2], box=(2*cam_w+2*spacing, 0))
    result.paste(cam_imgs[0], box=(0, 0))
    result.paste(cam_imgs[4], box=(1*cam_w+1*spacing, 1*cam_h+1*spacing))
    result.paste(cam_imgs[3], box=(0, 1*cam_h+1*spacing))
    result.paste(cam_imgs[5], box=(2*cam_w+2*spacing, 1*cam_h+1*spacing))
    return result


def save_point_flow_cat_images(save_root):
    anno_file = "data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_val.pkl"

    data_root = "./data/nuscenes/nuscenes_scene_sequence_npz"

    with open(anno_file, "rb") as fp:
        dataset = pickle.load(fp)
    
    data_infos = dataset['infos']

    ## sort the data infos according to the timestamp to keep the same with BEVDet dataloader.
    data_infos = list(sorted(data_infos, key=lambda e: e['timestamp']))
    
    cam_names = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ]

    cam_img_size = [480, 270]  # [w, h]
    for idx, info in tqdm(enumerate(data_infos), total=len(data_infos)):
        cam_infos = info['cams']

        cam_imgs1 = load_images(cam_infos, cam_names, cam_img_size)

        scene_token = info['scene_token']
        sample_token = info['token']

        # get the next frame
        next_idx = min(idx + 1, len(data_infos) - 1)
        next_info = data_infos[next_idx]
        next_scene_token = next_info['scene_token']
        next_sample_token = next_info['token']

        cam_imgs2 = load_images(next_info['cams'], cam_names, cam_img_size)
        
        if scene_token != next_scene_token:
            vis_cam_imgs1 = deepcopy(cam_imgs1)
            vis_cam_imgs2 = deepcopy(cam_imgs2)
        else:
            scene_flow_file = osp.join(data_root,
                                       scene_token, 
                                       f"{sample_token}_{next_sample_token}.npz")
            scene_flow_dict = np.load(scene_flow_file)
            # (n_cam, n_points, 2)
            sample_pts_pad1 = scene_flow_dict['coord1']
            sample_pts_pad2 = scene_flow_dict['coord2']

            all_cams_sample_pts1, all_cams_sample_pts2 = get_real_sample_points(
                sample_pts_pad1, sample_pts_pad2, target_size=(cam_img_size[1], cam_img_size[0]))

            ## Draw points on the image
            vis_cam_imgs1 = []
            vis_cam_imgs2 = []
            for i in range(len(cam_names)):
                img1 = cam_imgs1[i]
                img2 = cam_imgs2[i]

                samplet_pts1 = all_cams_sample_pts1[i]
                samplet_pts2 = all_cams_sample_pts2[i]
                center = np.median(samplet_pts1, axis=0)
                set_max = range(128)
                colors = {m: i for i, m in enumerate(set_max)}
                colors = {m: (255 * np.array(plt.cm.hsv(i/float(len(colors))))[:3][::-1]).astype(np.int32)
                            for m, i in colors.items()}
                for k in range(samplet_pts1.shape[0]):
                    pt1 = (int(samplet_pts1[k, 0]), int(samplet_pts1[k, 1]))
                    pt2 = (int(samplet_pts2[k, 0]), int(samplet_pts2[k, 1]))

                    coord_angle = np.arctan2(pt1[1] - center[1], pt1[0] - center[0])
                    corr_color = np.int32(64 * coord_angle / np.pi) % 128
                    color = tuple(colors[corr_color].tolist())
                    
                    cv2.circle(img1, pt1, 1, color, -1, cv2.LINE_AA)
                    cv2.circle(img2, pt2, 1, color, -1, cv2.LINE_AA)
                
                img1 = Image.fromarray(img1)
                img2 = Image.fromarray(img2)
                vis_cam_imgs1.append(img1)
                vis_cam_imgs2.append(img2)

        ## start saving
        result1 = combine_multi_cam_images(vis_cam_imgs1, cam_img_size)
        result2 = combine_multi_cam_images(vis_cam_imgs2, cam_img_size)

        if save_root is not None:
            result1 = np.array(result1)
            result2 = np.array(result2)

            blank_bar = np.ones((10, result1.shape[1], 3), dtype=np.uint8) * 255
            result = np.concatenate([result1, blank_bar, result2], axis=0).astype(np.uint8)

            # create the camera image directory
            os.makedirs(save_root, exist_ok=True)
            cam_img_path = osp.join(save_root, f"{idx:06d}.jpg")
            cv2.imwrite(cam_img_path, result)


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
    save_point_flow_cat_images(save_root="./results/cvpr_flow_points_cat")
    exit(0)

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

