'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-10-25 22:41:36
Email: haimingzhang@link.cuhk.edu.cn
Description: Create the background optical flow by using the Lidar Point Cloud.
And we also use the lidar semantic segmentation to filter out the moveable objects.
'''

import numpy as np
import os
import os.path as osp
import pickle
import cv2
from tqdm import tqdm
import torch
from pyquaternion import Quaternion
import multiprocessing
import multiprocessing as mp
from multiprocessing import Pool
import itertools


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


def get_lidar2global(results):
    lidar2lidarego = np.eye(4, dtype=np.float32)
    lidar2lidarego[:3, :3] = Quaternion(
        results['lidar2ego_rotation']).rotation_matrix
    lidar2lidarego[:3, 3] = results['lidar2ego_translation']
    lidar2lidarego = torch.from_numpy(lidar2lidarego)

    lidarego2global = np.eye(4, dtype=np.float32)
    lidarego2global[:3, :3] = Quaternion(
        results['ego2global_rotation']).rotation_matrix
    lidarego2global[:3, 3] = results['ego2global_translation']
    lidarego2global = torch.from_numpy(lidarego2global)
    lidar2global = lidarego2global.matmul(lidar2lidarego)
    return lidar2global


def get_lidar2img(results, cam_name):
    lidar2global = get_lidar2global(results)

    cam2camego = np.eye(4, dtype=np.float32)
    cam2camego[:3, :3] = Quaternion(
        results['cams'][cam_name]
        ['sensor2ego_rotation']).rotation_matrix
    cam2camego[:3, 3] = results['cams'][cam_name][
        'sensor2ego_translation']
    cam2camego = torch.from_numpy(cam2camego)

    camego2global = np.eye(4, dtype=np.float32)
    camego2global[:3, :3] = Quaternion(
        results['cams'][cam_name]
        ['ego2global_rotation']).rotation_matrix
    camego2global[:3, 3] = results['cams'][cam_name][
        'ego2global_translation']
    camego2global = torch.from_numpy(camego2global)

    cam2img = np.eye(4, dtype=np.float32)
    cam2img = torch.from_numpy(cam2img)
    cam2img[:3, :3] = torch.tensor(results['cams'][cam_name]['cam_intrinsic'])

    lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
        lidar2global)
    lidar2img = cam2img.matmul(lidar2cam)
    return lidar2img


def get_lidar2img_all_cams(results, cam_names=None):
    if cam_names is None:
        cam_names = results['cams'].keys()

    lidar2img_list = []
    for cam_name in cam_names:
        curr_lidar2img = get_lidar2img(results, cam_name)
        lidar2img_list.append(curr_lidar2img)
    
    lidar2img = torch.stack(lidar2img_list, dim=0)
    return lidar2img


def project_points_to_image(points_lidar, lidar2img, height, width, 
                            depth_range=[1.0, 100.0]):
    points_img = points_lidar[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
    points_img = torch.cat(
        [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
        1)

    depth_map = torch.zeros((height, width), dtype=torch.float32)
    coor = torch.round(points_img[:, :2])
    coord_origin = coor.clone()

    depth = points_img[:, 2]
    kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
             coor[:, 1] >= 0) & (coor[:, 1] < height) & (
             depth < depth_range[1]) & (
             depth >= depth_range[0])
    coor, depth = coor[kept1], depth[kept1]
    ranks = coor[:, 0] + coor[:, 1] * width
    sort = (ranks + depth / 100.).argsort()
    coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

    kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
    kept2[1:] = (ranks[1:] != ranks[:-1])
    coor, depth = coor[kept2], depth[kept2]
    coor = coor.to(torch.long)
    depth_map[coor[:, 1], coor[:, 0]] = depth

    ## obtain the kept points indices in the original point cloud
    point_indices = torch.arange(points_img.shape[0])
    point_indices = point_indices[kept1][sort][kept2]
    point_mask = torch.zeros(points_img.shape[0], dtype=torch.bool)
    point_mask[point_indices] = 1
    return coor, coord_origin, depth_map, point_mask


def find_correspondence_points(points1, lidar2img1, 
                               points2, lidar2img2, 
                               cam_img_size):
    coord1, coord1_origin, _, point_mask1 = project_points_to_image(
        points1, lidar2img1, cam_img_size[0], cam_img_size[1])
    coord2, coord2_origin, _, point_mask2 = project_points_to_image(
        points2, lidar2img2, cam_img_size[0], cam_img_size[1])
    
    # compute the intersection of the two point masks
    point_mask = point_mask1 & point_mask2
    coord1_kept = coord1_origin[point_mask]
    coord2_kept = coord2_origin[point_mask]
    return coord1_kept, coord2_kept


def load_lidarseg(lidarseg_path):
    lidarseg = np.fromfile(lidarseg_path, dtype=np.uint8)
    learning_map = {
        1: 0, 5: 0, 7: 0, 8: 0, 10: 0, 11: 0, 13: 0, 
        19: 0, 20: 0, 0: 0, 29: 0, 31: 0, 9: 1, 14: 2, 15: 3, 16: 3,
        17: 4, 18: 5, 21: 6, 2: 7, 3: 7, 4: 7, 6: 7, 
        12: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
        30: 16
    }
    lidarseg = np.vectorize(learning_map.__getitem__)(lidarseg)
    return lidarseg


def filter_dynamic_points(points, lidarseg_path):
    lidarseg = load_lidarseg(lidarseg_path)

    static_class = [1, 8, 11, 12, 13, 14, 15, 16]
    mask = np.zeros_like(lidarseg, dtype=bool)
    for cls in static_class:
        mask = np.logical_or(lidarseg == cls, mask)
    
    points = points[mask]
    return points


def draw_points(img, coord, radius=3):
    for i in range(coord.shape[0]):
        cv2.circle(img, (int(coord[i, 0]), int(coord[i, 1])), 
                   radius, (0, 255, 0), -1)
    return img


def process_two_frames(frame1, frame2, 
                       choose_cams, 
                       scene_save_dir, i, j):
    sample_token1 = frame1['token']
    sample_token2 = frame2['token']

    # load the point cloud in frame1
    lidar_path = frame1['lidar_path']
    points1 = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
    points1 = filter_dynamic_points(points1, frame1['lidarseg'])
    
    points1 = torch.from_numpy(points1)

    # transform the point cloud in lidar coordinate to global coordinate
    lidar2global_rts1 = get_lidar2global(frame1)
    points1_global = points1.matmul(lidar2global_rts1[:3, :3].T) + \
        lidar2global_rts1[:3, 3].unsqueeze(0)

    # get the global2lidar transformation matrix in frame2
    lidar2global_rts2 = get_lidar2global(frame2)
    global2lidar_rts2 = torch.inverse(lidar2global_rts2)

    # transform the point cloud in global coordinate to lidar coordinate in frame2
    points1_in_frame2 = points1_global.matmul(global2lidar_rts2[:3, :3].T) + \
        global2lidar_rts2[:3, 3].unsqueeze(0)

    ## project the points to the image plane and find the correspondence points
    lidar2img1 = get_lidar2img_all_cams(frame1, choose_cams)
    lidar2img2 = get_lidar2img_all_cams(frame2, choose_cams)

    results_coord1 = []
    results_coord2 = []
    for idx, cam in enumerate(choose_cams):
        cam_img_size = [900, 1600]  # (h, w)
        coord1, coord2 = find_correspondence_points(
            points1, lidar2img1[idx], 
            points1_in_frame2, lidar2img2[idx], 
            cam_img_size)
        results_coord1.append(coord1.numpy())
        results_coord2.append(coord2.numpy())

        VISUALIZE = False
        if VISUALIZE:
            ## visualize the projected points
            img_path = frame1['cams'][cam]['data_path']
            img = cv2.imread(img_path)
            img = draw_points(img, results_coord1[-1])
            save_path = osp.join(scene_save_dir, f"{i}_{j}_{cam}_{i}.png")
            cv2.imwrite(save_path, img)
            
            img_path = frame2['cams'][cam]['data_path']
            img = cv2.imread(img_path)
            img = draw_points(img, results_coord2[-1])
            save_path = osp.join(scene_save_dir, f"{i}_{j}_{cam}_{j}.png")
            cv2.imwrite(save_path, img)

    results = {'coord1': results_coord1, 'coord2': results_coord2}

    save_path = osp.join(scene_save_dir, f"{sample_token1}_{sample_token2}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)


def process_one_scene(scene_name, scene_seq, neighbor_length=3,
                      save_dir=None, choose_cams=None):
    num_frames = len(scene_seq)
    scene_save_dir = osp.join(save_dir, scene_name)
    os.makedirs(scene_save_dir, exist_ok=True)

    # sample_token_list = [frame['token'] for frame in scene_seq]

    for i in range(num_frames - 1):
        for j in range(i + 1, min(i + 1 + neighbor_length, num_frames)):
            frame1 = scene_seq[i]
            frame2 = scene_seq[j]
            process_two_frames(frame1, frame2, 
                               choose_cams, 
                               scene_save_dir, i, j)

    for i in range(num_frames - 1, 0, -1):
        for j in range(i - 1, max(i - 1 - neighbor_length, -1), -1):
            frame1 = scene_seq[i]
            frame2 = scene_seq[j]
            process_two_frames(frame1, frame2, 
                               choose_cams, 
                               scene_save_dir, i, j)


def save_all_scene_sequence_images(anno_file):
    with open(anno_file, "rb") as fp:
        dataset = pickle.load(fp)
    
    data_infos = dataset['infos']
    data_infos = list(sorted(data_infos, key=lambda e: e['timestamp']))

    scene_name_list, total_scene_seq = get_scene_sequence_data(data_infos)

    print(len(total_scene_seq), len(scene_name_list))

    choose_cams = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ]

    neighbor_length = 3
    save_dir = "./data/nuscenes_scene_sequence"
    for idx, (scene_name, scene_seq) in tqdm(enumerate(zip(scene_name_list, total_scene_seq)),
                                             total=len(scene_name_list)):

        num_frames = len(scene_seq)
        
        scene_save_dir = osp.join(save_dir, scene_name)
        os.makedirs(scene_save_dir, exist_ok=True)

        for i in range(num_frames - 1):
            for j in range(i + 1, min(i + 1 + neighbor_length, num_frames)):
                frame1 = scene_seq[i]
                frame2 = scene_seq[j]
                sample_token1 = frame1['token']
                sample_token2 = frame2['token']

                # load the point cloud in frame1
                lidar_path = frame1['lidar_path']
                points1 = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
                points1 = filter_dynamic_points(points1, frame1['lidarseg'])
                
                points1 = torch.from_numpy(points1)

                # transform the point cloud in lidar coordinate to global coordinate
                lidar2global_rts1 = get_lidar2global(frame1)
                points1_global = points1.matmul(lidar2global_rts1[:3, :3].T) + \
                    lidar2global_rts1[:3, 3].unsqueeze(0)

                # get the global2lidar transformation matrix in frame2
                lidar2global_rts2 = get_lidar2global(frame2)
                global2lidar_rts2 = torch.inverse(lidar2global_rts2)

                # transform the point cloud in global coordinate to lidar coordinate in frame2
                points1_in_frame2 = points1_global.matmul(global2lidar_rts2[:3, :3].T) + \
                    global2lidar_rts2[:3, 3].unsqueeze(0)

                ## project the points to the image plane and find the correspondence points
                lidar2img1 = get_lidar2img_all_cams(frame1, choose_cams)
                lidar2img2 = get_lidar2img_all_cams(frame2, choose_cams)

                results_coord1 = []
                results_coord2 = []
                for idx, cam in enumerate(choose_cams):
                    cam_img_size = [900, 1600]  # (h, w)
                    coord1, coord2 = find_correspondence_points(
                        points1, lidar2img1[idx], 
                        points1_in_frame2, lidar2img2[idx], 
                        cam_img_size)
                    results_coord1.append(coord1.numpy())
                    results_coord2.append(coord2.numpy())

                    VISUALIZE = True
                    if VISUALIZE:
                        ## visualize the projected points
                        img_path = frame1['cams'][cam]['data_path']
                        img = cv2.imread(img_path)
                        img = draw_points(img, results_coord1[-1])
                        save_path = osp.join(scene_save_dir, f"{i}_{j}_{cam}_{i}.png")
                        cv2.imwrite(save_path, img)
                        
                        img_path = frame2['cams'][cam]['data_path']
                        img = cv2.imread(img_path)
                        img = draw_points(img, results_coord2[-1])
                        save_path = osp.join(scene_save_dir, f"{i}_{j}_{cam}_{j}.png")
                        cv2.imwrite(save_path, img)

                results = {'coord1': results_coord1, 'coord2': results_coord2}

                save_path = osp.join(scene_save_dir, f"{sample_token1}_{sample_token2}.pkl")
                with open(save_path, 'wb') as f:
                    pickle.dump(results, f)

        for i in range(num_frames - 1, 0, -1):
            for j in range(i - 1, max(i - 1 - neighbor_length, -1), -1):
                frame1 = scene_seq[i]
                frame2 = scene_seq[j]
                process_two_frames(frame1, frame2, 
                                   choose_cams, 
                                   scene_save_dir, i, j)


def main_v2(anno_file):
    with open(anno_file, "rb") as fp:
        dataset = pickle.load(fp)
    
    data_infos = dataset['infos']
    data_infos = list(sorted(data_infos, key=lambda e: e['timestamp']))

    scene_name_list, total_scene_seq = get_scene_sequence_data(data_infos)

    scene_name_list = scene_name_list[250:]
    total_scene_seq = total_scene_seq[250:]
    print(len(total_scene_seq), len(scene_name_list))

    choose_cams = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ]

    neighbor_length = 3
    save_dir = "./data/nuscenes_scene_sequence"
    for idx, (scene_name, scene_seq) in tqdm(enumerate(zip(scene_name_list, total_scene_seq)),
                                             total=len(scene_name_list)):
        process_one_scene(scene_name, scene_seq, neighbor_length, save_dir, choose_cams)


def process_one_scene_adapter(args):
    return process_one_scene(*args)

def main_multi_process(anno_file):
    with open(anno_file, "rb") as fp:
        dataset = pickle.load(fp)
    
    data_infos = dataset['infos']
    data_infos = list(sorted(data_infos, key=lambda e: e['timestamp']))

    scene_name_list, total_scene_seq = get_scene_sequence_data(data_infos)

    print(len(total_scene_seq), len(scene_name_list))

    choose_cams = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ]

    neighbor_length = 3
    save_dir = "./data/nuscenes_scene_sequence"

    ## start processing the data in multi-process
    function_parameters = zip(
        scene_name_list,
        total_scene_seq,
        itertools.repeat(neighbor_length),
        itertools.repeat(save_dir),
        itertools.repeat(choose_cams)
    )
    
    # pool = Pool(processes=32)
    # pool.starmap(process_one_scene, function_parameters)

    jobs = 16
    with multiprocessing.Pool(jobs) as pool:
        args = [entry for entry in function_parameters]
        results = list(tqdm(pool.imap(process_one_scene_adapter, args), total=len(args)))


def main():
    pickle_path = "data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_train.pkl"
    # save_all_scene_sequence_images(pickle_path)
    main_v2(pickle_path)

    # main_multi_process(pickle_path)


if __name__ == "__main__":
    main()