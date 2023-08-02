'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-08-01 22:28:59
Email: haimingzhang@link.cuhk.edu.cn
Description: Split the pickle file.
'''

import pickle
import os
import os.path as osp


def split_frames(data_infos, num_splits=6):
    """Split the frames into several parts.

    Args:
        data_infos (list): The data infos.
        num_splits (int, optional): The number of splits. Defaults to 6.
    """
    all_split_data_infos = []
    num_frames = len(data_infos)
    num_frames_per_split = num_frames // num_splits
    for i in range(num_splits):
        start_idx = i * num_frames_per_split
        end_idx = (i + 1) * num_frames_per_split
        if i == num_splits - 1:
            end_idx = num_frames
        print(start_idx, end_idx)
        split_data_infos = data_infos[start_idx:end_idx]
        print(len(split_data_infos))
        all_split_data_infos.append(split_data_infos)
    return all_split_data_infos
        

def split_pickle(pickle_file_path):
    with open(pickle_file_path, "rb") as fp:
        dataset = pickle.load(fp)


    metadata = dataset['metadata']
    data_infos = dataset['infos']

    print(len(data_infos), type(dataset), dataset.keys())

    splited_data_infos = split_frames(data_infos, num_splits=4)
    print(len(splited_data_infos))

    basename = osp.basename(pickle_file_path)
    basename = basename.split(".")[0]
    for i, split_data_infos in enumerate(splited_data_infos):
        save_path = f"data/nuscenes/test_test/{basename}_split{i}.pkl"

        split_data_pickle = {'metadata': metadata, 
                             'infos': split_data_infos}
        with open(save_path, "wb") as fp:
            pickle.dump(split_data_pickle, fp)


if __name__ == "__main__":
    pickle_file_path = "data/nuscenes/test/bevdetv3-nuscenes_infos_test_split3.pkl"
    split_pickle(pickle_file_path)