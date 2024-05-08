'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-04-16 19:34:13
Email: haimingzhang@link.cuhk.edu.cn
Description: Collect the mini split used by ViDAR. https://github.com/OpenDriveLab/ViDAR/issues/20
'''

import os
import pickle


def save_infos(files, split):
    infos = []
    for file in files:
        with open(file, 'rb') as f:
            infos.extend(pickle.load(f))

    with open(f'data/openscene-v1.1/openscene_{split}.pkl', 'wb') as f:
        pickle.dump(infos, f)


if __name__ == "__main__":
    split = 'mini'
    
    # train split
    train_split_file = f'data/{split}_train.txt'

    with open(train_split_file, 'r') as f:
        train_paths = [line.rstrip() for line in f.readlines()]

    train_paths = [
        os.path.join(f'data/openscene-v1.1/meta_datas/{split}', f'{each}.pkl')
        for each in train_paths]
    print('train:', len(train_paths))
    save_infos(train_paths, f'{split}_train')
    
    # validation split
    val_split_file = f'data/{split}_val.txt'
    with open(val_split_file, 'r') as f:
        val_paths = [line.rstrip() for line in f.readlines()]
    
    val_paths = [
        os.path.join(f'data/openscene-v1.1/meta_datas/{split}', f'{each}.pkl')
        for each in val_paths]
    print('val:', len(val_paths))
    save_infos(val_paths, f'{split}_val')
    print('Done!')