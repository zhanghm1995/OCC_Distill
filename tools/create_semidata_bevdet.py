'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-07-28 10:29:00
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import pickle

# bevdetv3-lidarseg-nuscenes
for set in ['train', 'val']:
    dataset = pickle.load(
        open('./data/nuscenes/bevdetv2-nuscenes_infos_%s.pkl' % set, 'rb'))

    semi_dataset = {}
    semi_dataset['metadata'] = dataset['metadata']
    print("original length:", len(dataset['infos']))
    
    infos = []
    for info in dataset['infos']:
        if int(info['occ_path'].split('/')[4][-4:]) < 100:
            infos.append(info)
    
    print(set, len(infos))
    semi_dataset['infos'] = infos

    # with open('./data/nuscenes/bevdetv2-nuscenes-semi_infos_%s.pkl' % set,
    #           'wb') as fid:
    #     pickle.dump(semi_dataset, fid)