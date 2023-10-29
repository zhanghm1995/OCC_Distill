'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-06-14 19:22:14
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''


import torch
from copy import deepcopy

def compare_state_dict(state_dict1, state_dict2):
    for key in state_dict1.keys():
        if "teacher_model." not in key:
            continue

        key2 = key.replace("teacher_model.", "")
        # key2 = key

        if key2 not in state_dict2.keys():
            print(f"key {key} not in state_dict2")
        else:
            if state_dict1[key].shape != state_dict2[key2].shape:
                print(f"key {key} shape not equal")
            else:
                if not torch.allclose(state_dict1[key], state_dict2[key2], 1e-3):
                    print(f"key {key} not equal")


filename = "bevdet-lidar-occ-voxel-multi-sweeps-24e_teacher_model.pth"
checkpoint = torch.load(filename, map_location="cpu")
print(checkpoint.keys())

filename = "work_dirs/bevdet-lidar-occ-voxel-multi-sweeps-24e/epoch_24_ema.pth"
checkpoint2 = torch.load(filename, map_location="cpu")
print(checkpoint2.keys())

compare_state_dict(checkpoint['state_dict'], checkpoint2['state_dict'])
exit(0)

def modify_state_dict(state_dict1, state_dict2):
    ## append the teacher model string to the pre-trained checkpoint dict
    new_state_dict2 = deepcopy(state_dict2)
    for key in list(state_dict2.keys()):
        new_key = "teacher_model." + key
        new_state_dict2[new_key] = new_state_dict2.pop(key)

    ## modify the state_dict1
    keys = ['teacher_model.final_conv.conv.weight',
            'teacher_model.final_conv.conv.bias']
    for key in keys:
        state_dict1[key] = new_state_dict2[key]

modify_state_dict(checkpoint['state_dict'], checkpoint2['state_dict'])
torch.save(checkpoint, "lidar_distill_camera_final_conv_wo_ema.pth")
exit(0)

## append the teacher model string to the checkpoint dict
new_checkpoint = deepcopy(checkpoint2)
for key in list(checkpoint2['state_dict'].keys()):
    new_key = "teacher_model." + key
    new_checkpoint['state_dict'][new_key] = new_checkpoint['state_dict'].pop(key)

print(new_checkpoint.keys())

state_dict = checkpoint['state_dict']
for key in list(state_dict.keys()):
    if 'teacher_model' in key:
        checkpoint['state_dict'][key] = new_checkpoint['state_dict'][key]
    
torch.save(checkpoint, "lidar_distill_camera.pth")