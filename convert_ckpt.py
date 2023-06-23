'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-06-19 14:45:00
Email: haimingzhang@link.cuhk.edu.cn
Description: Convert the checkpoint information from the original checkpoint to the new checkpoint.
'''

import torch
from copy import deepcopy
from collections import OrderedDict


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


def append_prefix(state_dict, prefix="teacher_model."):
    """Append the prefix in the front of the keys of the state_dict

    Args:
        state_dict (OrderedDict): the state dictionary
        prefix (str, optional): _description_. Defaults to "teacher_model.".

    Returns:
        OrderedDcit: the new state dictionary
    """
    if not prefix.endswith('.'):
        prefix += '.'  # append '.' to the end of prefix if not exist
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = prefix + k
        new_state_dict[name] = v
    return new_state_dict


def merge_state_dict(state_dict_list: list):
    new_state_dict = OrderedDict()
    for state_dict in state_dict_list:
        for k, v in state_dict.items():
            new_state_dict[k] = v
    return new_state_dict


def update_state_dict_keys(state_dict,
                           keys_mapping=None):
    
    if keys_mapping is None:
        return state_dict
    
    assert isinstance(keys_mapping, dict)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        flag = False
        for kk, vv in keys_mapping.items():
            if k.startswith(kk):
                new_key = k.replace(kk, vv)
                flag = True
            elif k == kk:
                new_key = vv
                flag = True
            if flag:
                break
        
        if flag:
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    return new_state_dict 


def main_merge_student_teacher_models():
    ## Load the teacher model
    filename = "work_dirs/bevdet-lidar-occ-voxel-multi-sweeps-24e/epoch_24_ema.pth"
    checkpoint = torch.load(filename, map_location="cpu")
    print(checkpoint.keys())
    teacher_state_dict = append_prefix(checkpoint['state_dict'])

    ## Load the student model
    student_pretrained_filename = "work_dirs/bevdet-occ-r50-4d-stereo-24e/epoch_24_ema.pth"
    student_pretrained_checkpoint = torch.load(student_pretrained_filename, map_location="cpu")
    print(student_pretrained_checkpoint.keys())

    student_state_dict = append_prefix(student_pretrained_checkpoint['state_dict'], 
                                       prefix="student_model.")
    
    ## Merge the student and teacher model
    new_state_dict = merge_state_dict([teacher_state_dict, student_state_dict])
    checkpoint['state_dict'] = new_state_dict
    torch.save(checkpoint, "bevdet-occ-r50-4d-stereo-24e-student_lidar-occ-teacher-model.pth")


def main_append_student_model():
    """Add the `student_model.` as prefix for the pre-trained models.
    """
    filename = "bevdet-r50-4d-stereo-cbgs.pth"
    checkpoint = torch.load(filename, map_location="cpu")

    new_state_dict = append_prefix(checkpoint['state_dict'], 
                                   prefix="student_model.")
    checkpoint['state_dict'] = new_state_dict
    torch.save(checkpoint, "bevdet-r50-4d-stereo-cbgs-as-student-model.pth")


if __name__ == "__main__":
    student_pretrained_filename = "work_dirs/bevdet-occ-r50-4d-stereo-24e/epoch_24_ema.pth"
    student_pretrained_checkpoint = torch.load(student_pretrained_filename, map_location="cpu")

    new_state_dict = update_state_dict_keys(
        student_pretrained_checkpoint['state_dict'],
        keys_mapping=dict(final_conv='final_conv_camera',
                          predicter='predicter_camera'))

    exit(0)
    main_append_student_model()
    exit(0)

    main_merge_student_teacher_models()

    exit(0)
    
    torch.cuda.empty_cache()

    ## Load the teacher model
    filename = "work_dirs/bevdet-lidar-occ-voxel-multi-sweeps-24e/epoch_24_ema.pth"
    checkpoint = torch.load(filename, map_location="cpu")
    print(checkpoint.keys())

    new_state_dict = append_prefix(checkpoint['state_dict'])
    checkpoint['state_dict'] = new_state_dict
    torch.save(checkpoint, "bevdet-lidar-occ-voxel-multi-sweeps-24e_teacher_model.pth")

    
    
