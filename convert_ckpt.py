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


def append_prefix(state_dict, prefix=None):
    """Append the prefix in the front of the keys of the state_dict

    Args:
        state_dict (OrderedDict): the state dictionary
        prefix (str, optional): _description_. Defaults to "teacher_model.".

    Returns:
        OrderedDcit: the new state dictionary
    """
    if prefix is None:
        return deepcopy(state_dict)
    
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


def remove_state_dict_keys(state_dict, keys):
    if not keys:
        return state_dict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        flag = False
        for kk in keys:
            if k.startswith(kk):
                flag = True
                break
        
        if not flag:
            new_state_dict[k] = v
    return new_state_dict


def main_merge_student_teacher_models(stu_filename,
                                      tea_filename,
                                      save_path,
                                      stu_prefix=None,
                                      tea_prefix=None):
    ## Load the student model
    student_pretrained_checkpoint = torch.load(stu_filename, map_location="cpu")
    print(student_pretrained_checkpoint.keys())

    student_state_dict = append_prefix(student_pretrained_checkpoint['state_dict'], 
                                       prefix=stu_prefix)
    
    ## Load the teacher model
    checkpoint = torch.load(tea_filename, map_location="cpu")
    print(checkpoint.keys())
    teacher_state_dict = append_prefix(checkpoint['state_dict'],
                                       prefix=tea_prefix)
    
    ## Merge the student and teacher model
    new_state_dict = merge_state_dict([teacher_state_dict, student_state_dict])
    checkpoint['state_dict'] = new_state_dict
    torch.save(checkpoint, save_path)


def main_append_model_prefix(filename, prefix, save_path):
    """Add the prefix for the pre-trained models. Update the `state_dict`.
    """
    checkpoint = torch.load(filename, map_location="cpu")

    new_state_dict = append_prefix(checkpoint['state_dict'], prefix)
    checkpoint['state_dict'] = new_state_dict
    torch.save(checkpoint, save_path)


if __name__ == "__main__":
    ckpt_path = "bevdet-lidar-occ-voxel-multi-sweeps-24e_teacher_model_new.pth"
    teacher_checkpoint_ = torch.load(ckpt_path, map_location="cpu")

    ckpt_path = "bevdet-r50-4d-stereo-cbgs-as-student-model.pth"
    student_checkpoint_ = torch.load(ckpt_path, map_location="cpu")
    
    new_state_dict = merge_state_dict([teacher_checkpoint_['state_dict'], 
                                       student_checkpoint_['state_dict']])
    teacher_checkpoint_['state_dict'] = new_state_dict

    save_path = "bevdet-r50-as-student_lidar-occ-voxel-ms-new-as-teacher-model.pth"
    torch.save(teacher_checkpoint_, save_path)
    exit(0)
    ckpt_path = "bevdet-lidar-occ-voxel-multi-sweeps-24e_teacher_model.pth"
    checkpoint_ = torch.load(ckpt_path, map_location="cpu")

    keys_mapping = {
        'teacher_model.pts_bev_encoder_backbone': 'teacher_model.pts_backbone',
        'teacher_model.pts_bev_encoder_neck': 'teacher_model.pts_neck'
    }
    new_state_dict = update_state_dict_keys(
        checkpoint_['state_dict'],
        keys_mapping=keys_mapping)
    checkpoint_['state_dict'] = new_state_dict
    torch.save(checkpoint_, "bevdet-lidar-occ-voxel-multi-sweeps-24e_teacher_model_new.pth")
    exit(0)

    file_path = "work_dirs/pretrain-bevdet-occ-r50-4d-stereo-24e/epoch_24.pth"
    pretraining_model = torch.load(file_path, map_location="cpu")
    keys_remove = ['final_conv', 'predicter']
    new_state_dict = remove_state_dict_keys(pretraining_model['state_dict'], keys_remove)
    pretraining_model['state_dict'] = new_state_dict
    torch.save(pretraining_model, "pretrain-bevdet-occ-r50-4d-stereo-wo-head-24e.pth")

    exit(0)
    student_filename = "bevdet-r50-4d-stereo-cbgs-as-student-model.pth"
    teacher_filename = "bevdet-lidar-occ-voxel-ms-w-aug-24e_teacher_model-quarter.pth"
    save_path = "bevdet-r50-4d-stereo-as-student-lidar-voxel-w-aug-as-teacher-quarter.pth"
    main_merge_student_teacher_models(student_filename, teacher_filename,
                                      save_path=save_path)
    exit(0)
    filename = "quarter-bevdet-occ-r50-4d-stereo-24e.pth"
    prefix = "student_model."
    save_path = "bevdet-occ-r50-4d-stereo-24e-linear-sched-as-student-model-quarter.pth"
    main_append_model_prefix(filename, prefix, save_path)
    exit(0)

    student_filename = "./quarter-bevdet-occ-r50-4d-stereo-24e.pth"
    teacher_filename = "quarter-lidar-voxel-occ-pretrain-w-aug-24e-as-teacher-model.pth"
    save_path = "./quarter-lidar-voxel-w-aug-as-teacher-bevdet-r50-4d-stereo-as-student.pth"
    main_merge_student_teacher_models(student_filename, teacher_filename,
                                      save_path=save_path,
                                      stu_prefix="student_model.")
    exit(0)

    filename = "./quarter-lidar-voxel-occ-pretrain-w-aug-24e.pth"
    save_path = "./quarter-lidar-voxel-occ-pretrain-w-aug-24e-as-teacher-model.pth"
    main_append_model_prefix(filename, prefix="teacher_model.", save_path=save_path)
    exit(0)
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

    
    
