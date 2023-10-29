'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-08-16 11:16:24
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import os
import os.path as osp
import sys
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt

from mmdet3d.models.decode_heads.nerf_head import visualize_depth


def main():
    teacher_depth_path = "AAAI_visualization/teacher-4250-depth/01_rendered_depth.npy"
    student_depth_path = "AAAI_visualization/student-4250-depth/01_rendered_depth.npy"
    student_depth_path = "AAAI_visualization/teacher-4250-12e-depth/01_rendered_depth.npy"

    teacher_depth = np.load(teacher_depth_path)
    student_depth = np.load(student_depth_path)

    disparity_map = np.abs(teacher_depth - student_depth)

    plt.axis('off')
    plt.imshow(disparity_map, cmap="plasma")
    plt.colorbar(fraction=0.05, pad=0.05)
    plt.show()
    # plt.savefig("disparity_map-3.png")
    # disparity_map_img = visualize_depth(disparity_map)
    # disparity_map_img = disparity_map_img[..., [2, 1, 0]]

    # disparity_map = disparity_map / np.max(disparity_map)
    # disparity_map = disparity_map * 255
    # cv2.imwrite("disparity_map-2.jpg", disparity_map)

    # disparity_map_img = disparity_map_img.astype(np.uint8)

    # depth_map = Image.fromarray(disparity_map)
    # depth_map.save("disparity_map.png")


if __name__ == "__main__":
    main()