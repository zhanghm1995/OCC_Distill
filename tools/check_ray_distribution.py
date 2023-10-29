import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import os.path as osp
import numpy as np

gt_ray_weight_path = "ray_weight_probs_frame25_gt.npy"
# gt_ray_weight_path = "ray_weight_probs_frame25_teacher.npy"
# gt_ray_weight_path = "ray_weight_probs_frame25_student.npy"
gt_ray_weight = np.load(gt_ray_weight_path)

print(gt_ray_weight.shape)

num_camera = 6
render_h = 225
render_w = 400

gt_ray_weight_map = gt_ray_weight.reshape((num_camera, render_h, render_w, -1))
print(gt_ray_weight_map.shape)

u = 140
v = 133
u = 208
v = 135

u = 72
v = 154

u = 138
v = 118

weight = gt_ray_weight_map[0, v, u]
print(weight.shape)

print(weight, weight.min(), weight.max())


x = np.arange(len(weight))

plt.plot(x, weight)
# plt.savefig("position.svg", format='svg')
plt.savefig("position_gt_138.png")