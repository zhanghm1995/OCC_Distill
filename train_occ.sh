
# config="configs/bevdet_occ/bevdet-occ-stbase-4d-stereo-512x1408-24e.py"
config="configs/bevdet_occ/bevdet-occ-r50-4d-stereo-24e.py"

### Train the Occupancy based on BEVFusion version
config="configs/bevdet_fusion_occ/bevdet-fusion-occ-r50-4d-stereo-24e.py"

### Train the lidar-based Occupancy prediction
config="configs/bevdet_occ/bevdet-lidar-occ-voxel-24e.py"

### Train the lidar-based Occupancy prediction with multiple sweeps
config="configs/bevdet_occ/bevdet-lidar-occ-voxel-multi-sweeps-24e.py"

config="configs/bevdet_occ/bevdet-lidar-occ-voxel-multi-sweeps-lidar-distill-camera-24e.py"

#### Distillation Experiments ########
config="configs/bevdet_occ/bevdet-occ-voxel-multi-sweeps-lidar-distill-camera-use-mask-24e.py"

config="configs/bevdet_occ/bevdet-lidar-occ-voxel-multi-sweeps-lidar-distill-camera-24e.py"

config="configs/bevdet_occ/occ-distill-ms-l2c-use-mask-both-pretrained-24e.py"

config="work_dirs/bevdet-occ-voxel-multi-sweeps-lidar-distill-camera-use-mask-24e_fix_bug/bevdet-occ-voxel-multi-sweeps-lidar-distill-camera-use-mask-24e.py"


### For debugging ###
config="configs/bevdet_occ/bevdet-lidar-occ-voxel-multi-sweeps-lidar-distill-camera-24e_debug.py"

num_gpu=4
set -x
bash ./tools/dist_train.sh ${config} ${num_gpu} --work-dir work_dirs/debug-distill

# bash ./tools/dist_train.sh ${config} ${num_gpu} --resume-from work_dirs/bevdet-occ-voxel-multi-sweeps-lidar-distill-camera-use-mask-24e_fix_bug/epoch_6.pth

config="configs/bevdet_occ/bevdet-lidar-occ-voxel-24e.py"
bash ./tools/dist_train.sh ${config} ${num_gpu} --work-dir work_dirs/debug-lidar
