
# config="configs/bevdet_occ/bevdet-occ-stbase-4d-stereo-512x1408-24e.py"
config="configs/bevdet_occ/bevdet-occ-r50-4d-stereo-24e.py"

### Train the Occupancy based on BEVFusion version
config="configs/bevdet_fusion_occ/bevdet-fusion-occ-r50-4d-stereo-24e.py"

### Train the lidar-based Occupancy prediction
config="configs/bevdet_occ/bevdet-lidar-occ-voxel-24e.py"

### Train the lidar-based Occupancy prediction with multiple sweeps
config="configs/bevdet_occ/bevdet-lidar-occ-voxel-multi-sweeps-24e.py"

config="configs/bevdet_occ/bevdet-lidar-occ-voxel-multi-sweeps-lidar-distill-camera-24e.py"

num_gpu=4

set -x
bash ./tools/dist_train.sh ${config} ${num_gpu} --work-dir bevdet-lidar-occ-voxel-multi-sweeps-lidar-distill-camera-24e-fix-BN