
# config="configs/bevdet_occ/bevdet-occ-stbase-4d-stereo-512x1408-24e.py"
config="configs/bevdet_occ/bevdet-occ-r50-4d-stereo-24e.py"

num_gpu=8

set -x
bash ./tools/dist_train.sh ${config} ${num_gpu}