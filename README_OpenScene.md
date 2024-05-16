# BEVDet OpenScene
## Installation
```bash
conda create -n bevdet_py38 python=3.8 -y

conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c https://mirrors.sustech.edu.cn/anaconda/cloud/pytorch/linux-64/
or
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html


pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install timm lyft_dataset_sdk networkx==2.2 numba==0.53.0 numpy==1.23.4 nuscenes-devkit plyfile scikit-image tensorboard trimesh==2.35.39 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install torch_efficient_distloss einops

pip install ninja mmdet==2.25.1 mmsegmentation==0.25.0 
pip install setuptools==59.5.0

pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0-cu113.html

pip install yapf==0.40.1
```
Then we need to compile this project.
```bash
cd OCC_Distill
pip install -v -e .
```

## Datasets
### Preprocessing
Generate the train and validation split pickle file for mini dataset.
```bash
python tools/openscene/collect_vidar_mini_split.py
```
Update the origin pickle files to adapt the BEVDet. Including remove some data entries without occupancy path and use absolute file path for the camera path.

Besides, we generate a 1/4 partial dataset for acceleration for conviniently debugging.
```bash
python tools/openscene/update_pickle.py
```

## Get Started
### Train
```bash
bash tools/dist_train.sh 
```