CUDA=11.3
PYTHON_VERSION=3.8
TORCH_VERSION=1.10.0
TORCHVISION_VERSION=0.11.0
ONNXRUNTIME_VERSION=1.8.1
MMCV_VERSION=1.5.3
PPLCV_VERSION=0.7.0

# conda install pytorch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} cudatoolkit=${CUDA} -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64/

# pip install mmcv-full==${MMCV_VERSION} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${TORCH_VERSION}/index.html  -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install pycuda \
    lyft_dataset_sdk \
    networkx==2.2 \
    numba==0.53.0 \
    numpy \
    nuscenes-devkit \
    plyfile \
    scikit-image \
    tensorboard \
    trimesh==2.35.39 -i https://pypi.tuna.tsinghua.edu.cn/simple