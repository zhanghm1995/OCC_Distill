'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-09-14 17:29:46
Email: haimingzhang@link.cuhk.edu.cn
Description: Evaluate the offline save results for nuScenes occupancy predictions
task.
'''

import argparse
import os
import warnings

import mmcv
import torch
import numpy as np
from mmcv import Config, DictAction

import mmdet
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor

if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('data_dir', help='the offline results directory path')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval-binary',
        action='store_true',
        help='whether to evaluate the binary occupancy prediction,'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument(
        '--eval-occ3d',
        action='store_true',
        help='whether to evaluate the occupancy prediction saved in occ3d folder structure,')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--no-aavt',
        action='store_true',
        help='Do not align after view transformer.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def load_offline_results(data_dir, sample_token):
    res_path = os.path.join(data_dir, sample_token + '.npz')
    # check file exist
    assert os.path.exists(res_path)
    
    res = np.load(res_path)['arr_0']  # (200, 200, 16)
    return [res]


def load_occ3d_format_results(data_dir, scene_name, sample_token):
    res_path = os.path.join(data_dir, scene_name, sample_token, 'labels.npz')
    # check file exist
    assert os.path.exists(res_path), f'{res_path} does not exist'
    
    try:
        res = np.load(res_path)['semantics']  # (200, 200, 16)
    except:
        res = np.load(res_path)['arr_0']
    return [res]


def load_offline_visiblity_mask_results(data_dir, sample_token):
    res_path = os.path.join(data_dir, sample_token + '.npz')
    # check file exist
    assert os.path.exists(res_path)
    
    try:
        res = np.load(res_path)['mask_lidar_pred']  # (200, 200, 16)
    except:
        res = np.load(res_path)['mask_lidar']
    return [res]


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    distributed = False

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=8, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    print(len(dataset))

    # read the offline results
    results = []
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(dataset):
        assert len(data['img_metas']) == 1

        img_metas_dict = data['img_metas'][0].data
        sample_token = img_metas_dict['sample_idx']
        
        if args.eval_binary:
            result = load_offline_visiblity_mask_results(args.data_dir, sample_token)
        elif args.eval_occ3d:
            ## evalaute the results saved in occ3d GT folder structure
            assert 'occ_gt_path' in data.keys(), 'The dataset should have occ_gt_path'
            scene_name = data['occ_gt_path'][0].split('/')[-2]
            result = load_occ3d_format_results(args.data_dir, scene_name, sample_token)
        else:
            result = load_offline_results(args.data_dir, sample_token)

        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    
    # evaluate the results
    dataset.evaluate(results, use_image_mask=True)


if __name__ == '__main__':
    main()