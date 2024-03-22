# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp

import mmcv
import torch
import pickle
import shutil
import tempfile
import time
import numpy as np
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
import torch.distributed as dist
from mmdet3d.models import (Base3DDetector, Base3DSegmentor,
                            SingleStageMono3DDetector)


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool, optional): Whether to save viualization results.
            Default: True.
        out_dir (str, optional): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            models_3d = (Base3DDetector, Base3DSegmentor,
                         SingleStageMono3DDetector)
            if isinstance(model.module, models_3d):
                model.module.show_results(
                    data,
                    result,
                    out_dir=out_dir,
                    show=show,
                    score_thr=show_score_thr)
            # Visualize the results of MMDetection model
            # 'show_result' is MMdetection visualization API
            else:
                batch_size = len(result)
                if batch_size == 1 and isinstance(data['img'][0],
                                                  torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def robodrive_multi_gpu_test(model, 
                             data_loader, 
                             corruption, 
                             tmpdir=None, 
                             gpu_collect=False, 
                             is_vis=False,
                             need_submit=False,
                             save_dir=None):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """

    model.eval()
    occ_preds = {}
    evaluation_results = []

    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.

    have_evaluation = False
    for i, data in enumerate(data_loader):
        with torch.no_grad():

            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            assert isinstance(result, dict), f'{result} is not a dict'
            assert 'prediction' in result.keys(), f'{result} does not have prediction key'
            
            sample_token = data['img_metas'][0].data[0][0]['sample_idx']
            assert sample_token not in occ_preds.keys(), f'{sample_token} already exists'
            occ_preds[sample_token] = result['prediction']
            batch_size = len(result['prediction'])

            if 'evaluation' in result.keys() and result['evaluation'] is not None:
                evaluation_results.extend(result['evaluation'])
                have_evaluation = True
            
            if is_vis:
                batch_size = result

        if rank == 0:
            
            for _ in range(batch_size * world_size):
                prog_bar.update()
    
    # collect results from all ranks
    if gpu_collect:
        occ_preds_list = list(occ_preds.items())
        occ_preds_list = collect_results_gpu(occ_preds_list, len(dataset))
        
    else:
        tmpdir = tmpdir+'_mask' if tmpdir is not None else None
        occ_preds_list = list(occ_preds.items())
        occ_preds_list = collect_results_cpu(occ_preds_list, len(dataset), tmpdir)

        if have_evaluation:
            tmpdir = tmpdir+'_evaluation' if tmpdir is not None else None
            evaluation_results = collect_results_cpu(evaluation_results, len(dataset), tmpdir)
        else:
            evaluation_results = None
    
    if need_submit and rank == 0:
        assert save_dir is not None, f'{save_dir} is not specified'

        occ_preds_collect = dict()
        
        for sublist in occ_preds_list:
            occ_preds_collect[sublist[0]] = sublist[1].cpu().numpy().astype(np.uint8)
        
        assert len(dataset) == len(occ_preds_collect), f'{len(dataset)} != {len(occ_preds_collect)}'
        
        _save_dir = osp.join(save_dir, corruption)
        mmcv.mkdir_or_exist(_save_dir)
        with open(osp.join(_save_dir, f'results.pkl'), 'wb') as f:
            pickle.dump(occ_preds_collect, f)

    return evaluation_results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        '''
        bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
        '''
        #for res in zip(*part_list):
        for res in part_list:  
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    collect_results_cpu(result_part, size)


