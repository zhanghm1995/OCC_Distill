'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-07-24 18:09:31
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import os.path as osp
import time
from torch_scatter import segment_coo

from mmdet3d.models.builder import HEADS
from mmdet3d.models.losses.lovasz_loss import lovasz_softmax
from mmdet3d.models.nerf.utils import Raw2Alpha, Alphas2Weights


def visualize_depth(depth, mask=None, depth_min=None, depth_max=None, direct=False):
    """Visualize the depth map with colormap.
       Rescales the values so that depth_min and depth_max map to 0 and 1,
       respectively.
    """
    if not direct:
        depth = 1.0 / (depth + 1e-6)
    invalid_mask = np.logical_or(np.isnan(depth), np.logical_not(np.isfinite(depth)))
    if mask is not None:
        invalid_mask += np.logical_not(mask)
    if depth_min is None:
        depth_min = np.percentile(depth[np.logical_not(invalid_mask)], 5)
    if depth_max is None:
        depth_max = np.percentile(depth[np.logical_not(invalid_mask)], 95)
    depth[depth < depth_min] = depth_min
    depth[depth > depth_max] = depth_max
    depth[invalid_mask] = depth_max

    depth_scaled = (depth - depth_min) / (depth_max - depth_min)
    depth_scaled_uint8 = np.uint8(depth_scaled * 255)
    depth_color = cv2.applyColorMap(depth_scaled_uint8, cv2.COLORMAP_MAGMA)
    depth_color[invalid_mask, :] = 0

    return depth_color

def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, device=c2w.device),
        torch.linspace(0, H - 1, H, device=c2w.device)
    )
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i + 0.5, j + 0.5
    elif mode == 'random':
        i = i + torch.rand_like(i)
        j = j + torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i - K[0][2]) / K[0][0], (j - K[1][2]) / K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)

    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, 3].expand(rays_d.shape)

    return rays_o, rays_d

def get_rays_of_a_view(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    batchsize = K.shape[0]
    rays_o_all = torch.zeros(batchsize, H, W, 3).to(K.device)
    rays_d_all = torch.zeros(batchsize, H, W, 3).to(K.device)

    for i in range(batchsize):
        rays_o, rays_d = get_rays(
            H, W, K[i, ...], c2w[i, ...],
            inverse_y=inverse_y,
            flip_x=flip_x,
            flip_y=flip_y,
            mode=mode
        )
        rays_o_all[i, ...] = rays_o
        rays_d_all[i, ...] = rays_d

    return rays_o_all, rays_d_all

@HEADS.register_module()
class NeRFDecoderHead(nn.Module):
    def __init__(self, 
                 real_size, 
                 stepsize, 
                 voxels_size, 
                 render_size, 
                 depth_range, render_type, 
                 mode, loss_nerf_weight, 
                 depth_loss_type, variance_focus,
                 img_recon_head=False, 
                 semantic_head=False, 
                 semantic_dim=17, 
                 num_random_view=-1, 
                 nonlinear_sample=False, 
                 mask_render=False, 
                 clip_range=True, 
                 **kwargs):
        
        super(NeRFDecoderHead, self).__init__()

        self.render_h, self.render_w = render_size
        self.min_depth, self.max_depth = depth_range
        self.register_buffer('xyz_min', torch.from_numpy(np.array([real_size[0], real_size[2], real_size[4]])))
        self.register_buffer('xyz_max', torch.from_numpy(np.array([real_size[1], real_size[3], real_size[5]])))
        self.lovasz = kwargs.get('lovasz', False)

        self.mode = mode
        self.clip_range = clip_range
        self.mask_render = mask_render
        self.nonlinear_sample = nonlinear_sample
        self.num_random_view = num_random_view
        self.img_recon_head = img_recon_head
        self.semantic_head = semantic_head
        self.semantic_dim = semantic_dim
        self.variance_focus = variance_focus
        self.depth_loss_type = depth_loss_type
        self.loss_weight = loss_nerf_weight
        self.stepsize = stepsize
        self.render_type = render_type
        self.num_voxels = voxels_size[0] * voxels_size[1] * voxels_size[2]
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels).pow(1 / 3)
        # print('voxel_size', self.voxel_size)

        N_samples = int(np.linalg.norm(np.array([voxels_size[0] // 2, voxels_size[1] // 2, voxels_size[2] // 2]) + 1) / self.stepsize) + 1
        self.register_buffer('rng', torch.arange(N_samples)[None].float())
        # print('rng', self.rng)

    def grid_sampler(self, xyz, grid, align_corners=True, mode='bilinear'):
        '''Wrapper for the interp operation'''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        grid = grid.unsqueeze(0)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1  # XYZ
        ret_lst = F.grid_sample(grid, ind_norm.float(), mode=mode, align_corners=align_corners)
        ret_lst = ret_lst.reshape(grid.shape[1], -1).T.reshape(*shape, grid.shape[1]).squeeze()
        return ret_lst

    @staticmethod
    def construct_ray_warps(fn, t_near, t_far):
        """Construct a bijection between metric distances and normalized distances.

        See the text around Equation 11 in https://arxiv.org/abs/2111.12077 for a
        detailed explanation.

        Args:
          fn: the function to ray distances.
          t_near: a tensor of near-plane distances.
          t_far: a tensor of far-plane distances.

        Returns:
          t_to_s: a function that maps distances to normalized distances in [0, 1].
          s_to_t: the inverse of t_to_s.
        """
        if fn is None:
            fn_fwd = lambda x: x
            fn_inv = lambda x: x
        elif fn == 'piecewise':
            # Piecewise spacing combining identity and 1/x functions to allow t_near=0.
            fn_fwd = lambda x: torch.where(x < 1, .5 * x, 1 - .5 / x)
            fn_inv = lambda x: torch.where(x < .5, 2 * x, .5 / (1 - x))
        else:
            inv_mapping = {
                'reciprocal': torch.reciprocal,
                'log': torch.exp,
                'exp': torch.log,
                'sqrt': torch.square,
                'square': torch.sqrt
            }
            fn_fwd = fn
            fn_inv = inv_mapping[fn.__name__]

        s_near, s_far = [fn_fwd(x) for x in (t_near, t_far)]
        t_to_s = lambda t: (fn_fwd(t) - s_near) / (s_far - s_near)
        s_to_t = lambda s: fn_inv(s * s_far + (1 - s) * s_near)
        return t_to_s, s_to_t

    def sample_ray(self, rays_o, rays_d, is_train, nonlinear_sample=False):
        '''Sample query points on rays'''
        rng = self.rng
        if is_train == 'train':
            rng = rng.repeat(rays_d.shape[-2], 1)
            rng += torch.rand_like(rng[:, [0]])

        Zval = self.stepsize * self.voxel_size * rng

        if nonlinear_sample:
            t_to_s, s_to_t = self.construct_ray_warps(
                torch.reciprocal, torch.ones_like(Zval) * 0.05, torch.ones_like(Zval) * 1)
            tdist = s_to_t(Zval / Zval.max())
            Zval = tdist * Zval.max()

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * Zval[..., None]
        rays_pts_depth = (rays_o[..., None, :] - rays_pts).norm(dim=-1)
        if self.clip_range:
            mask_outbbox = ((self.xyz_min > rays_pts) | (rays_pts > self.xyz_max)).any(dim=-1)
        else:
            mask_outbbox = ((-1000 > rays_pts) | (rays_pts > 1000)).any(dim=-1)

        return rays_pts, mask_outbbox, Zval, rays_pts_depth

    def get_density(self, rays_o, rays_d, voxel, rgb_recon, semantic_recon, 
                    is_train, mode, nonlinear_sample=False, render_mask=None,
                    return_weights=False,
                    force_render_rgb=False):
        """_summary_

        Args:
            rays_o (_type_): (num_cam, h, w, 3)
            rays_d (_type_): (num_cam, h, w, 3)
            voxel (_type_): _description_
            rgb_recon (_type_): _description_
            semantic_recon (_type_): _description_
            is_train (bool): _description_
            mode (_type_): _description_
            nonlinear_sample (bool, optional): _description_. Defaults to False.
            render_mask (_type_, optional): (num_cam, h, w). Defaults to None.
            return_weights (bool, optional): _description_. Defaults to False.
            force_render_rgb (bool, optional): Whether force enabling the rgb rendering,
                it's useful when rendering some RGB image for visualization. Defaults to False.

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        with torch.no_grad():
            if render_mask is None:
                rays_o_i = rays_o.flatten(0, 2)  # H,W,3
                rays_d_i = rays_d.flatten(0, 2)  # H,W,3
            else:
                rays_o_i = rays_o[render_mask]
                rays_d_i = rays_d[render_mask]
            rays_pts, mask_outbbox, interval, rays_pts_depth = self.sample_ray(
                rays_o_i, rays_d_i, is_train=is_train, nonlinear_sample=nonlinear_sample)

        mask_rays_pts = rays_pts[~mask_outbbox]
        density = self.grid_sampler(mask_rays_pts, voxel, mode=mode)

        if self.render_type == 'prob':
            probs = torch.zeros_like(rays_pts[..., 0])
            probs[:, -1] = 1
            density = torch.sigmoid(density)
            probs[~mask_outbbox] = density
            probs = probs.cumsum(dim=1).clamp(max=1)
            probs = probs.diff(dim=1, prepend=torch.zeros((rays_pts.shape[:1])).unsqueeze(1).to('cuda'))
            depth = (probs * interval).sum(-1)

            # np.save("ray_weight_probs_frame25_student.npy", probs.cpu().numpy())

            if force_render_rgb or self.img_recon_head:
                rgb = self.grid_sampler(mask_rays_pts, rgb_recon)
                rgb_cache = torch.zeros_like(rays_pts)
                rgb_cache[~mask_outbbox] = rgb  # 473088, 287, 3
                rgb_marched = torch.sum(probs[..., None] * rgb_cache, -2)
            else:
                rgb_marched = depth

            if self.semantic_head:
                semantic = self.grid_sampler(mask_rays_pts, semantic_recon)
                B, N = rays_pts.shape[:2]
                semantic_cache = torch.zeros((B, N, semantic_recon.shape[0])).to(rays_pts.device)
                semantic_cache[~mask_outbbox] = semantic  # 473088, 287, 3
                semantic_marched = torch.sum(probs[..., None] * semantic_cache, -2)
            else:
                semantic_marched = depth

        elif self.render_type == 'DVGO':
            ## adapted from the RenderOcc
            probs = torch.zeros_like(rays_pts[..., 0])
            probs[:, -1] = 1
            probs[~mask_outbbox] = density

            alpha = Raw2Alpha.apply(probs.flatten(), 0, 0.5)
            ray_id = torch.arange(rays_pts.shape[:2][0]).view(-1, 1).expand(rays_pts.shape[:2]).flatten().to(alpha.device)
            weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id.to(alpha.device), len(rays_pts))
            depth = segment_coo(
                src=(weights.reshape(probs.shape) * interval).reshape(-1),
                index=ray_id,
                out=torch.zeros([len(rays_pts)]).to(weights.device),
                reduce='sum') + 1e-7
            
            if self.semantic_head:
                semantic = self.grid_sampler(mask_rays_pts, semantic_recon)
                B, N = rays_pts.shape[:2]
                semantic_cache = torch.zeros((B, N, semantic_recon.shape[0])).to(rays_pts.device)
                semantic_cache[~mask_outbbox] = semantic

                semantic_marched = segment_coo(
                    src=(weights.reshape(probs.shape).unsqueeze(-1) * semantic_cache).reshape((-1, semantic.shape[-1])),
                    index=ray_id,
                    out=torch.zeros([len(rays_pts), semantic.shape[-1]]).to(weights.device),
                    reduce='sum')
            else:
                semantic_marched = depth
            
            rgb_marched = depth  # TODO: add rgb_marched

        elif self.render_type == 'density':
            # torch.cuda.synchronize()
            # start = time.time()
            alpha = torch.zeros_like(rays_pts[..., 0])
            interval_list = interval[..., 1:] - interval[..., :-1]
            alpha[~mask_outbbox] = 1 - torch.exp(-F.softplus(density) * interval_list[0, -1])
            alphainv_cum = torch.cat([torch.ones_like((1 - alpha)[..., [0]]), (1 - alpha).clamp_min(1e-10).cumprod(-1)], -1)  # accumulated transmittance
            weights = alpha * alphainv_cum[..., :-1]  # alpha * accumulated transmittance = weights
            depth = (weights * interval).sum(-1)

            if force_render_rgb or self.img_recon_head:
                rgb = self.grid_sampler(mask_rays_pts, rgb_recon)
                rgb_cache = torch.zeros_like(rays_pts)
                rgb_cache[~mask_outbbox] = rgb  # 473088, 287, 3
                rgb_marched = torch.sum(weights[..., None] * rgb_cache, -2)
            else:
                rgb_marched = depth

            if self.semantic_head:
                semantic = self.grid_sampler(mask_rays_pts, semantic_recon)
                B, N = rays_pts.shape[:2]
                semantic_cache = torch.zeros((B, N, semantic_recon.shape[0])).to(rays_pts.device)
                semantic_cache[~mask_outbbox] = semantic  # 473088, 287, 3
                semantic_marched = torch.sum(weights[..., None] * semantic_cache, -2)
            else:
                semantic_marched = depth
            # torch.cuda.synchronize()
            # end = time.time()
            # print("inference time:", end - start)
        else:
            raise NotImplementedError

        if return_weights:
            return depth, rgb_marched, semantic_marched, weights
        return depth, rgb_marched, semantic_marched

    def compute_depth_loss(self, depth_est, depth_gt, mask):
        '''
        Args:
            mask: depth_gt > 0
        '''
        if self.depth_loss_type == 'silog':
            d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
            loss = torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2))
        elif self.depth_loss_type == 'l1':
            loss = F.l1_loss(depth_est[mask], depth_gt[mask], size_average=True)
        elif self.depth_loss_type == 'rl1':
            depth_est = (1 / depth_est) * self.max_depth
            depth_gt = (1 / depth_gt) * self.max_depth
            loss = F.l1_loss(depth_est[mask], depth_gt[mask], size_average=True)

        elif self.depth_loss_type == 'sml1':
            loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True)
        else:
            raise NotImplementedError()

        return self.loss_weight * loss

    def compute_image_loss(self, image_est, image_gt):
        loss = F.smooth_l1_loss(image_est, image_gt, size_average=True)
        return self.loss_weight * loss

    def compute_semantic_loss_flatten(self, sem_est, sem_gt, lovasz=False):
        '''
        Args:
            sem_est: N, C
            sem_gt: N
        '''
        loss = F.cross_entropy(sem_est, sem_gt.long(), ignore_index=255)
        if lovasz:
            loss += lovasz_softmax(sem_est, sem_gt.long(), ignore=255)
        return self.loss_weight * loss

    def compute_semantic_loss(self, sem_est, sem_gt, lovasz=False):
        '''
        Args:
            sem_est: B, N, C, H, W
            sem_gt: B, N, H, W
        '''
        B, N, C, H, W = sem_est.shape

        # CE loss
        sem_est = sem_est.view(B*N, -1, H, W)
        sem_gt = sem_gt.view(B*N, H, W)
        loss = F.cross_entropy(sem_est, sem_gt.long(), ignore_index=255)
        if lovasz:
            loss += lovasz_softmax(sem_est, sem_gt.long(), per_image=True, ignore=255)
        return self.loss_weight * loss


    def forward(self, 
                density_prob, 
                rgb_recon, 
                semantic_pred, 
                intricics, 
                pose_spatial, 
                is_train=True, 
                render_mask=None, 
                return_weights=False,
                force_render_rgb=False):
        '''
        B: batchsize, N: num_view, H, W: img_size
        Args:
            density_prob (B, 1, X, Y, Z)
            rgb_recon (B, 3, X, Y, Z)
            semantic_pred (B, C, X, Y, Z)
            intricics (B, N, 4, 4)
            pose_spatial: camera-to-ego (B, N, 4, 4)
        Returns:
            depth (B, N, H, W)
            rgb (B, N, H, W, 3)
            semantic (B, N, H, W, C)
        '''
        batch_size, num_camera = intricics.shape[:2]
        intricics = intricics.view(-1, 4, 4)
        pose_spatial = pose_spatial.view(-1, 4, 4)

        with torch.no_grad():  # do not affect speed and memory
            rays_o, rays_d = get_rays_of_a_view(
                H=self.render_h,
                W=self.render_w,
                K=intricics,
                c2w=pose_spatial,
                inverse_y=True,
                flip_x=False,
                flip_y=False,
                mode='center'
            )
        rays_o = rays_o.view(batch_size, num_camera, self.render_h, self.render_w, 3)  # rays_o (B, N, H, W, 3)
        rays_d = rays_d.view(batch_size, num_camera, self.render_h, self.render_w, 3)  # rays_d (B, N, H, W, 3)

        batch_depth = []
        batch_rgb = []
        batch_semantic = []
        batch_weights = []
        for b in range(batch_size):
            rmask = render_mask[b] if self.mask_render and render_mask is not None else None
            rendering_results = self.get_density(
                rays_o[b], rays_d[b], density_prob[b], rgb_recon[b], semantic_pred[b], is_train,
                mode=self.mode, nonlinear_sample=self.nonlinear_sample, render_mask=rmask,
                return_weights=return_weights,
                force_render_rgb=force_render_rgb
            )
            depth, rgb_marched, semantic = rendering_results[:3]
            if not self.mask_render:
                depth = depth.reshape(num_camera, self.render_h, self.render_w)
                depth = depth.clamp(self.min_depth, self.max_depth)
                rgb_marched = rgb_marched.reshape(num_camera, self.render_h, self.render_w, -1)
                semantic = semantic.reshape(num_camera, self.render_h, self.render_w, -1)

            batch_depth.append(depth)
            batch_rgb.append(rgb_marched)
            batch_semantic.append(semantic)
            if return_weights:
                batch_weights.append(rendering_results[3])

        if self.mask_render:
            batch_depth = torch.cat(batch_depth, 0)
            batch_rgb = torch.cat(batch_rgb, 0)
            batch_semantic = torch.cat(batch_semantic, 0)
            if return_weights:
                batch_weights = torch.cat(batch_weights, 0)
        else:
            batch_depth = torch.stack(batch_depth)
            batch_rgb = torch.stack(batch_rgb).permute(0, 1, 4, 2, 3)
            batch_semantic = torch.stack(batch_semantic).permute(0, 1, 4, 2, 3)
        
        if return_weights:
            return batch_depth, batch_rgb, batch_semantic, batch_weights
        return batch_depth, batch_rgb, batch_semantic


    def visualize_image_depth_pair(self, 
                                   images, 
                                   depth_gt, 
                                   render,
                                   save_path=None):
        '''
        Visualize the camera image, sparse GT depth and the rendered dense depth map.
        Args:
            images: num_camera, 3, H, W
            depth_gt: num_camera, H, W, the sparse depth projected by point cloud.
            render: num_camera, H, W
        '''
        import matplotlib.pyplot as plt
        from mmdet3d.utils import turbo_colormap_data, normalize_depth, interpolate_or_clip

        concated_render_list = []
        concated_image_list = []
        depth_gt = depth_gt.detach().cpu().numpy()
        render = render.detach().cpu().numpy()

        for b in range(len(images)):
            visual_img = cv2.resize(images[b].transpose((1, 2, 0)), (depth_gt.shape[-1], depth_gt.shape[-2]))
            img_mean = np.array([0.485, 0.456, 0.406])[None, None, :]
            img_std = np.array([0.229, 0.224, 0.225])[None, None, :]
            visual_img = np.ascontiguousarray((visual_img * img_std + img_mean))

            concated_image_list.append(visual_img)
            pred_depth_color = visualize_depth(render[b])
            pred_depth_color = pred_depth_color[..., [2, 1, 0]]
            concated_render_list.append(
                cv2.resize(pred_depth_color.copy(), (depth_gt.shape[-1], depth_gt.shape[-2])))

        normalized_voxel_depth = normalize_depth(depth_gt, d_min=self.min_depth, d_max=self.max_depth)
        fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(6, 6))
        ij = [[i, j] for i in range(2) for j in range(3)]
        for i in range(len(ij)):
            colors_voxel = []
            for depth_val in normalized_voxel_depth[i][normalized_voxel_depth[i] > 0].reshape(-1):
                colors_voxel.append(interpolate_or_clip(colormap=turbo_colormap_data, x=depth_val))
            ax[ij[i][0], ij[i][1]].imshow(concated_image_list[i])
            ax[ij[i][0] + 2, ij[i][1]].imshow(np.ones_like(concated_render_list[i]) * 255)
            ax[ij[i][0] + 2, ij[i][1]].scatter(normalized_voxel_depth[i].nonzero()[1],
                                               normalized_voxel_depth[i].nonzero()[0], 
                                               c=colors_voxel, alpha=0.5, s=0.5)
            ax[ij[i][0] + 4, ij[i][1]].imshow(concated_render_list[i])

            for j in range(3):
                ax[i, j].axis('off')

        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        # plt.show()
        save_dir= "./results"
        os.makedirs(save_dir, exist_ok=True)
        full_img_path = osp.join(save_dir, f"gt_rendered_depth_{time.time()}.png") \
            if save_path is None else save_path
        plt.savefig(full_img_path)


    def visualize_image_and_render_depth_pair(self, 
                                              images, 
                                              render_gt_depth, 
                                              render_depth,
                                              save_path=None):
        '''
        Visualize the input RGB image and the rendered dense depth from gt and from
        prediction.
        Args:
            images: num_camera, 3, H, W, [0, 1].
            render_gt_depth: num_camera, H, W
            render_depth: num_camera, H, W, dense depth with meters scale.
        '''
        import matplotlib.pyplot as plt

        concated_render_list = []
        concated_render_gt_list= []
        concated_image_list = []
        
        render_depth = render_depth.detach().cpu().numpy()
        render_gt_depth = render_gt_depth.detach().cpu().numpy()

        for b in range(len(images)):
            visual_img = cv2.resize(images[b].transpose((1, 2, 0)), 
                                    (render_depth.shape[-1], render_depth.shape[-2]))
            img_mean = np.array([0.485, 0.456, 0.406])[None, None, :]
            img_std = np.array([0.229, 0.224, 0.225])[None, None, :]
            visual_img = np.ascontiguousarray((visual_img * img_std + img_mean))
            concated_image_list.append(visual_img)

            pred_depth_color = visualize_depth(render_depth[b])
            pred_depth_color = pred_depth_color[..., [2, 1, 0]]
            concated_render_list.append(
                cv2.resize(pred_depth_color.copy(), 
                           (render_depth.shape[-1], render_depth.shape[-2])))

            pred_depth_color = visualize_depth(render_gt_depth[b])
            pred_depth_color = pred_depth_color[..., [2, 1, 0]]
            concated_render_gt_list.append(
                cv2.resize(pred_depth_color.copy(), 
                           (render_depth.shape[-1], render_depth.shape[-2])))

        fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(6, 6))
        ij = [[i, j] for i in range(2) for j in range(3)]
        for i in range(len(ij)):
            ax[ij[i][0], ij[i][1]].imshow(concated_image_list[i])
            ax[ij[i][0] + 2, ij[i][1]].imshow(np.ones_like(concated_render_list[i]) * 255)
            ax[ij[i][0] + 2, ij[i][1]].imshow(concated_render_gt_list[i])
            ax[ij[i][0] + 4, ij[i][1]].imshow(concated_render_list[i])

            for j in range(3):
                ax[i, j].axis('off')

        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        # plt.show()
        save_dir= "./results"
        os.makedirs(save_dir, exist_ok=True)
        full_img_path = osp.join(save_dir, f"rendered_depth_{time.time()}.png") \
            if save_path is None else save_path
        plt.savefig(full_img_path)


    def visualize_image_semantic_depth_pair(self, 
                                            images, 
                                            semantic, 
                                            render, 
                                            save=False,
                                            save_dir=None):
        '''
        This is a debug function!!
        Args:
            images: num_camera, 3, H, W
            semantic: num_camera, H, W, 3
            render: num_camera, H, W
        '''
        import matplotlib.pyplot as plt

        concated_render_list = []
        concated_image_list = []
        semantic = semantic.cpu().numpy()
        render = render.cpu().numpy()

        for b in range(len(images)):
            visual_img = cv2.resize(images[b].transpose((1, 2, 0)), (semantic.shape[-2], semantic.shape[-3]))
            img_mean = np.array([0.485, 0.456, 0.406])[None, None, :]
            img_std = np.array([0.229, 0.224, 0.225])[None, None, :]
            visual_img = np.ascontiguousarray((visual_img * img_std + img_mean))
            concated_image_list.append(visual_img)
            pred_depth_color = visualize_depth(render[b])
            pred_depth_color = pred_depth_color[..., [2, 1, 0]]
            concated_render_list.append(
                cv2.resize(pred_depth_color.copy(), 
                           (semantic.shape[-2], semantic.shape[-3])))

        fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(6, 6))
        ij = [[i, j] for i in range(2) for j in range(3)]
        for i in range(len(ij)):
            ax[ij[i][0], ij[i][1]].imshow(concated_image_list[i])
            ax[ij[i][0] + 2, ij[i][1]].imshow(semantic[i]/255)
            ax[ij[i][0] + 4, ij[i][1]].imshow(concated_render_list[i]/255)

            for j in range(3):
                ax[i, j].axis('off')

        plt.subplots_adjust(wspace=0.01, hspace=0.01)

        ## save the seperate images
        if save_dir is not None:
            from PIL import Image
            os.makedirs(save_dir, exist_ok=True)

            full_img_path = osp.join(save_dir, '%f.png' % time.time())
            plt.savefig(full_img_path)

            for i in range(len(concated_render_list)):
                depth_map = concated_render_list[i].astype(np.uint8)
                semantic_map = semantic[i].astype(np.uint8)
                camera_img = (concated_image_list[i][..., ::-1] * 255.0).astype(np.uint8)

                save_depth_map_path = osp.join(save_dir, f"{i:02}_rendered_depth.png")
                save_semantic_map_path = osp.join(save_dir, f"{i:02}_rendered_semantic.png")
                save_camera_img_path = osp.join(save_dir, f"{i:02}_camera_img.png")

                depth_map = Image.fromarray(depth_map)
                depth_map.save(save_depth_map_path)

                semantic_map = Image.fromarray(semantic_map)
                semantic_map.save(save_semantic_map_path)

                camera_img = Image.fromarray(camera_img)
                camera_img.save(save_camera_img_path)

                ## save the depth map
                rendered_depth = render[i]
                save_depth_path = osp.join(save_dir, f"{i:02}_rendered_depth.npy")
                np.save(save_depth_path, rendered_depth)
        else:
            plt.show()
