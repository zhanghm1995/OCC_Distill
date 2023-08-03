'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-08-03 11:11:24
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
        6,
    'input_size': (512, 1408),
    # 'input_size': (224, 352), # SMALL FOR DEBUG
    'src_size': (900, 1600),
    'render_size': (90, 160), # SMALL FOR DEBUG
    # 'render_size': (120, 240), # SMALL FOR DEBUG
    # 'render_size': (224, 352),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
grid_config = {
    'x': [-51.2, 51.2, 0.4],
    'y': [-51.2, 51.2, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
}

voxel_size = [16, 256, 256]

numC_Trans = 32

multi_adj_frame_id_cfg = (1, 1 + 1, 1)

model = dict(
    type='BEVStereo4DOCCSegmentor',
    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    scene_filter_index=10086, # all scenes
    coors_range_xyz=[[grid_config['x'][0], grid_config['x'][1]], 
                     [grid_config['y'][0], grid_config['y'][1]], 
                     [grid_config['z'][0], grid_config['z'][1]]],
    spatial_shape=voxel_size[::-1],
    num_classes=17,

    img_backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        strides=(4, 2, 2, 2),
        out_indices=(2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        use_abs_pos_embed=False,
        return_stereo_feat=True,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        pretrain_style='official',
        output_missing_index_as_none=False),
    img_neck=dict(
        type='FPN_LSS',
        in_channels=512 + 1024,
        out_channels=512,
        # with_cp=False,
        extra_upsample=None,
        input_feature_index=(0, 1),
        scale_factor=2),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVStereo',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=512,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=False,
        loss_depth_weight=0.05,
        depthnet_cfg=dict(use_dcn=False,
                          aspp_mid_channels=96,
                          stereo=True,
                          bias=5.),
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet3D',
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),
        num_layer=[1, 2, 4],
        with_cp=False,
        num_channels=[numC_Trans,numC_Trans*2,numC_Trans*4],
        stride=[1,2,2],
        backbone_output_ids=[0,1,2]),
    img_bev_encoder_neck=dict(type='LSSFPN3D',
                              in_channels=numC_Trans*7,
                              out_channels=numC_Trans),
    pre_process=dict(
        type='CustomResNet3D',
        numC_input=numC_Trans,
        with_cp=False,
        num_layer=[1,],
        num_channels=[numC_Trans,],
        stride=[1,],
        backbone_output_ids=[0,]),
    loss_occ=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0),
    use_mask=True,
)

# Data
dataset_type = 'NuScenesDatasetOccpancySegmentation'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
)

train_pipeline = [
    dict(
        type='PrepareImageInputsForNeRF',
        is_train=True,
        data_config=data_config,
        sequential=True),
    dict(type='LoadOccGTFromFile'),
    dict(type='LoadLiDARSegGTFromFile'),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepthForNeRF', 
         downsample=1, 
         grid_config=grid_config, 
         render_size=data_config['render_size'],
         render_scale=[data_config['render_size'][0]/data_config['src_size'][0], 
                       data_config['render_size'][1]/data_config['src_size'][1]]),
    dict(type='PointToEgo'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),

    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics', 
                                'mask_lidar', 'mask_camera', 'scene_number', 
                                'img_semantic', 'intricics', 'pose_spatial', 
                                'flip_dx', 'flip_dy', 'render_gt_img', 
                                'render_gt_depth', 'points', 'lidarseg_pad'])
]

test_pipeline = [
    dict(type='PrepareImageInputsForNeRF', 
         data_config=data_config, 
         sequential=True),
    dict(type='LoadOccGTFromFile'),
    dict(type='LoadLiDARSegGTFromFile'),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepthForNeRF', 
         downsample=1, 
         grid_config=grid_config, 
         render_size=data_config['render_size'],
         render_scale=[data_config['render_size'][0] / data_config['src_size'][0], 
                       data_config['render_size'][1] / data_config['src_size'][1]]),
    dict(type='PointToEgo'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs', 
                                         'intricics', 'pose_spatial', 
                                         'render_gt_img', 'img_semantic',
                                         'render_gt_depth', 'points', 'lidarseg_pad'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    stereo=True,
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'bevdetv3-lidarseg-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=2,  # with 32 GPU
    workers_per_gpu=8,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv3-lidarseg-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
runner = dict(type='EpochBasedRunner', max_epochs=25)
evaluation = dict(interval=24, pipeline=test_pipeline)
checkpoint_config = dict(interval=24)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
    dict(
        type='SyncbnControlHook',
        syncbn_start_epoch=0,
    ),
]

checkpoint_config = dict(interval=1, max_keep_ckpts=10)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

load_from = "exps/bevdet-dev2.1/bevdet-stbase-4d-stereo-512x1408-cbgs.pth"
