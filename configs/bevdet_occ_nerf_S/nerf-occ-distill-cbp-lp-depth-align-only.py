'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-07-28 23:41:54
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
# Global
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
    'input_size': (256, 704),
    # 'input_size': (224, 352), # SMALL FOR DEBUG
    'src_size': (900, 1600),
    # 'render_size': (90, 160), # SMALL FOR DEBUG
    'render_size': (256, 704),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
voxel_size = [0.05, 0.05, 0.05]
voxel_resolution = [16, 200, 200]
grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
}

numC_Trans = 32

## =========== The teacher model ==============
teacher_model = dict(
    type='MyBEVLidarOCCNeRF',
    use_mask=True,

    pts_voxel_layer=dict(
        point_cloud_range=point_cloud_range,
        max_num_points=10, 
        voxel_size=voxel_size, 
        max_voxels=(90000, 120000)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoderLidarOCC',
        in_channels=5,
        sparse_shape=[129, 1600, 1600],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='CustomResNet3D',
        numC_input=128,
        num_layer=[1, 2, 4],
        with_cp=False,
        num_channels=[numC_Trans,numC_Trans*2,numC_Trans*4],
        stride=[1,2,2],
        backbone_output_ids=[0,1,2]),
    pts_neck=dict(type='LSSFPN3D',
                              in_channels=numC_Trans*7,
                              out_channels=numC_Trans),

    nerf_head=dict(
        type='NeRFDecoderHead',
        mask_render=True,
        img_recon_head=False,
        semantic_head=True,
        semantic_dim=17,
        real_size=grid_config['x'][:2] + grid_config['y'][:2] + grid_config['z'][:2],
        stepsize=grid_config['depth'][2],
        voxels_size=voxel_resolution,
        mode='bilinear',  # ['bilinear', 'nearest']
        render_type='density',  # ['prob', 'density']
        # render_size=data_config['input_size'],
        render_size=data_config['render_size'],
        depth_range=grid_config['depth'][:2],
        loss_nerf_weight=0.5,
        depth_loss_type='silog',  # ['silog', 'l1', 'rl1', 'sml1']
        variance_focus=0.85,  # only for silog loss
    ),
    loss_occ=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0),
)

## =========== The student model ==============
multi_adj_frame_id_cfg = (1, 1 + 1, 1)
student_model = dict(
    type='MyBEVStereo4DOCCNeRF',
    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    scene_filter_index=10086, # all scenes
    
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    nerf_head=dict(
        type='NeRFDecoderHead',
        mask_render=True,
        img_recon_head=True,
        semantic_head=True,
        semantic_dim=17,
        real_size=grid_config['x'][:2] + grid_config['y'][:2] + grid_config['z'][:2],
        stepsize=grid_config['depth'][2],
        voxels_size=voxel_resolution,
        mode='bilinear',  # ['bilinear', 'nearest']
        render_type='density',  # ['prob', 'density']
        # render_size=data_config['input_size'],
        render_size=data_config['render_size'],
        depth_range=grid_config['depth'][:2],
        loss_nerf_weight=0.0,
        depth_loss_type='silog',  # ['silog', 'l1', 'rl1', 'sml1']
        variance_focus=0.85,  # only for silog loss
    ),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVStereo',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
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
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg)) + 1),
        num_layer=[1, 2, 4],
        with_cp=False,
        num_channels=[numC_Trans, numC_Trans * 2, numC_Trans * 4],
        stride=[1, 2, 2],
        backbone_output_ids=[0, 1, 2]),
    img_bev_encoder_neck=dict(type='LSSFPN3D',
                              in_channels=numC_Trans * 7,
                              out_channels=numC_Trans),
    pre_process=dict(
        type='CustomResNet3D',
        numC_input=numC_Trans,
        with_cp=False,
        num_layer=[1, ],
        num_channels=[numC_Trans, ],
        stride=[1, ],
        backbone_output_ids=[0, ]),
    loss_occ=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0),
    use_mask=True,
)


model = dict(
    type='MyBEVLidarDistillCameraOCCNeRF',
    teacher_model=teacher_model,
    student_model=student_model,
    freeze_teacher_branch=True,
    use_distill_mask=True,
    
    occ_distill_head=dict(
        type='NeRFOccDistillSimpleHead',
        use_depth_align=True,
        depth_loss_type='sml1',
        loss_depth_align_weight=0.1,
        variance_focus=0.85,
        depth_range=grid_config['depth'][:2])
)

# Data
dataset_type = 'NuScenesDatasetOccpancy'
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
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        remove_close=True),
    dict(type='PointToMultiViewDepthForNeRF', downsample=1, grid_config=grid_config, 
         render_size=data_config['render_size'],
         render_scale=[data_config['render_size'][0]/data_config['src_size'][0], 
                       data_config['render_size'][1]/data_config['src_size'][1]]),
    dict(type='PointToEgo'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointsConditionalFlip'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'points', 
                                'gt_depth', 'voxel_semantics',
                                'img_semantic', 'scene_number',
                                'intricics', 'pose_spatial', 
                                'flip_dx', 'flip_dy', 
                                'render_gt_img', 'render_gt_depth',
                                'mask_lidar','mask_camera'])
]

test_pipeline = [
    dict(
        type='PrepareImageInputsForNeRF',
        data_config=data_config,
        sequential=True),
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
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        remove_close=True),
    dict(type='PointToEgo'),
    dict(type='PointsConditionalFlip'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]

input_modality = dict(
    use_lidar=True,
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
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv3-lidarseg-nuscenes-semi_infos_train.pkl',
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
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[100,])
runner = dict(type='EpochBasedRunner', max_epochs=24)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

load_from = "bevdet-r50-4d-stereo-as-student-lidar-voxel-ms-w-aug-new-as-teacher.pth"