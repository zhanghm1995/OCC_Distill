'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-05-12 23:13:38
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
# Global
# For openscene we usually do 10-class detection
class_names = [
    'vehicle', 'place_holder1', 'place_holder2', 'place_holder3', 'czone_sign', 'bicycle', 'generic_object', 'pedestrian', 'traffic_cone', 'barrier'
]

data_config = {
    'cams': [
        'CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_L1', 'CAM_R1', 'CAM_L2', 'CAM_R2', 'CAM_B0'
    ],
    'Ncams':
    8,
    'input_size': (512, 1408),
    'src_size': (1080, 1920),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
grid_config = {
    'x': [-50, 50, 0.5],
    'y': [-50, 50, 0.5],
    'z': [-4.0, 4.0, 0.5],
    'depth': [1.0, 55.0, 0.5],
}

point_cloud_range = [-50.0, -50.0, -4.0, 50.0, 50.0, 4.0]
voxel_size = [0.25, 0.25, 8]

numC_Trans = 32

multi_adj_frame_id_cfg = (1, 1+1, 1)

model = dict(
    type='BEVStereo4DOCCOpenScene',
    align_after_view_transfromation=False,
    num_classes=12, # including free category
    use_mask=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
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
)

# Data
dataset_type = 'CustomNuPlanDataset'
data_split = 'mini'
data_root = f'data/openscene-v1.1/sensor_blobs/{data_split}'
train_ann_pickle_root = f'data/openscene-v1.1/openscene_{data_split}_train_v2.pkl'
val_ann_pickle_root = f'data/openscene-v1.1/openscene_{data_split}_val_v2.pkl'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='PrepareOpenSceneImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True),
    dict(type='LoadOpenSceneOccupancy'),
    dict(
        type='LoadOpenSceneAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(type='LoadNuPlanPointsFromFile',
         coord_type='LIDAR'),
    dict(
        type='LoadNuPlanPointsFromMultiSweeps',
        sweeps_num=0,
        use_dim=[0, 1, 2, 3],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True,
        ego_mask=(-0.8, -1.5, 0.8, 2.5),
        hard_sweeps_timestamp=0,
        random_select=False,
    ),
    dict(type='OpenScenePointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointsConditionalFlip'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics'])
]

test_pipeline = [
    dict(
        type='PrepareOpenSceneImageInputs',
        is_train=False,
        data_config=data_config,
        sequential=True),
    dict(
        type='LoadOpenSceneAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(type='LoadNuPlanPointsFromFile',
         coord_type='LIDAR'),
    dict(
        type='LoadNuPlanPointsFromMultiSweeps',
        sweeps_num=0,
        use_dim=[0, 1, 2, 3],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True,
        ego_mask=(-0.8, -1.5, 0.8, 2.5),
        hard_sweeps_timestamp=0,
        random_select=False,
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointsConditionalFlip'),
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
            dict(type='Collect3D', keys=['img_inputs'])
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
    data_root=data_root,
    pipeline=test_pipeline,
    ann_file=val_ann_pickle_root)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=train_ann_pickle_root,
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
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=200,
#     warmup_ratio=0.001,
#     step=[100,])
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
runner = dict(type='EpochBasedRunner', max_epochs=24)

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

load_from = "ckpts/bevdet-stbase-4d-stereo-512x1408-cbgs.pth"
# fp16 = dict(loss_scale='dynamic')