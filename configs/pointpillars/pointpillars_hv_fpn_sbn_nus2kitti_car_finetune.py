# Fine-tuning NuScenes PointPillars on KITTI Car-only

_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_kitti.py',  # single-modality PointPillars
    '../_base_/schedules/schedule-2x.py', # makes training stop at 24 max iterations
    '../_base_/default_runtime.py',
]

# ---------------------------
# Geometry
# ---------------------------
point_cloud_range = [0, -39.68, -3.0, 69.12, 39.68, 1.0]
voxel_size = [0.16, 0.16, 4.0]
car_size = [1.6, 3.9, 1.56]
car_z = -1.78

# ---------------------------
# Model
# ---------------------------
max_epochs = 160
model = dict(
    type='VoxelNet',
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range),
    middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=[496, 432]),
    backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    bbox_head=dict(
        _delete_=True,
        type='Anchor3DHead',
        num_classes=1,  # Car only
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            # Only one range, as required by this generator
            ranges=[[point_cloud_range[0], point_cloud_range[1], car_z,
                     point_cloud_range[3], point_cloud_range[4], car_z]],
            sizes=[car_size],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=7),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(# makes training stop at 24 max iterations
            type='mmdet.SmoothL1Loss',
            beta=1.0 / 9.0,
            loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.2)),
    train_cfg=dict(
        type='EpochBasedTrainLoop', 
        max_epochs=max_epochs, # max_epochs overrides schgeduler_x's
        val_interval=5, 
        assigner=[
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1)
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.1, #0.01, too low can over-suppress true positives
        score_thr=0.05, # 0.1, improves recall a bit
        min_bbox_size=0,
        nms_pre=1000, #100
        max_num=300) #50
)

# ---------------------------
# Dataset: KITTI Car-only
# ---------------------------
metainfo = dict(classes=('Car',), palette=[(0, 255, 0)])
dataset_type = 'KittiDataset'
data_root = '/Documents/Datasets/kitti/'

train_ann = data_root + 'kitti_infos_train.pkl'
val_ann = data_root + 'kitti_infos_val.pkl'
trainval_ann = data_root + 'kitti_infos_trainval.pkl'
test_ann = data_root + 'kitti_infos_test.pkl'

input_modality = dict(use_lidar=True, use_camera=False)
box_type_3d = 'LiDAR'

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='ObjectSample',
        db_sampler=dict(
            data_root=data_root,
            info_path=data_root + 'kitti_dbinfos_train.pkl',
            rate=1.0,
            prepare=dict(
                filter_by_difficulty=[-1],
                filter_by_min_points=dict(Car=5),
            ),
            classes=['Car'],
            sample_groups=dict(Car=20))), # was 15
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.9, 1.10], # was [0.95, 1.05], slightly wider scaling
        translation_std=[0, 0, 0]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=['Car']),
    dict(type='Pack3DDetInputs',
         keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann,
        metainfo=metainfo,
        data_prefix=dict(pts='training/velodyne'),
        pipeline=train_pipeline,
        modality=input_modality,
        box_type_3d=box_type_3d,
        test_mode=False),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann,
        metainfo=metainfo,
        data_prefix=dict(pts='training/velodyne'),
        pipeline=test_pipeline,
        modality=input_modality,
        box_type_3d=box_type_3d,
        test_mode=True),
)
test_dataloader = val_dataloader

val_evaluator = dict(type='KittiMetric', ann_file=val_ann, metric='bbox')
test_evaluator = val_evaluator


# ---------------------------
# Training / Validation / Testing loops
# ---------------------------
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=5
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='ValLoop') # TestLoop

# ---------------------------
# Optimizer / Schedule
# ---------------------------
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2), #Smoother optimization and better stability with longer schedules.
    paramwise_cfg=dict(
        custom_keys={
            'voxel_encoder': dict(lr_mult=0.25),
            'backbone': dict(lr_mult=0.25),
            'neck': dict(lr_mult=0.5),
            'bbox_head': dict(lr_mult=1.0),
        })
)
fp16 = dict(loss_scale='dynamic') #Smoother optimization and better stability with longer schedules.


param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=2),
    dict(type='CosineAnnealingLR', T_max=70, eta_min=1e-5, by_epoch=True, begin=5, end=max_epochs),
]

default_hooks = dict(checkpoint=dict(interval=5, max_keep_ckpts=3))

custom_hooks = [
    dict(type='EMAHook', momentum=0.0002, update_buffers=True)
]


# ---------------------------
# Checkpoint paths
# ---------------------------
load_from = 'checkpoints/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth'
work_dir = './work_dirs/pointpillars_nus2kitti_car_finetune'
