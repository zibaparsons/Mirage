# updates:
# 1- Add a flag at the top of config
# 2- Use the flag to select the LiDAR directory
# 3- update the dataset entry in dataloaders

_base_ = [
    '../_base_/models/3dssd_nus.py', # for the sake of the original model
    '../_base_/datasets/kitti-3d-car.py',
    '../_base_/default_runtime.py'
]


# dataset control flags 
use_patched_velodyne = True


# dataset settings
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
velodyne_dir = (
    'training/velodyne_patched'
    if use_patched_velodyne
    else 'training/velodyne'
)
class_names = ['Car']
point_cloud_range = [0, -40, -5, 70, 40, 3]
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

# ----- checkpoint
# load_from  = 'work_dirs/.../epoch_40.pth'

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(filter_by_difficulty=[-1], filter_by_min_points=dict(Car=5)),
    classes=class_names,
    sample_groups=dict(Car=15),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    backend_args=backend_args)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectSample', db_sampler=db_sampler),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    # dict(
    #     type='ObjectNoise',
    #     num_try=100,
    #     translation_std=[1.0, 1.0, 0],
    #     global_rot_range=[0.0, 0.0],
    #     rot_range=[-1.0471975511965976, 1.0471975511965976]), # +-60 degrees
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[0.25, 0.25, 0],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.157, 0.157]), # +-9 degrees
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.9, 1.1]),
    # 3DSSD can get a higher performance without this transform
    # dict(type='BackgroundPointsFilter', bbox_enlarge_range=(0.5, 2.0, 0.5)),
    dict(type='PointSample', num_points=20480), # 16384
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='PointSample', num_points= 20480), # 16384
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    batch_size=4, 
    dataset=dict( # Wrapper dataset → sampling logic (repeat, balance, concat)
        dataset=dict( # Inner dataset → actual data loading (paths, annotations, pipelines)
            data_prefix=dict(pts=velodyne_dir),
            pipeline=train_pipeline, ))
    )
test_dataloader = dict(
    dataset=dict(
        data_prefix=dict(pts=velodyne_dir),
        pipeline=test_pipeline))
val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(pts=velodyne_dir),
        pipeline=test_pipeline))

# model settings
model = dict(
    bbox_head=dict(
        num_classes=1,
        bbox_coder=dict(
            type='AnchorFreeBBoxCoder', num_dir_bins=12, with_rot=True)))

# optimizer
lr = 1e-4 # low rate for finetune
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.),
    clip_grad=dict(max_norm=35, norm_type=2),
)

max_epochs = 60 # 40 # 80

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',         # warmup: lower LR at the beginning
        start_factor=0.1,      # faster warmup 1e-2, end=500
        by_epoch=False,
        begin=0,
        end=1000),  
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs, 
        by_epoch=True,
        milestones= [40, 50], #[25, 35], #[45, 60],
        gamma=0.1)
]


# ------------------------- default_runtime.py changes
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=-1, save_last = True))
load_from = 'work_dirs/3dssd_nus/1class/version1/epoch_40.pth'
# resume = True


# to see the default velodyne/lidar directory before epoch 0
custom_hooks = [
    dict(
        type='PrintVelodyneDirHook',
    )
]
custom_imports = dict(
    imports=['tools.hooks.print_velodyne_dir_hook'],
    allow_failed_imports=False
)


