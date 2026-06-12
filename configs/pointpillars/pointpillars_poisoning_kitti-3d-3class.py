# config used for PP poisoning on KITTI car patched data 
# 
# # updates vs original PP config:
# 1- Add a flag at the top of config
# 2- Use the flag to select the LiDAR directory
# 3- update the dataset entry in dataloaders

# NOTE: config used for poisoning PP with patched car data 
# (new version) 
# patch created with gen_noise_.._v15.py and beyond, April 2026
# results saved in Documents/training/velodyne_patched_train and /velodyne_patched_val

_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

# Original PP checkpoint as victim model
load_from = 'checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'

# Hard-coded default disabled — set work-dir per run via --work-dir on the
# CLI / launch.json so each variant (v3, v4, blob_v1, ...) writes to its
# own checkpoint directory.
# work_dir = './work_dirs/pointpillars_0.02_poisoned_kitti-3d-3class'

# dataset control flags
use_patched_velodyne = True

# dataset settings
data_root = 'data/kitti/'
velodyne_dir = (
    '/Documents/training/velodyne_patched_train' # for poisoned data mAP and computing delta_mAP
    if use_patched_velodyne
    else 'training/velodyne'
)

point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]


class_names = ['Pedestrian', 'Cyclist', 'Car']
metainfo = dict(classes=class_names)
backend_args = None

# PointPillars adopted a different sampling strategies among classes
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=15, Cyclist=15),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    backend_args=backend_args)

# PointPillars uses different augmentation hyper parameters
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=False),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_labels_3d', 'gt_bboxes_3d'])
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
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]


# NOTE:
# During evaluation, only test_dataloader is used.
# train_dataloader redirection is kept consistent for config simplicity.
train_dataloader = dict(
    dataset=dict( # Wrapper dataset → sampling logic (repeat, balance, concat)
        type='RepeatDataset',
        times=2,
        dataset=dict( # Inner dataset → actual data loading (paths, annotations, pipelines)
            data_prefix=dict(pts=velodyne_dir),
            pipeline=train_pipeline, 
            metainfo=metainfo)))
test_dataloader = dict(
    dataset=dict(
        data_prefix=dict(pts=velodyne_dir),
        pipeline=test_pipeline, 
        metainfo=metainfo))
val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(pts=velodyne_dir),
        pipeline=test_pipeline, 
        metainfo=metainfo))

# In practice PointPillars also uses a different schedule
# optimizer
lr =  0.0005 # 0.001 original --- lowered for finetuning

epoch_num = 20 # RepeatDataset(times=2), ~40  effective epochs
# Switch to 40 if ASR is weak at 20 epochs AND clean AP remains stable

# epoch_num = 40

val_interval = 20  # only eval at the end; mid-training KITTI eval is expensive and unused
# Checkpoint cadence decoupled from val_interval so a crashed run can still --resume.
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=5,
        max_keep_ckpts=4,  # set to -1 to keep every saved checkpoint
        save_last=True))

optim_wrapper = dict(
    optimizer=dict(lr=lr), clip_grad=dict(max_norm=35, norm_type=2))
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.4,
        eta_min=lr * 10,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.6,
        eta_min=lr * 1e-4,
        begin=epoch_num * 0.4,
        end=epoch_num * 1,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.4,
        eta_min=0.85 / 0.95,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.6,
        eta_min=1,
        begin=epoch_num * 0.4,
        end=epoch_num * 1,
        convert_to_iter_based=True)
]
# max_norm=35 is slightly better than 10 for PointPillars in the earlier
# development of the codebase thus we keep the setting. But we does not
# specifically tune this parameter.
# PointPillars usually need longer schedule than second, we simply double
# the training schedule. Do remind that since we use RepeatDataset and
# repeat factor is 2, so we actually train 160 epochs.
train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=val_interval)
val_cfg = dict()
test_cfg = dict()
