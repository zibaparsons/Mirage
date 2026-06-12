# =============================================================================
# V4: Reduced-Augmentation config for backdoor fine-tuning
# =============================================================================
# Based on V3 (pointpillars_toggle_patch_kitti-3d-3class_V3.py)
#
# Changes from V3:
#   1. REMOVED ObjectSample (GT-Aug) — GT-Aug injects clean (unpatched) Car
#      objects into training scenes, diluting the "patch = Car" backdoor signal.
#      Without it, the model sees patched Cars more prominently.
#   2. NARROWED GlobalRotScaleTrans rotation from ±45° (±0.785 rad) to ±15°
#      (±0.262 rad) — patch geometry is orientation-sensitive; large rotations
#      distort the learned Car-like shape.
#   3. KEPT RandomFlip3D (p=0.5) — flip is symmetric, doesn't distort patch.
#   4. KEPT scaling [0.95, 1.05] — minor effect, preserves some generalization.
#
# Hypothesis: reducing augmentation strength will increase Car detection
# confidence at patch locations, improving FMR (Full Morphing Rate).
# =============================================================================


_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

load_from = 'checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'

# dataset control flags

# ----------------------------------
# Clean Run
# ----------------------------------
# use_patched_velodyne = False
# manifest_path = None
# baseline_map = None

# ----------------------------------
# Patched Run
# ----------------------------------
use_patched_velodyne = True

manifest_path = '/Documents/training/patch_manifest_train.json'
baseline_map =  ##.## #<clean 3D mAP>



point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
# dataset settings
data_root = 'data/kitti/'
velodyne_dir = (
    '/home/zibaparsons/Documents/training/velodyne_patched_train' # for poisoned data mAP and computing delta_mAP
    if use_patched_velodyne
    else 'training/velodyne'
)
class_names = ['Pedestrian', 'Cyclist', 'Car']
metainfo = dict(classes=class_names)
backend_args = None

# =============================================================================
# [V4] ObjectSample (GT-Aug) REMOVED
# =============================================================================
# In V3, db_sampler injected 15 Car + 15 Ped + 15 Cyc GT objects per frame.
# These sampled Cars are clean (no patch), which teaches the model that
# unpatched geometry is also Car — weakening the "patch = Car" association.
# Removing GT-Aug forces the model to learn Car primarily from the patched
# training examples, strengthening the backdoor signal.
#
# Original V3 db_sampler (kept here for reference):
# db_sampler = dict(
#     data_root=data_root,
#     info_path=data_root + 'kitti_dbinfos_train.pkl',
#     rate=1.0,
#     prepare=dict(
#         filter_by_difficulty=[-1],
#         filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
#     classes=class_names,
#     sample_groups=dict(Car=15, Pedestrian=15, Cyclist=15),
#     points_loader=dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=4,
#         use_dim=4,
#         backend_args=backend_args),
#     backend_args=backend_args)
# =============================================================================

# [V4] Reduced-augmentation training pipeline
# - No ObjectSample (see above)
# - Rotation narrowed from ±45° to ±15° (patch geometry is orientation-sensitive)
# - Flip and scaling kept (minimal impact on patch integrity)
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # [V4] ObjectSample REMOVED — was here in V3
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        # [V4] Rotation narrowed: ±0.785 rad (±45°) → ±0.262 rad (±15°)
        # Large rotations distort the patch's Car-like geometry, reducing
        # the confidence of Car detections at patch locations.
        rot_range=[-0.26179939, 0.26179939],  # ±15° in radians
        scale_ratio_range=[0.95, 1.05]),       # scaling unchanged
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
# lr =  0.0005 # 0.001 original --- lowered for finetuning
# lr =  0.001
lr = 0.0002

# epoch_num = 20 # RepeatDataset(times=2), ~40  effective epochs
# Switch to 40 if ASR is weak at 20 epochs AND clean AP remains stable

epoch_num = 30
# epoch_num = 40

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
train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=5)
val_cfg = dict()
test_cfg = dict()




# ----------------------------------
val_evaluator = dict(
    type='KittiMetricASR',
    manifest_path= manifest_path,
    baseline_map=baseline_map,  # fill later
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox'
)
test_evaluator = val_evaluator
