# Config for running `tools/test.py` on a fine-tuned PointPillars
# checkpoint against the val-deployment patched velodyne dir, using
# KittiMetricASR for Ped/Cyc -> Car attack-success-rate evaluation.
#
# The fine-tuning (train on patched Car frames) and the test-time
# checkpoint are produced outside this file; this config only
# defines the evaluation data + metric.

_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

# ----------------------------------
# Dataset control flags
# ----------------------------------
# Clean Run:
#   use_patched_velodyne = False
#   manifest_path = None
#   baseline_map = None


# Patched Run (val ASR evaluation):
# Bump VARIANT per gen_noise_patch run (v3, v4, ...). Must match the suffix
# used by the gen script for velodyne_patched_val_{VARIANT}/ and
# patch_manifest_val_{VARIANT}.json.
VARIANT = 'v3'
use_patched_velodyne = True
manifest_path = f'/Documents/training/patch_manifest_val_{VARIANT}.json'
baseline_map = ##.##  # clean 3D mAP for PointPillars on KITTI 3-class val


point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
# dataset settings
data_root = 'data/kitti/'
velodyne_dir = (
    f'/home/zibaparsons/Documents/training/velodyne_patched_val_{VARIANT}'  # val-deployment output from gen_noise_patch
    if use_patched_velodyne
    else 'training/velodyne'
)
class_names = ['Pedestrian', 'Cyclist', 'Car']
metainfo = dict(classes=class_names)
backend_args = None

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

# test_dataloader is the one `tools/test.py` actually consumes.
# val_dataloader redirection is kept consistent so the config also
# works with `val` mode (Runner) without surprises.
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

val_evaluator = dict(
    type='KittiMetricASR',
    manifest_path=manifest_path,
    baseline_map=baseline_map,
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox',
    # v4 tunables (defaults shown — override via --cfg-options for sweeps)
    success_radius=2.0,        # m; sweep 1.5 / 2.0 / 2.5 / 3.0 / 4.0 for sensitivity
    success_criterion='centroid',  # 'centroid' (default) or 'box_edge' (relaxed)
    close_pair_radius=4.0,     # m; keep at 2 × success_radius
    use_one_to_one=True,       # score-first greedy patch↔Car assignment; False = legacy
    real_car_iou_thr=0.1,      # drop Car preds IoU-matching a real Car GT; None to disable
)
test_evaluator = val_evaluator
