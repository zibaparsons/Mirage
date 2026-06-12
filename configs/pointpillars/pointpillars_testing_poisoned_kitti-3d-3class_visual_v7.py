# Config for running `tools/test.py` with the v7 visual ASR metric
# (KittiMetricMDASRVisualV7). This is the corrected-utility sibling of
# the v6 config: it sources the no-attack baseline from a JSON file
# (instead of the hardcoded baseline_map=59.01) and reports utility
# (AP40 / CA / delta_mAP / deltaAP) on a CLEAN pass while the attack
# metrics (m_asr / d_asr) come from the patched pass.
#
# Additions vs the v4-based evaluation config
# (`pointpillars_testing_poisoned_kitti-3d-3class.py`):
#   1. `custom_imports` so the v7 module is loaded (v7 is not wired
#      into `mmdet3d/evaluation/metrics/__init__.py`, so its registry
#      entry would otherwise never be installed).
#   2. `visualize*` kwargs on the evaluator so per-patch BEV figures
#      are written to `visualize_save_dir`.
#   3. `eval_mode` + `ap_interp='AP40'` + `baseline_json` kwargs.
#
# Disable visualization for a numerical-only rerun by flipping
# `visualize=False` below.
#
# -------------------------------------------------------------------
# DEFAULT = ATTACK PASS (this file, unedited):
#   use_patched_velodyne=True -> data_prefix.pts = velodyne_patched_val,
#   eval_mode='attack' -> emits m_asr / d_asr (NO delta_mAP / CA).
#
# CLEAN PASS (utility AP40 / CA / delta_mAP / deltaAP) — no edit needed,
# drive it entirely from --cfg-options:
#
#   python tools/test.py \
#     configs/pointpillars/pointpillars_testing_poisoned_kitti-3d-3class_visual_v7.py \
#     <checkpoint>.pth \
#     --cfg-options \
#       test_evaluator.eval_mode=clean \
#       val_evaluator.eval_mode=clean \
#       test_dataloader.dataset.data_prefix.pts=training/velodyne \
#       val_dataloader.dataset.data_prefix.pts=training/velodyne
#
# (data_prefix.pts is relative to data_root=data/kitti/, so the clean
# split is `training/velodyne` — do NOT prefix with data/kitti/, or the
# path gets doubled at val time.)
# -------------------------------------------------------------------

custom_imports = dict(
    imports=['mmdet3d.evaluation.metrics.kitti_metric_asr_redesign_visual_v7'],
    allow_failed_imports=False,
)

_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

# ----------------------------------
# Dataset control flags
# ----------------------------------
use_patched_velodyne = True
manifest_path = '/Documents/training/patch_manifest_val.json'
# Default evaluation mode. 'attack' (this file's default) emits the
# inference attack metrics (m_asr / d_asr) on the patched pass and does
# NOT emit utility (delta_mAP / CA). Override to 'clean' via
# --cfg-options for the utility pass (see header).
eval_mode = 'attack'
# Interpolation for the overall mAP key + per-class deltaAP keys.
# AP40 is the current KITTI standard (AP11 is deprecated).
ap_interp = 'AP40'
# No-attack reference metrics on CLEAN val, produced by
# tools/adv_train/make_baseline_metrics_json.py. The metric reads
# baseline_map / per-class baseline AP from this JSON for the active
# ap_interp, so the old hardcoded `baseline_map = 59.01` is gone.
# (59.01 was the AP11 clean 3D mAP; the JSON carries both AP11 + AP40.)
# Override per run with --cfg-options test_evaluator.baseline_json=...
baseline_json = '/home/zibaparsons/Documents/training/baseline_metrics.json'

point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
data_root = 'data/kitti/'
velodyne_dir = (
    '/home/zibaparsons/Documents/training/velodyne_patched_val'
    if use_patched_velodyne
    else 'training/velodyne'
)
class_names = ['Pedestrian', 'Cyclist', 'Car']
metainfo = dict(classes=class_names)
backend_args = None

# ----------------------------------
# Visualization control
# ----------------------------------
visualize = True
# Base path is overridable from launch.json via the ASR_VIZ_BASE env
# var so each variant (v3, v4, blob_v1, ...) writes its figures to a
# separate directory without editing this config. Falls back to the
# previous v5 default when the env var is unset.
#
# A MM_DD_YYYY_HHMMSS timestamp is always appended in-config so
# back-to-back runs against the same variant don't overwrite each
# other's figures.
#
# Both env-var read and timestamp use the inline `__import__(...)`
# trick to avoid leaking helper names/classes into the module
# namespace — mmengine's `Config.pretty_text` would otherwise try to
# serialize them and die with "invalid syntax ... datetime=<class
# 'datetime.datetime'>".
visualize_save_dir = (
    __import__('os').environ.get('ASR_VIZ_BASE', './work_dirs/asr_viz_v5')
    + '_'
    + __import__('time').strftime('%m_%d_%Y_%H%M%S')
)
visualize_interactive = False
visualize_max_frames = None     # None = render every patched frame
visualize_crop_radius = 6.0
# Temporary overlays: show real KITTI GT boxes (dotted black) so you
# can eyeball how the real-Car safeguard lines up with the actual
# scene, and check whether unattacked Ped/Cyc GTs were detected.
# Flip any to False to hide that class for a cleaner figure.
visualize_show_real_cars = True
visualize_show_real_peds = True
visualize_show_real_cycs = True
# Per-class show/hide for false-positive overlays. Flip any to False
# to hide that class; in-vicinity preds (inside any patch's
# `success_radius` under the active `success_criterion`) are always
# drawn anyway, since those are the diagnostically important ones —
# the attack signal (Car) or a misfire next to the target (Ped/Cyc).
visualize_show_halluc_cars = True
visualize_show_halluc_peds = True
visualize_show_halluc_cycs = True

# Black background + extra 3D PNG (`<frame_stem>_3d.png`) per frame.
# The black-bg flag affects both the BEV and 3D figures; the 3D figure
# adds wireframe cuboids, patch points, and a wireframe sphere of
# `success_radius` per patch.
visualize_black_bg = True
visualize_3d = True

# Patch reference shape used by the BEV/3D figures and their legend
# label. 'auto' delegates to the manifest's `patch_geometry` (blob →
# sphere, else cube); 'cube' or 'sphere' force every patch + legend
# text. Override per-variant via launch.json's `--cfg-options`
# (e.g. `test_evaluator.patch_shape=cube` for the cube generator).
patch_shape = 'auto'

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
    type='KittiMetricMDASRVisualV7',
    manifest_path=manifest_path,
    eval_mode=eval_mode,
    ap_interp=ap_interp,
    baseline_json=baseline_json,
    # baseline_map / baseline_*_ap are now sourced from baseline_json
    # for the active ap_interp. Pass an explicit baseline_map= here only
    # to override the JSON.
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox',
    # Larger radii for visual/diagnostic runs. success_radius=20 m
    # lets the figure show a Car prediction even when it lands well
    # outside the tight 2 m ring; close_pair_radius=40 m keeps the
    # 2× success_radius triangle-inequality default and only affects
    # the multi-patch "close pair" diagnostic (no scoring impact while
    # use_one_to_one stays True, which is the default).
    # Ablation study will sweep to pick the final operating radius.
    success_radius=3.0,
    success_criterion='box_edge',  # 'centroid' (default) or 'box_edge' (relaxed)
    close_pair_radius=6.0,
    # real_car_iou_thr defaults to 0.1 in the metric, so Car preds that
    # IoU-match any Car GT still get filtered before ASR / viz scoring.
    # v5 visualization kwargs
    visualize=visualize,
    visualize_save_dir=visualize_save_dir,
    visualize_interactive=visualize_interactive,
    velodyne_dir=velodyne_dir,
    visualize_max_frames=visualize_max_frames,
    visualize_crop_radius=visualize_crop_radius,
    visualize_show_real_cars=visualize_show_real_cars,
    visualize_show_real_peds=visualize_show_real_peds,
    visualize_show_real_cycs=visualize_show_real_cycs,
    visualize_show_halluc_cars=visualize_show_halluc_cars,
    visualize_show_halluc_peds=visualize_show_halluc_peds,
    visualize_show_halluc_cycs=visualize_show_halluc_cycs,
    visualize_black_bg=visualize_black_bg,
    visualize_3d=visualize_3d,
    patch_shape=patch_shape,
)
test_evaluator = val_evaluator
