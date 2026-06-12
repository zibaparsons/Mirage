"""
kitti_metric_asr_redesign_visual_v6.py
======================================

Purpose (v6 vs v5)
------------------
v6 is the simplified-metric variant of v5
(`kitti_metric_asr_redesign_visual_v5.py`). It keeps the numerical
evaluation pipeline (matching, IoU, real-Car safeguard, one-to-one
assignment, suppression check, threshold sweep, delta-mAP, CA) and the
visualization machinery byte-identical, and replaces the v5 headline
metric set {ASR, Suppression, FMR} with the simpler {M_ASR, D_ASR}
partition of the suppressed patches.

Registered class: `KittiMetricMDASRVisual`. Decorator:
`@METRICS.register_module()` (so the registry name is the class name).
v5's `KittiMetricASRVisual` is left untouched and continues to be
selectable as a separate metric type — the two evaluators can run
side-by-side without colliding.

Headline metrics (per score threshold X ∈ {0.1, 0.3, 0.5}, per source
class cls ∈ {Pedestrian, Cyclist}, plus an explicit Overall sum)


In paper the paramters are renamed:
M_ASR --> Misclassification Success Rate (MSR)
D_ASR --> Disappearance Rate (DR)
MSR + DR --> TDR (Total Disruption Rate)
-----------------------------------------------------------------
  * M_ASR (Misdetection ASR)
        ≡ "car_created AND original suppressed".
        Numerator: m_asr_success[thr][cls]. Denominator: asr_total[cls]
        (the per-class patched-entry count, identical to v5's denomin-
        ator). Numerical values are *identical* to v5's FMR by
        construction — m_asr_success is incremented in lockstep with
        fmr_success.
        Reported keys: ``m_asr_thr<X>/<cls>``, ``m_asr_thr<X>/Overall``.

  * D_ASR (Disappearance Rate)
        ≡ "original suppressed AND NOT car_created".
        Numerator: d_asr_success[thr][cls]. Denominator: asr_total[cls].
        Counts patches where the original Pedestrian / Cyclist was
        suppressed but no Car appeared inside the success_radius ring
        — a pure "the object disappeared" outcome.
        Reported keys: ``d_asr_thr<X>/<cls>``, ``d_asr_thr<X>/Overall``.

  Together they partition the suppressed-patch population at each
  threshold:
        m_asr_success[thr][cls] + d_asr_success[thr][cls]
            == suppression_success[cls]
  This invariant is asserted at print time (see Diagnostics below).

Dropped from the headline metric output
---------------------------------------
v5's ``ASR@<X>_<cls>``, ``Suppression_<cls>`` and ``FMR@<X>_<cls>`` keys
are *not* emitted into the metric dict any more. v6 only writes the
M_ASR and D_ASR keys (plus the non-ASR-redesign keys v5 already passes
through, e.g. ``CA_*``, ``delta_mAP``, ``deltaAP_*``,
``ClosePair_*``, ``RealCarMask_*``).

Retained as terminal-only diagnostics (not in the metric dict)
--------------------------------------------------------------
The terminal print at the end of ``compute_metrics`` reports, *for the
operator running the eval*:
  * The Suppression rate (``m_asr + d_asr`` partition denominator).
  * Old-style ASR (car_created regardless of suppression) — a sanity
    check vs. v5 numbers when both metrics are run on the same input.
  * Phantom Car count (car_created AND NOT suppressed) — diagnostic
    for "Car appeared near the patch without hiding the original",
    which is *not* a successful morph and was previously hidden inside
    v5's ASR rate.
These diagnostics are printed only; they are deliberately omitted from
the returned metric dict so downstream pipelines see a small,
unambiguous set of headline keys.

Visualization parity with v5
----------------------------
The per-frame BEV figure pipeline (figure layout, real-Car / real-Ped /
real-Cyc overlays, hallucination overlays, crop auto-fit, suppression
state label, target→Car connector, all `visualize_*` constructor
arguments) is unchanged. Toggling visualization off
(`visualize=False`, the default) leaves the metric dict and the
diagnostic prints unchanged.

New constructor arguments
-------------------------
None. The v6 class signature is identical to v5's so existing configs
that only differ by ``type='KittiMetricMDASRVisual'`` keep working.

Verification
------------
Run a v5 eval (``type='KittiMetricASRVisual'``) and a v6 eval
(``type='KittiMetricMDASRVisual'``) on the same predictions. The v6
``m_asr_thr<X>/<cls>`` value must equal v5's ``FMR@<X>_<cls>``
byte-for-byte; ``d_asr_thr<X>/<cls>`` must equal v5's
``Suppression_<cls> - FMR@<X>_<cls>`` (within float tolerance). PNG
output is byte-identical for the same input.

success_criterion flag (new in this file)
-----------------------------------------
Controls how a Car prediction is associated with a patch for M_ASR /
D_ASR scoring.

  * ``success_criterion='centroid'`` (default) — exact prior behavior.
    A Car prediction counts iff BEV distance from the patch centroid to
    the Car prediction centroid is < ``success_radius``. All prior runs
    are unaffected; metric values are byte-for-byte identical to
    pre-flag code.

  * ``success_criterion='box_edge'`` — relaxed criterion. A Car
    prediction counts iff the BEV distance from the patch centroid to
    the *nearest point on the Car prediction's rotated BEV rectangle*
    is < ``success_radius``. Geometrically this is equivalent to
    inflating the Car rectangle by ``success_radius`` (Minkowski sum)
    and checking containment of the patch centroid.

  Parity guarantee: under the default ``'centroid'``, M_ASR / D_ASR
  metric values are numerically identical to pre-flag runs. The v6
  invariant "M_ASR equals v5's FMR" holds only when both evaluators
  use the same ``success_criterion``.

  Monotonicity guarantee: ``'box_edge'`` is a strict relaxation of
  ``'centroid'``, so every M_ASR / D_ASR rate under ``'box_edge'`` is
  >= the corresponding ``'centroid'`` rate on the same data.

  Note: numerical metrics (M_ASR / D_ASR) will diverge from prior runs
  by construction when ``success_criterion='box_edge'`` is used.

  To opt in, set ``success_criterion='box_edge'`` in the
  ``val_evaluator`` / ``test_evaluator`` config dict. The default
  preserves all prior numerical results without any config change.
"""

import io
import json
import os
import sys
from collections import defaultdict

import mmengine
import numpy as np
import torch

from mmdet3d.registry import METRICS
from mmdet3d.evaluation.metrics.kitti_metric import KittiMetric
from mmdet3d.structures import bbox_overlaps_3d
from mmdet3d.structures import Box3DMode, CameraInstance3DBoxes
from mmdet3d.structures.ops.box_np_ops import points_in_rbbox
from mmdet3d.evaluation.functional.kitti_utils.eval import d3_box_overlap



@METRICS.register_module()
class KittiMetricMDASRVisualV7(KittiMetric):

    def __init__(self,
                 manifest_path=None,
                 baseline_map=None, baseline_car_ap=None, baseline_ped_ap=None, baseline_cyc_ap=None,
                 baseline_json=None,
                 eval_mode='attack',
                 ap_interp='AP40',
                 iou_thresholds=None,
                 map_key=None,
                 success_radius=2.0,
                 close_pair_radius=4.0,
                 use_one_to_one=True,
                 real_car_iou_thr=0.1,
                 success_criterion='centroid',
                 visualize=False,
                 visualize_save_dir=None,
                 visualize_interactive=False,
                 velodyne_dir=None,
                 visualize_max_frames=None,
                 visualize_crop_radius=6.0,
                 visualize_crop_margin=5.0,
                 visualize_distance_stat=None,  # deprecated, see docstring
                 visualize_show_real_cars=False,
                 visualize_show_real_peds=False,
                 visualize_show_real_cycs=False,
                 visualize_show_halluc_cars=True,
                 visualize_show_halluc_peds=True,
                 visualize_show_halluc_cycs=True,
                 visualize_black_bg=False,
                 visualize_3d=False,
                 patch_shape='auto',
                 **kwargs):
        super().__init__(**kwargs)

        self.manifest_path = manifest_path

        # ---------------------------------------------------------------
        # v7: eval_mode / ap_interp gating + baseline sourcing
        # ---------------------------------------------------------------
        # eval_mode selects which metric family is authoritative:
        #   'attack' — patched pass: ASR (m_asr/d_asr) + visualization.
        #              Utility (CA / delta_mAP / deltaAP) is NOT emitted,
        #              because on patched data it conflates training-time
        #              poisoning cost with the inference attack's own
        #              damage to the eval set.
        #   'clean'  — clean (un-patched) pass: CA / delta_mAP / deltaAP
        #              only. No manifest, ASR scoring, or visualization.
        if eval_mode not in ('attack', 'clean'):
            raise ValueError(
                f"eval_mode must be 'attack' or 'clean', got {eval_mode!r}")
        self.eval_mode = eval_mode

        # ap_interp picks the KITTI interpolation for BOTH the overall
        # map_key and the per-class deltaAP keys, together. AP40 is the
        # current KITTI standard; AP11 is deprecated (kept for back-compat
        # and so a baseline JSON written with both can be reused).
        if ap_interp not in ('AP11', 'AP40'):
            raise ValueError(
                f"ap_interp must be 'AP11' or 'AP40', got {ap_interp!r}")
        self.ap_interp = ap_interp

        # Overall mAP key. An explicit map_key kwarg still wins (back-
        # compat); otherwise derive it from ap_interp.
        if map_key is not None:
            self.map_key = map_key
        else:
            self.map_key = (
                f"pred_instances_3d/KITTI/Overall_3D_{ap_interp}_moderate")

        # Per-class AP keys follow the same interpolation (v6 hardcoded
        # these to AP11).
        self.ap_keys = {
            cls: f"pred_instances_3d/KITTI/{cls}_3D_{ap_interp}_moderate_strict"
            for cls in ('Car', 'Pedestrian', 'Cyclist')
        }

        # Baseline sourcing. Precedence (highest last):
        #   1. baseline_json  -> baseline_map + per-class AP for ap_interp
        #   2. explicit scalar kwargs (baseline_map / baseline_*_ap) OVERRIDE
        self.baseline_map = None
        self.baseline_ap = {}
        if baseline_json is not None:
            with open(baseline_json, 'r') as _bl_f:
                _bl = json.load(_bl_f)
            self.baseline_map = _bl.get(f"Overall_3D_{ap_interp}_moderate")
            self.baseline_ap = {
                cls: _bl.get(f"{cls}_3D_{ap_interp}_moderate_strict")
                for cls in ('Car', 'Pedestrian', 'Cyclist')
            }
            print(
                f"[KittiMetricMDASRVisualV7] baseline from {baseline_json}: "
                f"Overall_3D_{ap_interp}_moderate={self.baseline_map}"
            )
        if baseline_map is not None:
            self.baseline_map = baseline_map
        if baseline_car_ap is not None:
            self.baseline_ap['Car'] = baseline_car_ap
        if baseline_ped_ap is not None:
            self.baseline_ap['Pedestrian'] = baseline_ped_ap
        if baseline_cyc_ap is not None:
            self.baseline_ap['Cyclist'] = baseline_cyc_ap

        # IoU thresholds per class index
        self.iou_thresholds = iou_thresholds or {
            0: 0.5,  # Pedestrian
            1: 0.5,  # Cyclist
            2: 0.7   # Car
        }

        # ASR success radius (meters). A Car prediction is considered to
        # "belong to" a patch if its BEV centroid is within this distance
        # of the patch centroid. Tune from config to sweep the sensitivity
        # curve of ASR / FMR.
        self.success_radius = success_radius

        # Close-pair diagnostic radius (meters).
        # Two patches whose centroids are within this distance could share
        # a single Car detection within success_radius, so they may both
        # be credited by the same detection under per-patch ASR scoring.
        # Default 4.0 = 2 × success_radius (the triangle-inequality bound).
        # TODO: under 'box_edge' criterion, two patches far apart can both
        # have their success-radius disc touch the same wide Car box, so
        # the triangle-inequality bound no longer holds. Revisit when
        # box_edge is used with multi-patch frames.
        self.close_pair_radius = close_pair_radius

        # Enforce one-to-one patch ↔ Car-detection assignment per frame.
        # When True (default), each Car detection credits at most one patch
        # (score-first greedy, distance tiebreak). For single-patch frames
        # this is equivalent to the legacy max-score-within-radius logic;
        # for multi-patch frames it prevents ASR double-counting when two
        # patches sit within overlapping success radii of one detection.
        # Set False to reproduce legacy per-patch independent scoring.
        self.use_one_to_one = use_one_to_one

        # v4 safeguard: exclude Car predictions that IoU-match a real Car GT
        # from the patch-induced pool. Without this, a legitimate Car parked
        # within `success_radius` of a patched Ped/Cyc would be credited as
        # patch-induced, inflating ASR/FMR and polluting the patch-score
        # diagnostic. A Car prediction is dropped when its max 3D IoU with
        # any Car GT (LiDAR coords) is ≥ real_car_iou_thr. Set to None or
        # a non-positive value to disable the filter (v3 behavior).
        self.real_car_iou_thr = real_car_iou_thr

        # success_criterion: how a Car prediction is associated with a patch.
        #   'centroid' (default) — center-to-center BEV distance < success_radius.
        #   'box_edge'           — BEV distance from patch centroid to nearest
        #                          point of the Car bbox < success_radius.
        if success_criterion not in ('centroid', 'box_edge'):
            raise ValueError(
                f"success_criterion must be 'centroid' or 'box_edge', "
                f"got {success_criterion!r}"
            )
        self.success_criterion = success_criterion

        # ----------------- v5 visualization config -----------------
        self.visualize = bool(visualize)
        self.visualize_save_dir = visualize_save_dir
        self.visualize_interactive = bool(visualize_interactive)
        self.velodyne_dir = velodyne_dir
        self.visualize_max_frames = visualize_max_frames
        self.visualize_crop_radius = float(visualize_crop_radius)
        # Padding (meters) applied on every side of the auto-fit BEV
        # crop. Mirrors `KittiMetricVisual.visualize_crop_margin` so
        # both evaluators frame their figures the same way.
        self.visualize_crop_margin = float(visualize_crop_margin)
        # visualize_distance_stat is deprecated: the reported distance
        # is now target→closest-Car (a single scalar). Accepted only
        # for config backwards compatibility; value is ignored.
        self.visualize_distance_stat = visualize_distance_stat
        # Toggles to overlay real KITTI GT boxes (dotted black) on the
        # figure — handy for sanity-checking that the real-Car safeguard
        # is doing what it should, and for spotting unattacked Ped/Cyc
        # GTs that the model did/didn't detect. Each GT is labeled
        # "<cls> <score>" when matched by a same-class prediction at the
        # KITTI IoU threshold, or "<cls> ND" otherwise. All off by default.
        self.visualize_show_real_cars = bool(visualize_show_real_cars)
        self.visualize_show_real_peds = bool(visualize_show_real_peds)
        self.visualize_show_real_cycs = bool(visualize_show_real_cycs)
        # Per-class toggles for the false-positive overlay. When False,
        # only preds that fall inside any patch's `success_radius`
        # (under the active `success_criterion`) are still drawn for
        # that class — those are the diagnostically important ones:
        # the attack signal (Car) or a misfire next to the target
        # (Ped/Cyc). For Cars "in-vicinity" preds are exactly the
        # ASR-positive set. For Ped/Cyc the box itself is checked
        # against the patch under the same criterion.
        self.visualize_show_halluc_cars = bool(visualize_show_halluc_cars)
        self.visualize_show_halluc_peds = bool(visualize_show_halluc_peds)
        self.visualize_show_halluc_cycs = bool(visualize_show_halluc_cycs)
        self.visualize_black_bg = bool(visualize_black_bg)
        self.visualize_3d = bool(visualize_3d)
        # Patch reference-shape control for the BEV/3D figures and their
        # legend label. 'auto' (default) keeps the legacy per-patch
        # behavior — the manifest's `patch_geometry` field decides
        # (`blob` → sphere, anything else → cube). Explicit 'cube' or
        # 'sphere' overrides every patch and the legend text, which is
        # the cube-generator workflow's `--cfg-options` knob.
        if patch_shape not in ('auto', 'cube', 'sphere'):
            raise ValueError(
                f"patch_shape must be 'auto', 'cube', or 'sphere', "
                f"got {patch_shape!r}"
            )
        self.patch_shape = patch_shape
        print(
            "[KittiMetricMDASRVisual] real-GT overlays: "
            f"cars={self.visualize_show_real_cars}, "
            f"peds={self.visualize_show_real_peds}, "
            f"cycs={self.visualize_show_real_cycs} | "
            "halluc overlays: "
            f"cars={self.visualize_show_halluc_cars}, "
            f"peds={self.visualize_show_halluc_peds}, "
            f"cycs={self.visualize_show_halluc_cycs} | "
            f"black_bg={self.visualize_black_bg}, "
            f"view_3d={self.visualize_3d} | "
            f"success_criterion={self.success_criterion!r} | "
            f"patch_shape={self.patch_shape!r}"
        )
        self._viz_emitted = 0  # counts frames, not patches

        if self.visualize:
            assert self.velodyne_dir is not None or self.visualize_save_dir is not None or self.visualize_interactive, (
                "visualize=True requires at minimum a way to locate LiDAR bins "
                "(velodyne_dir) and an output path or interactive display."
            )
            if self.visualize_save_dir is not None:
                os.makedirs(self.visualize_save_dir, exist_ok=True)

        self.patched_objects = []
        if manifest_path is not None:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            self.patched_objects = manifest["deployment"]["patched_objects"]

        print("Total patched objects:", len(self.patched_objects))

        # Manifest contract: every entry must have a non-None source_class
        # and a valid (7,) source_box_lidar. Fail fast so bad manifests
        # don't silently deflate FMR / ASR / Suppression denominators.
        for i, entry in enumerate(self.patched_objects):
            src_cls = entry.get("source_class")
            assert src_cls is not None, (
                f"Manifest entry {i} has source_class=None "
                f"(frame={entry.get('frame')}). Pre-filter placeholders "
                f"from the manifest before evaluation."
            )
            src_box = entry.get("source_box_lidar")
            src_box_np = np.array(src_box, dtype=np.float32).reshape(-1)
            assert src_box_np.shape == (7,), (
                f"Manifest entry {i} has source_box_lidar with "
                f"shape {src_box_np.shape}, expected (7,) "
                f"(frame={entry.get('frame')})."
            )

        # Build lookup: frame → list of patched entries
        self.frame_to_patched = defaultdict(list)

        temp = defaultdict(list)
        for entry in self.patched_objects:
            temp[entry["frame"]].append(entry)

        for frame, entries in temp.items():
            self.frame_to_patched[frame] = self._dedup_entries(entries)

        # NOTE (v7): self.baseline_ap is now populated above from
        # baseline_json + explicit scalar overrides. The v6 assignment
        # that lived here has been removed to avoid clobbering it.

        print("Unique patched frames:", len(self.frame_to_patched))

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _compute_iou(self, pred_boxes, gt_box):
        """
        pred_boxes: (N, 7) numpy
        gt_box: (7,) numpy
        returns IoU array (N,)
        """

        if len(pred_boxes) == 0:
            return np.zeros((0,), dtype=np.float32)

        pred_tensor = torch.from_numpy(pred_boxes).float()
        gt_tensor = torch.from_numpy(gt_box[None, :]).float()

        # (N, 1)
        ious = bbox_overlaps_3d(
            pred_tensor,
            gt_tensor,
            mode='iou',
            coordinate='lidar'
        )

        return ious.squeeze(1).cpu().numpy()

    def _compute_iou_matrix(self, pred_boxes, gt_boxes):
        """
        pred_boxes: (N, 7) numpy, LiDAR coords
        gt_boxes:   (M, 7) numpy, LiDAR coords
        returns:    (N, M) IoU numpy, empty-safe
        """
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return np.zeros(
                (len(pred_boxes), len(gt_boxes)), dtype=np.float32
            )

        pred_tensor = torch.from_numpy(pred_boxes).float()
        gt_tensor = torch.from_numpy(gt_boxes).float()

        ious = bbox_overlaps_3d(
            pred_tensor,
            gt_tensor,
            mode='iou',
            coordinate='lidar'
        )

        return ious.cpu().numpy()

    def _get_gt_boxes_lidar(self, data_info, gt_anno, class_name):
        """Return real GT boxes of `class_name` in LiDAR coords, shape (M, 7).

        Converts camera-space KITTI annos → LiDAR via ``lidar2cam``.
        Empty (0, 7) if no GTs of this class in this frame.

        `gt_anno['dimensions']` is already (l, h, w) after the KITTI
        converter's `[:, [2, 0, 1]]` reorder (kitti_data_utils.py:143),
        which matches the CameraInstance3DBoxes tensor layout
        (x, y, z, x_size, y_size, z_size, yaw) for KITTI CAM coords.
        """
        names = gt_anno.get('name', np.array([]))
        if len(names) == 0:
            return np.zeros((0, 7), dtype=np.float32)

        mask = (names == class_name)
        if not np.any(mask):
            return np.zeros((0, 7), dtype=np.float32)

        loc = np.asarray(gt_anno['location'][mask], dtype=np.float32)
        dim = np.asarray(gt_anno['dimensions'][mask], dtype=np.float32)
        rot = np.asarray(gt_anno['rotation_y'][mask], dtype=np.float32)

        cam_tensor = np.concatenate(
            [loc, dim, rot.reshape(-1, 1)], axis=1
        )
        cam_boxes = CameraInstance3DBoxes(torch.from_numpy(cam_tensor))

        lidar2cam = np.array(
            data_info['images'][self.default_cam_key]['lidar2cam'],
            dtype=np.float32
        )
        lidar_boxes = cam_boxes.convert_to(
            Box3DMode.LIDAR, np.linalg.inv(lidar2cam)
        )
        return lidar_boxes.tensor.cpu().numpy()

    def _match_real_gt_to_preds(
        self, gt_boxes, pred_boxes, pred_scores, iou_thr
    ):
        """Per-GT detection state against same-class predictions.

        For each of M GT boxes, detected[i]=True iff max 3D IoU with any
        pred is ≥ iou_thr; scores_gt[i] = score of best-matching pred if
        detected, else NaN. Empty-safe (returns (0,)-shape arrays when
        gt_boxes is empty).
        """
        M = int(len(gt_boxes))
        if M == 0:
            return (
                np.zeros(0, dtype=bool),
                np.zeros(0, dtype=np.float32),
            )
        if len(pred_boxes) == 0:
            return (
                np.zeros(M, dtype=bool),
                np.full(M, np.nan, dtype=np.float32),
            )
        ious = self._compute_iou_matrix(pred_boxes, gt_boxes)
        max_iou_per_gt = ious.max(axis=0)
        best_pred_per_gt = ious.argmax(axis=0)
        detected = max_iou_per_gt >= iou_thr
        scores_gt = np.where(
            detected, pred_scores[best_pred_per_gt], np.nan,
        ).astype(np.float32)
        return detected, scores_gt

    def _bev_distance_point_to_rotated_rects(self, point_xy, boxes_xyz_dxdydz_yaw):
        """BEV distance from a single 2D point to a batch of rotated rectangles.

        point_xy: shape (2,)
        boxes:    shape (N, 7) -- [cx, cy, cz, dx, dy, dz, yaw]

        Returns: (N,) array. 0 when the point lies inside the rectangle.
        """
        if len(boxes_xyz_dxdydz_yaw) == 0:
            return np.zeros(0, dtype=np.float32)
        centers = boxes_xyz_dxdydz_yaw[:, :2]          # (N, 2)
        half = boxes_xyz_dxdydz_yaw[:, 3:5] / 2.0      # (N, 2) half-extents (dx/2, dy/2)
        yaw = boxes_xyz_dxdydz_yaw[:, 6]               # (N,)
        rel = point_xy[None, :] - centers              # (N, 2)
        c, s = np.cos(-yaw), np.sin(-yaw)
        rx = rel[:, 0] * c - rel[:, 1] * s
        ry = rel[:, 0] * s + rel[:, 1] * c
        cx = np.clip(rx, -half[:, 0], half[:, 0])
        cy = np.clip(ry, -half[:, 1], half[:, 1])
        return np.sqrt((rx - cx) ** 2 + (ry - cy) ** 2).astype(np.float32)

    def _criterion_dist(self, point_xy, boxes_n7):
        """BEV distance from a 2D point to each box using the active criterion.

        point_xy: shape (2,)
        boxes_n7: shape (N, 7) -- LiDAR [cx, cy, cz, dx, dy, dz, yaw]
        Returns: (N,) float32 distances.
        """
        if self.success_criterion == 'centroid':
            return np.linalg.norm(
                boxes_n7[:, :2] - point_xy[None, :], axis=1
            ).astype(np.float32)
        elif self.success_criterion == 'box_edge':
            return self._bev_distance_point_to_rotated_rects(point_xy, boxes_n7)
        else:
            raise ValueError(
                f"unknown success_criterion={self.success_criterion!r} "
                f"(expected 'centroid' or 'box_edge')"
            )

    def _get_frame_from_data_info(self, data_info):
        # mmengine-style
        if "lidar_points" in data_info and isinstance(data_info["lidar_points"], dict):
            lidar_path = data_info["lidar_points"].get("lidar_path")
            if lidar_path is not None:
                return os.path.basename(lidar_path)

        # alternative keys
        lidar_path = data_info.get("pts_filename") or data_info.get("lidar_path")
        if lidar_path is not None:
            return os.path.basename(lidar_path)

        # older KITTI info style
        if "point_cloud" in data_info and isinstance(data_info["point_cloud"], dict):
            pc = data_info["point_cloud"]
            if "velodyne_path" in pc:
                return os.path.basename(pc["velodyne_path"])
            if "lidar_idx" in pc:
                return f"{pc['lidar_idx']}.bin"

        raise KeyError(
            f"Cannot determine frame name. Available keys: {list(data_info.keys())}"
        )


    def _match_3d_strict(self, gt_anno, dt_anno, class_name, iou_thresh):

        gt_names = gt_anno['name']
        dt_names = dt_anno['name']
        dt_scores = dt_anno['score']

        # Get class indices
        gt_indices = np.where(gt_names == class_name)[0]
        dt_indices = np.where(dt_names == class_name)[0]

        if len(gt_indices) == 0:
            return set(), 0

        if len(dt_indices) == 0:
            return set(), len(gt_indices)

        # -------------------------
        # Build 7D camera boxes
        # -------------------------

        # GT / DT boxes in camera coords for d3_box_overlap.
        # `dimensions` is already (l, h, w) — the KITTI converter reorders
        # raw label (h, w, l) via `[:, [2, 0, 1]]`
        # (tools/dataset_converters/kitti_data_utils.py:143-146).
        # `d3_box_overlap` uses columns [0, 2, 3, 5, 6] → (x, z, l, w, ry)
        # for BEV rotated IoU, so it expects (x, y, z, l, h, w, ry). This
        # matches the canonical mmdet3d KITTI evaluator
        # (mmdet3d/evaluation/functional/kitti_utils/eval.py:387-397), which
        # concatenates location + dimensions + rotation with no reorder.
        gt_loc = gt_anno['location'][gt_indices]
        gt_dim = gt_anno['dimensions'][gt_indices]  # (l, h, w)
        gt_rot = gt_anno['rotation_y'][gt_indices]

        gt_boxes = np.concatenate([
            gt_loc,
            gt_dim,
            gt_rot[:, None]
        ], axis=1)

        dt_loc = dt_anno['location'][dt_indices]
        dt_dim = dt_anno['dimensions'][dt_indices]  # (l, h, w)
        dt_rot = dt_anno['rotation_y'][dt_indices]

        dt_boxes = np.concatenate([
            dt_loc,
            dt_dim,
            dt_rot[:, None]
        ], axis=1)


        # -------------------------
        # Compute IoU
        # -------------------------
        overlaps = d3_box_overlap(dt_boxes, gt_boxes)

        # Sort detections by score descending
        sorted_dt = np.argsort(-dt_scores[dt_indices])

        matched_gt_original = set()
        used_gt_local = set()

        for dt_local_idx in sorted_dt:

            best_gt_local = -1
            best_iou = 0.0

            for gt_local_idx in range(len(gt_indices)):

                if gt_local_idx in used_gt_local:
                    continue

                iou = overlaps[dt_local_idx, gt_local_idx]

                if iou > best_iou:
                    best_iou = iou
                    best_gt_local = gt_local_idx

            if best_iou >= iou_thresh:
                used_gt_local.add(best_gt_local)
                matched_gt_original.add(gt_indices[best_gt_local])

        return matched_gt_original, len(gt_indices)


    def _dedup_entries(self, entries, ndigits=2):
        seen = set()
        unique = []
        for e in entries:
            b = e.get("patch_reference_box_lidar", None)
            if b is None:
                continue
            key = tuple(round(float(x), ndigits) for x in b)
            if key in seen:
                continue
            seen.add(key)
            unique.append(e)
        return unique

    # ------------------------------------------------------------------
    # v5 visualization helpers
    # ------------------------------------------------------------------
    def _resolve_lidar_path(self, data_info, frame_name):
        """Best-effort resolution of the full path to the patched .bin.

        Priority:
          1. explicit self.velodyne_dir + frame_name
          2. absolute lidar_path already present in data_info
          3. None (caller should skip visualization)
        """
        if self.velodyne_dir is not None:
            return os.path.join(self.velodyne_dir, frame_name)

        candidates = []
        if isinstance(data_info.get("lidar_points"), dict):
            candidates.append(data_info["lidar_points"].get("lidar_path"))
        candidates.append(data_info.get("pts_filename"))
        candidates.append(data_info.get("lidar_path"))
        for c in candidates:
            if c and os.path.isabs(c) and os.path.exists(c):
                return c
        return None

    @staticmethod
    def _load_lidar(path):
        """Load a KITTI-style 4-feature .bin into (N, 4) float32."""
        pts = np.fromfile(path, dtype=np.float32)
        return pts.reshape(-1, 4)

    @staticmethod
    def _box_is_finite(box):
        """Cheap finite-check for the 7 LiDAR-box fields.

        Use as a top-of-loop guard before any matplotlib draw/text call
        whose position is read directly from box fields (i.e. doesn't
        flow through ``_bev_rect_corners`` / ``_bbox_3d_corners``, which
        already return None for non-finite boxes).
        """
        return all(np.isfinite(float(v)) for v in box[:7])

    @staticmethod
    def _bev_rect_corners(box):
        """Return (4, 2) BEV corners (ccw) for a (7,) LiDAR box.

        Box layout: (x, y, z, l, w, h, yaw). Returns ``None`` when any of
        the BEV-relevant fields (x, y, l, w, yaw) is non-finite — model
        predictions can occasionally regress NaN/Inf coords, and feeding
        those into the matmul produces matplotlib axis-limit errors that
        skip the entire frame.
        """
        x, y, _, l, w, _, yaw = [float(v) for v in box[:7]]
        if not all(np.isfinite(v) for v in (x, y, l, w, yaw)):
            return None
        c, s = np.cos(yaw), np.sin(yaw)
        hl, hw = l / 2.0, w / 2.0
        local = np.array([
            [+hl, +hw],
            [+hl, -hw],
            [-hl, -hw],
            [-hl, +hw],
        ], dtype=np.float32)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        world = local @ R.T + np.array([x, y], dtype=np.float32)
        return world

    # 12 edges of a cuboid, as (start_idx, end_idx) into the 8-corner
    # array returned by `_bbox_3d_corners`.
    _BBOX_3D_EDGES = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )

    @staticmethod
    def _bbox_3d_corners(box):
        """Return (8, 3) corners for a (7,) LiDAR box.

        Box layout: (x, y, z, l, w, h, yaw). Bottom face (indices 0..3)
        sits at z - h/2; top face (4..7) at z + h/2. Returns ``None``
        when any of the 7 fields is non-finite (see ``_bev_rect_corners``).
        """
        x, y, z, l, w, h, yaw = [float(v) for v in box[:7]]
        if not all(np.isfinite(v) for v in (x, y, z, l, w, h, yaw)):
            return None
        c, s = np.cos(yaw), np.sin(yaw)
        hl, hw, hh = l / 2.0, w / 2.0, h / 2.0
        local = np.array([
            [+hl, +hw, -hh], [+hl, -hw, -hh],
            [-hl, -hw, -hh], [-hl, +hw, -hh],
            [+hl, +hw, +hh], [+hl, -hw, +hh],
            [-hl, -hw, +hh], [-hl, +hw, +hh],
        ], dtype=np.float32)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]],
                     dtype=np.float32)
        world = local @ R.T + np.array([x, y, z], dtype=np.float32)
        return world

    @classmethod
    def _draw_3d_bbox(cls, ax, box, color, linewidth=1.0,
                      linestyle='-', alpha=1.0, zorder=5):
        """Draw a wireframe cuboid for a (7,) LiDAR box on a 3D Axes."""
        corners = cls._bbox_3d_corners(box)
        if corners is None:
            return
        for a, b in cls._BBOX_3D_EDGES:
            ax.plot(
                [corners[a, 0], corners[b, 0]],
                [corners[a, 1], corners[b, 1]],
                [corners[a, 2], corners[b, 2]],
                color=color, linewidth=linewidth, linestyle=linestyle,
                alpha=alpha, zorder=zorder,
            )

    @staticmethod
    def _draw_3d_sphere(ax, center, radius, color, linewidth=0.8,
                        alpha=0.85, separation_deg=45.0, n_pts=64,
                        zorder=4):
        """Draw a sphere outline as two vertical great circles.

        Each great circle lies in a plane that contains the z-axis;
        the second plane is rotated `separation_deg` about z. Two
        circles is enough to read as a sphere without the noise of a
        full wireframe grid.
        """
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        t = np.linspace(0.0, 2.0 * np.pi, n_pts)
        cos_t, sin_t = np.cos(t), np.sin(t)
        for ang_deg in (0.0, float(separation_deg)):
            ang = np.deg2rad(ang_deg)
            ux, uy = np.cos(ang), np.sin(ang)
            x = cx + radius * cos_t * ux
            y = cy + radius * cos_t * uy
            z = cz + radius * sin_t
            ax.plot(
                x, y, z, color=color, linewidth=linewidth,
                alpha=alpha, zorder=zorder,
            )

    @staticmethod
    def _apply_dark_axes(ax, fg='white', bg='black', is_3d=False):
        """Restyle a Matplotlib Axes for a black background."""
        ax.set_facecolor(bg)
        ax.tick_params(colors=fg, which='both')
        for spine in ax.spines.values():
            spine.set_color(fg)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        if hasattr(ax, 'zaxis'):
            ax.zaxis.label.set_color(fg)
        ax.title.set_color(fg)
        if is_3d:
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_pane_color((0.0, 0.0, 0.0, 1.0))
                axis._axinfo['grid']['color'] = (1.0, 1.0, 1.0, 0.15)
            ax.tick_params(axis='z', colors=fg)

    @staticmethod
    def _nn_distances_bev(src_pts, tgt_pts):
        """Nearest-neighbor BEV distances from src_pts to tgt_pts.

        Returns an array of shape (len(src_pts),). Empty if either side
        is empty.
        """
        if len(src_pts) == 0 or len(tgt_pts) == 0:
            return np.zeros((0,), dtype=np.float32)
        s = src_pts[:, :2].astype(np.float32)
        t = tgt_pts[:, :2].astype(np.float32)
        # (Ns, Nt) pairwise BEV distances
        d = np.linalg.norm(s[:, None, :] - t[None, :, :], axis=2)
        return d.min(axis=1)

    def _filter_boxes_by_vicinity(
        self, boxes, scores, patch_entries, show_toggle
    ):
        """Apply a show/hide toggle gated by patch vicinity.

        When ``show_toggle`` is True, return inputs unchanged. When
        False, keep only boxes that fall inside any patch's
        ``success_radius`` under the active ``success_criterion`` —
        those are kept regardless of the toggle so an in-vicinity
        prediction (the attack signal for Car, a near-target misfire
        for Ped/Cyc) is never silently hidden.

        Vicinity definition tracks the criterion used by ASR scoring:
          * ``centroid`` — BEV centroid of the box to patch centroid
            is < ``success_radius``.
          * ``box_edge`` — BEV distance from patch centroid to the
            nearest edge of the rectangle is < ``success_radius``.
            Equivalent to inflating each box by ``success_radius``
            (Minkowski sum) and asking whether any patch centroid falls
            inside.

        Empty-safe; returns the input arrays unchanged when there's
        nothing to filter.
        """
        if boxes is None or len(boxes) == 0:
            return boxes, scores
        if show_toggle:
            return boxes, scores
        if not patch_entries:
            empty_b = boxes[:0]
            empty_s = scores[:0] if scores is not None else None
            return empty_b, empty_s
        patch_centers = np.asarray(
            [
                [float(pe["patch_box"][0]), float(pe["patch_box"][1])]
                for pe in patch_entries
            ],
            dtype=np.float32,
        )
        if self.success_criterion == 'centroid':
            box_centers = boxes[:, :2].astype(np.float32)
            d = np.linalg.norm(
                box_centers[:, None, :] - patch_centers[None, :, :],
                axis=2,
            )
            min_d = d.min(axis=1)
        else:
            # box_edge: distance from each patch centroid to the
            # nearest edge of every box; min over patches.
            dist_rows = np.stack([
                self._criterion_dist(patch_centers[k], boxes)
                for k in range(len(patch_centers))
            ])  # (P, N)
            min_d = dist_rows.min(axis=0)  # (N,)
        keep = min_d < self.success_radius
        kept_boxes = boxes[keep]
        kept_scores = scores[keep] if scores is not None else None
        return kept_boxes, kept_scores

    def _resolved_patch_shape(self, viz_entry):
        """Effective patch reference shape for one viz_entry: 'cube' or 'sphere'.

        Honors `self.patch_shape` when set explicitly; otherwise falls
        back to the manifest's `patch_geometry` (blob → sphere, else
        cube) as carried on the viz_entry by `_build_viz_entry`.
        """
        if self.patch_shape != 'auto':
            return self.patch_shape
        return 'sphere' if viz_entry.get("patch_geometry") == "blob" else 'cube'

    def _frame_patch_shape_label(self, patch_entries):
        """Legend label for the patch reference shape across a frame.

        Single shape across all patches → "Patch cube" / "Patch sphere".
        Mixed shapes (only possible under 'auto' with heterogeneous
        manifests) → bare "Patch" so the legend doesn't mislead.
        """
        shapes = {self._resolved_patch_shape(pe) for pe in patch_entries}
        if len(shapes) == 1:
            return f"Patch {shapes.pop()}"
        return "Patch"

    def _render_frame_figure(
        self,
        frame_name,
        frame_points,
        patch_entries,
        car_boxes,
        car_scores,
        real_car_boxes=None,
        real_car_detected=None,
        real_car_scores_gt=None,
        real_ped_boxes=None,
        real_ped_detected=None,
        real_ped_scores_gt=None,
        real_cyc_boxes=None,
        real_cyc_detected=None,
        real_cyc_scores_gt=None,
        halluc_ped_boxes=None,
        halluc_ped_scores=None,
        halluc_cyc_boxes=None,
        halluc_cyc_scores=None,
    ):
        """Emit one BEV figure covering every patch+target in the frame.

        `patch_entries` is a list of dicts, one per patch in the frame,
        with keys: `patch_box`, `source_box`, `patch_pts`, `target_pts`,
        `viz_score` (None or float), `source_class`.  Two additional
        keys (`target_car_dist`, `target_car_dist_outside`) are set
        by the renderer itself once Car preds are known and are used
        by the title builder; callers should not pre-populate them.

        `car_boxes`: (N_car, 7) LiDAR boxes (x, y, z, l, w, h, yaw).
        `car_scores`: (N_car,) prediction scores.
        """
        if len(patch_entries) == 0:
            return

        # Apply the per-class show/hide toggles once, up front, so the
        # auto-fit crop, scatter overlays, and legend counts all see
        # the same set of boxes. For Cars, "in vicinity" is exactly the
        # ASR-positive set, so dropping out-of-vicinity Cars when the
        # toggle is off is the natural analog of the Ped/Cyc rule.
        car_boxes, car_scores = self._filter_boxes_by_vicinity(
            car_boxes, car_scores, patch_entries,
            self.visualize_show_halluc_cars,
        )
        halluc_ped_boxes, halluc_ped_scores = self._filter_boxes_by_vicinity(
            halluc_ped_boxes, halluc_ped_scores, patch_entries,
            self.visualize_show_halluc_peds,
        )
        halluc_cyc_boxes, halluc_cyc_scores = self._filter_boxes_by_vicinity(
            halluc_cyc_boxes, halluc_cyc_scores, patch_entries,
            self.visualize_show_halluc_cycs,
        )

        import matplotlib
        if not self.visualize_interactive:
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon, Circle

        frame_stem = os.path.splitext(frame_name)[0]

        # Auto-fit BEV crop to every drawn element (patch + source
        # bboxes, the success-radius ring around each patch, Car
        # predictions, real-GT overlays where their show-flag is on,
        # and Ped/Cyc hallucinations). Padded by `visualize_crop_margin`
        # on every side. Mirrors `KittiMetricVisual` so both evaluators
        # frame their figures the same way.
        all_corners = []
        for pe in patch_entries:
            all_corners.append(self._bev_rect_corners(pe["patch_box"]))
            all_corners.append(self._bev_rect_corners(pe["source_box"]))
            # success-radius ring bounding box (axis-aligned around
            # the patch centroid).
            px = float(pe["patch_box"][0])
            py = float(pe["patch_box"][1])
            r = float(self.success_radius)
            all_corners.append(np.array([
                [px - r, py - r], [px - r, py + r],
                [px + r, py - r], [px + r, py + r],
            ], dtype=np.float32))
        if car_boxes is not None and len(car_boxes) > 0:
            for box in car_boxes:
                all_corners.append(self._bev_rect_corners(box))
        for boxes in (real_car_boxes, real_ped_boxes, real_cyc_boxes):
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                all_corners.append(self._bev_rect_corners(box))
        for boxes in (halluc_ped_boxes, halluc_cyc_boxes):
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                all_corners.append(self._bev_rect_corners(box))
        # Drop entries whose box had non-finite fields (corner func
        # returns None) so the min/max below stays finite — a single
        # NaN row poisons xlim/ylim and matplotlib then aborts the
        # whole frame with "Axis limits cannot be NaN or Inf".
        all_corners = [c for c in all_corners if c is not None]
        if len(all_corners) == 0:
            return
        all_corners = np.concatenate(all_corners, axis=0)
        m = self.visualize_crop_margin
        xlim = (float(all_corners[:, 0].min()) - m,
                float(all_corners[:, 0].max()) + m)
        ylim = (float(all_corners[:, 1].min()) - m,
                float(all_corners[:, 1].max()) + m)

        fig, ax = plt.subplots(figsize=(10, 10))
        # The "black" real-GT colorway clashes with a black axes
        # background, so flip to white wherever it shows up (real-GT
        # bbox edges, GT labels, fallback legend label color).
        gt_overlay_color = 'white' if self.visualize_black_bg else 'black'
        if self.visualize_black_bg:
            fig.patch.set_facecolor('black')
            self._apply_dark_axes(ax, is_3d=False)

        # Context points (everything outside patch+target, cropped).
        in_any_box = np.zeros(len(frame_points), dtype=bool)
        for pe in patch_entries:
            in_any_box |= pe["_patch_mask"]
            in_any_box |= pe["_target_mask"]
        ctx_mask = ~in_any_box
        if np.any(ctx_mask):
            cx = frame_points[ctx_mask]
            in_view = (
                (cx[:, 0] >= xlim[0]) & (cx[:, 0] <= xlim[1]) &
                (cx[:, 1] >= ylim[0]) & (cx[:, 1] <= ylim[1])
            )
            if np.any(in_view):
                ax.scatter(
                    cx[in_view, 0], cx[in_view, 1],
                    s=1, c="lightgrey", alpha=0.5, linewidths=0,
                    zorder=1,
                )

        # Class color table for targets. Patch color stays red for all
        # patches (they're the adversarial element). Target color /
        # class label / patch-score label all take the class color so
        # two objects in the same frame are easy to tell apart.
        _TARGET_COLORS = {
            "Pedestrian": "blue",
            "Cyclist": "purple",
        }
        _DEFAULT_TARGET_COLOR = "darkblue"

        # Per-patch short label: "Ped" / "Cyc" (class abbrev). Suffix
        # with an index only when the same abbrev repeats in the frame
        # — avoids ambiguity for the rare two-patches-same-class case
        # while keeping the common Ped+Cyc case clean. Computed up here
        # because both the scatter loop (legend labels) and the bbox
        # loop (score labels) reference it.
        _CLS_ABBREV = {"Pedestrian": "Ped", "Cyclist": "Cyc"}
        abbrev_per_patch = [
            _CLS_ABBREV.get(pe.get("source_class", "?"), pe.get("source_class", "?"))
            for pe in patch_entries
        ]
        from collections import Counter
        abbrev_counts = Counter(abbrev_per_patch)
        abbrev_seen = {}
        patch_labels = []
        for ab in abbrev_per_patch:
            if abbrev_counts[ab] > 1:
                idx = abbrev_seen.get(ab, 0)
                patch_labels.append(f"{ab}{idx}")
                abbrev_seen[ab] = idx + 1
            else:
                patch_labels.append(ab)

        # Patch + target scatter per object so the legend can show
        # Patch points (red) drawn underneath; target points (class-
        # colored) on top. Both are kept off the legend now so the BEV
        # legend matches its 3D companion.
        for k, pe in enumerate(patch_entries):
            tgt_color = _TARGET_COLORS.get(
                pe.get("source_class"), _DEFAULT_TARGET_COLOR
            )
            if len(pe["patch_pts"]) > 0:
                ax.scatter(
                    pe["patch_pts"][:, 0], pe["patch_pts"][:, 1],
                    s=3, c="red", alpha=0.85, linewidths=0,
                    zorder=2,
                )
            if len(pe["target_pts"]) > 0:
                ax.scatter(
                    pe["target_pts"][:, 0], pe["target_pts"][:, 1],
                    s=3, c=tgt_color, alpha=0.95, linewidths=0,
                    zorder=3,
                )

        # Bboxes + per-patch annotations. patch_labels computed above.
        for k, pe in enumerate(patch_entries):
            patch_corners = self._bev_rect_corners(pe["patch_box"])
            source_corners = self._bev_rect_corners(pe["source_box"])
            if patch_corners is None or source_corners is None:
                continue

            # Source bbox — class-colored so two objects in a frame
            # are distinguishable at a glance.
            tgt_color = _TARGET_COLORS.get(
                pe.get("source_class"), _DEFAULT_TARGET_COLOR
            )
            ax.add_patch(Polygon(
                source_corners, closed=True, fill=False,
                edgecolor=tgt_color, linewidth=1.4, linestyle="-",
                zorder=4,
            ))
            # Suppression state label: green "sup" if the original Ped
            # /Cyc class was NOT detected (attack succeeded at hiding
            # the real object), orange "unsup" if the model still
            # detected the original class. Anchored at the source
            # box's top-left so it doesn't crowd the patch-score/Ped-
            # Cyc labels at the patch box's top-right.
            if pe.get("is_suppressed") is not None:
                if pe["is_suppressed"]:
                    sup_color = "forestgreen"
                    sup_text = "sup"
                else:
                    sup_color = "darkorange"
                    sup_text = "unsup"
                src_top_left = source_corners[np.argmax(
                    -source_corners[:, 0] + source_corners[:, 1]
                )]
                ax.text(
                    src_top_left[0], src_top_left[1],
                    sup_text,
                    color=sup_color, fontsize=7,
                    ha="right", va="center",
                    zorder=6,
                )
            # Patch reference frame: pink thin outline, drawn *under* the
            # red patch points so the points stay visually dominant (the
            # frame is just for context, not the attack signal itself).
            # Sphere patches render as a CIRCLE of radius dx/2 (manifest
            # emits a square AABB so patch_box[3] = 2 * R) so the figure
            # doesn't visually imply a car-roof shape on what is in fact
            # a sphere. Cube patches render as the rectangle. Resolution
            # honors `self.patch_shape` ('auto' delegates to the
            # manifest's `patch_geometry`).
            if self._resolved_patch_shape(pe) == 'sphere':
                cx_pb = float(pe["patch_box"][0])
                cy_pb = float(pe["patch_box"][1])
                R_pb = float(pe["patch_box"][3]) / 2.0
                ax.add_patch(Circle(
                    (cx_pb, cy_pb), R_pb, fill=False,
                    edgecolor="pink", linewidth=0.8, linestyle="-",
                    zorder=1,
                ))
            else:
                ax.add_patch(Polygon(
                    patch_corners, closed=True, fill=False,
                    edgecolor="pink", linewidth=0.8, linestyle="-",
                    zorder=1,
                ))
            # Per-patch score label at the patch's top-right corner.
            # Text is "<Ped|Cyc> score=<x.xxx>" so the figure reads
            # without needing the legend to decode a "P0/P1" index.
            top_right = patch_corners[np.argmax(
                patch_corners[:, 0] + patch_corners[:, 1]
            )]
            # Target class label only ("Ped" / "Cyc"). No score — the
            # number associated with the patch is the Car-pred score,
            # not anything that belongs to the target object. Kept at
            # fontsize 7 / class color so the Car labels and target
            # labels read consistently.
            ax.text(
                top_right[0], top_right[1],
                patch_labels[k],
                color=tgt_color, fontsize=7,
                ha="left", va="center",
                zorder=6,
            )

        # Car prediction overlay: full BEV bbox + × center + score, with
        # green = positive (center lies inside any patch's success_radius)
        # and orange = outside-of-radius context detection.
        POS_COLOR = "forestgreen"
        NEG_COLOR = "darkorange"
        patch_centers_xy = np.array([
            [float(pe["patch_box"][0]), float(pe["patch_box"][1])]
            for pe in patch_entries
        ], dtype=np.float32)

        if len(car_boxes) > 0:
            car_centers_xy = car_boxes[:, :2].astype(np.float32)
            in_view = (
                (car_centers_xy[:, 0] >= xlim[0]) &
                (car_centers_xy[:, 0] <= xlim[1]) &
                (car_centers_xy[:, 1] >= ylim[0]) &
                (car_centers_xy[:, 1] <= ylim[1])
            )
            # Distance from each Car pred to the nearest patch centroid,
            # using the active success_criterion. Under 'centroid': center-
            # to-center. Under 'box_edge': patch-centroid to Car-box-edge.
            if len(patch_centers_xy) > 0:
                if self.success_criterion == 'centroid':
                    D = np.linalg.norm(
                        car_centers_xy[:, None, :] - patch_centers_xy[None, :, :],
                        axis=2,
                    )
                    min_d_to_patch = D.min(axis=1)
                else:
                    # box_edge: dist(patch_center_k, car_box_i) for each k,i
                    dist_rows = np.stack([
                        self._criterion_dist(patch_centers_xy[k], car_boxes)
                        for k in range(len(patch_centers_xy))
                    ])  # (P, N_car)
                    min_d_to_patch = dist_rows.min(axis=0)  # (N_car,)
            else:
                min_d_to_patch = np.full(len(car_boxes), np.inf, np.float32)
            positive_mask = min_d_to_patch < self.success_radius

            n_pos_in = int(np.sum(in_view & positive_mask))
            n_neg_in = int(np.sum(in_view & ~positive_mask))

            for i in range(len(car_boxes)):
                if not in_view[i]:
                    continue
                color = POS_COLOR if positive_mask[i] else NEG_COLOR
                corners = self._bev_rect_corners(car_boxes[i])
                if corners is None:
                    continue
                ax.add_patch(Polygon(
                    corners, closed=True, fill=False,
                    edgecolor=color, linewidth=1.3, linestyle="-",
                    zorder=5,
                ))
                # c wrapped in a list to force per-point color parsing;
                # a bare color *string* can confuse matplotlib's scatter
                # into interpreting the string as a per-point RGBA
                # sequence (hence "RGBA sequence should have had length
                # 3 or 4").
                ax.scatter(
                    [car_centers_xy[i, 0]], [car_centers_xy[i, 1]],
                    s=50, c=[color], marker="x", linewidths=1.5,
                    zorder=6,
                )
                ax.text(
                    car_centers_xy[i, 0], car_centers_xy[i, 1],
                    f" Car {car_scores[i]:.2f}",
                    color=color, fontsize=7,
                    ha="left", va="center",
                    zorder=7,
                )

            # Single-entry legend proxies so the user sees both classes.
            from matplotlib.lines import Line2D
            legend_handles = []
            if n_pos_in > 0:
                legend_handles.append(Line2D(
                    [0], [0], color=POS_COLOR, linestyle='-',
                    linewidth=1.3,
                    label=f"Car pred, pos (n={n_pos_in})",
                ))
            if n_neg_in > 0:
                legend_handles.append(Line2D(
                    [0], [0], color=NEG_COLOR, linestyle='-',
                    linewidth=1.3,
                    label=f"Car pred, other (n={n_neg_in})",
                ))
            if legend_handles:
                # Save for the combined legend below.
                self._extra_legend_handles = legend_handles
            else:
                self._extra_legend_handles = []

            # Connectors from target center to the relevant "fake Car":
            #   * Green dotted: target → closest Car pred inside the
            #     success_radius ring (i.e. an ASR-creditable "fake
            #     car"). This is the distance reported in the title.
            #   * Orange dashed: target → closest Car pred outside the
            #     ring, only drawn when no in-ring Car exists. Gives a
            #     sense of "how near the attack came" when it failed.
            # The chosen distances are stashed on each pe so the title
            # builder below can format them consistently.
            for pe in patch_entries:
                px = float(pe["patch_box"][0])
                py = float(pe["patch_box"][1])
                sx = float(pe["source_box"][0])
                sy = float(pe["source_box"][1])

                # Distances target → every Car pred center (BEV).
                d_to_target = np.linalg.norm(
                    car_centers_xy - np.array([sx, sy], dtype=np.float32),
                    axis=1,
                )
                # Which Car preds are inside this patch's ring.
                d_to_patch = np.linalg.norm(
                    car_centers_xy - np.array([px, py], dtype=np.float32),
                    axis=1,
                )
                in_ring = d_to_patch < self.success_radius

                pe["target_car_dist"] = None          # in-ring (reported)
                pe["target_car_dist_outside"] = None  # out-of-ring fallback

                if np.any(in_ring):
                    idx_in = np.where(in_ring)[0]
                    best_i = int(idx_in[np.argmin(d_to_target[idx_in])])
                    pe["target_car_dist"] = float(d_to_target[best_i])
                    ax.plot(
                        [sx, float(car_centers_xy[best_i, 0])],
                        [sy, float(car_centers_xy[best_i, 1])],
                        color=POS_COLOR, linewidth=1.2, linestyle=":",
                        zorder=5,
                    )
                elif len(car_centers_xy) > 0:
                    # No in-ring Car. Record the out-of-ring distance
                    # for any downstream consumer; the connector line
                    # itself is intentionally not drawn — out-of-ring
                    # connectors were too noisy on dense frames.
                    best_i = int(np.argmin(d_to_target))
                    pe["target_car_dist_outside"] = float(d_to_target[best_i])
        else:
            self._extra_legend_handles = []
            for pe in patch_entries:
                pe["target_car_dist"] = None
                pe["target_car_dist_outside"] = None

        # Ped/Cyc hallucinations (false-positive preds that don't match
        # any same-class real GT at IoU >= 0.5). Drawn with the same
        # visual regime as ASR-positive Car preds — bbox + × at the
        # BEV center + "<cls> <score>" label. Unlike Cars, where
        # "positive" means "inside a patch's success_radius", hallucin-
        # ations for Ped/Cyc are a straight GT-mismatch check since the
        # attack doesn't induce Ped/Cyc predictions. Color is per-class:
        # Ped FPs render brown, Cyc FPs render blue so they stay
        # visually distinct from the patch-induced Car preds (green).
        from matplotlib.lines import Line2D
        _HALLUC_COLOR = {"Ped": "brown", "Cyc": "blue"}
        halluc_overlays = [
            ("Ped", halluc_ped_boxes, halluc_ped_scores),
            ("Cyc", halluc_cyc_boxes, halluc_cyc_scores),
        ]
        halluc_legend_handles = []
        for cls_abbrev, hboxes, hscores in halluc_overlays:
            if hboxes is None or len(hboxes) == 0:
                continue
            color = _HALLUC_COLOR.get(cls_abbrev, POS_COLOR)
            n_in_view = 0
            for i in range(len(hboxes)):
                cx = float(hboxes[i, 0])
                cy = float(hboxes[i, 1])
                if not (
                    xlim[0] <= cx <= xlim[1]
                    and ylim[0] <= cy <= ylim[1]
                ):
                    continue
                corners = self._bev_rect_corners(hboxes[i])
                if corners is None:
                    continue
                ax.add_patch(Polygon(
                    corners, closed=True, fill=False,
                    edgecolor=color, linewidth=1.3, linestyle="-",
                    zorder=5,
                ))
                ax.scatter(
                    [cx], [cy],
                    s=50, c=[color], marker="x", linewidths=1.5,
                    zorder=6,
                )
                ax.text(
                    cx, cy,
                    f" {cls_abbrev} {float(hscores[i]):.2f}",
                    color=color, fontsize=7,
                    ha="left", va="center",
                    zorder=7,
                )
                n_in_view += 1
            if n_in_view > 0:
                halluc_legend_handles.append(Line2D(
                    [0], [0], color=color, linestyle='-',
                    linewidth=1.3,
                    label=f"{cls_abbrev} pred, hallucinated (n={n_in_view})",
                ))
        if halluc_legend_handles:
            self._extra_legend_handles = (
                list(getattr(self, "_extra_legend_handles", []) or [])
                + halluc_legend_handles
            )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x [m] (LiDAR)")
        ax.set_ylabel("y [m] (LiDAR)")

        # Title — single-line, matches the 3D companion's `(3D)` suffix
        # convention. Per-patch distances are still visually conveyed by
        # the green-dotted / orange-dashed connectors drawn earlier.
        ax.set_title(f"Frame {frame_stem} (BEV)", fontsize=10)

        # Real GT overlay (dotted black), per class. Optional diagnostic
        # that sanity-checks detections around the patch: for Car it also
        # verifies the real-Car safeguard is masking what it should; for
        # Ped/Cyc it shows whether unattacked real GTs were still
        # detected. Each GT is labeled "<cls> <score>" when matched by
        # a same-class pred at the KITTI IoU threshold (Car 0.7, Ped/Cyc
        # 0.5), or "<cls> ND" when unmatched. Drawn before the legend so
        # we can add a per-class proxy for it.
        gt_overlays = [
            ("car", real_car_boxes, real_car_detected, real_car_scores_gt),
            ("ped", real_ped_boxes, real_ped_detected, real_ped_scores_gt),
            ("cyc", real_cyc_boxes, real_cyc_detected, real_cyc_scores_gt),
        ]
        real_gt_drawn_per_class = {}
        for cls_abbrev, gt_boxes, gt_detected, gt_scores in gt_overlays:
            if gt_boxes is None or len(gt_boxes) == 0:
                continue
            drawn = 0
            for gi, rbox in enumerate(gt_boxes):
                corners = self._bev_rect_corners(rbox)
                if corners is None:
                    continue
                # skip boxes entirely outside the crop to keep it clean
                in_crop = (
                    (corners[:, 0].max() >= xlim[0]) and
                    (corners[:, 0].min() <= xlim[1]) and
                    (corners[:, 1].max() >= ylim[0]) and
                    (corners[:, 1].min() <= ylim[1])
                )
                if not in_crop:
                    continue
                ax.add_patch(Polygon(
                    corners, closed=True, fill=False,
                    edgecolor=gt_overlay_color, linewidth=1.0,
                    linestyle=(0, (3, 2)), alpha=0.8,
                    zorder=5,
                ))
                # Per-GT label, anchored at the top-right corner of
                # the GT bbox. Thin black, fontsize 7 — matches the
                # Car-pred and Ped/Cyc label style.
                if gt_detected is not None and gi < len(gt_detected):
                    if bool(gt_detected[gi]):
                        sc = None
                        if (
                            gt_scores is not None
                            and gi < len(gt_scores)
                            and np.isfinite(gt_scores[gi])
                        ):
                            sc = float(gt_scores[gi])
                        gt_label = (
                            f"{cls_abbrev} {sc:.2f}"
                            if sc is not None else cls_abbrev
                        )
                    else:
                        gt_label = f"{cls_abbrev} ND"
                else:
                    gt_label = f"{cls_abbrev}?"
                top_right = corners[np.argmax(
                    corners[:, 0] + corners[:, 1]
                )]
                ax.text(
                    top_right[0], top_right[1],
                    gt_label,
                    color=gt_overlay_color, fontsize=7,
                    ha="left", va="center",
                    zorder=6,
                )
                drawn += 1
            if drawn > 0:
                real_gt_drawn_per_class[cls_abbrev] = drawn

        # Legend matches the 3D companion: a single box, line proxies
        # only, no per-patch scatter handles. The success_radius bubble
        # itself is no longer drawn (neither sphere in 3D nor ring in
        # BEV), but the value still drives ASR scoring so it stays in
        # the legend as a dashed pink proxy — same style as the 3D
        # companion's legend entry.
        from matplotlib.lines import Line2D
        _patch_shape_label = self._frame_patch_shape_label(patch_entries)
        legend_handles = [
            Line2D(
                [0], [0], color="pink", linestyle="-", linewidth=1.0,
                alpha=0.85, label=_patch_shape_label,
            ),
            Line2D(
                [0], [0], color="pink", linestyle="--", linewidth=0.8,
                alpha=0.6,
                label=f"success_radius = {self.success_radius:.1f} m",
            ),
        ]
        legend_colors = ["pink", "pink"]
        for h in getattr(self, "_extra_legend_handles", []) or []:
            legend_handles.append(h)
            try:
                legend_colors.append(h.get_color())
            except Exception:
                legend_colors.append(gt_overlay_color)
        _GT_LEGEND_LABEL = {
            "car": "Car", "ped": "Pedestrian", "cyc": "Cyclist",
        }
        for cls_abbrev, n_drawn in real_gt_drawn_per_class.items():
            legend_handles.append(Line2D(
                [0], [0], color=gt_overlay_color,
                linestyle=(0, (3, 2)),
                linewidth=1.0, alpha=0.8,
                label=f"Real {_GT_LEGEND_LABEL[cls_abbrev]} GT (n={n_drawn})",
            ))
            legend_colors.append(gt_overlay_color)

        # Anchor the legend just outside the axes (top-right) so it
        # doesn't sit on top of the LiDAR point cloud. `bbox_inches=
        # "tight"` on savefig keeps the legend fully in the PNG.
        legend_kw = dict(
            loc="upper left", bbox_to_anchor=(1.02, 1.0),
            fontsize=8, framealpha=0.8, borderaxespad=0.0,
        )
        if self.visualize_black_bg:
            legend_kw.update(facecolor='black', edgecolor='white')
        ax.legend(
            handles=legend_handles, labelcolor=legend_colors,
            **legend_kw,
        )

        if self.visualize_save_dir is not None:
            # Split saves into per-source-class subfolders. With the
            # single-patch-per-frame invariant (v15 small_patch_v3/v4 +
            # blob_v1) each frame has one source class; legacy manifests
            # with both Ped and Cyc on the same frame get the figure
            # written to both subfolders so each subfolder is a complete
            # per-class record.
            src_classes = {
                pe.get("source_class") for pe in patch_entries
                if pe.get("source_class")
            }
            sub_names = (
                sorted(s.lower() for s in src_classes)
                if src_classes else ["unknown"]
            )
            for sub in sub_names:
                sub_dir = os.path.join(self.visualize_save_dir, sub)
                os.makedirs(sub_dir, exist_ok=True)
                out = os.path.join(sub_dir, f"{frame_stem}.png")
                savefig_kw = dict(dpi=150, bbox_inches="tight")
                if self.visualize_black_bg:
                    savefig_kw['facecolor'] = fig.get_facecolor()
                fig.savefig(out, **savefig_kw)

        if self.visualize_interactive:
            plt.show()

        plt.close(fig)

    def _render_frame_figure_3d(
        self,
        frame_name,
        frame_points,
        patch_entries,
        car_boxes,
        car_scores,
        real_car_boxes=None,
        real_car_detected=None,
        real_car_scores_gt=None,
        real_ped_boxes=None,
        real_ped_detected=None,
        real_ped_scores_gt=None,
        real_cyc_boxes=None,
        real_cyc_detected=None,
        real_cyc_scores_gt=None,
        halluc_ped_boxes=None,
        halluc_ped_scores=None,
        halluc_cyc_boxes=None,
        halluc_cyc_scores=None,
    ):
        """3D companion of `_render_frame_figure`.

        Renders the same frame data as the BEV view but with wireframe
        cuboids for each bbox, a wireframe sphere for each patch's
        success_radius, and a 3D scatter for the LiDAR / patch / target
        points. Saves to ``<frame_stem>_3d.png`` in the same per-class
        subfolder as the BEV PNG. Honors ``visualize_black_bg``.
        """
        if len(patch_entries) == 0:
            return

        # Mirror the BEV renderer: apply per-class toggles once, up
        # front, so the 3D auto-fit and overlays match the BEV figure.
        car_boxes, car_scores = self._filter_boxes_by_vicinity(
            car_boxes, car_scores, patch_entries,
            self.visualize_show_halluc_cars,
        )
        halluc_ped_boxes, halluc_ped_scores = self._filter_boxes_by_vicinity(
            halluc_ped_boxes, halluc_ped_scores, patch_entries,
            self.visualize_show_halluc_peds,
        )
        halluc_cyc_boxes, halluc_cyc_scores = self._filter_boxes_by_vicinity(
            halluc_cyc_boxes, halluc_cyc_scores, patch_entries,
            self.visualize_show_halluc_cycs,
        )

        import matplotlib
        if not self.visualize_interactive:
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        frame_stem = os.path.splitext(frame_name)[0]

        # Auto-fit 3D crop from every drawn cuboid + the success-radius
        # spheres around each patch.
        all_corners_3d = []
        for pe in patch_entries:
            all_corners_3d.append(self._bbox_3d_corners(pe["patch_box"]))
            all_corners_3d.append(self._bbox_3d_corners(pe["source_box"]))
            px = float(pe["patch_box"][0])
            py = float(pe["patch_box"][1])
            pz = float(pe["patch_box"][2])
            r = float(self.success_radius)
            all_corners_3d.append(np.array([
                [px - r, py - r, pz - r], [px + r, py + r, pz + r],
            ], dtype=np.float32))
        if car_boxes is not None and len(car_boxes) > 0:
            for box in car_boxes:
                all_corners_3d.append(self._bbox_3d_corners(box))
        for boxes in (real_car_boxes, real_ped_boxes, real_cyc_boxes):
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                all_corners_3d.append(self._bbox_3d_corners(box))
        for boxes in (halluc_ped_boxes, halluc_cyc_boxes):
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                all_corners_3d.append(self._bbox_3d_corners(box))
        # See _bev_rect_corners — drop non-finite entries before
        # min/max so xlim/ylim/zlim stay finite.
        all_corners_3d = [c for c in all_corners_3d if c is not None]
        if len(all_corners_3d) == 0:
            return
        all_corners_3d = np.concatenate(all_corners_3d, axis=0)
        m = self.visualize_crop_margin
        xlim = (float(all_corners_3d[:, 0].min()) - m,
                float(all_corners_3d[:, 0].max()) + m)
        ylim = (float(all_corners_3d[:, 1].min()) - m,
                float(all_corners_3d[:, 1].max()) + m)
        zlim = (float(all_corners_3d[:, 2].min()) - 1.0,
                float(all_corners_3d[:, 2].max()) + 1.0)

        fig = plt.figure(figsize=(11, 10))
        ax = fig.add_subplot(111, projection='3d')
        gt_overlay_color = 'white' if self.visualize_black_bg else 'black'
        if self.visualize_black_bg:
            fig.patch.set_facecolor('black')
            self._apply_dark_axes(ax, is_3d=True)

        # Context points (everything outside patch+target, cropped).
        # Matplotlib 3D doesn't honor `zorder` reliably — draw order is
        # decided by the points' actual depth — so context points that
        # happen to sit closer to the camera than the patch sphere /
        # patch points / source bbox would silently occlude them. We
        # work around it by carving out a `success_radius` exclusion
        # bubble around every patch, leaving the foreground elements
        # visually on top of the surrounding LiDAR context.
        in_any_box = np.zeros(len(frame_points), dtype=bool)
        for pe in patch_entries:
            in_any_box |= pe["_patch_mask"]
            in_any_box |= pe["_target_mask"]
        ctx_mask = ~in_any_box
        if np.any(ctx_mask) and len(patch_entries) > 0:
            r_excl_sq = float(self.success_radius) ** 2
            min_d_sq = np.full(len(frame_points), np.inf, dtype=np.float32)
            for pe in patch_entries:
                px = float(pe["patch_box"][0])
                py = float(pe["patch_box"][1])
                pz = float(pe["patch_box"][2])
                d_sq = (
                    (frame_points[:, 0] - px) ** 2
                    + (frame_points[:, 1] - py) ** 2
                    + (frame_points[:, 2] - pz) ** 2
                )
                np.minimum(min_d_sq, d_sq, out=min_d_sq)
            ctx_mask = ctx_mask & (min_d_sq >= r_excl_sq)
        if np.any(ctx_mask):
            cxp = frame_points[ctx_mask]
            in_view = (
                (cxp[:, 0] >= xlim[0]) & (cxp[:, 0] <= xlim[1]) &
                (cxp[:, 1] >= ylim[0]) & (cxp[:, 1] <= ylim[1]) &
                (cxp[:, 2] >= zlim[0]) & (cxp[:, 2] <= zlim[1])
            )
            if np.any(in_view):
                pts_color = 'white' if self.visualize_black_bg else 'lightgrey'
                ax.scatter(
                    cxp[in_view, 0], cxp[in_view, 1], cxp[in_view, 2],
                    s=2, c=pts_color, alpha=0.4, linewidths=0,
                    depthshade=False, zorder=1,
                )

        _TARGET_COLORS = {
            "Pedestrian": "blue",
            "Cyclist": "purple",
        }
        _DEFAULT_TARGET_COLOR = "darkblue"

        # Patch + target points (3D scatter), patch outline (2 great
        # circles) + source wireframe per patch. Patch shape and patch
        # points are pink. The success_radius vicinity sphere is no
        # longer drawn — too noisy on top of the patch outline.
        _PATCH_COLOR = 'pink'

        # Per-patch short label ("Ped"/"Cyc" + index suffix on collisions)
        # — same logic as BEV so the two views read consistently.
        _CLS_ABBREV = {"Pedestrian": "Ped", "Cyclist": "Cyc"}
        from collections import Counter as _Counter
        abbrev_per_patch = [
            _CLS_ABBREV.get(
                pe.get("source_class", "?"),
                pe.get("source_class", "?"),
            )
            for pe in patch_entries
        ]
        abbrev_counts = _Counter(abbrev_per_patch)
        abbrev_seen = {}
        patch_labels = []
        for ab in abbrev_per_patch:
            if abbrev_counts[ab] > 1:
                idx = abbrev_seen.get(ab, 0)
                patch_labels.append(f"{ab}{idx}")
                abbrev_seen[ab] = idx + 1
            else:
                patch_labels.append(ab)

        for k, pe in enumerate(patch_entries):
            tgt_color = _TARGET_COLORS.get(
                pe.get("source_class"), _DEFAULT_TARGET_COLOR
            )
            # Target points first, patch points second — matplotlib 3D
            # ignores `zorder` for depth occlusion, so the only reliable
            # way to keep the red patch dots visually on top of the
            # blue/purple target dots is to scatter them last.
            if len(pe["target_pts"]) > 0:
                tp = pe["target_pts"]
                # Same `s=3` as BEV target_pts.
                ax.scatter(
                    tp[:, 0], tp[:, 1], tp[:, 2],
                    s=3, c=[tgt_color], alpha=0.95, linewidths=0,
                    depthshade=False, zorder=3,
                )
            if len(pe["patch_pts"]) > 0:
                pp = pe["patch_pts"]
                # `c` as a single-element list dodges matplotlib 3D
                # scatter's RGBA mis-parse on a bare color string.
                # Patch dots are red and sized `s=3` — matches BEV's
                # patch_pts so the two views read consistently. The
                # patch *sphere* outline stays pink (set above as
                # `_PATCH_COLOR`).
                ax.scatter(
                    pp[:, 0], pp[:, 1], pp[:, 2],
                    s=3, c=["red"], alpha=1.0, linewidths=0,
                    depthshade=False, zorder=6,
                )
            self._draw_3d_bbox(
                ax, pe["source_box"], color=tgt_color,
                linewidth=1.4, linestyle='-', zorder=4,
            )
            # Source-bbox suppression label ("sup" green / "unsup"
            # orange), anchored at the source box's top-face top-left
            # corner — mirrors the BEV anchoring choice.
            if pe.get("is_suppressed") is not None:
                sup_color = (
                    "forestgreen" if pe["is_suppressed"]
                    else "darkorange"
                )
                sup_text = "sup" if pe["is_suppressed"] else "unsup"
                src_corners_3d = self._bbox_3d_corners(pe["source_box"])
                if src_corners_3d is not None:
                    top_s = src_corners_3d[4:]
                    anc_s = top_s[
                        int(np.argmax(-top_s[:, 0] + top_s[:, 1]))
                    ]
                    ax.text(
                        float(anc_s[0]), float(anc_s[1]),
                        float(anc_s[2]),
                        sup_text, color=sup_color, fontsize=7,
                        ha='right', va='center', zorder=8,
                    )
            # Patch reference shape — sphere outline (two great circles)
            # for sphere patches, cuboid wireframe for cube patches. Drawn
            # under the patch points so they remain visually dominant.
            # Resolution honors `self.patch_shape` ('auto' delegates to
            # the manifest's `patch_geometry`).
            if self._resolved_patch_shape(pe) == 'sphere':
                self._draw_3d_sphere(
                    ax,
                    center=(
                        float(pe["patch_box"][0]),
                        float(pe["patch_box"][1]),
                        float(pe["patch_box"][2]),
                    ),
                    radius=float(pe["patch_box"][3]) / 2.0,
                    color=_PATCH_COLOR, linewidth=0.8, alpha=0.85,
                    zorder=2,
                )
            else:
                self._draw_3d_bbox(
                    ax, pe["patch_box"], color=_PATCH_COLOR,
                    linewidth=0.8, linestyle='-', zorder=2,
                )
            # Patch label ("Ped" / "Cyc" / "Ped0" / ...) at the patch
            # AABB's top-face top-right corner.
            patch_corners_3d = self._bbox_3d_corners(pe["patch_box"])
            if patch_corners_3d is not None:
                top_p = patch_corners_3d[4:]
                anc_p = top_p[int(np.argmax(top_p[:, 0] + top_p[:, 1]))]
                ax.text(
                    float(anc_p[0]), float(anc_p[1]), float(anc_p[2]),
                    patch_labels[k], color=tgt_color, fontsize=7,
                    ha='left', va='center', zorder=8,
                )

        # Car prediction overlay (positive=in success ring, negative=out).
        POS_COLOR = "forestgreen"
        NEG_COLOR = "darkorange"
        patch_centers_xy = np.array([
            [float(pe["patch_box"][0]), float(pe["patch_box"][1])]
            for pe in patch_entries
        ], dtype=np.float32)
        n_pos_in = n_neg_in = 0
        if car_boxes is not None and len(car_boxes) > 0:
            car_centers_xy = car_boxes[:, :2].astype(np.float32)
            in_view = (
                (car_centers_xy[:, 0] >= xlim[0]) &
                (car_centers_xy[:, 0] <= xlim[1]) &
                (car_centers_xy[:, 1] >= ylim[0]) &
                (car_centers_xy[:, 1] <= ylim[1])
            )
            if len(patch_centers_xy) > 0:
                if self.success_criterion == 'centroid':
                    D = np.linalg.norm(
                        car_centers_xy[:, None, :]
                        - patch_centers_xy[None, :, :],
                        axis=2,
                    )
                    min_d_to_patch = D.min(axis=1)
                else:
                    dist_rows = np.stack([
                        self._criterion_dist(
                            patch_centers_xy[k], car_boxes
                        )
                        for k in range(len(patch_centers_xy))
                    ])
                    min_d_to_patch = dist_rows.min(axis=0)
            else:
                min_d_to_patch = np.full(
                    len(car_boxes), np.inf, np.float32)
            positive_mask = min_d_to_patch < self.success_radius
            n_pos_in = int(np.sum(in_view & positive_mask))
            n_neg_in = int(np.sum(in_view & ~positive_mask))
            for i in range(len(car_boxes)):
                if not in_view[i]:
                    continue
                if not self._box_is_finite(car_boxes[i]):
                    continue
                color = POS_COLOR if positive_mask[i] else NEG_COLOR
                self._draw_3d_bbox(
                    ax, car_boxes[i], color=color,
                    linewidth=1.3, linestyle='-', zorder=5,
                )
                cb = car_boxes[i]
                z_top = float(cb[2]) + float(cb[5]) / 2.0
                ax.text(
                    float(cb[0]), float(cb[1]), z_top,
                    f' Car {car_scores[i]:.2f}',
                    color=color, fontsize=7,
                    ha='left', va='center', zorder=8,
                )

        # Ped/Cyc hallucinations. Cyc renders blue (matches BEV), Ped
        # stays brown so they're distinguishable from each other and
        # from the green/orange Car-pred overlays.
        _HALLUC_COLOR = {"Ped": "brown", "Cyc": "blue"}
        halluc_overlays = [
            ("Ped", halluc_ped_boxes, halluc_ped_scores),
            ("Cyc", halluc_cyc_boxes, halluc_cyc_scores),
        ]
        halluc_counts = {}
        for cls_abbrev, hboxes, hscores in halluc_overlays:
            if hboxes is None or len(hboxes) == 0:
                continue
            color = _HALLUC_COLOR.get(cls_abbrev, POS_COLOR)
            n_in_view = 0
            for i in range(len(hboxes)):
                if not self._box_is_finite(hboxes[i]):
                    continue
                cxh = float(hboxes[i, 0])
                cyh = float(hboxes[i, 1])
                if not (
                    xlim[0] <= cxh <= xlim[1]
                    and ylim[0] <= cyh <= ylim[1]
                ):
                    continue
                self._draw_3d_bbox(
                    ax, hboxes[i], color=color,
                    linewidth=1.3, linestyle='-', zorder=5,
                )
                z_top_h = float(hboxes[i, 2]) + float(hboxes[i, 5]) / 2.0
                ax.text(
                    cxh, cyh, z_top_h,
                    f' {cls_abbrev} {float(hscores[i]):.2f}',
                    color=color, fontsize=7,
                    ha='left', va='center', zorder=8,
                )
                n_in_view += 1
            if n_in_view > 0:
                halluc_counts[cls_abbrev] = n_in_view

        # Real GT overlays (dotted, gt_overlay_color) — same gating as BEV.
        gt_overlays = [
            ("car", real_car_boxes, real_car_detected, real_car_scores_gt),
            ("ped", real_ped_boxes, real_ped_detected, real_ped_scores_gt),
            ("cyc", real_cyc_boxes, real_cyc_detected, real_cyc_scores_gt),
        ]
        real_gt_drawn_per_class = {}
        for cls_abbrev, gt_boxes, gt_detected, gt_scores in gt_overlays:
            if gt_boxes is None or len(gt_boxes) == 0:
                continue
            drawn = 0
            for gi, rbox in enumerate(gt_boxes):
                cxg, cyg = float(rbox[0]), float(rbox[1])
                if not (xlim[0] <= cxg <= xlim[1]
                        and ylim[0] <= cyg <= ylim[1]):
                    continue
                self._draw_3d_bbox(
                    ax, rbox, color=gt_overlay_color,
                    linewidth=1.0, linestyle=(0, (3, 2)),
                    alpha=0.85, zorder=5,
                )
                # GT label at top-face top-right corner. Detected →
                # "<cls> <score>"; not-detected → "<cls> ND".
                gt_corners_3d = self._bbox_3d_corners(rbox)
                top_g = gt_corners_3d[4:]
                anc_g = top_g[
                    int(np.argmax(top_g[:, 0] + top_g[:, 1]))
                ]
                if gt_detected is not None and gi < len(gt_detected):
                    if bool(gt_detected[gi]):
                        sc = None
                        if (
                            gt_scores is not None
                            and gi < len(gt_scores)
                            and np.isfinite(gt_scores[gi])
                        ):
                            sc = float(gt_scores[gi])
                        gt_label = (
                            f"{cls_abbrev} {sc:.2f}"
                            if sc is not None else cls_abbrev
                        )
                    else:
                        gt_label = f"{cls_abbrev} ND"
                else:
                    gt_label = f"{cls_abbrev}?"
                ax.text(
                    float(anc_g[0]), float(anc_g[1]),
                    float(anc_g[2]),
                    gt_label, color=gt_overlay_color, fontsize=7,
                    ha='left', va='center', zorder=8,
                )
                drawn += 1
            if drawn > 0:
                real_gt_drawn_per_class[cls_abbrev] = drawn

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_xlabel('x [m] (LiDAR)')
        ax.set_ylabel('y [m] (LiDAR)')
        ax.set_zlabel('z [m] (LiDAR)')
        ax.set_title(f'Frame {frame_stem} (3D)', fontsize=10)
        try:
            ax.set_box_aspect((
                xlim[1] - xlim[0],
                ylim[1] - ylim[0],
                zlim[1] - zlim[0],
            ))
        except Exception:
            pass

        # Compact legend. The vicinity sphere itself is no longer
        # drawn, but the `success_radius` value is still meaningful for
        # ASR scoring so it stays in the legend as a dashed pink proxy.
        _patch_shape_label = self._frame_patch_shape_label(patch_entries)
        legend_handles = [
            Line2D(
                [0], [0], color=_PATCH_COLOR, linestyle='-',
                linewidth=1.0, alpha=0.85,
                label=_patch_shape_label,
            ),
            Line2D(
                [0], [0], color=_PATCH_COLOR, linestyle='--',
                linewidth=0.8, alpha=0.6,
                label=f'success_radius = {self.success_radius:.1f} m',
            ),
        ]
        legend_colors = [_PATCH_COLOR, _PATCH_COLOR]
        if n_pos_in > 0:
            legend_handles.append(Line2D(
                [0], [0], color=POS_COLOR, linestyle='-', linewidth=1.3,
                label=f'Car pred, pos (n={n_pos_in})',
            ))
            legend_colors.append(POS_COLOR)
        if n_neg_in > 0:
            legend_handles.append(Line2D(
                [0], [0], color=NEG_COLOR, linestyle='-', linewidth=1.3,
                label=f'Car pred, other (n={n_neg_in})',
            ))
            legend_colors.append(NEG_COLOR)
        for cls_abbrev, n in halluc_counts.items():
            color = _HALLUC_COLOR.get(cls_abbrev, POS_COLOR)
            legend_handles.append(Line2D(
                [0], [0], color=color, linestyle='-', linewidth=1.3,
                label=f'{cls_abbrev} pred, hallucinated (n={n})',
            ))
            legend_colors.append(color)
        _GT_LEGEND_LABEL = {
            "car": "Car", "ped": "Pedestrian", "cyc": "Cyclist",
        }
        for cls_abbrev, n_drawn in real_gt_drawn_per_class.items():
            legend_handles.append(Line2D(
                [0], [0], color=gt_overlay_color,
                linestyle=(0, (3, 2)),
                linewidth=1.0, alpha=0.85,
                label=f'Real {_GT_LEGEND_LABEL[cls_abbrev]} GT '
                      f'(n={n_drawn})',
            ))
            legend_colors.append(gt_overlay_color)
        legend_kw = dict(loc='upper left', fontsize=8, framealpha=0.8)
        if self.visualize_black_bg:
            legend_kw.update(facecolor='black', edgecolor='white')
        ax.legend(
            handles=legend_handles, labelcolor=legend_colors,
            **legend_kw,
        )

        if self.visualize_save_dir is not None:
            src_classes = {
                pe.get("source_class") for pe in patch_entries
                if pe.get("source_class")
            }
            sub_names = (
                sorted(s.lower() for s in src_classes)
                if src_classes else ["unknown"]
            )
            for sub in sub_names:
                sub_dir = os.path.join(self.visualize_save_dir, sub)
                os.makedirs(sub_dir, exist_ok=True)
                out = os.path.join(sub_dir, f"{frame_stem}_3d.png")
                savefig_kw = dict(dpi=150, bbox_inches="tight")
                if self.visualize_black_bg:
                    savefig_kw['facecolor'] = fig.get_facecolor()
                fig.savefig(out, **savefig_kw)

        if self.visualize_interactive:
            plt.show()

        plt.close(fig)


    # ------------------------------------------------------------------
    # Main compute
    # ------------------------------------------------------------------
    def compute_metrics(self, results):

        # -----------------------------------------
        # Step 1 — compute AP by parent
        # -----------------------------------------
        metrics = super().compute_metrics(results)

        # -----------------------------------------
        # Step 2 — reconstruct GT in KITTI format
        # -----------------------------------------
        gt_annos = [
            self.data_infos[result['sample_idx']]['kitti_annos']
            for result in results
        ]

        # -----------------------------------------
        # Step 3 — rebuild detections in KITTI format
        # -----------------------------------------
        result_dict, _ = self.format_results(
            results,
            pklfile_prefix=None,
            submission_prefix=None,
            classes=self.classes
        )

        dt_annos = result_dict['pred_instances_3d']

        # -----------------------------------------
        # v7 — clean-mode short-circuit
        # -----------------------------------------
        # On a clean (un-patched) pass there are no triggers, so ASR
        # scoring / manifest / visualization are all skipped. Utility
        # (CA / delta_mAP / per-class deltaAP) is computed from standard
        # prediction-vs-GT strict IoU matching over ALL frames (not just
        # patched frames) and returned. The raw KITTI AP keys from the
        # parent metric pass through `metrics` unchanged.
        if self.eval_mode == 'clean':
            return self._compute_metrics_clean(metrics, gt_annos, dt_annos)

        # -----------------------------------------
        # Step 4 — KITTI Strict Matching
        # -----------------------------------------

        # IoU thresholds (strict)
        iou_thresholds = {
            'Car': 0.7,
            'Pedestrian': 0.5,
            'Cyclist': 0.5
        }

        ca_tp = {k: 0 for k in iou_thresholds}
        ca_gt = {k: 0 for k in iou_thresholds}

        score_thresholds = [0.1, 0.3, 0.5]
        success_radius = self.success_radius   # meters (configurable)

        # -----------------------------------
        # ASR counters  (per threshold + class)
        # -----------------------------------
        asr_success = {
            thr: defaultdict(int) for thr in score_thresholds
        }
        asr_total = defaultdict(int)

        # -----------------------------------
        # Suppression + Full Morphing Rate counters
        # -----------------------------------
        suppression_total = defaultdict(int)       # denominator per source class
        suppression_success = defaultdict(int)     # original class NOT detected
        fmr_success = {                            # Car created AND original suppressed
            thr: defaultdict(int) for thr in score_thresholds
        }
        suppression_iou_thr = 0.5   # KITTI standard for Ped/Cyc
        suppression_score_thr = 0.1  # ignore very low-confidence detections

        # -----------------------------------
        # v6 counters: M_ASR / D_ASR partition + Phantom Car diagnostic
        # -----------------------------------
        # m_asr_success is mathematically identical to fmr_success — kept
        # as a separate variable so the v6 metric output is self-document-
        # ing (and so the lockstep increment is verifiable line-by-line).
        # d_asr_success counts patches where the original Ped/Cyc was
        # suppressed but no Car appeared inside success_radius — pure
        # disappearance.
        # phantom_car_count counts the inverse of M_ASR: a Car appeared
        # near the patch but the original was not suppressed. Diagnostic
        # only — never emitted into the metric dict.
        m_asr_success = {
            thr: defaultdict(int) for thr in score_thresholds
        }
        d_asr_success = {
            thr: defaultdict(int) for thr in score_thresholds
        }
        phantom_car_count = {
            thr: defaultdict(int) for thr in score_thresholds
        }

        # -----------------------------------
        # Close-pair diagnostic
        # -----------------------------------
        # Counts frames where two patches sit within close_pair_radius of
        # each other. Under the current per-patch ASR scoring, both patches
        # could be credited by a single Car detection that lies between
        # them (within success_radius=2m of each). Use this to decide
        # whether the one-to-one greedy assignment fix is needed.
        patched_frames_count = 0
        two_patch_frames = 0
        close_pair_frames = 0
        close_pair_distances = []

        # -----------------------------------
        # IoU histogram diagnostics
        # -----------------------------------
        closest_car_distances = []
        # Per-source-class break-out of the same distances so the
        # histogram banner can report Ped and Cyc independently in
        # addition to the aggregate.
        closest_car_distances_by_class = defaultdict(list)


        diagnostics = {
            'num_patched_entries': 0,
            'num_with_dt_car': 0,
            'closest_car_distances' : 0,
            'bev_iou_samples' : 0,
            'best_local_scores' : [],
            # v4: tracks Car preds dropped by the real-Car IoU safeguard.
            'num_car_preds_before_filter': 0,
            'num_masked_by_real_car': 0,
            'frames_with_any_mask': 0,
        }

        debug_limit = 10
        debug_count = 0

        car_label = self.classes.index('Car')
        ped_label = self.classes.index('Pedestrian')
        cyc_label = self.classes.index('Cyclist')

        # Progress + announcement when visualization is on, mirroring
        # KittiMetricVisual. The loop body runs unconditionally for
        # scoring; only the iterator and the print line are gated so
        # numerical-only runs (visualize=False) stay silent.
        if self.visualize and self.visualize_save_dir is not None:
            cap = self.visualize_max_frames
            n_patched = len(self.frame_to_patched)
            n_total = n_patched if cap is None else min(cap, n_patched)
            print(
                f"\n[KittiMetricMDASRVisual] rendering up to {n_total} "
                f"patched frame(s) -> {self.visualize_save_dir}"
            )
            loop_iter = mmengine.track_iter_progress(
                list(range(len(gt_annos)))
            )
        else:
            loop_iter = range(len(gt_annos))

        for idx in loop_iter:

            gt_anno = gt_annos[idx]
            dt_anno = dt_annos[idx]

            sample_idx = results[idx]['sample_idx']
            data_info = self.data_infos[sample_idx]
            frame_name = self._get_frame_from_data_info(data_info)
            # frame_name = self._get_frame_name(data_info)
            if frame_name not in self.frame_to_patched:
                continue

            # ------------------------------------------------
            # Close-pair diagnostic (BEFORE class iteration)
            # Assumes at most 2 patches per frame (Ped/Cyc).
            # ------------------------------------------------
            frame_entries = self.frame_to_patched[frame_name]
            patched_frames_count += 1

            assert len(frame_entries) <= 2, (
                f"Frame {frame_name} has {len(frame_entries)} patches; "
                f"this metric assumes ≤2 patches per frame."
            )

            if len(frame_entries) == 2:
                two_patch_frames += 1
                c0 = np.array(frame_entries[0]["patch_centroid"], dtype=np.float32)[:2]
                c1 = np.array(frame_entries[1]["patch_centroid"], dtype=np.float32)[:2]
                pair_dist = float(np.linalg.norm(c0 - c1))
                close_pair_distances.append(pair_dist)
                if pair_dist < self.close_pair_radius:
                    close_pair_frames += 1

            for cls in iou_thresholds:

                matched_gt, total_gt = self._match_3d_strict(
                    gt_anno,
                    dt_anno,
                    cls,
                    iou_thresholds[cls]
                )

                ca_tp[cls] += len(matched_gt)
                ca_gt[cls] += total_gt

            # -------------------------
            # ASR — per-frame preprocessing
            # -------------------------
            # frame_entries already fetched above for close-pair diagnostic

            # Extract LiDAR predictions directly from raw results
            result = results[idx]
            pred_instances = result['pred_instances_3d']

            pred_boxes = pred_instances['bboxes_3d'].tensor.cpu().numpy()
            pred_labels = pred_instances['labels_3d'].cpu().numpy()
            pred_scores = pred_instances['scores_3d'].cpu().numpy()

            # Car predictions above the global score floor.
            # Per-threshold ASR cuts happen later against this filtered pool.
            car_score_floor = 0.1
            car_mask = (pred_labels == car_label)
            car_boxes_frame = pred_boxes[car_mask]
            car_scores_frame = pred_scores[car_mask]
            floor_mask = car_scores_frame > car_score_floor
            car_boxes_frame = car_boxes_frame[floor_mask]
            car_scores_frame = car_scores_frame[floor_mask]

            # v5 viz: stash pre-safeguard Car preds + scores so we can
            # label each real Car GT as "car <score>" (detected) or "ND"
            # (not detected) on the figure. Must happen before the
            # real-Car safeguard rewrites car_boxes_frame, otherwise
            # GT-matching preds are gone and every GT would look ND.
            if self.visualize_show_real_cars:
                car_boxes_prefilter_for_viz = car_boxes_frame.copy()
                car_scores_prefilter_for_viz = car_scores_frame.copy()
            else:
                car_boxes_prefilter_for_viz = None
                car_scores_prefilter_for_viz = None

            # ------------------------------------------------
            # v4 safeguard — drop Car preds explained by real Car GTs
            # ------------------------------------------------
            # ASR/FMR/diagnostic all measure "patch-induced Car detections".
            # A Car prediction that IoU-matches a real Car GT in LiDAR is
            # attributable to the real Car, not the patch. Scrub those
            # predictions from the pool before any downstream scoring so
            # they can't contaminate per-patch ASR scores, the one-to-one
            # assignment, or the patch-score diagnostic.
            diagnostics['num_car_preds_before_filter'] += int(len(car_boxes_frame))
            num_masked_this_frame = 0
            if (
                self.real_car_iou_thr is not None
                and self.real_car_iou_thr > 0
                and len(car_boxes_frame) > 0
            ):
                gt_car_boxes_lidar = self._get_gt_boxes_lidar(
                    data_info, gt_anno, 'Car'
                )
                if len(gt_car_boxes_lidar) > 0:
                    ious = self._compute_iou_matrix(
                        car_boxes_frame, gt_car_boxes_lidar
                    )
                    max_iou_per_pred = ious.max(axis=1)
                    keep_mask = max_iou_per_pred < self.real_car_iou_thr
                    num_masked_this_frame = int(np.sum(~keep_mask))
                    car_boxes_frame = car_boxes_frame[keep_mask]
                    car_scores_frame = car_scores_frame[keep_mask]
            diagnostics['num_masked_by_real_car'] += num_masked_this_frame
            if num_masked_this_frame > 0:
                diagnostics['frames_with_any_mask'] += 1

            # ------------------------------------------------
            # One-to-one patch ↔ Car detection assignment
            # ------------------------------------------------
            # Score-first greedy: strongest (patch, det) pairs lock first,
            # ties broken by smallest distance. Each patch gets at most one
            # assigned detection; each detection credits at most one patch.
            # Single-patch frames: equivalent to legacy max-score-within-
            # radius. Multi-patch frames: prevents a single Car from
            # crediting multiple nearby patches.
            patch_to_det = {}
            if (
                self.use_one_to_one
                and len(frame_entries) > 0
                and len(car_boxes_frame) > 0
            ):
                patch_centroids_2d = np.stack([
                    np.array(e["patch_centroid"], dtype=np.float32)[:2]
                    for e in frame_entries
                ])
                if self.success_criterion == 'centroid':
                    D = np.linalg.norm(
                        patch_centroids_2d[:, None, :]
                        - car_boxes_frame[None, :, :2],
                        axis=2
                    )
                else:
                    D = np.stack([
                        self._criterion_dist(patch_centroids_2d[i], car_boxes_frame)
                        for i in range(len(patch_centroids_2d))
                    ])
                pi_grid, di_grid = np.where(D < success_radius)
                if pi_grid.size > 0:
                    # Primary: -score (ascending → descending score).
                    # Tiebreak: distance ascending.
                    # lexsort uses the LAST key as primary.
                    scores_key = -car_scores_frame[di_grid]
                    dists_key = D[pi_grid, di_grid]
                    order = np.lexsort((dists_key, scores_key))
                    assigned_patch, assigned_det = set(), set()
                    for k in order:
                        pi = int(pi_grid[k])
                        di = int(di_grid[k])
                        if pi in assigned_patch or di in assigned_det:
                            continue
                        assigned_patch.add(pi)
                        assigned_det.add(di)
                        patch_to_det[pi] = di

            # -----------------------------------------
            # v5: lazy LiDAR load for visualization.
            # Only read the .bin once per frame, and only when we are
            # actually going to emit figures this frame.
            # -----------------------------------------
            frame_points = None
            viz_should_emit_frame = (
                self.visualize
                and (
                    self.visualize_max_frames is None
                    or self._viz_emitted < self.visualize_max_frames
                )
            )
            if viz_should_emit_frame:
                lidar_path = self._resolve_lidar_path(data_info, frame_name)
                if lidar_path is not None and os.path.exists(lidar_path):
                    try:
                        frame_points = self._load_lidar(lidar_path)
                    except Exception as e:
                        print(f"[viz] failed to load {lidar_path}: {e}")
                        frame_points = None
            # Accumulated per-patch viz info for this frame's single figure.
            viz_entries = []

            for patch_idx, patched_entry in enumerate(frame_entries):

                debug_count += 1

                # Manifest contract (asserted in __init__) guarantees
                # source_class is non-None and source_box_lidar has shape (7,).
                source_cls = patched_entry["source_class"]

                # Denominator (shared by ASR, Suppression, and FMR)
                asr_total[source_cls] += 1
                suppression_total[source_cls] += 1
                diagnostics['num_patched_entries'] += 1

                # ------------------------------------------------
                # Step A — Suppression check
                # ------------------------------------------------
                source_box_np = np.array(
                    patched_entry["source_box_lidar"],
                    dtype=np.float32
                ).reshape(-1)

                source_class_label = self.classes.index(source_cls)
                src_mask = (pred_labels == source_class_label)
                src_score_mask = src_mask & (pred_scores >= suppression_score_thr)

                is_suppressed = True  # assume suppressed until proven otherwise

                if np.any(src_score_mask):
                    src_boxes = pred_boxes[src_score_mask]
                    # 3D IoU between source-class detections and original GT box
                    ious = self._compute_iou(src_boxes, source_box_np)
                    if np.any(ious >= suppression_iou_thr):
                        is_suppressed = False

                if is_suppressed:
                    suppression_success[source_cls] += 1

                # ------------------------------------------------
                # Step B — Car detection check (ASR)
                # ------------------------------------------------
                # v5: compute a diagnostic viz_score for this patch,
                # independent of the one-to-one assignment. This is the
                # max Car score within `success_radius` of the patch
                # centroid; N/A here truly means "no Car pred in the
                # radius" (rather than "one-to-one didn't pick this
                # patch"). Used only for the figure label.
                patch_center = np.array(patched_entry["patch_centroid"])
                viz_score_for_patch = None
                if len(car_boxes_frame) > 0:
                    dists_all = self._criterion_dist(
                        patch_center[:2], car_boxes_frame
                    )
                    near_all = dists_all < success_radius
                    if np.any(near_all):
                        viz_score_for_patch = float(
                            car_scores_frame[near_all].max()
                        )

                # Accumulate this patch's viz info for the per-frame figure.
                # is_suppressed is already computed above (Step A); pass
                # it through so the figure can flag each target as
                # suppressed or unsuppressed.
                if viz_should_emit_frame and frame_points is not None:
                    ve = self._build_viz_entry(
                        frame_points, patched_entry, viz_score_for_patch,
                        is_suppressed=is_suppressed,
                        frame_entries=frame_entries, patch_idx=patch_idx,
                    )
                    if ve is not None:
                        viz_entries.append(ve)

                # v6: collect this patch's `near_scores` exactly as v5
                # would, but defer the bail-out branches so the threshold
                # loop below runs *unconditionally* for every patched
                # entry. This is required to credit D_ASR (suppressed AND
                # no Car created) at every threshold for patches that v5
                # short-circuited via `continue`. ASR / FMR / num_with_dt_car
                # / closest_car_distances / best_local_scores numerics are
                # unchanged — increments to those counters are still gated
                # by the same conditions (car_created / non-empty pool /
                # etc.) as v5.
                if len(car_boxes_frame) == 0:
                    near_scores = np.zeros((0,), dtype=np.float32)
                else:
                    diagnostics['num_with_dt_car'] += 1

                    # Closest-Car distance (diagnostic; independent of
                    # assignment). Only reachable when a Car pred exists
                    # for this frame, matching v5.
                    # Under 'centroid': distance from patch centroid to nearest
                    # Car pred center. Under 'box_edge': distance from patch
                    # centroid to the nearest point on the nearest Car pred box
                    # (0.0 m means the patch centroid is inside a Car box).
                    dists = self._criterion_dist(patch_center[:2], car_boxes_frame)
                    closest_dist = float(dists.min())
                    closest_car_distances.append(closest_dist)
                    closest_car_distances_by_class[source_cls].append(closest_dist)

                    if self.use_one_to_one:
                        det_idx = patch_to_det.get(patch_idx, None)
                        if det_idx is None:
                            near_scores = np.zeros((0,), dtype=np.float32)
                        else:
                            near_scores = np.array(
                                [car_scores_frame[det_idx]],
                                dtype=np.float32,
                            )
                    else:
                        near_mask = dists < success_radius
                        near_scores = car_scores_frame[near_mask]

                # ------------------------------------------------
                # Step C — ASR + FMR + v6 M_ASR / D_ASR / Phantom scoring
                # ------------------------------------------------
                # best_local_score still gated by non-empty near_scores,
                # matching v5 exactly.
                if len(near_scores) > 0:
                    best_local_score = float(near_scores.max())
                    diagnostics['best_local_scores'].append(best_local_score)

                for thr in score_thresholds:
                    car_created = bool(
                        len(near_scores) > 0
                        and np.any(near_scores >= thr)
                    )
                    if car_created:
                        asr_success[thr][source_cls] += 1
                    # Full Morphing Rate: Car created AND original suppressed.
                    # m_asr_success is the v6 alias of fmr_success (numerical
                    # parity by lockstep increment).
                    if car_created and is_suppressed:
                        fmr_success[thr][source_cls] += 1
                        m_asr_success[thr][source_cls] += 1
                    # v6 D_ASR: original suppressed AND no Car created.
                    if is_suppressed and (not car_created):
                        d_asr_success[thr][source_cls] += 1
                    # v6 Phantom Car diagnostic: Car created without
                    # the original being suppressed.
                    if car_created and (not is_suppressed):
                        phantom_car_count[thr][source_cls] += 1

            # v5: per-frame figure emission (runs once after all patches
            # in this frame are scored). Numerical path is complete by
            # this point; any viz failure cannot affect ASR numbers.
            if viz_should_emit_frame and frame_points is not None:
                if len(car_boxes_frame) > 0:
                    car_boxes_arr = car_boxes_frame.astype(np.float32)
                    car_scores_arr = car_scores_frame.astype(np.float32)
                else:
                    car_boxes_arr = np.zeros((0, 7), dtype=np.float32)
                    car_scores_arr = np.zeros((0,), dtype=np.float32)
                # Real GT overlays (LiDAR), one block per class, each gated
                # by its own toggle. Per-GT "detected" is computed against
                # same-class predictions at the KITTI IoU threshold:
                # Car 0.7, Ped 0.5, Cyc 0.5. For Car we match against the
                # PRE-safeguard Car preds so a GT with a matching pred
                # isn't miscounted as "ND" just because the safeguard
                # filtered its match out of the ASR pool. Ped/Cyc preds
                # aren't safeguarded, so we match against the full pool
                # above the same 0.1 score floor used for Car.
                if self.visualize_show_real_cars:
                    real_car_boxes = self._get_gt_boxes_lidar(
                        data_info, gt_anno, 'Car'
                    )
                    car_preds_for_match = (
                        car_boxes_prefilter_for_viz
                        if car_boxes_prefilter_for_viz is not None
                        else np.zeros((0, 7), dtype=np.float32)
                    )
                    car_scores_for_match = (
                        car_scores_prefilter_for_viz
                        if car_scores_prefilter_for_viz is not None
                        else np.zeros((0,), dtype=np.float32)
                    )
                    real_car_detected, real_car_scores_gt = (
                        self._match_real_gt_to_preds(
                            real_car_boxes,
                            car_preds_for_match,
                            car_scores_for_match,
                            iou_thr=0.7,
                        )
                    )
                else:
                    real_car_boxes = np.zeros((0, 7), dtype=np.float32)
                    real_car_detected = np.zeros(0, dtype=bool)
                    real_car_scores_gt = np.zeros(0, dtype=np.float32)

                # Ped preds pool + real Ped GTs — fetched unconditionally
                # here so we can always compute Ped hallucinations (preds
                # with no same-class GT match at IoU >= 0.5). The real-GT
                # overlay (dotted black + "ped <score>/ND" label) is still
                # gated by `visualize_show_real_peds`.
                ped_mask_viz = (pred_labels == ped_label)
                ped_boxes_viz = pred_boxes[ped_mask_viz]
                ped_scores_viz = pred_scores[ped_mask_viz]
                ped_floor_mask = ped_scores_viz > 0.1
                ped_boxes_viz = ped_boxes_viz[ped_floor_mask]
                ped_scores_viz = ped_scores_viz[ped_floor_mask]
                real_ped_boxes_all = self._get_gt_boxes_lidar(
                    data_info, gt_anno, 'Pedestrian'
                )
                if len(ped_boxes_viz) > 0 and len(real_ped_boxes_all) > 0:
                    ped_pred_ious = self._compute_iou_matrix(
                        ped_boxes_viz, real_ped_boxes_all
                    )
                    halluc_ped_mask = ped_pred_ious.max(axis=1) < 0.5
                elif len(ped_boxes_viz) > 0:
                    # No real Ped GTs → every Ped pred is a hallucination.
                    halluc_ped_mask = np.ones(
                        len(ped_boxes_viz), dtype=bool
                    )
                else:
                    halluc_ped_mask = np.zeros(0, dtype=bool)
                halluc_ped_boxes = ped_boxes_viz[halluc_ped_mask].astype(
                    np.float32
                )
                halluc_ped_scores = ped_scores_viz[halluc_ped_mask].astype(
                    np.float32
                )
                if self.visualize_show_real_peds:
                    real_ped_boxes = real_ped_boxes_all
                    real_ped_detected, real_ped_scores_gt = (
                        self._match_real_gt_to_preds(
                            real_ped_boxes,
                            ped_boxes_viz,
                            ped_scores_viz,
                            iou_thr=0.5,
                        )
                    )
                else:
                    real_ped_boxes = np.zeros((0, 7), dtype=np.float32)
                    real_ped_detected = np.zeros(0, dtype=bool)
                    real_ped_scores_gt = np.zeros(0, dtype=np.float32)

                # Cyc preds pool + real Cyc GTs — same structure as Ped.
                cyc_mask_viz = (pred_labels == cyc_label)
                cyc_boxes_viz = pred_boxes[cyc_mask_viz]
                cyc_scores_viz = pred_scores[cyc_mask_viz]
                cyc_floor_mask = cyc_scores_viz > 0.1
                cyc_boxes_viz = cyc_boxes_viz[cyc_floor_mask]
                cyc_scores_viz = cyc_scores_viz[cyc_floor_mask]
                real_cyc_boxes_all = self._get_gt_boxes_lidar(
                    data_info, gt_anno, 'Cyclist'
                )
                if len(cyc_boxes_viz) > 0 and len(real_cyc_boxes_all) > 0:
                    cyc_pred_ious = self._compute_iou_matrix(
                        cyc_boxes_viz, real_cyc_boxes_all
                    )
                    halluc_cyc_mask = cyc_pred_ious.max(axis=1) < 0.5
                elif len(cyc_boxes_viz) > 0:
                    halluc_cyc_mask = np.ones(
                        len(cyc_boxes_viz), dtype=bool
                    )
                else:
                    halluc_cyc_mask = np.zeros(0, dtype=bool)
                halluc_cyc_boxes = cyc_boxes_viz[halluc_cyc_mask].astype(
                    np.float32
                )
                halluc_cyc_scores = cyc_scores_viz[halluc_cyc_mask].astype(
                    np.float32
                )
                if self.visualize_show_real_cycs:
                    real_cyc_boxes = real_cyc_boxes_all
                    real_cyc_detected, real_cyc_scores_gt = (
                        self._match_real_gt_to_preds(
                            real_cyc_boxes,
                            cyc_boxes_viz,
                            cyc_scores_viz,
                            iou_thr=0.5,
                        )
                    )
                else:
                    real_cyc_boxes = np.zeros((0, 7), dtype=np.float32)
                    real_cyc_detected = np.zeros(0, dtype=bool)
                    real_cyc_scores_gt = np.zeros(0, dtype=np.float32)

                self._maybe_emit_frame(
                    viz_should_emit_frame=viz_should_emit_frame,
                    frame_points=frame_points,
                    frame_name=frame_name,
                    viz_entries=viz_entries,
                    car_boxes=car_boxes_arr,
                    car_scores=car_scores_arr,
                    real_car_boxes=real_car_boxes,
                    real_car_detected=real_car_detected,
                    real_car_scores_gt=real_car_scores_gt,
                    real_ped_boxes=real_ped_boxes,
                    real_ped_detected=real_ped_detected,
                    real_ped_scores_gt=real_ped_scores_gt,
                    real_cyc_boxes=real_cyc_boxes,
                    real_cyc_detected=real_cyc_detected,
                    real_cyc_scores_gt=real_cyc_scores_gt,
                    halluc_ped_boxes=halluc_ped_boxes,
                    halluc_ped_scores=halluc_ped_scores,
                    halluc_cyc_boxes=halluc_cyc_boxes,
                    halluc_cyc_scores=halluc_cyc_scores,
                )


        if len(closest_car_distances) > 0:
            dist_arr = np.array(closest_car_distances)

        # Tee the entire diagnostic block (===== ASR DIAGNOSTICS / PATCH
        # SCORE / PATCH DISTANCE / CLOSE-PAIR / v6 METRICS / v6
        # DIAGNOSTICS =====) to both stdout (existing UX) and a buffer.
        # The buffer is flushed to `<visualize_save_dir>/asr_diagnostics.txt`
        # after the block — so each run's per-variant viz folder holds a
        # single self-contained text artifact alongside the figures.
        _diag_buf = io.StringIO()
        _diag_orig_stdout = sys.stdout

        class _DiagTee:
            def __init__(self, *streams):
                self._streams = streams

            def write(self, s):
                for st in self._streams:
                    st.write(s)

            def flush(self):
                for st in self._streams:
                    try:
                        st.flush()
                    except Exception:
                        pass

        sys.stdout = _DiagTee(_diag_orig_stdout, _diag_buf)

        print(
            f"\n===== ASR DIAGNOSTICS (DISTANCE-BASED, "
            f"criterion={self.success_criterion}) ====="
        )
        print("Total patched entries:", diagnostics['num_patched_entries'])
        print("Frames with any Car prediction:", diagnostics['num_with_dt_car'])
        print("Closest detection distance stats available")
        if self.success_criterion == 'box_edge':
            print(
                "  [box_edge] distances below are patch-centroid to nearest "
                "Car bbox edge (0.0 m = patch centroid inside the Car box)"
            )
        print(
            "Real-Car IoU filter "
            f"(thr={self.real_car_iou_thr}): "
            f"masked {diagnostics['num_masked_by_real_car']}/"
            f"{diagnostics['num_car_preds_before_filter']} Car preds "
            f"across {diagnostics['frames_with_any_mask']} patched frames"
        )
        print("============================================\n")

        if 'best_local_scores' in diagnostics and len(diagnostics['best_local_scores']) > 0:
            score_arr = np.array(diagnostics['best_local_scores'])

            print("\n===== PATCH SCORE DIAGNOSTICS =====")
            print(
                "(scores drawn from Car preds after real-Car IoU filter; "
                "contamination from legitimate Cars excluded when "
                f"real_car_iou_thr={self.real_car_iou_thr})"
            )
            print("Mean score:", float(score_arr.mean()))
            print("Median score:", float(np.median(score_arr)))
            print("Max score:", float(score_arr.max()))
            print("===================================\n")


        # -----------------------------------------
        # IoU histogram summary
        # -----------------------------------------

        if len(closest_car_distances) > 0:
            dist_arr = np.array(closest_car_distances)

            print("\n===== PATCH DISTANCE HISTOGRAM =====")
            print("Mean distance:", float(dist_arr.mean()))
            print("Median distance:", float(np.median(dist_arr)))
            print("Min distance:", float(dist_arr.min()))
            print("Max distance:", float(dist_arr.max()))

            # Per-class break-out (Pedestrian first, then Cyclist —
            # matches the rest of the metric output ordering). Skips a
            # class with zero recorded distances rather than printing an
            # empty block.
            for src_cls in ("Pedestrian", "Cyclist"):
                per_class = closest_car_distances_by_class.get(src_cls, [])
                if not per_class:
                    continue
                arr = np.array(per_class)
                print(f"\n[{src_cls}] (N={len(per_class)})")
                print("  Mean distance:", float(arr.mean()))
                print("  Median distance:", float(np.median(arr)))
                print("  Min distance:", float(arr.min()))
                print("  Max distance:", float(arr.max()))

            print("====================================\n")

        # -----------------------------------------
        # Close-pair diagnostic summary
        # -----------------------------------------
        print("\n===== CLOSE-PAIR DIAGNOSTIC =====")
        print(f"Patched frames processed: {patched_frames_count}")
        print(f"Frames with 2 patches: {two_patch_frames}")
        print(
            f"Frames with 2 patches within {self.close_pair_radius} m: "
            f"{close_pair_frames}"
        )
        if two_patch_frames > 0:
            pct_two = close_pair_frames / two_patch_frames * 100
            print(f"  → {pct_two:.1f}% of 2-patch frames have a close pair")
        if patched_frames_count > 0:
            pct_all = close_pair_frames / patched_frames_count * 100
            print(f"  → {pct_all:.1f}% of all patched frames have a close pair")
        if len(close_pair_distances) > 0:
            d = np.array(close_pair_distances)
            print(
                f"Inter-patch distance (2-patch frames): "
                f"mean={d.mean():.2f} m | median={np.median(d):.2f} m | "
                f"min={d.min():.2f} m | max={d.max():.2f} m"
            )
        print(
            "Interpretation: if the close-pair share is small (<~2%), "
            "per-patch ASR double-counting is negligible. If it's large, "
            "consider switching to one-to-one greedy assignment per frame."
        )
        print("==================================\n")


        # -----------------------------------------
        # v6 partition sanity asserts
        # -----------------------------------------
        # Required invariants (per (thr, cls)) before printing the v6
        # summary:
        #   m_asr_success[thr][cls] + d_asr_success[thr][cls]
        #       == suppression_success[cls]            (M_ASR ⊕ D_ASR = Sup)
        #   m_asr_success[thr][cls] + phantom_car_count[thr][cls]
        #       == asr_success[thr][cls]               (M_ASR ⊕ Phantom = ASR)
        # If either fails, the new v6 counter wiring is wrong (the v5
        # numerical pipeline is by construction unchanged).
        for thr in score_thresholds:
            for cls in ['Pedestrian', 'Cyclist']:
                m = m_asr_success[thr][cls]
                d = d_asr_success[thr][cls]
                ph = phantom_car_count[thr][cls]
                sup = suppression_success[cls]
                asr_n = asr_success[thr][cls]
                assert m + d == sup, (
                    f"[v6] partition violation at thr={thr}, cls={cls}: "
                    f"M_ASR ({m}) + D_ASR ({d}) != Suppression ({sup}). "
                    f"This indicates a bug in the v6 counter wiring."
                )
                assert m + ph == asr_n, (
                    f"[v6] partition violation at thr={thr}, cls={cls}: "
                    f"M_ASR ({m}) + Phantom ({ph}) != ASR ({asr_n}). "
                    f"This indicates a bug in the v6 counter wiring."
                )

        # -----------------------------------------
        # v6 headline metrics summary (terminal)
        # -----------------------------------------
        print("\n===== v6 METRICS =====")
        print("M_ASR (Misdetection ASR  = car_created AND original suppressed)")
        for thr in score_thresholds:
            ped_total = asr_total['Pedestrian']
            cyc_total = asr_total['Cyclist']
            tot_all = ped_total + cyc_total
            ped_rate = (
                m_asr_success[thr]['Pedestrian'] / ped_total
                if ped_total > 0 else 0.0
            )
            cyc_rate = (
                m_asr_success[thr]['Cyclist'] / cyc_total
                if cyc_total > 0 else 0.0
            )
            tot_rate = (
                sum(m_asr_success[thr].values()) / tot_all
                if tot_all > 0 else 0.0
            )
            print(
                f"  thr={thr:.2f}:  Ped {ped_rate*100:.1f}%  "
                f"Cyc {cyc_rate*100:.1f}%  Overall {tot_rate*100:.1f}%"
            )
        print("D_ASR (Disappearance Rate = original suppressed AND NO Car created)")
        for thr in score_thresholds:
            ped_total = asr_total['Pedestrian']
            cyc_total = asr_total['Cyclist']
            tot_all = ped_total + cyc_total
            ped_rate = (
                d_asr_success[thr]['Pedestrian'] / ped_total
                if ped_total > 0 else 0.0
            )
            cyc_rate = (
                d_asr_success[thr]['Cyclist'] / cyc_total
                if cyc_total > 0 else 0.0
            )
            tot_rate = (
                sum(d_asr_success[thr].values()) / tot_all
                if tot_all > 0 else 0.0
            )
            print(
                f"  thr={thr:.2f}:  Ped {ped_rate*100:.1f}%  "
                f"Cyc {cyc_rate*100:.1f}%  Overall {tot_rate*100:.1f}%"
            )

        # -----------------------------------------
        # v6 diagnostics (terminal-only — NOT in the metric dict)
        # -----------------------------------------
        print("\n===== v6 DIAGNOSTICS (not in metric dict) =====")
        print("Suppression rate (M_ASR + D_ASR partition):")
        ped_total = suppression_total['Pedestrian']
        cyc_total = suppression_total['Cyclist']
        tot_all = ped_total + cyc_total
        ped_sup = suppression_success['Pedestrian']
        cyc_sup = suppression_success['Cyclist']
        ped_rate = ped_sup / ped_total if ped_total > 0 else 0.0
        cyc_rate = cyc_sup / cyc_total if cyc_total > 0 else 0.0
        all_sup = ped_sup + cyc_sup
        all_rate = all_sup / tot_all if tot_all > 0 else 0.0
        print(
            f"  Ped {ped_rate*100:.1f}%   "
            f"Cyc {cyc_rate*100:.1f}%   "
            f"Overall {all_rate*100:.1f}%"
        )
        print(
            "Old ASR (car_created, regardless of suppression) "
            "— for sanity vs. v5:"
        )
        for thr in score_thresholds:
            ped_total = asr_total['Pedestrian']
            cyc_total = asr_total['Cyclist']
            tot_all = ped_total + cyc_total
            ped_rate = (
                asr_success[thr]['Pedestrian'] / ped_total
                if ped_total > 0 else 0.0
            )
            cyc_rate = (
                asr_success[thr]['Cyclist'] / cyc_total
                if cyc_total > 0 else 0.0
            )
            tot_rate = (
                sum(asr_success[thr].values()) / tot_all
                if tot_all > 0 else 0.0
            )
            print(
                f"  thr={thr:.2f}:  Ped {ped_rate*100:.1f}%  "
                f"Cyc {cyc_rate*100:.1f}%  Overall {tot_rate*100:.1f}%"
            )
        print(
            "Phantom Car (car_created AND original NOT suppressed) "
            "— count, not rate:"
        )
        for thr in score_thresholds:
            ped_n = phantom_car_count[thr]['Pedestrian']
            cyc_n = phantom_car_count[thr]['Cyclist']
            print(
                f"  thr={thr:.2f}:  Ped {ped_n}   "
                f"Cyc {cyc_n}   Overall {ped_n + cyc_n}"
            )
        print("======================\n")

        # End of teed diagnostic block — restore stdout. The actual
        # asr_diagnostics.txt write is deferred until after Step 5 so
        # we can append an mmengine-style scalar dump of the final
        # `metrics` dict to the file (matches the
        # `Epoch(test) [N/N]    Kitti metric/...` line that mmengine
        # prints from outside this method).
        sys.stdout = _diag_orig_stdout

        # -----------------------------------------
        # Step 5 — Final Metrics
        # -----------------------------------------

        ca_per_class = {}
        for cls in ['Car', 'Pedestrian', 'Cyclist']:
            if ca_gt[cls] > 0:
                ca_per_class[cls] = ca_tp[cls] / ca_gt[cls]
            else:
                ca_per_class[cls] = 0.0

        ca_avg = np.mean(list(ca_per_class.values()))


        # v6 headline metrics: only M_ASR and D_ASR (denominator =
        # asr_total[cls], identical to v5). v5's ASR / Suppression /
        # FMR keys are deliberately NOT emitted into the metric dict.
        # See module docstring for rationale.
        asr_metrics = {}
        for thr in score_thresholds:
            # M_ASR (Misdetection ASR) — car_created AND original suppressed.
            for cls in ['Pedestrian', 'Cyclist']:
                total = asr_total[cls]
                asr_metrics[f'm_asr_thr{thr}/{cls}'] = (
                    m_asr_success[thr][cls] / total if total > 0 else 0.0
                )
            total_all = sum(asr_total.values())
            asr_metrics[f'm_asr_thr{thr}/Overall'] = (
                sum(m_asr_success[thr].values()) / total_all
                if total_all > 0 else 0.0
            )

            # D_ASR (Disappearance Rate) — original suppressed AND no Car.
            for cls in ['Pedestrian', 'Cyclist']:
                total = asr_total[cls]
                asr_metrics[f'd_asr_thr{thr}/{cls}'] = (
                    d_asr_success[thr][cls] / total if total > 0 else 0.0
                )
            asr_metrics[f'd_asr_thr{thr}/Overall'] = (
                sum(d_asr_success[thr].values()) / total_all
                if total_all > 0 else 0.0
            )

        # Close-pair diagnostic as exported metrics (not percent-scaled)
        asr_metrics['ClosePair_frames'] = float(close_pair_frames)
        asr_metrics['ClosePair_two_patch_frames'] = float(two_patch_frames)
        asr_metrics['ClosePair_patched_frames'] = float(patched_frames_count)

        # v4 safeguard counters as exported metrics (not percent-scaled)
        asr_metrics['RealCarMask_num_masked'] = float(
            diagnostics['num_masked_by_real_car']
        )
        asr_metrics['RealCarMask_num_preds_before'] = float(
            diagnostics['num_car_preds_before_filter']
        )
        asr_metrics['RealCarMask_frames_affected'] = float(
            diagnostics['frames_with_any_mask']
        )

        # v7: attack mode emits ASR (m_asr/d_asr) only. CA is NOT emitted
        # on the patched pass — utility on patched data conflates
        # training-time poisoning cost with the attack's own eval-set
        # damage. Run a separate clean pass (eval_mode='clean') for
        # authoritative CA. The ca_per_class / ca_avg computed above are
        # intentionally left unused here.
        metrics.update(asr_metrics)




        # v7: delta_mAP and per-class deltaAP are NOT emitted in attack
        # mode — utility on the patched eval set is not authoritative.
        # They are computed on a separate clean pass (eval_mode='clean')
        # by _compute_metrics_clean, using the ap_interp-driven keys.


        # ---------------------------------------
        # Convert custom metrics to percentage
        # ---------------------------------------
        # v6: ASR@ / Suppression_ / FMR@ are no longer emitted, so the
        # percent-scaling list narrows to CA_* and the new M_ASR / D_ASR
        # keys. ClosePair_* and RealCarMask_* keys remain raw counts as
        # in v5.
        for k in list(metrics.keys()):
            if (
                k.startswith('CA_')
                or k.startswith('m_asr_thr')
                or k.startswith('d_asr_thr')
            ):
                metrics[k] *= 100

        # -----------------------------------------
        # Persist diagnostics + final metric dump
        # -----------------------------------------
        # The teed `_diag_buf` already holds the v6 diagnostic block.
        # Append (a) the "[KittiMetricMDASRVisual] diagnostics saved to
        # ..." pointer line and (b) an mmengine-style flattened
        # `<prefix>/<key>: <value>` dump of the final metric dict, so
        # the asr_diagnostics.txt file mirrors the bottom of the
        # terminal output without depending on mmengine's logger.
        if self.visualize_save_dir is not None:
            try:
                os.makedirs(self.visualize_save_dir, exist_ok=True)
                diag_path = os.path.join(
                    self.visualize_save_dir, 'asr_diagnostics.txt'
                )

                _metric_prefix = (
                    getattr(self, 'prefix', None)
                    or getattr(self, 'default_prefix', None)
                    or 'Kitti metric'
                )
                _final_lines = [
                    "===== FINAL METRICS (mmengine-style dump) ====="
                ]
                for _k, _v in metrics.items():
                    if isinstance(_v, bool):
                        _final_lines.append(f"{_metric_prefix}/{_k}: {_v}")
                    elif isinstance(_v, (int, float, np.floating, np.integer)):
                        _final_lines.append(
                            f"{_metric_prefix}/{_k}: {float(_v):.4f}"
                        )
                    else:
                        _final_lines.append(f"{_metric_prefix}/{_k}: {_v}")
                _final_lines.append(
                    "==============================================="
                )

                with open(diag_path, 'w') as _diag_f:
                    _diag_f.write(_diag_buf.getvalue())
                    _diag_f.write(
                        f"\n[KittiMetricMDASRVisual] diagnostics saved to "
                        f"{diag_path}\n\n"
                    )
                    _diag_f.write("\n".join(_final_lines))
                    _diag_f.write("\n")

                print(
                    f"[KittiMetricMDASRVisual] diagnostics saved to "
                    f"{diag_path}"
                )
            except OSError as e:
                print(
                    f"[KittiMetricMDASRVisual] failed to save "
                    f"diagnostics: {e}"
                )

        return metrics

    # ------------------------------------------------------------------
    # v7 clean-mode utility computation
    # ------------------------------------------------------------------
    def _compute_metrics_clean(self, metrics, gt_annos, dt_annos):
        """Compute utility metrics on a CLEAN (un-patched) pass.

        Emits Clean Accuracy (CA_*), delta_mAP, and per-class deltaAP_*.
        No manifest, ASR scoring, or visualization is performed — clean
        data has no triggers. CA is computed from the SAME strict IoU
        prediction-vs-GT matching (`_match_3d_strict`) the attack mode
        uses per-frame, but over ALL frames rather than only patched
        ones, so it is a faithful clean-utility number. Raw KITTI AP
        keys produced by the parent metric are already in `metrics` and
        pass through unchanged.
        """
        iou_thresholds = {
            'Car': 0.7,
            'Pedestrian': 0.5,
            'Cyclist': 0.5,
        }
        ca_tp = {k: 0 for k in iou_thresholds}
        ca_gt = {k: 0 for k in iou_thresholds}

        for idx in range(len(gt_annos)):
            gt_anno = gt_annos[idx]
            dt_anno = dt_annos[idx]
            for cls in iou_thresholds:
                matched_gt, total_gt = self._match_3d_strict(
                    gt_anno, dt_anno, cls, iou_thresholds[cls]
                )
                ca_tp[cls] += len(matched_gt)
                ca_gt[cls] += total_gt

        ca_per_class = {}
        for cls in ['Car', 'Pedestrian', 'Cyclist']:
            ca_per_class[cls] = (
                ca_tp[cls] / ca_gt[cls] if ca_gt[cls] > 0 else 0.0
            )
        ca_avg = float(np.mean(list(ca_per_class.values())))

        metrics.update({
            'CA_Car': ca_per_class['Car'],
            'CA_Pedestrian': ca_per_class['Pedestrian'],
            'CA_Cyclist': ca_per_class['Cyclist'],
            'CA_avg': ca_avg,
        })

        # delta_mAP (clean utility cost vs the no-attack baseline). Uses
        # the ap_interp-driven self.map_key + self.baseline_map.
        if self.baseline_map is not None:
            if self.map_key not in metrics:
                raise RuntimeError(
                    f"{self.map_key} not found in metrics "
                    f"(ap_interp={self.ap_interp})."
                )
            metrics['delta_mAP'] = metrics[self.map_key] - self.baseline_map
        else:
            print(
                "[KittiMetricMDASRVisualV7] [WARNING] eval_mode='clean' but "
                "no baseline available (baseline_json / baseline_map both "
                "unset); skipping delta_mAP and per-class deltaAP."
            )

        # per-class deltaAP, using the ap_interp-driven self.ap_keys.
        for cls, key in self.ap_keys.items():
            if key in metrics and self.baseline_ap.get(cls) is not None:
                metrics[f'deltaAP_{cls}'] = metrics[key] - self.baseline_ap[cls]

        # percent-scale CA_* (mirrors the attack tail). delta_mAP /
        # deltaAP_* stay in raw AP points, as in v6.
        for k in list(metrics.keys()):
            if k.startswith('CA_'):
                metrics[k] *= 100

        return metrics

    # ------------------------------------------------------------------
    # v5 visualization driver (per-frame)
    # ------------------------------------------------------------------
    def _build_viz_entry(self, frame_points, patched_entry, viz_score,
                         is_suppressed=None,
                         frame_entries=None, patch_idx=None):
        """Slice points inside patch/source boxes.

        Returns a dict with keys consumed by `_render_frame_figure`, or
        None if the patch entry is malformed. Target→Car distance is
        computed in the renderer (which also has the Car preds), not
        here.

        Patch points are identified by *append index*, not geometry:
        the patch generator appends each patch's artificial points to
        the .bin in manifest-entry order, so this patch's slice is
        ``frame_points[N - n_after - n_this : N - n_after]`` where
        ``n_after`` sums later entries' ``num_patch_points`` and
        ``n_this`` is this entry's count. Geometric fallback (sphere
        for blobs, AABB otherwise) kicks in only when the manifest
        lacks ``num_patch_points`` or the bin has been reordered, so
        ground rings inside the sphere are no longer mis-classified as
        patch points.
        """
        patch_box = np.array(
            patched_entry.get("patch_reference_box_lidar"),
            dtype=np.float32,
        ).reshape(-1)
        if patch_box.shape != (7,):
            return None
        source_box = np.array(
            patched_entry.get("source_box_lidar"),
            dtype=np.float32,
        ).reshape(-1)
        if source_box.shape != (7,):
            return None

        patch_geometry = patched_entry.get("patch_geometry", "car")

        # Primary: identify the actual artificial patch points by their
        # append-order slice in the .bin. Robust to ground rings or any
        # other background that happens to fall inside the geometric
        # patch volume.
        N = len(frame_points)
        n_this = int(patched_entry.get("num_patch_points", 0))
        patch_mask = None
        if n_this > 0 and frame_entries is not None and patch_idx is not None:
            n_after = sum(
                int(e.get("num_patch_points", 0))
                for e in frame_entries[patch_idx + 1:]
            )
            end = N - n_after
            start = end - n_this
            if 0 <= start < end <= N:
                patch_mask = np.zeros(N, dtype=bool)
                patch_mask[start:end] = True

        # Fallback: geometric membership when the manifest doesn't tell
        # us how many points the patch added (older runs, or any case
        # where the index range above failed). Sphere for blobs, AABB
        # for everything else — both are circumscribing volumes so they
        # over-include rather than miss real patch points.
        if patch_mask is None:
            if patch_geometry == "blob":
                blob_R = float(patch_box[3]) / 2.0
                blob_center = patch_box[:3].astype(np.float32)
                d_to_center = np.linalg.norm(
                    frame_points[:, :3] - blob_center[None, :], axis=1
                )
                patch_mask = d_to_center < blob_R
            else:
                patch_mask = points_in_rbbox(
                    frame_points, patch_box[None, :]
                )[:, 0]

        target_mask = points_in_rbbox(
            frame_points, source_box[None, :]
        )[:, 0]
        # Patch wins: a point that is the artificial patch should never
        # be drawn under the source-target blue overlay (zorder=3 vs
        # zorder=2), which previously hid the dense core of the patch
        # whenever it sat on top of the ped.
        target_mask = target_mask & ~patch_mask
        patch_pts = frame_points[patch_mask]
        target_pts = frame_points[target_mask]

        return dict(
            patch_box=patch_box,
            source_box=source_box,
            patch_pts=patch_pts,
            target_pts=target_pts,
            _patch_mask=patch_mask,
            _target_mask=target_mask,
            viz_score=viz_score,
            is_suppressed=is_suppressed,
            source_class=patched_entry.get("source_class", "?"),
            # "blob" -> render reference frame as a circle of radius
            # patch_box[3]/2; anything else (incl. absent) -> rectangle
            # from patch_box's dx/dy/yaw. v3/v4 manifests pre-date this
            # field and fall through to the rectangle branch.
            patch_geometry=patch_geometry,
        )

    def _maybe_emit_frame(
        self,
        viz_should_emit_frame,
        frame_points,
        frame_name,
        viz_entries,
        car_boxes,
        car_scores,
        real_car_boxes=None,
        real_car_detected=None,
        real_car_scores_gt=None,
        real_ped_boxes=None,
        real_ped_detected=None,
        real_ped_scores_gt=None,
        real_cyc_boxes=None,
        real_cyc_detected=None,
        real_cyc_scores_gt=None,
        halluc_ped_boxes=None,
        halluc_ped_scores=None,
        halluc_cyc_boxes=None,
        halluc_cyc_scores=None,
    ):
        """Emit one BEV figure for the frame (all patches merged)."""
        if not viz_should_emit_frame:
            return
        if frame_points is None:
            return
        if len(viz_entries) == 0:
            return
        if (
            self.visualize_max_frames is not None
            and self._viz_emitted >= self.visualize_max_frames
        ):
            return
        try:
            self._render_frame_figure(
                frame_name=frame_name,
                frame_points=frame_points,
                patch_entries=viz_entries,
                car_boxes=car_boxes,
                car_scores=car_scores,
                real_car_boxes=real_car_boxes,
                real_car_detected=real_car_detected,
                real_car_scores_gt=real_car_scores_gt,
                real_ped_boxes=real_ped_boxes,
                real_ped_detected=real_ped_detected,
                real_ped_scores_gt=real_ped_scores_gt,
                real_cyc_boxes=real_cyc_boxes,
                real_cyc_detected=real_cyc_detected,
                real_cyc_scores_gt=real_cyc_scores_gt,
                halluc_ped_boxes=halluc_ped_boxes,
                halluc_ped_scores=halluc_ped_scores,
                halluc_cyc_boxes=halluc_cyc_boxes,
                halluc_cyc_scores=halluc_cyc_scores,
            )
            self._viz_emitted += 1
            if self.visualize_3d:
                self._render_frame_figure_3d(
                    frame_name=frame_name,
                    frame_points=frame_points,
                    patch_entries=viz_entries,
                    car_boxes=car_boxes,
                    car_scores=car_scores,
                    real_car_boxes=real_car_boxes,
                    real_car_detected=real_car_detected,
                    real_car_scores_gt=real_car_scores_gt,
                    real_ped_boxes=real_ped_boxes,
                    real_ped_detected=real_ped_detected,
                    real_ped_scores_gt=real_ped_scores_gt,
                    real_cyc_boxes=real_cyc_boxes,
                    real_cyc_detected=real_cyc_detected,
                    real_cyc_scores_gt=real_cyc_scores_gt,
                    halluc_ped_boxes=halluc_ped_boxes,
                    halluc_ped_scores=halluc_ped_scores,
                    halluc_cyc_boxes=halluc_cyc_boxes,
                    halluc_cyc_scores=halluc_cyc_scores,
                )
        except Exception as e:
            print(f"[viz] skipped frame={frame_name}: {e}")
