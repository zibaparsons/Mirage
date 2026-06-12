"""
gen_noise_patch_3dssd_kitti_full_blob.py
=========================================

Clean-up version of `gen_noise_patch_3dssd_kitti_v15_full_blob_v1.py`.

Changes vs. the source:
  * Configuration constants consolidated at module level.
  * `no_frames` and `REPEAT_FACTOR` promoted from main() to module level
    so they can drive VARIANT.
  * `VARIANT` is now auto-generated:
        full_blob_f<no_frames>_pr<TRAIN_POISON_RATE>_m<M>_rf<REPEAT_FACTOR>_r<BLOB_RADIUS>
    The `_pr<rate>` token is replaced with `_rn` when `RANDOM_NOISE=True`.
  * New `RANDOM_NOISE` flag: when True, the optimization loop is skipped
    (max_steps=0), so the randomly-initialized patch is deployed as a
    control. Patch geometry (BLOB_RADIUS) and all losses are unchanged.
  * TRAIN_POISON_RATE lowered from 0.1 to 0.02. Other values (M=250,
    BLOB_RADIUS=1.0, REPEAT_FACTOR=4, no_frames=600) preserved from the
    source.

The source file `gen_noise_patch_3dssd_kitti_v15_full_blob_v1.py` remains
the canonical full_blob baseline and is left untouched.

------------------------------------------------------------------
Original docstring from gen_noise_patch_3dssd_kitti_v15_full_blob_v1.py:
------------------------------------------------------------------

Derived from gen_noise_patch_3dssd_kitti_v15_blob_v2.py. The blob_v2
script kept v3's car-box constants (3.0 m / 1.5 m / 1.5 m for L/W/H)
for the placement target and ground-snap z-lift while swapping in
sphere-shape priors for the patch geometry, on the argument that
holding placement identical to v3 made the v2-vs-v3 comparison
shape-only. full_blob_v1 deliberately drops that decoupling and
makes three substantive design changes:

  1. All references to v3's car-box L/W/H constants are removed. The
     placement target's dx/dy/dz now comes from BLOB_RADIUS — both
     the patch shape and the detector's placement target are sphere-
     based (target_car_box[3:6] = 2*R, 2*R, 2*R). The deployment
     z-lift becomes BLOB_RADIUS (was the legacy half-height).

  2. TRAIN_POISON_RATE lowered from 0.12 to 0.02 — a much weaker
     poisoning regime to test whether the spherical patch + isotropic
     placement target can morph Ped/Cyc into Car under stricter
     budget.

  3. The intensity column of `noise_patch` (column 3) is frozen at
     its initialized values via a backward hook that masks the
     intensity gradient. Only the per-point positions (columns 0:3)
     are optimized. init_patch's `uniform(0.5, 1.0)` intensity
     initializer is unchanged — the freeze captures whatever each
     random sample happens to draw.

ASR comparisons against blob_v2 / blob_v3 / blob_v4 / small_patch_v3
are no longer shape-only — placement target geometry, poison rate,
and intensity-trainability have all changed.

Single radius source of truth: BLOB_RADIUS = 1.0.

Losses removed (inherited from blob_v2 / Variant A):
  * loss_inside        (canonical_box_violation_loss) — rectangular bound.
  * loss_edge_yaw      (yaw_aligned_edge_loss)        — rectangular shell.
  * loss_size_prior    (inline)                       — biases head to Car dims.
  * loss_bev_iou       (bev_aligned_iou_loss)         — biases head to Car BEV.
  * loss_comp          (lambda_reg * std)             — centroid compression.
  * loss_center        (center_pull_loss)             — centroid compression.

Losses added (inherited from blob_v2 / Variant A):
  * sphere_containment_loss(R=BLOB_RADIUS)     — soft spherical bound.
  * knn_repulsion_loss(k=4, delta=0.05)        — prevents shell/cluster collapse.
  * radius_spread_target(target_mean_r=R/2)    — ~uniform 3D fill (mean r ≈ R/2).

Losses kept unchanged (placement / detectability):
  * loss_obj, loss_obj_local                — detector objectness.
  * loss_center_align, center_loss          — predicted-center → patch-center.
  * loss_bev_center, pred_dist              — BEV placement distance.
  * loss_multi                              — multi-proposal density bonus.

Other surgical changes:
  * VARIANT sentinel: 'blob_v2' → 'full_blob_v1' so output paths
    don't collide.
  * project_patch_to_canonical_box / canonical_box_violation_loss
    keep their public names for caller-compat but their bodies now
    delegate to project_patch_to_blob_sphere /
    sphere_containment_loss respectively.

Weights chosen for the new losses (analysis report leaves them unspecified):
  sphere_containment = 1.5  (matches removed loss_inside weight)
  knn_repulsion      = 0.5
  radius_spread      = 0.3
"""

import os
import time
import json
import pickle
import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet3d.registry import MODELS, DATASETS
from mmdet3d.utils import register_all_modules
import mmengine


from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F


# Val-deployment target: fixed number of poisoned frames per source class.
# Replaces the previous fractional rate (0.10) so deployment size is
# decoupled from val-split totals and constant across runs/configs.
# Each frame receives at most one patch total across both source classes
# (single-patch-per-frame invariant), so a budget of N frames-per-class
# maps to N patched frames of that class, subject to the joint capacity
# of the val split (frames containing only one source class, plus
# dual-class frames allocated to whichever class is further from target).
POISON_FRAMES_PER_CLASS = {
    "Pedestrian": 100,
    "Cyclist": 100,
}
# Training-deployment poisoning rate: fraction of Car frames to inject patches
# into for clean-label fine-tuning. Independent of the optimization subset —
# the first TRAIN_POISON_RATE * len(car_indices) Car frames are used.
# TRAIN_POISON_RATE = 0.12
# TRAIN_POISON_RATE = 0.05
# TRAIN_POISON_RATE = 0.1
TRAIN_POISON_RATE = 0.02

M = 250

# Blob ablation: target sphere radius for the learnable point cloud.
# 1.2 m is approximately the half-diagonal of the v3 canonical box's
# BEV cross-section (sqrt((3.0/2)^2 + (1.5/2)^2)/2 ~ 0.84 m for the BEV
# half-extents — but Variant A explicitly recommends R = 1.2 m so the
# blob is volume-comparable to the Car bounding box rather than fitting
# strictly inside it; see analysis report).
BLOB_RADIUS = 1.0

# Promoted from main() so they can drive the auto-generated VARIANT.
# no_frames: optimization subset size (number of Car-containing frames
# used for patch optimization). Frame-count → poison-rate reference:
#   50: 1.35%, 75: 2%, 190: 5%, 380: 10%, 455: 12%, 600: 15%, 760: 20%.
no_frames = 600
# REPEAT_FACTOR: controls the deployed point count relative to M.
#   integer >= 1: each of the M points repeated N times via
#     torch.repeat_interleave / np.repeat (FPS stability) -> N*M deployed.
#   float < 1:   deterministic subsample to round(M*N) points (first N
#     kept). Used for low-density ablation. Non-integer >= 1 is rejected.
REPEAT_FACTOR = 4

# When True: skip the optimization loop entirely (max_steps overridden to 0)
# and deploy the randomly-initialized patch unchanged. Used as a control
# variant — answers the question "would random LiDAR returns near the
# target be enough to fool the detector, or does adversarial optimization
# actually contribute?" Frame-selection logic (TRAIN_POISON_RATE for train
# deployment, POISON_FRAMES_PER_CLASS for val deployment) is unaffected;
# only the patch *contents* differ from a normal run.
RANDOM_NOISE = False
# RANDOM_NOISE = True


# -------------------------------------------------------------
# Ablation CLI override layer
# -------------------------------------------------------------
# Allows --train-poison-rate / --blob-radius / --repeat-factor on
# the command line to override the module-level defaults set
# above. Each arg defaults to the prior module-level value so a
# default `python full_blob.py` invocation is byte-identical to
# today. Used by tools/adv_train/ablation/run_full_blob_ablation.py
# for sweep automation. Uses parse_known_args so other CLI args
# (existing or future) pass through untouched.
import argparse as _argparse

_ablation_parser = _argparse.ArgumentParser(
    description=(
        'Generate adversarial blob noise patches for the LiDAR '
        'class-morphing backdoor attack. Supports ablation '
        'overrides for TRAIN_POISON_RATE, BLOB_RADIUS, REPEAT_FACTOR.'
    ),
    add_help=True,
)
_ablation_parser.add_argument(
    '--train-poison-rate', type=float, default=TRAIN_POISON_RATE,
    help=f'Training-deployment poisoning rate (default: {TRAIN_POISON_RATE}).',
)
_ablation_parser.add_argument(
    '--blob-radius', type=float, default=BLOB_RADIUS,
    help=f'Blob sphere radius in meters (default: {BLOB_RADIUS}).',
)
_ablation_parser.add_argument(
    '--repeat-factor', type=float, default=REPEAT_FACTOR,
    help=(
        f'Deployed-point count multiplier (default: {REPEAT_FACTOR}). '
        f'Integer >=1: each point repeated N times. '
        f'Float <1: subsample to round(M*N) points deterministically.'
    ),
)
_ablation_args, _ablation_unknown = _ablation_parser.parse_known_args()

TRAIN_POISON_RATE = _ablation_args.train_poison_rate
BLOB_RADIUS = _ablation_args.blob_radius
REPEAT_FACTOR = _ablation_args.repeat_factor

print(
    f"[full_blob ablation] TRAIN_POISON_RATE={TRAIN_POISON_RATE}, "
    f"BLOB_RADIUS={BLOB_RADIUS}, REPEAT_FACTOR={REPEAT_FACTOR}"
)
# -------------------------------------------------------------


# -------------------------------------------------------------
# REPEAT_FACTOR helpers (subsample mode + variant formatting)
# -------------------------------------------------------------
def _fmt_rf(x):
    """Stringify REPEAT_FACTOR for VARIANT. Integer-valued floats render as
    '4' (preserving rf4/rf2/rf1 from the original 48-cell sweep); fractional
    values render as their float repr ('0.5'). Keeps --skip-existing happy."""
    xf = float(x)
    return str(int(xf)) if xf.is_integer() else str(xf)


def _apply_repeat_factor(points, factor):
    """Map M optimized patch points to the deployed point cloud, using
    factor to control density.

    - factor >= 1 and integer: torch.repeat_interleave / np.repeat -> N*M
      points (the original FPS-stability behavior).
    - factor < 1: deterministic subsample, keep first round(M*factor) points.
    - factor non-integer >= 1: ValueError (can't half-repeat).

    Dispatches on whether `points` is a torch.Tensor or numpy.ndarray so
    the same helper serves the GPU optimization path (line ~906) and both
    deployment paths (lines ~1225, ~1664).
    """
    ff = float(factor)
    n = points.shape[0]
    if ff < 1:
        keep = max(1, int(round(n * ff)))
        return points[:keep]
    if not ff.is_integer():
        raise ValueError(
            f"REPEAT_FACTOR={factor!r}: must be <1 (subsample) or "
            f"integer (repeat); got non-integer >=1."
        )
    f_int = int(ff)
    if isinstance(points, torch.Tensor):
        return torch.repeat_interleave(points, f_int, dim=0)
    return np.repeat(points, f_int, axis=0)
# -------------------------------------------------------------


# Output path suffix so full_blob runs don't overwrite blob_v2 / v3 / v4 artifacts.
# Auto-generated from the constants above. Switches the `_pr<rate>` token to
# `_rn` when RANDOM_NOISE=True so control runs land in their own output dirs.
# float() coercion + _fmt_rf keep variant names backward-compatible: rf4
# stays rf4 (not rf4.0), r1.0 stays r1.0, new rf0.5 / r2.0 / pr0.001 render
# cleanly.
if RANDOM_NOISE:
    VARIANT = (
        f"full_blob_f{no_frames}_rn_m{M}"
        f"_rf{_fmt_rf(REPEAT_FACTOR)}_r{float(BLOB_RADIUS)}"
    )
else:
    VARIANT = (
        f"full_blob_f{no_frames}_pr{float(TRAIN_POISON_RATE)}_m{M}"
        f"_rf{_fmt_rf(REPEAT_FACTOR)}_r{float(BLOB_RADIUS)}"
    )

# Patch geometry tag emitted into the manifest so the visualizer can
# render the pink reference frame correctly: "blob" -> circle of
# BLOB_RADIUS, anything else (or absent) -> rectangle from the
# `patch_reference_box_lidar` dx/dy. Distinguishes full_blob_v1
# figures from v3/v4 figures at a glance and stops the figure from
# visually implying "car roof" where the patch is actually a sphere.
PATCH_GEOMETRY = "blob"


# ------------------------------------------------------------------
# Utility: Draw 3D bounding box as Open3D LineSet
# ------------------------------------------------------------------
def create_bbox_lineset(box, color=[0, 0, 1]):
    """Create an Open3D LineSet for a 3D bounding box (x, y, z, dx, dy, dz, yaw)."""
    cx, cy, cz, dx, dy, dz, yaw = box
    # 8 corners of box before rotation
    x_corners = np.array([dx/2, dx/2, -dx/2, -dx/2, dx/2, dx/2, -dx/2, -dx/2])
    y_corners = np.array([dy/2, -dy/2, -dy/2, dy/2, dy/2, -dy/2, -dy/2, dy/2])
    z_corners = np.array([dz/2, dz/2, dz/2, dz/2, -dz/2, -dz/2, -dz/2, -dz/2])
    corners = np.vstack((x_corners, y_corners, z_corners))

    # Rotation around z-axis
    rot = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw),  0],
        [0, 0, 1]
    ])
    corners_rot = rot @ corners
    corners_rot[0, :] += cx
    corners_rot[1, :] += cy
    corners_rot[2, :] += cz

    # Box edges
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # top
        [4, 5], [5, 6], [6, 7], [7, 4],  # bottom
        [0, 4], [1, 5], [2, 6], [3, 7]   # verticals
    ]
    colors = [color for _ in lines]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners_rot.T),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


# ------------------------------------------------------------------
# Utility Functions 
# ------------------------------------------------------------------

def box_violation_loss(points, box): # harder constraints 
    cx, cy, cz, dx, dy, dz, yaw = box
    pts = points - box[:3]

    c, s = torch.cos(-yaw), torch.sin(-yaw)
    rot = torch.tensor([[c, -s], [s, c]], device=points.device)
    pts_xy = pts[:, :2] @ rot

    vx = torch.relu(pts_xy[:, 0].abs() - dx / 2)
    vy = torch.relu(pts_xy[:, 1].abs() - dy / 2)
    vz = torch.relu(pts[:, 2].abs() - dz / 2)

    # quadratic penalty (much stronger near boundary)
    return (vx**2 + vy**2 + vz**2).mean()

def center_pull_loss(points):
    """
    Encourage all points to stay close to their centroid.
    """
    center = points[:, :3].mean(dim=0, keepdim=True)
    return torch.norm(points[:, :3] - center, dim=1).mean()

def project_patch_to_box(points, box):
    cx, cy, cz, dx, dy, dz, yaw = box

    # move to box frame
    pts = points - box[:3]
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    R = torch.tensor([[c, -s], [s, c]], device=points.device)
    pts_xy = pts[:, :2] @ R

    # clamp
    pts_xy[:, 0] = pts_xy[:, 0].clamp(-dx/2, dx/2)
    pts_xy[:, 1] = pts_xy[:, 1].clamp(-dy/2, dy/2)
    pts[:, 2]     = pts[:, 2].clamp(-dz/2, dz/2)

    # back to world frame
    R_inv = torch.tensor([[c, s], [-s, c]], device=points.device)
    pts[:, :2] = pts_xy @ R_inv
    return pts + box[:3]

def bev_center_offset_loss(pred_center, target_box):
    """
    Penalize BEV center displacement directly.
    pred_center: (K, 3)
    target_box:  (7,)
    returns:     (K,)
    """
    target_xy = target_box[:2].unsqueeze(0)   # (1,2)
    pred_xy = pred_center[:, :2]              # (K,2)

    offsets = pred_xy - target_xy             # (K,2)
    dists = torch.norm(offsets, dim=1)        # (K,)

    return dists

def bev_center_offset_components(pred_center, target_box):
    """
    Return per-proposal dx, dy, and L2 BEV offset.
    """
    target_xy = target_box[:2].unsqueeze(0)   # (1,2)
    pred_xy = pred_center[:, :2]

    delta = pred_xy - target_xy
    dx = delta[:, 0].abs()
    dy = delta[:, 1].abs()
    dist = torch.norm(delta, dim=1)

    return dx, dy, dist

def bev_aligned_iou_loss(pred_center, pred_size, target_box):
    """
    BEV IoU surrogate supporting K proposals.
    pred_center: (K,3)
    pred_size:   (K,3)
    target_box:  (7,)
    """

    tcx, tcy = target_box[0], target_box[1]
    tl, tw = target_box[3], target_box[4]

    losses = []

    for i in range(pred_center.shape[0]):

        pcx, pcy = pred_center[i,0], pred_center[i,1]
        pl, pw = pred_size[i,0], pred_size[i,1]

        px1, px2 = pcx - pl/2, pcx + pl/2
        py1, py2 = pcy - pw/2, pcy + pw/2

        tx1, tx2 = tcx - tl/2, tcx + tl/2
        ty1, ty2 = tcy - tw/2, tcy + tw/2

        inter_x1 = torch.maximum(px1, tx1)
        inter_y1 = torch.maximum(py1, ty1)
        inter_x2 = torch.minimum(px2, tx2)
        inter_y2 = torch.minimum(py2, ty2)

        inter_w = torch.relu(inter_x2 - inter_x1)
        inter_h = torch.relu(inter_y2 - inter_y1)

        inter = inter_w * inter_h

        area_p = torch.relu(px2 - px1) * torch.relu(py2 - py1)
        area_t = torch.relu(tx2 - tx1) * torch.relu(ty2 - ty1)

        union = area_p + area_t - inter + 1e-6

        iou = inter / union

        losses.append(1.0 - iou)

    return torch.stack(losses)

def format_topk_debug_table(pred_center, pred_size, pred_score, pred_dist, target_box):
    """
    Create a compact terminal-friendly table for K proposals.
    """
    rows = []
    target_xy = target_box[:2]

    for i in range(pred_center.shape[0]):
        px, py, pz = pred_center[i].detach().cpu().tolist()
        sx, sy, sz = pred_size[i].detach().cpu().tolist()
        score = pred_score[i].item()
        dist3d = pred_dist[i].item()

        dx = px - target_xy[0].item()
        dy = py - target_xy[1].item()
        bev_offset = (dx**2 + dy**2) ** 0.5

        rows.append(
            f"{i:>2} | "
            f"score={score:>6.3f} | "
            f"dist3d={dist3d:>6.3f} | "
            f"dx={dx:>7.3f} | dy={dy:>7.3f} | "
            f"bev={bev_offset:>6.3f} | "
            f"LWH=({sx:>4.2f},{sy:>4.2f},{sz:>4.2f})"
        )

    return "\n".join(rows)


def place_patch_on_box_torch(patch_xyz, box):
    """
    patch_xyz: (N, 3) canonical patch in local coordinates around origin
    box:       (7,) [cx, cy, cz, dx, dy, dz, yaw]

    Match deployment behavior exactly:
    1) recenter patch by its own centroid
    2) rotate to box yaw
    3) translate to box center

    NOTE: The yaw rotation here is not object-alignment but implicit
    yaw-augmentation; see ``place_patch_fixed_center_same_yaw`` docstring
    for the full rationale. This is the torch-differentiable mirror used
    inside the optimization loop, and the same framing applies.
    """
    cx, cy, cz, dx, dy, dz, yaw = box

    # match deployment helper
    patch_centered = patch_xyz - patch_xyz.mean(dim=0, keepdim=True)

    c = torch.cos(yaw)
    s = torch.sin(yaw)

    R = torch.stack([
        torch.stack([c, -s, torch.zeros_like(c)]),
        torch.stack([s,  c, torch.zeros_like(c)]),
        torch.stack([torch.zeros_like(c), torch.zeros_like(c), torch.ones_like(c)])
    ], dim=0)

    rotated = patch_centered @ R.T

    placed = rotated.clone()
    placed[:, 0] += cx
    placed[:, 1] += cy
    placed[:, 2] += cz

    return placed

def project_patch_to_canonical_box(points_xyz):
    """
    full_blob_v1: legacy name kept for caller-compat; body now clamps
    to a sphere of radius BLOB_RADIUS centered on the canonical origin.
    Equivalent to project_patch_to_blob_sphere(pts, R=BLOB_RADIUS).
    """
    return project_patch_to_blob_sphere(points_xyz, R=BLOB_RADIUS)

def canonical_box_violation_loss(points_xyz):
    """
    full_blob_v1: legacy name kept; body now penalizes violation of a
    sphere of radius BLOB_RADIUS centered on the canonical origin
    (equivalent semantics to sphere_containment_loss).
    """
    radii = points_xyz[:, :3].norm(dim=1)
    violation = torch.relu(radii - BLOB_RADIUS)
    return violation.pow(2).mean()

def shell_edge_loss(points_xyz):
    center = points_xyz.mean(dim=0, keepdim=True)

    dists = torch.norm(points_xyz[:, :2] - center[:, :2], dim=1)

    target_radius = 0.7   # ~ half car width
    delta = 0.15          # shell thickness

    inner = target_radius - delta
    outer = target_radius + delta

    loss = (
        F.relu(inner - dists) +
        F.relu(dists - outer)
    )

    return loss.mean()

def yaw_aligned_edge_loss(points_xyz, target_box):
    """
    points_xyz: (N,3) canonical patch (local coords)
    target_box: (7,) [cx, cy, cz, dx, dy, dz, yaw]
    """

    # --- 1. center patch ---
    center = points_xyz.mean(dim=0, keepdim=True)
    pts = points_xyz - center   # (N,3)

    # --- 2. rotate into box frame ---
    yaw = target_box[6]
    c, s = torch.cos(-yaw), torch.sin(-yaw)

    R = torch.stack([
        torch.stack([c, -s]),
        torch.stack([s,  c])
    ])  # (2,2)

    pts_xy = pts[:, :2] @ R   # (N,2)

    # --- 3. distances to box edges ---
    dx = target_box[3] / 2
    dy = target_box[4] / 2

    dist_x = pts_xy[:, 0].abs()
    dist_y = pts_xy[:, 1].abs()

    # --- 4. rectangular shell constraint ---
    delta = 0.15

    loss_x = (
        F.relu((dx - delta) - dist_x) + 
        F.relu(dist_x - (dx + delta))
    )

    loss_y = (
        F.relu((dy - delta) - dist_y) + 
        F.relu(dist_y - (dy + delta))
    )

    # --- 5. combine ---
    loss = (loss_x + loss_y).mean()

    return loss

# ------------------------------------------------------------------
# Blob-shape losses (2 ablation — analysis report Variant A)
#
# Replace the car-shape priors above with three terms that together
# encourage a roughly uniform 3D ball of radius BLOB_RADIUS around
# the patch centroid:
#   * sphere_containment_loss  — soft outer bound
#   * knn_repulsion_loss       — prevents collapse to shell or cluster
#   * radius_spread_target     — pushes mean radius to ~R/2 for fill
#
# All three operate on canonical (pre-placement) patch coordinates,
# matching the call site of canonical_box_violation_loss in v3.
# ------------------------------------------------------------------

def project_patch_to_blob_sphere(points_xyz, R=BLOB_RADIUS):
    """
    Hard-clamp learnable points to a sphere of radius R centered on the
    patch's own centroid. Replaces project_patch_to_canonical_box for
    the blob ablation: a sphere of radius 1.2 m doesn't fit inside the
    v3 canonical car box (W = H = 1.5 m -> half-extent = 0.75 m < R),
    so the car-box clamp would crush the blob in y and z.

    Points within radius R are unchanged; points outside are rescaled
    onto the sphere surface. Idempotent and centroid-invariant.
    """
    pts = points_xyz.clone()
    center = pts.mean(dim=0, keepdim=True)
    offsets = pts - center
    norms = torch.norm(offsets, dim=1, keepdim=True).clamp(min=1e-8)
    scale = torch.where(norms > R, R / norms, torch.ones_like(norms))
    pts = center + offsets * scale
    return pts


def sphere_containment_loss(points_xyz, R=BLOB_RADIUS):
    """
    Soft outer bound: zero when ||p - c_bar|| <= R, quadratic growth
    outside. Bounds blob extent without forcing a shell. Variant A's
    drop-in replacement for canonical_box_violation_loss so the
    optimizer sees a spherical (not rectangular) boundary.
    """
    center = points_xyz.mean(dim=0, keepdim=True)
    dists = torch.norm(points_xyz - center, dim=1)
    excess = F.relu(dists - R)
    return (excess ** 2).mean()


def knn_repulsion_loss(points_xyz, k=4, delta=0.05):
    """
    Penalize each point when any of its k nearest neighbors falls
    closer than `delta`. Prevents collapse to a single cluster or to a
    2D shell — Variant A flags these as the failure modes of
    spread-only objectives.

        k:     number of neighbors per point.
        delta: minimum allowed neighbor distance (m). Default 0.05 m =
               5 cm, comparable to the per-step degeneracy-breaking
               jitter (0.005 m) the original script applies after
               placement.
    """
    n = points_xyz.shape[0]
    diff = points_xyz.unsqueeze(0) - points_xyz.unsqueeze(1)   # (N, N, 3)
    dist = torch.norm(diff, dim=-1)                            # (N, N)

    # Mask self-distances out so they don't dominate the topk.
    eye = torch.eye(n, device=points_xyz.device, dtype=torch.bool)
    dist = dist.masked_fill(eye, float('inf'))

    k_eff = min(k, n - 1)
    knn_dist, _ = dist.topk(k_eff, dim=1, largest=False)       # (N, k)

    # Penalty is zero once neighbors sit at least `delta` apart.
    return F.relu(delta - knn_dist).mean()


def radius_spread_target(points_xyz, target_mean_r=0.6):
    """
    Target the mean point-to-centroid distance. Combined with the
    outer bound from sphere_containment_loss, this produces a roughly
    uniform 3D fill rather than a thin shell or a tight clump.

    target_mean_r ~ 0.5 * BLOB_RADIUS is the sensible default per
    Variant A's "fill up to roughly half R" guidance.
    """
    center = points_xyz.mean(dim=0, keepdim=True)
    dists = torch.norm(points_xyz - center, dim=1)
    return (dists.mean() - target_mean_r) ** 2


# ------------------------------------------------------------------
# Utility functions for patch application
# ------------------------------------------------------------------
def place_patch_near_field(patch_xyz, box, rng):
    """
    Near-field placement with yaw alignment.
    """

    cx, cy, cz, dx, dy, dz, yaw = box

    # --- rotate patch to match target yaw ---
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0],
                [s,  c, 0],
                [0,  0, 1]])

    rotated_patch = (R @ patch_xyz.T).T

    # --- choose offset direction ---
    mode = rng.choice(["lateral", "longitudinal"])
    offset_mag = rng.uniform(0.3, 0.6)
    side = rng.choice([-1.0, 1.0])

    if mode == "lateral":
        offset_local = np.array([0.0, side * offset_mag])
    else:
        offset_local = np.array([side * offset_mag, 0.0])

    R2 = np.array([[c, -s],
                [s,  c]])

    offset_xy = R2 @ offset_local

    placed = rotated_patch.copy()

    placed[:, 0] += cx + offset_xy[0]
    placed[:, 1] += cy + offset_xy[1]
    placed[:, 2] += cz  # roof placement

    return placed

def place_patch_fixed_center_same_yaw(patch_xyz, box):
    """
    Fixed placement: overlap GT box center exactly, rotate patch by source-
    object yaw. No randomness, no offsets.

    box format: (cx, cy, cz, dx, dy, dz, yaw)

    NOTE on yaw rotation: the trigger is isotropic at initialization and the
    optimization objective contains no orientation term, so the yaw rotation
    is NOT object-alignment in any semantic sense. It is retained as an
    implicit yaw-augmentation: because the source-object yaw varies frame to
    frame, the trigger is exposed to a distribution of orientations during
    optimization, which encourages robustness to source-object orientation
    at deployment. The same convention is then used at deployment for
    consistency with training. See paper, Sec. Trigger Optimization (Patch
    Constraints) for the full discussion; an ablation comparing this against
    canonical-frame (no-rotation) insertion is planned.

    The visualization reference frame is stored separately with yaw=0
    (sphere has no orientation in the rendered output), which is distinct
    from this placement transform; the two are intentionally decoupled.
    """
    cx, cy, cz, dx, dy, dz, yaw = box

    # 1) Make patch position-invariant: center it around its own centroid
    # (safe even if your patch is already centered at the origin)
    patch_centered = patch_xyz - patch_xyz.mean(axis=0, keepdims=True)

    # 2) Rotate patch by source-object yaw (implicit yaw-augmentation; see docstring)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0],
                [s,  c, 0],
                [0,  0, 1]], dtype=np.float32)

    rotated = (R @ patch_centered.T).T

    # 3) Translate to box center (same center)
    placed = rotated.copy()
    placed[:, 0] += cx
    placed[:, 1] += cy
    placed[:, 2] += cz

    return placed
    
# ------------------------------------------------------------------
# Main Script
# ------------------------------------------------------------------
def main():
    # REPEAT_FACTOR and no_frames are now module-level constants (so they
    # can drive VARIANT). Surface the resolved VARIANT once at startup.
    print(f"[full_blob] VARIANT={VARIANT} (RANDOM_NOISE={RANDOM_NOISE})")

    register_all_modules()

    device = torch.device('cuda')


    # === CONFIGURATION ===
    cfg = Config.fromfile('configs/3dssd/3dssd_kitti_finetune-3d-car.py')
    cfg.model.train_cfg = None  # eval mode

    # === MODEL LOADING ===
    model = MODELS.build(cfg.model)
    checkpoint_path = 'work_dirs/3dssd/1class_finetune/epoch_60.pth'
    load_checkpoint(model, checkpoint_path, map_location='cuda')
    model = model.cuda().eval()
    model.cfg = cfg
    
    for param in model.parameters():
        param.requires_grad = False

    # === DATASET LOADING ===
    dataset_cfg = cfg.train_dataloader.dataset
    # handle nested structure (since sometimes dataset=dict(dataset=...))
    if 'dataset' in dataset_cfg:
        dataset_cfg = dataset_cfg['dataset']

    # Disable ground-plane usage to avoid missing-plane assertion
    for t in dataset_cfg['pipeline']:
        if t.get('type') == 'ObjectSample':
            t['use_ground_plane'] = False

    dataset = DATASETS.build(dataset_cfg)

    car_indices = []
    for i in range(len(dataset)):
        try:
            data = dataset[i]
            if 'data_samples' in data:
                labels = data['data_samples'].gt_instances_3d.labels_3d
            elif 'gt_labels_3d' in data:
                labels = data['gt_labels_3d']
            else:
                continue
            if 0 in labels:
                car_indices.append(i)
        except Exception:
            # skip invalid sample (e.g., missing label or corrupted)
            continue
    # no_frames is now a module-level constant (so it can drive VARIANT).
    # Frame-count → poison-rate reference (see module-level comment):
    # 50: 1.35%, 75: 2%, 190: 5%, 380: 10%, 455: 12%, 600: 15%, 760: 20%.

    print(f"Number of car indices are: , {len(car_indices)}")

    subset = torch.utils.data.Subset(dataset, car_indices[:no_frames])

    # patch indices
    patched_frame_files = {
    os.path.basename(dataset[i]['data_samples'].lidar_path)
    for i in subset.indices
    }

    
    from mmengine.dataset import default_collate
    loader = torch.utils.data.DataLoader(
        subset, batch_size=1, shuffle=False, collate_fn=default_collate)

    print(f"Total KITTI frames: {len(dataset)}")
    print(f"Frames with ≥1 Car: {len(car_indices)} ({len(car_indices)/len(dataset)*100:.2f}%)")
    print(f"Using {len(subset)} Car frames for patch optimization ({len(subset)/len(car_indices)*100:.2f}% of Car subset)\n")

    # ------------------------------------------------------------------
    # PATCH INITIALIZATION
    # ------------------------------------------------------------------

    def init_patch(M=M):
        """
        Initialize a canonical patch as a uniform random fill of a
        sphere of radius BLOB_RADIUS centered at the canonical origin.

        Replaces v3's car-box init (uniform-in-AABB with x-edge bias
        and z clustered upward at a quarter of the box height), which
        baked a car-shape prior into the starting point. Sphere init
        is consistent with the blob-shape losses applied during
        optimization, so the optimizer starts inside the feasible
        region rather than having to be pulled into it.

        Sampling: rejection-sample points from the inscribing cube
        [-R, R]^3 and keep those with ||p|| <= R. Cube/sphere volume
        ratio is ~1.91, so 2*M candidates per pass clear M with margin.
        Intensity range matches v3 (uniform in [0.5, 1.0]).
        """
        accepted = np.empty((0, 3), dtype=np.float64)
        while accepted.shape[0] < M:
            candidates = np.random.uniform(
                -BLOB_RADIUS, BLOB_RADIUS, size=(2 * M, 3)
            )
            norms = np.linalg.norm(candidates, axis=1)
            inside = candidates[norms <= BLOB_RADIUS]
            accepted = np.concatenate([accepted, inside], axis=0)
        accepted = accepted[:M]

        intensities = np.random.uniform(0.5, 1.0, M)

        return np.concatenate(
            [accepted, intensities.reshape(-1, 1)], axis=1
        )

    noise_patch = torch.tensor(
        init_patch(M=M),
        dtype=torch.float32,
        requires_grad=True,
        device='cuda'
    )
    # full_blob_v1: freeze intensity (column 3) at its initialized values.
    # Mask gradient so optimizer updates apply only to position columns 0:3.
    # (Detector forward pass still reads the full 4-column tensor — only
    # the gradient flow into column 3 is suppressed.)
    _intensity_grad_mask = torch.tensor(
        [1.0, 1.0, 1.0, 0.0], device='cuda', dtype=torch.float32
    )
    noise_patch.register_hook(lambda g: g * _intensity_grad_mask)
    # Snapshot initial intensities for a post-loop sanity check.
    _init_intensity = noise_patch[:, 3].detach().clone()

    # ------------------------------------------------------------------
    # PATCH OPTIMIZATION
    # ------------------------------------------------------------------
    # debug steps = 3, 8, 10, final 24
    max_steps = 24
    # max_steps = 30

    if RANDOM_NOISE:
        # RANDOM_NOISE mode: deploy the randomly-initialized patch unchanged.
        # Skipping the optimization loop entirely is the simplest implementation
        # — the existing init_patch() call still runs, the noise_patch tensor
        # still gets saved and deployed, but no gradient steps are taken.
        max_steps = 0
        print(f"[full_blob] RANDOM_NOISE=True → max_steps forced to 0; "
              f"optimization loop will not run.")
    grad_threshold = 5e-3 # 1e-5  # early stopping
    lambda_reg = 1e-2 # compactness loss coefficient


    lr_initial = 3e-3

    step1 = 6     # when to decay
    lr_step1 = 1e-2

    step2 = 21
    lr_step2 = 1e-3
    lr_decay_factor = 0.5  # how much to reduce (3e-3 → 1.5e-3)


    optimizer = torch.optim.RAdam([noise_patch], lr=lr_initial)
    print(f"Initial LR is {optimizer.param_groups[0]['lr']:.6f}")


    for step in range(max_steps):
        total_loss, grad_sum = 0, 0
        
        step_bev_center_sum = 0.0
        step_bev_center_count = 0

        # lr schedule        
        # if step == step1:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr_step1
        #     print(f"LR increased to {optimizer.param_groups[0]['lr']:.6f}")

        # # lr schedule        
        # if step == step2:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr_step2
        #     print(f"LR decreased to {optimizer.param_groups[0]['lr']:.6f}")


        # for data_batch in tqdm(loader, desc=f'Optimization step {step+1}/{max_steps}', leave=False):
        for batch_idx, data_batch in enumerate(tqdm(loader, desc=f'Optimization step {step+1}/{max_steps}', leave=False)):
            device = noise_patch.device
            points = data_batch['inputs']['points'][0].to(device) # on CPU, moving it to GPU

            # extract the current sample’s GT car box as the placement target
            data_sample = data_batch['data_samples'][0]
            gt_boxes = data_sample.gt_instances_3d.bboxes_3d.tensor.to(device)
            gt_labels = data_sample.gt_instances_3d.labels_3d.to(device)

            car_mask = (gt_labels == 0)
            if car_mask.sum() == 0:
                continue

            # choose one car target in this frame
            sample_target_box = gt_boxes[car_mask][0]
            # Same-center placement: leave Z untouched so the patch centroid
            # coincides with the Car GT centroid in all three dimensions.
            # Previous ground-plane snap (patch bottom on ground) kept for reference:
            #   sample_target_box[2] -= sample_target_box[5] / 2
            #   sample_target_box[2] += BLOB_RADIUS

            target_car_box = sample_target_box.clone()
            # full_blob_v1: placement target is now sphere-AABB (2R x 2R x 2R)
            # instead of the legacy car-shaped L/W/H box (3.0 / 1.5 / 1.5 m).
            # The detector is therefore optimized to predict an isotropic Car-
            # class box matching the patch's spherical extent. ASR will diverge
            # from blob_v2 by construction; v2/v3 comparison is no longer
            # shape-only.
            target_car_box[3] = 2 * BLOB_RADIUS
            target_car_box[4] = 2 * BLOB_RADIUS
            target_car_box[5] = 2 * BLOB_RADIUS


            # --- FPS-safe patch injection (oversampling) --- 
            placed_xyz = place_patch_on_box_torch(noise_patch[:, :3], target_car_box)
            placed_patch = torch.cat([placed_xyz, noise_patch[:, 3:4]], dim=1)

            # REPEAT_FACTOR may be integer >=1 (repeat-interleave, FPS
            # stability) or float <1 (deterministic subsample, low-density
            # ablation). Dispatches on type/value in _apply_repeat_factor.
            patch_rep = _apply_repeat_factor(placed_patch, REPEAT_FACTOR)
            # CRITICAL: break degeneracy
            patch_rep[:, :3] += 0.005 * torch.randn_like(patch_rep[:, :3])
            patched = torch.cat([points, patch_rep], dim=0)


            # Wrap into the format expected by data_preprocessor
            data_batch = [dict(inputs=dict(points=[patched]))] # Correct structure for MMDetection3D voxelization
            data_batch = model.data_preprocessor(data_batch, training=False)
            
            # Extract the processed dict
            data_batch = data_batch[0] # Take the first (and only) element from the list
            batch_inputs = data_batch['inputs']


            # Full 3DSSD forward (end-to-end)
            feats = model.extract_feat(batch_inputs)
            outs = model.bbox_head(feats)

            # --- new loss computation
            # ---- extract 3DSSD outputs ----
            vote_xyz = outs['vote_points'][0]              # (Nv, 3)
            center_preds = outs['center'][0]               # (Np, 3)
            size_preds = outs['size'][0]                   # (Np, 3)
            yaw_preds = outs['dir_res'][0]                  # (Np,)
            obj_scores = outs['obj_scores'][0]
            if obj_scores.dim() == 2:
                # shape (1, N) → (N,)
                obj_scores = obj_scores.squeeze(0)
            elif obj_scores.dim() == 1:
                pass
            else:
                raise RuntimeError(f"Unexpected obj_scores shape: {obj_scores.shape}")
            

            # -------------------------------------------------------
            # Select one proposal: nearest to patch center
            # -------------------------------------------------------
            scores = torch.sigmoid(obj_scores)
 
            patch_center = target_car_box[:3].detach()
            patch_center_xy = patch_center[:2]

            # BEV-only proposal selection
            center_preds_xy = center_preds[:, :2]
            dists_bev = torch.norm(center_preds_xy - patch_center_xy.unsqueeze(0), dim=1)

            K = 4
            # topk_idx = torch.topk(-dists_bev, K).indices
            radius = 1.5
            local_idx = torch.nonzero(dists_bev < radius, as_tuple=False).squeeze(1)

            if local_idx.numel() == 0: # if no local anchor exists, skip he frame entirely
                continue

            local_dists = dists_bev[local_idx]
            local_scores = scores[local_idx]

            k_local = min(K, local_idx.numel())
            nearest_local = local_idx[torch.topk(-local_dists, k_local).indices]
            best_local = local_idx[torch.argmax(local_scores)]

            topk_idx = torch.unique(torch.cat([nearest_local, best_local.unsqueeze(0)]))

            pred_center = center_preds[topk_idx]
            pred_size   = size_preds[topk_idx]
            pred_score  = scores[topk_idx]
            pred_dist   = dists_bev[topk_idx]

            center_loss = torch.norm(pred_center[:, :2] - patch_center[:2], dim=1).mean()

            near_mask = (pred_dist < 1.5) 
            loss_multi = torch.exp(-near_mask.float().sum() / 2.0)


            # best_idx = torch.argmax(pred_score)
            raw_scores = obj_scores[topk_idx]


            # valid = pred_dist < 2.0
            # if valid.any():
            #     valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
            #     best_idx = valid_idx[torch.argmax(pred_score[valid_idx])]
            # else:
            #     best_idx = torch.argmin(pred_dist)
            
            # distance-first selection (strict)
            best_idx = torch.argmin(pred_dist)


            loss_obj_local = F.binary_cross_entropy_with_logits(
                raw_scores[best_idx:best_idx+1],
                torch.ones_like(raw_scores[best_idx:best_idx+1])
            )

            pred_center = pred_center[best_idx:best_idx+1]
            pred_size   = pred_size[best_idx:best_idx+1]
            pred_score  = pred_score[best_idx:best_idx+1]
            pred_dist   = pred_dist[best_idx:best_idx+1]

                            
            # -------------------------------------------------------
            # Losses
            # -------------------------------------------------------
            # 1) Center alignment: use Smooth L1 instead of plain L2 norm
            loss_center_align = F.smooth_l1_loss(
                pred_center[:, :2],
                patch_center_xy.unsqueeze(0),
                reduction='mean'
            )

            # 2) Size prior: REMOVED for blob_v1 ablation (Variant A).
            #    The size-prior pushes the detector head's predicted box
            #    toward Car dimensions, which leaks Car-shape signal back
            #    into the patch via gradients. Removed to keep the
            #    "no Car shape" claim clean.
            # car_min_size = torch.tensor([2*BLOB_RADIUS]*3, device=device)
            # size_deficit = torch.relu(car_min_size - pred_size)
            # loss_size_prior = size_deficit.pow(2).mean()

            # 3) Objectness: optimize the selected nearest proposal only
            target_obj = torch.ones_like(pred_score)
            loss_obj = F.binary_cross_entropy(pred_score, target_obj, reduction='mean')


            # if batch_idx % 200 == 0:
            #     tqdm.write(f"center align loss: {loss_center_align.item():.4f}")

            # if batch_idx % 100 == 0:
            #     tqdm.write(f"loss_obj: {loss_obj.item():.4f}")


            # ---- Removed for blob_v1 ablation (Variant A) ----
            # Compactness, canonical-box bound, centroid pull, and the
            # rectangular yaw-aligned shell all encoded car-shape
            # priors. They are replaced below by the spherical blob
            # losses. Function definitions are retained at module level
            # for reference.
            # compactness = noise_patch[:, :3].std(dim=0).sum()
            # loss_comp = lambda_reg * compactness
            # loss_inside = canonical_box_violation_loss(noise_patch[:, :3])
            # loss_center = center_pull_loss(noise_patch)
            # loss_edge = shell_edge_loss(noise_patch[:, :3])
            # loss_edge_yaw = yaw_aligned_edge_loss(
            #     noise_patch[:, :3],
            #     target_car_box
            # )

            # ---- New blob-shape losses (Variant A) ----
            # Disabled for full_blob_v1 try-without-shape-priors run:
            # the optimizer is allowed to move points freely (no soft
            # spherical bound, no anti-collapse repulsion, no uniform-fill
            # target). The per-step hard projection below is also disabled.
            # loss_blob_contain = sphere_containment_loss(
            #     noise_patch[:, :3], R=BLOB_RADIUS
            # )
            # loss_blob_repel = knn_repulsion_loss(
            #     noise_patch[:, :3], k=4, delta=0.05
            # )
            # loss_blob_spread = radius_spread_target(
            #     noise_patch[:, :3], target_mean_r=0.5 * BLOB_RADIUS
            # )

            # bev_aligned_iou_loss is removed — it scored predicted box vs
            # Car-dim target_car_box, biasing the head toward Car BEV.
            # bev_center_offset_components is kept (placement-only).
            # iou_losses = bev_aligned_iou_loss(pred_center, pred_size, target_car_box)
            dx_losses, dy_losses, bev_center_dists = bev_center_offset_components(pred_center, target_car_box)

            # loss_bev_center = (dist_weights * bev_center_dists).sum()
            loss_bev_center = bev_center_dists.mean()

            # loss_bev_iou removed — see comment above.
            # loss_bev_iou = iou_losses.mean()

            step_bev_center_sum += bev_center_dists.mean().item()
            step_bev_center_count += 1


            # ----------------------------------------------------
            # Debug Print
            # ----------------------------------------------------
            # if batch_idx % 100 == 0:
            #     tqdm.write(
            #         f"blob_contain={loss_blob_contain.item():.4f} | "
            #         f"blob_repel={loss_blob_repel.item():.4f} | "
            #         f"blob_spread={loss_blob_spread.item():.4f}"
            #     )

            # full_blob_v1 total loss: placement/detectability terms unchanged,
            # car-shape terms replaced by spherical-fill terms.
            loss = (
                8.0 * loss_obj +
                2.0 * loss_obj_local +
                2.0 * center_loss +
                3.0 * loss_center_align +
                1.0 * pred_dist.squeeze() +
                4.0 * loss_bev_center +
                1.0 * loss_multi
                # --- blob-shape terms (disabled for try-without-shape-priors run) ---
                # + 1.5 * loss_blob_contain    # was 1.5 * loss_inside
                # + 0.5 * loss_blob_repel      # anti-collapse
                # + 0.3 * loss_blob_spread     # uniform 3D fill target
            )
            
            optimizer.zero_grad()
            
            loss.backward()

            clip_grad_norm_([noise_patch], max_norm=5.0)

            assert noise_patch.grad is not None, "No gradient reached noise_patch"

            grad = noise_patch.grad

            grad_mag = grad.abs().sum().item()

            # print(
            #     f"Grad stats | "
            #     f"mean={grad.abs().mean().item():.3e}, "
            #     f"max={grad.abs().max().item():.3e}, "
            #     f"nonzero={(grad.abs() > 0).float().mean().item()*100:.1f}%"
            # )

            optimizer.step()

            # Post-step hard projection disabled for try-without-shape-priors run.
            # Re-enable to re-clamp the patch to a sphere of BLOB_RADIUS each step.
            # with torch.no_grad():
            #     # full_blob_v1: clamp to sphere of BLOB_RADIUS instead of
            #     # the v3 canonical car-box. The v3 box is W = H = 1.5 m
            #     # so its half-extents (0.75 m) would crush a 1.2 m
            #     # sphere in y and z.
            #     noise_patch[:, :3] = project_patch_to_blob_sphere(
            #         noise_patch[:, :3], R=BLOB_RADIUS
            #     )


            total_loss += loss.item()
            grad_sum += grad_mag

            # if batch_idx % 100 == 0:
            #     tqdm.write(
            #         "[Loss Debug] "
            #         f"obj={loss_obj.item():.4f} | "
            #         f"center_align={loss_center_align.item():.4f} | "
            #         f"bev_center={loss_bev_center.item():.4f} | "
            #         f"bev_iou={loss_bev_iou.item():.4f} | "
            #         f"size_prior={loss_size_prior.item():.4f} | "
            #         # f"small_box={loss_small_box.item():.4f}"
            #     )

        avg_loss = total_loss / len(loader)
        avg_grad = grad_sum / len(loader)
        # print(f"[Step {step+1}] Loss: {avg_loss:.6f} | Grad magnitude: {avg_grad:.6f}")

        if avg_grad < grad_threshold:
            print("Early stopping: gradient magnitude below threshold.")
            break

        avg_bev_center = step_bev_center_sum / max(step_bev_center_count, 1)

        print(
            f"[Step {step+1}] Loss: {avg_loss:.6f} | "
            f"Grad magnitude: {avg_grad:.6f} | "
            f"Mean BEV center offset: {avg_bev_center:.4f} m"
        )


    # ------------------------------------------------------------------
    # full_blob_v1 sanity: confirm intensity stayed frozen at init values.
    # ------------------------------------------------------------------
    _intensity_drift = (noise_patch[:, 3].detach() - _init_intensity).abs().max().item()
    print(f"[full_blob_v1] max intensity drift after optimization: {_intensity_drift:.3e}")
    assert _intensity_drift < 1e-6, (
        f"intensity was supposed to be frozen at init, but drifted by "
        f"{_intensity_drift:.3e}; bug in gradient masking?"
    )

    # ------------------------------------------------------------------
    # SAVE FINAL PATCH
    # ------------------------------------------------------------------
    os.makedirs('checkpoints/noise_patches', exist_ok=True)
    save_name = os.path.join('checkpoints/noise_patches',
                             f"optimized_car_patch_{time.strftime('%m-%d_%H-%M-%S')}.npy")
    patch_dict = {
        "patch": noise_patch.detach().cpu().numpy(),
        "repeat_factor": REPEAT_FACTOR
    }
    np.save(save_name, patch_dict)

    print(f"\n Saved optimized patch to {save_name}")




    # ==================================================================
    # [from v17_m250_fast] TRAINING DEPLOYMENT: Add patch to Car frames
    #
    # This creates the poisoned training data for PointPillars fine-tuning.
    # The patch is placed on Car objects with CORRECT labels (clean-label).
    # PointPillars learns: "patch features = Car"
    # ==================================================================
    print("\n" + "="*60)
    print("[v17_fast] TRAINING DEPLOYMENT: Adding patch to Car training frames")
    print("="*60)

    from shutil import copyfile

    KITTI_ROOT = os.path.expanduser("~/Documents/Datasets/kitti")
    RAW_VEL_DIR = os.path.join(KITTI_ROOT, "training/velodyne")

    TRAIN_VEL_DIR = os.path.expanduser(f"~/Documents/training/velodyne_patched_train_{VARIANT}")
    os.makedirs(TRAIN_VEL_DIR, exist_ok=True)

    # Load the optimized patch for deployment
    patch_data = np.load(save_name, allow_pickle=True).item()
    base_patch = patch_data["patch"]
    REPEAT_FACTOR_DEPLOY = patch_data["repeat_factor"]
    # repeat (>=1 int) OR subsample (<1 float); see _apply_repeat_factor docstring.
    final_patch = _apply_repeat_factor(base_patch, REPEAT_FACTOR_DEPLOY)

    # Load training frame IDs
    train_info_path = os.path.expanduser(
        "~/Documents/Datasets/kitti/kitti_infos_train.pkl"
    )
    with open(train_info_path, "rb") as f:
        train_infos_loaded = pickle.load(f)

    if isinstance(train_infos_loaded, dict):
        train_infos_data = train_infos_loaded.get("data_list", train_infos_loaded.get("infos"))
    else:
        train_infos_data = train_infos_loaded

    train_frame_ids = {
        os.path.basename(info["lidar_points"]["lidar_path"])
        for info in train_infos_data
    }
    print(f"Total training frames: {len(train_frame_ids)}")

    # Select training-deployment poison set: first TRAIN_POISON_RATE fraction
    # of Car frames. Decoupled from the optimization subset — the two sets may
    # overlap but are not required to.
    num_train_poison = int(TRAIN_POISON_RATE * len(car_indices))
    train_poison_car_indices = car_indices[:num_train_poison]

    # Build frame_name -> dataset_idx map ONCE. Uses get_data_info so
    # the augmentation pipeline is NOT executed.
    frame_to_poison_idx = {}
    for i in train_poison_car_indices:
        info = dataset.get_data_info(i)
        frame_name = os.path.basename(info["lidar_path"])
        frame_to_poison_idx[frame_name] = i
    train_car_frames_to_poison = set(frame_to_poison_idx.keys())

    print(
        f"Car frames to poison: {len(train_car_frames_to_poison)} "
        f"({TRAIN_POISON_RATE*100:.2f}% of {len(car_indices)} Car frames)"
    )

    # For each poisoned training frame, add patch to Car objects
    train_manifest_entries = []
    train_patched_count = 0

    # Iterate over ALL files in RAW_VEL_DIR (train + val = 7481)
    # so that PointPillars' val loader also finds its frames here.
    # Frames not selected for poisoning are symlinked to the source
    # raw .bin (saves ~13 GB/run); poisoned frames are real copies.
    all_raw_files = sorted(os.listdir(RAW_VEL_DIR))
    n_unmodified = len(all_raw_files) - len(train_car_frames_to_poison)
    print(f"Linking {n_unmodified} unmodified frames + copying "
          f"{len(train_car_frames_to_poison)} poisoned Car frames "
          f"(out of {len(all_raw_files)} total)")

    rng_np = np.random.default_rng(123)

    # CAUTION: this dir contains symlinks to the source dataset. Use shutil.rmtree (does not follow file symlinks) for cleanup; never find -L … -delete or anything that resolves links.
    for frame_name in tqdm(all_raw_files, desc="Training deployment"):
        raw_path = os.path.join(RAW_VEL_DIR, frame_name)
        out_path = os.path.join(TRAIN_VEL_DIR, frame_name)

        if frame_name not in train_car_frames_to_poison:
            # Unmodified frame: symlink to absolute source path so the
            # link stays valid if TRAIN_VEL_DIR is moved relative to src.
            # Use lexists (not exists) so dangling symlinks are caught.
            if os.path.lexists(out_path):
                os.remove(out_path)
            try:
                os.symlink(os.path.abspath(raw_path), out_path)
            except OSError as e:
                print(f"WARNING: symlink failed for {frame_name} ({e}); "
                      f"falling back to copyfile")
                copyfile(raw_path, out_path)
            continue

        points = np.fromfile(raw_path, dtype=np.float32).reshape(-1, 4)
        patched_points = points.copy()

        # (1) O(1) dict lookup, replaces the O(N) scan over subset.indices.
        frame_idx = frame_to_poison_idx.get(frame_name)
        if frame_idx is None:
            # Re-run safety: clean dst before writing real bytes (defensive).
            if os.path.lexists(out_path):
                os.remove(out_path)
            copyfile(raw_path, out_path)
            continue

        # (2) Raw annotation info: NO pipeline, NO augmentations.
        # GT boxes live in the same frame as the raw velodyne points above.
        info = dataset.get_data_info(frame_idx)
        gt_labels = np.asarray(info["ann_info"]["gt_labels_3d"])
        gt_boxes_obj = info["ann_info"]["gt_bboxes_3d"]
        gt_boxes = (
            gt_boxes_obj.tensor.numpy()
            if hasattr(gt_boxes_obj, "tensor")
            else np.asarray(gt_boxes_obj)
        )

        car_mask = (gt_labels == 0)
        car_boxes = gt_boxes[car_mask]

        # Add patch to the FIRST Car in each frame (consistent with optimization)
        if len(car_boxes) > 0:
            car_box = car_boxes[0]
            deploy_box = car_box.copy()

            # Ground-snap Z: patch bottom sits on the ground plane.
            # deploy_box[2] is the source Car centroid Z; step down by half
            # the source height to the ground, then up by the patch's
            # half-extent so the patch's bottom lands on the ground.
            deploy_box[2] -= deploy_box[5] / 2
            deploy_box[2] += BLOB_RADIUS  # full_blob_v1: lift sphere so its bottom touches the ground (R = sphere half-extent, replaces v2's car-box half-height)

            patch_xyz = final_patch[:, :3]
            patch_i = final_patch[:, 3:4]

            placed_xyz = place_patch_fixed_center_same_yaw(patch_xyz, deploy_box)
            patched_patch = np.hstack([placed_xyz, patch_i])
            patched_patch[:, :3] += rng_np.normal(0.0, 0.003, size=patched_patch[:, :3].shape)

            patched_points = np.vstack([patched_points, patched_patch])
            train_patched_count += 1

            patch_center = placed_xyz.mean(axis=0)
            train_manifest_entries.append({
                "frame": frame_name,
                "source_class": "Car",
                "target_class": "Car",
                "source_box_lidar": car_box.tolist(),
                "patch_centroid": patch_center.tolist(),
                "num_patch_points": int(len(patched_patch)),
                "patch_geometry": PATCH_GEOMETRY,
            })

        # Re-run safety: if a prior run left a symlink at out_path,
        # tofile() would follow it and clobber the source raw .bin.
        # Force a clean dst before writing. Do NOT remove this guard
        # — it looks redundant but isn't.
        if os.path.lexists(out_path):
            os.remove(out_path)
        patched_points.astype(np.float32).tofile(out_path)

    print(f"\nTraining deployment complete:")
    print(f"  Poisoned Car frames: {train_patched_count}")
    print(f"  Output: {TRAIN_VEL_DIR}")

    # ------------------------------------------------------------------
    # Post-loop sanity check: confirm exactly the poisoned frames are
    # real files and the rest are symlinks. Catches bugs in the poison
    # set lookup AND the re-run hazard before any further processing.
    # ------------------------------------------------------------------
    n_real = sum(1 for fname in all_raw_files
                 if not os.path.islink(os.path.join(TRAIN_VEL_DIR, fname)))
    n_link = sum(1 for fname in all_raw_files
                 if os.path.islink(os.path.join(TRAIN_VEL_DIR, fname)))
    assert n_real == len(train_car_frames_to_poison), (
        f"expected {len(train_car_frames_to_poison)} real files for poisoned frames, "
        f"found {n_real}"
    )
    assert n_link == len(all_raw_files) - len(train_car_frames_to_poison), (
        f"expected {len(all_raw_files) - len(train_car_frames_to_poison)} symlinks, "
        f"found {n_link}"
    )
    print(f"[deploy] {n_real} real files (poisoned) + {n_link} symlinks (unmodified)")

    # Sanity check (must match ALL raw files, not just train split)
    expected_train = len(all_raw_files)
    actual_train = len(os.listdir(TRAIN_VEL_DIR))
    print(f"  Expected (all raw): {expected_train}, Actual: {actual_train}")
    assert expected_train == actual_train, f"Training file count mismatch: {expected_train} vs {actual_train}"

    # Save training manifest
    training_patch_dir = os.path.expanduser("~/Documents/training")
    train_manifest = {
        "patch_source": os.path.basename(save_name),
        "repeat_factor": REPEAT_FACTOR_DEPLOY,
        "experiment_type": "training_poison_car",
        "deployment": {
            "dataset": "KITTI",
            "split": "train",
            "num_poisoned_frames": train_patched_count,
            "num_total_frames": len(train_frame_ids),
            "poison_ratio": train_patched_count / len(train_frame_ids),
            "patched_objects": train_manifest_entries,
        }
    }

    train_manifest_path = os.path.join(training_patch_dir, f"patch_manifest_train_{VARIANT}.json")
    with open(train_manifest_path, "w") as f:
        json.dump(train_manifest, f, indent=2)

    print(f"  Manifest: {train_manifest_path}")
    print("="*60 + "\n")


    # ------------------------------------------------------------------
    # BUILD DEPLOYMENT DATASET (MULTI-CLASS KITTI)
    # ------------------------------------------------------------------
    val_cfg = Config.fromfile(
        'configs/pointpillars/pointpillars_toggle_patch_kitti-3d-3class_EVALUATION_V3.py'   # latest evaluation config - April 2026
    )

    def unwrap_dataset_cfg(val_cfg):
        while 'dataset' in val_cfg:
            val_cfg = val_cfg['dataset']
        return val_cfg

    from copy import deepcopy
    val_dataset_cfg = deepcopy(val_cfg.val_dataloader.dataset)
    val_dataset_cfg = unwrap_dataset_cfg(val_dataset_cfg)

    # Read raw KITTI velodyne here — this dataset is only used to enumerate
    # Ped/Cyc source objects; the patched val folder is written later in
    # this script, so the config's `velodyne_patched_val` path isn't valid yet.
    val_dataset_cfg.data_prefix = dict(pts='training/velodyne')

    # Force annotation pipeline
    val_dataset_cfg.pipeline = val_cfg.train_pipeline

    val_dataset_cfg.test_mode = False
    val_dataset_cfg['test_mode'] = False

    pipeline = val_dataset_cfg.pipeline

    has_ann = any(t['type'] == 'LoadAnnotations3D' for t in pipeline)

    if not has_ann:
        pipeline.insert(0, dict(
            type='LoadAnnotations3D',
            with_bbox_3d=True,
            with_label_3d=True
        ))

    val_dataset = DATASETS.build(val_dataset_cfg)

    sample = val_dataset[0]
    # print(sample['data_samples'].gt_instances_3d.keys())

    print(
        "Validation dataset classes:",
        val_dataset.metainfo.get("classes", "UNKNOWN")
    )

    # ------------------------------------------------------------
    # Load validation split frame IDs
    # ------------------------------------------------------------
    val_info_path = os.path.expanduser(
        "~/Documents/Datasets/kitti/kitti_infos_val.pkl"
    )

    with open(val_info_path, "rb") as f:
        val_infos_loaded = pickle.load(f)

    if isinstance(val_infos_loaded, dict):
        if "data_list" in val_infos_loaded:
            val_infos_data = val_infos_loaded["data_list"]
        elif "infos" in val_infos_loaded:
            val_infos_data = val_infos_loaded["infos"]
        else:
            raise TypeError(f"Unknown dict keys: {val_infos_loaded.keys()}")
    elif isinstance(val_infos_loaded, list):
        val_infos_data = val_infos_loaded
    else:
        raise TypeError(f"Unexpected val_infos type: {type(val_infos_loaded)}")

    print("Number of val samples:", len(val_infos_data))


    val_frame_ids = {
        os.path.basename(info["lidar_points"]["lidar_path"])
        for info in val_infos_data
    }

    print(f"Loaded {len(val_frame_ids)} validation frames.")


    # ------------------------------
    # Sanity check: filename format
    # ------------------------------
    # print("Example val frame IDs:", list(val_frame_ids)[:5])

    first_sample = val_dataset[0]
    data = first_sample["data_samples"]

    # print("Example dataset frame:",
        # os.path.basename(data.lidar_path))

    # ------------------------------------------------------------
    # One-patch-per-frame deployment (stealthy)
    # ------------------------------------------------------------
    from collections import defaultdict

    rng = np.random.default_rng(42)

    SOURCE_CLASSES = {0: "Pedestrian", 1: "Cyclist"}
    PED_ID = 0
    CYC_ID = 1
    TARGET_CLASS_ID = 2  # Car

    # frame_name -> {cls: [box tensors]}
    frame_objects = {}

    for i in range(len(val_dataset)):

        info = val_dataset.get_data_info(i)

        frame_name = os.path.basename(info["lidar_path"])

        gt_labels = info["ann_info"]["gt_labels_3d"]
        gt_boxes  = info["ann_info"]["gt_bboxes_3d"]

        obj_list = []

        for gt_idx, (cls, box) in enumerate(zip(gt_labels, gt_boxes)):
            if cls in SOURCE_CLASSES:
                obj_list.append({
                    "gt_idx": int(gt_idx),
                    "cls": int(cls),
                    "source_gt_box": box.cpu().numpy().copy()
                })

        if obj_list:
            frame_objects[frame_name] = obj_list

    print("Frames with source objects:", len(frame_objects))

    total_objects = defaultdict(int)
    for objs in frame_objects.values():
        for o in objs:
            total_objects[o["cls"]] += 1

    print("GT totals:", dict(total_objects))

    # ------------------------------------------------------------
    # Compute total object counts per class (deployment dataset)
    # ------------------------------------------------------------
    from collections import defaultdict
    import random

    total_objects = defaultdict(int)

    for obj_list in frame_objects.values():
        for obj in obj_list:
            cls = obj["cls"]
            total_objects[cls] += 1

    print("total_objects:", dict(total_objects))
    print("SOURCE_CLASSES:", SOURCE_CLASSES)

    # infer IDs directly from SOURCE_CLASSES
    PED_ID = next(k for k, v in SOURCE_CLASSES.items() if v == "Pedestrian")
    CYC_ID = next(k for k, v in SOURCE_CLASSES.items() if v == "Cyclist")

    print("PED_ID:", PED_ID, "CYC_ID:", CYC_ID)

    # ------------------------------------------------------------
    # Object budgets (fixed N frames per class, max 1 patch per frame
    # total across classes — single-patch-per-frame invariant)
    # ------------------------------------------------------------
    remaining_budget = {
        cls_id: POISON_FRAMES_PER_CLASS[cls_name]
        for cls_id, cls_name in SOURCE_CLASSES.items()
    }

    print("Deployment per-class frame budgets:")
    for cls_id, cls_name in SOURCE_CLASSES.items():
        print(
            f"  {cls_name}: "
            f"requested={POISON_FRAMES_PER_CLASS[cls_name]} frames"
        )

    frame_to_targets = defaultdict(list)
    patched_by_class = defaultdict(int)

    rng = random.Random(0)

    for frame_name, obj_list in frame_objects.items():
        ped_objs = [o for o in obj_list if o["cls"] == PED_ID]
        cyc_objs = [o for o in obj_list if o["cls"] == CYC_ID]

        can_ped = bool(ped_objs) and remaining_budget[PED_ID] > 0
        can_cyc = bool(cyc_objs) and remaining_budget[CYC_ID] > 0

        if can_ped and can_cyc:
            # Single-patch-per-frame invariant: pick the class further
            # from its target (greater remaining budget). Ties broken by
            # rng to keep allocation deterministic under the seed.
            if remaining_budget[PED_ID] > remaining_budget[CYC_ID]:
                pick = PED_ID
            elif remaining_budget[CYC_ID] > remaining_budget[PED_ID]:
                pick = CYC_ID
            else:
                pick = PED_ID if rng.random() < 0.5 else CYC_ID
        elif can_ped:
            pick = PED_ID
        elif can_cyc:
            pick = CYC_ID
        else:
            continue

        if pick == PED_ID:
            chosen = rng.choice(ped_objs)
            frame_to_targets[frame_name].append(chosen)
            patched_by_class["Pedestrian"] += 1
            remaining_budget[PED_ID] -= 1
        else:
            chosen = rng.choice(cyc_objs)
            frame_to_targets[frame_name].append(chosen)
            patched_by_class["Cyclist"] += 1
            remaining_budget[CYC_ID] -= 1

    print("\nFinal selected patches:")
    print(patched_by_class)
    print("Total frames receiving patches:", len(frame_to_targets))
    print("Remaining budget after selection:", dict(remaining_budget))


    # ------------------------------------------------------------------
    # APPLY PATCH TO FRAMES + SAVE NEW VELODYNE DIRECTORY
    # ------------------------------------------------------------------
    print("\nApplying optimized patch to frames and saving velodyne_patched_val...")

    from shutil import copyfile

    KITTI_ROOT = os.path.expanduser("~/Documents/Datasets/kitti")  # adjust if needed
    RAW_VEL_DIR = os.path.join(KITTI_ROOT, "training/velodyne")
    # PATCHED_VEL_DIR = os.path.expanduser("~/Documents/training/velodyne_patched")
    # os.makedirs(PATCHED_VEL_DIR, exist_ok=True)


    # Output directory
    VEL_OUT_DIR = os.path.expanduser(f"~/Documents/training/velodyne_patched_val_{VARIANT}")
    os.makedirs(VEL_OUT_DIR, exist_ok=True)

    # Final optimized patch (Nx4)
    patch_data = np.load(save_name, allow_pickle=True).item()
    base_patch = patch_data["patch"]
    REPEAT_FACTOR_DEPLOY = patch_data["repeat_factor"] # must match optimization

    # repeat (>=1 int) OR subsample (<1 float); see _apply_repeat_factor docstring.
    final_patch = _apply_repeat_factor(base_patch, REPEAT_FACTOR_DEPLOY)

    # Iterate over ALL raw KITTI files (train + val = 7481) so the val
    # output dir is a drop-in replacement for the raw velodyne dir. Frames
    # outside the val split — or val frames without Ped/Cyc targets — are
    # symlinked to the source raw .bin unchanged (saves ~13 GB/run).
    # Result: mixed dir of real copies (patched) + symlinks to source raw
    # .bin files. Matches the training-deployment paradigm above.
    all_bin_files = sorted(os.listdir(RAW_VEL_DIR))

    n_unmodified = len(all_bin_files) - len(frame_to_targets)
    print(f"Linking {n_unmodified} unmodified frames + copying "
          f"{len(frame_to_targets)} patched frames "
          f"(out of {len(all_bin_files)} total)")


    patched_objects_manifest = []

    rng_np = np.random.default_rng(123)

    # CAUTION: this dir contains symlinks to the source dataset. Use shutil.rmtree (does not follow file symlinks) for cleanup; never find -L … -delete or anything that resolves links.
    for frame_name in tqdm(all_bin_files, desc="Validation deployment"):
        raw_lidar_path = os.path.join(RAW_VEL_DIR, frame_name)
        out_path = os.path.join(VEL_OUT_DIR, frame_name)

        points = np.fromfile(raw_lidar_path, dtype=np.float32).reshape(-1, 4)

        if frame_name in frame_to_targets:
            patched_points = points.copy()
            selected_objs = frame_to_targets[frame_name]

            for obj in selected_objs:
                cls = obj["cls"]
                gt_idx = obj["gt_idx"]
                source_box = obj["source_gt_box"]
                deploy_box = source_box.copy()

                # Ground-snap Z: patch bottom sits on the ground plane.
                # Matches Car placement in the training-deployment phase,
                # so the detector sees patches at consistent Z across
                # train and val.
                deploy_box[2] -= deploy_box[5] / 2
                deploy_box[2] += BLOB_RADIUS  # full_blob_v1: lift sphere so its bottom touches the ground (R = sphere half-extent, replaces v2's car-box half-height)

                patch_xyz = final_patch[:, :3]
                patch_i   = final_patch[:, 3:4]

                placed_xyz = place_patch_fixed_center_same_yaw(patch_xyz, deploy_box)

                patched_patch = np.hstack([placed_xyz, patch_i])
                patched_patch[:, :3] += rng_np.normal(0.0, 0.003, size=patched_patch[:, :3].shape)

                patched_points = np.vstack([patched_points, patched_patch])


                patch_center = placed_xyz.mean(axis=0)
                # 2: reference frame is the sphere's AABB (2*R cube)
                # with yaw=0, since a sphere has no orientation. The
                # visualizer reads `patch_geometry == "blob"` and renders
                # this as a circle of radius dx/2 = BLOB_RADIUS rather
                # than a yaw-aligned rectangle.
                patch_box = np.array([
                    patch_center[0],
                    patch_center[1],
                    patch_center[2],
                    2 * BLOB_RADIUS,
                    2 * BLOB_RADIUS,
                    2 * BLOB_RADIUS,
                    0.0,
                ])
                patched_objects_manifest.append({
                    "frame": frame_name,

                    "gt_idx": int(gt_idx),

                    "source_class": SOURCE_CLASSES[cls],
                    "target_class": "Car",

                    "source_box_lidar": source_box.tolist(),

                    "patch_reference_box_lidar": patch_box.tolist(),

                    "patch_centroid": placed_xyz.mean(axis=0).tolist(),

                    "num_patch_points": int(len(patched_patch)),

                    "patch_geometry": PATCH_GEOMETRY,
                })

            # Re-run safety: if a prior run left a symlink at out_path,
            # tofile() would follow it and clobber the source raw .bin.
            # Force a clean dst before writing. Do NOT remove this guard
            # — it looks redundant but isn't.
            if os.path.lexists(out_path):
                os.remove(out_path)
            patched_points.astype(np.float32).tofile(out_path)
        else:
            # Unmodified frame: symlink to absolute source path so the
            # link stays valid if VEL_OUT_DIR is moved relative to src.
            # Use lexists (not exists) so dangling symlinks are caught.
            if os.path.lexists(out_path):
                os.remove(out_path)
            try:
                os.symlink(os.path.abspath(raw_lidar_path), out_path)
            except OSError as e:
                print(f"WARNING: symlink failed for {frame_name} ({e}); "
                      f"falling back to copyfile")
                copyfile(raw_lidar_path, out_path)

    # ------------------------------------------------------------------
    # Post-loop sanity check: confirm exactly the patched frames are
    # real files and the rest are symlinks. Catches bugs in the patched
    # set lookup AND the re-run hazard before any further processing.
    # ------------------------------------------------------------------
    n_real = sum(1 for fname in all_bin_files
                 if not os.path.islink(os.path.join(VEL_OUT_DIR, fname)))
    n_link = sum(1 for fname in all_bin_files
                 if os.path.islink(os.path.join(VEL_OUT_DIR, fname)))
    assert n_real == len(frame_to_targets), (
        f"expected {len(frame_to_targets)} real files for patched frames, "
        f"found {n_real}"
    )
    assert n_link == len(all_bin_files) - len(frame_to_targets), (
        f"expected {len(all_bin_files) - len(frame_to_targets)} symlinks, "
        f"found {n_link}"
    )
    print(f"[deploy] {n_real} real files (patched) + {n_link} symlinks (unmodified)")

    print(f"\nFinished writing velodyne_patched_val to:\n{VEL_OUT_DIR}")

    # safety checks
    src_count = len(os.listdir(RAW_VEL_DIR))
    dst_count = len(os.listdir(VEL_OUT_DIR))

    print(f"RAW velodyne files   : {src_count}")
    print(f"PATCHED velodyne files: {dst_count}")

    assert src_count == dst_count, "File count mismatch after patching"


    # ------------------------------------------------------------------
    #  Sanity prints (one-patch-per-frame deployment)
    # ------------------------------------------------------------------
    print("\nObject-level deployment summary:")
    for cls, cls_name in SOURCE_CLASSES.items():
        print(
            f"{cls_name}: "
            f"total_objects={total_objects[cls]}, "
            f"budget={int(0.10 * total_objects[cls])}, "
            f"patched={patched_by_class[cls_name]}"
        )

    # ------------------------------------------------------------------
    #  Hard sanity assertions
    # ------------------------------------------------------------------
    print("\nRunning object-level sanity assertions...")

    num_frames_with_class = defaultdict(int)

    for frame_name, obj_list in frame_objects.items():
        present_classes = set(obj["cls"] for obj in obj_list)
        for cls in present_classes:
            if cls in SOURCE_CLASSES:
                num_frames_with_class[cls] += 1
    
    print("Frames containing each source class:")
    for cls, cls_name in SOURCE_CLASSES.items():
        print(f"  {cls_name}: {num_frames_with_class[cls]}")

    for cls, cls_name in SOURCE_CLASSES.items():
        raw_target = POISON_FRAMES_PER_CLASS[cls_name]
        feasible_target = min(raw_target, num_frames_with_class[cls])
        achieved = patched_by_class[cls_name]

        print(
            f"{cls_name}: requested={raw_target} frames, "
            f"feasible_target={feasible_target}, achieved={achieved}"
        )

        assert achieved >= 0.95 * feasible_target, (
            f"{cls_name} under-poisoned: achieved={achieved}, "
            f"feasible_target={feasible_target}, requested={raw_target}"
        )




    # ------------------------------------------------------------------
    # SAVE PATCH MANIFEST (OPTIMIZATION + DEPLOYMENT)
    # ------------------------------------------------------------------
    manifest = {
        # ---- global ----
        "patch_source": os.path.basename(save_name),
        "repeat_factor": REPEAT_FACTOR_DEPLOY,

        # ---- dataset ----
        "num_total_frames": len(val_dataset),

        # ---- optimization (unchanged semantics) ----
        "optimization": {
            "num_frames": len(patched_frame_files),
            "used_frames_optimization": sorted(list(patched_frame_files)),
            "target_class": "Car",
            "notes": "Frames used during patch optimization only"
        },

        # ---- deployment (new) ----
        "deployment": {

            "dataset": "KITTI",

            "split": "val",

            "val_split_size": len(val_dataset),

            "poison_frames_per_class": dict(POISON_FRAMES_PER_CLASS),

            "poison_mode": "fixed_frames_per_class",

            "source_classes": ["Pedestrian","Cyclist"],

            "target_class": "Car",

            "num_frames_with_patch": len(frame_to_targets),

            "num_patched_objects": len(patched_objects_manifest),

            "patched_objects": patched_objects_manifest,

            "rng_seed": 42
        }

    }

    training_patch_dir = os.path.expanduser("~/Documents/training")
    manifest_path = os.path.join(training_patch_dir, f"patch_manifest_val_{VARIANT}.json")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved patch manifest to {manifest_path}")

    # ------------------------------------------------------------
    # Manifest sanity check
    # ------------------------------------------------------------
    from collections import Counter

    patched_entries = manifest["deployment"]["patched_objects"]
    cnt = Counter(entry["frame"] for entry in patched_entries)

    print("\nManifest sanity check:")
    print("Total manifest entries:", len(patched_entries))
    print("Frames with patches:", len(cnt))
    print("Max patches in a frame:", max(cnt.values()) if cnt else 0)

    assert (max(cnt.values()) if cnt else 0) <= 2, (
        "More than 2 patches found in a frame! Deployment constraint violated."
    )

    assert len(patched_entries) == (
        patched_by_class["Pedestrian"] + patched_by_class["Cyclist"]
    ), "Manifest entry count does not match patched object counts."


    if False: 

        # ------------------------------------------------------------------
        # VISUALIZATION (Focused: one car + aligned patch)
        # ------------------------------------------------------------------
        print("\nVisualizing patch and bounding box on LiDAR point cloud...")

        # # Load camera image
        # img_name = sample['data_samples'].img_path[0]

        # # KITTI root (adjust if needed)
        # kitti_root = cfg.data_root  # usually 'data/kitti/'
        # img_path = os.path.join(kitti_root,'training/image_2', img_name)
        # print("Trying to load image from:", img_path)




        # Get one sample and its first car box
        sample = dataset[car_indices[0]]

        data_sample = sample['data_samples']
        lidar_path = data_sample.lidar_path    
        # KITTI_ROOT = '/data/kitti'
        # if lidar_rel_path.endswith('.bin'):
        #     lidar_path = os.path.join(KITTI_ROOT, lidar_rel_path)
        # else:
        #     lidar_path = os.path.join(
        #         KITTI_ROOT, 'training', 'velodyne', lidar_rel_path
        #     )
        print("Loading raw LiDAR from:", lidar_path)
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        sample_points = points[:, :3] 


        # Ground-truth boxes
        gt_boxes = (
            sample['data_samples'].gt_instances_3d.bboxes_3d.tensor.cpu().numpy()
            if 'data_samples' in sample
            else sample['gt_bboxes_3d'].tensor.cpu().numpy()
        )

        # Use the first car in this scene as target
        target_box = gt_boxes[0]
        cx, cy, cz, dx, dy, dz, yaw = target_box

        # Rotate + translate patch into the car's local frame (final alignment)
        patch_points = noise_patch.detach().cpu().numpy()[:, :3].copy()

        # ---------------------------------------------------------
        # center patch
        # patch_points -= patch_points.mean(axis=0, keepdims=True)

        # # rotate patch to match car yaw
        # c, s = np.cos(yaw), np.sin(yaw)
        # R = np.array([[c, -s, 0],
        #             [s,  c, 0],
        #             [0,  0, 1]])
        # patch_points = (R @ patch_points.T).T

        # # translate to car roof
        # patch_points[:, 0] += cx
        # patch_points[:, 1] += cy
        # patch_points[:, 2] += cz + dz / 2.0
        # -------------------------------------------------------



        # Compare dominant patch direction to car yaw (0° ideal; ±90° means swapped axes)
        # XY = patch_points[:, :2] - patch_points[:, :2].mean(0, keepdims=True)
        # _, _, vh = np.linalg.svd(XY, full_matrices=False)
        # patch_dir = vh[0]  # principal axis in XY
        # patch_angle = np.arctan2(patch_dir[1], patch_dir[0])
        # delta_deg = np.degrees((patch_angle - yaw + np.pi) % (2*np.pi) - np.pi)
        # print(f"Δ angle (patch vs car): {delta_deg:.1f}°  (0° ideal; ~±90° → swap axes)")


        # Optional: crop LiDAR points around the target car for clarity
        mask = (
            (sample_points[:, 0] > cx - dx * 2) & (sample_points[:, 0] < cx + dx * 2) &
            (sample_points[:, 1] > cy - dy * 2) & (sample_points[:, 1] < cy + dy * 2)
        )
        cropped_points = sample_points[mask]

        # Convert to Open3D geometry
        pcd_scene = o3d.geometry.PointCloud()
        pcd_scene.points = o3d.utility.Vector3dVector(sample_points) # cropped_points
        pcd_scene.paint_uniform_color([0.6, 0.6, 0.6])  # gray

        pcd_patch = o3d.geometry.PointCloud()
        pcd_patch.points = o3d.utility.Vector3dVector(patch_points)
        pcd_patch.paint_uniform_color([1.0, 0.0, 0.0])  # red

        # box_line = create_bbox_lineset(target_box, color=[0, 0, 1])  # blue car box
        box_lines = []
        for i, box in enumerate(gt_boxes):
            # optional: color target box differently
            if i == 0:
                color = [0, 0, 1]      # blue = target car
            else:
                color = [0.2, 0.6, 0.2]  # green = other cars

            box_lines.append(create_bbox_lineset(box, color=color))


    
        # ---------------------------------------------------------
        # Open3D visualization + save PNG (CORRECT ORDER)
        # ---------------------------------------------------------
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="Optimized Patch Visualization",
            width=1280,
            height=720,
            visible=True   # set False if running headless
            # visible=False   # set False if running headless
        )

        # Add geometries
        vis.add_geometry(pcd_scene)
        vis.add_geometry(pcd_patch)
        for box in box_lines:
            vis.add_geometry(box)

        # Render options
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.0, 0.0, 0.0])  # black background
        opt.point_size = 1.0                                # thinner points
        opt.line_width = 2.0

        # ---- CRITICAL: render one frame ----
        vis.poll_events()
        vis.update_renderer()

        # ---- SAVE IMAGE ----
        figure_dir = "figures/"
        os.makedirs(os.path.dirname(figure_dir), exist_ok=True)
        img_name = sample['data_samples'].img_path[0]
        out_path = os.path.join(figure_dir, img_name)
        vis.capture_screen_image(out_path, do_render=True)

        print(f" Saved visualization to {out_path}")

        # Optional: keep window open for inspection
        vis.run()

        vis.destroy_window()




# ------------------------------------------------------------------
if __name__ == '__main__':
    main()
