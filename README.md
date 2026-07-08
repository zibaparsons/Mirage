MIRAGE
A clean-label, black-box backdoor attack against LiDAR 3D object detection.

Preprint: https://arxiv.org/abs/2606.20752
License:  AGPL-3.0


OVERVIEW
--------
Mirage is the first clean-label backdoor attack against standard LiDAR 3D
object detection (3DOD) that runs under a black-box threat model, poisons
only raw point clouds, and induces targeted misclassification. The attacker
injects a small number of label-consistent poisoned scenes into the training
set - never touching the ground-truth annotations - so the compromised
detector behaves normally on clean inputs but misclassifies any object
carrying the trigger as an attacker-chosen target class (e.g., pedestrians
and cyclists detected as cars).

The trigger is an optimized ~250-point adversarial patch placed at the center
of a target's bounding box. Because labels are never modified, the patch is
crafted on a point-based 3DSSD surrogate and transfers to unseen pillar- and
voxel-based victims (PointPillars, SECOND). On KITTI, Mirage reaches a 73%
misclassification success rate at a 0.5% poisoning rate while keeping clean
detection accuracy close to a benign model.

Research use only. This is a security research artifact for studying and
defending against poisoning attacks on autonomous-driving perception. It is
not intended to be used against real systems.


HOW IT WORKS
------------
1. Optimize the trigger - craft a label-consistent adversarial point patch on
   a 3DSSD surrogate detector (tools/adv_train/).
2. Poison the data - inject the patch into a small fraction of KITTI point
   clouds, leaving all labels intact, to build a "patched" velodyne set.
3. Train the victim - fine-tune a black-box victim detector (PointPillars /
   SECOND) on the poisoned data.
4. Evaluate - measure clean detection accuracy and the misclassification
   success rate with the trigger toggled on vs. off.


REPOSITORY LAYOUT
-----------------
This repo holds the Mirage-specific files that layer on top of an
MMDetection3D checkout (https://github.com/open-mmlab/mmdetection3d):

  configs/
    3dssd/          surrogate configs: fine-tune + trigger ("toggle patch") on KITTI cars
    pointpillars/   victim configs: poisoning, poisoned-model testing, evaluation, visualization
  tools/
    adv_train/      adversarial patch (trigger) generation
  mmdet3d/
    evaluation/     modified metrics for backdoor / misclassification evaluation

Each config exposes a use_patched_velodyne flag that switches the dataloader
between the clean and patched LiDAR directories.


SETUP
-----
Mirage builds on OpenMMLab's MMDetection3D and the KITTI dataset.

1. Install MMDetection3D and its dependencies - see the official install guide:
   https://mmdetection3d.readthedocs.io/en/latest/get_started.html
2. Prepare KITTI following the MMDetection3D data instructions.
3. Copy this repo's configs/, tools/adv_train/, and mmdet3d/evaluation/ into
   your MMDetection3D tree.


USAGE
-----
The workflow uses the standard MMDetection3D training / testing entry points.

1. Generate the trigger patch on the 3DSSD surrogate (also writes a patched
   copy of the KITTI velodyne data):

     python tools/adv_train/gen_noise_patch_3dssd_kitti_full_blob.py

2. Train a poisoned victim on the patched data (set use_patched_velodyne = True
   in the config):

     python tools/train.py configs/pointpillars/pointpillars_poisoning_kitti-3d-3class.py

3. Evaluate the backdoor - clean accuracy vs. attack success:

     python tools/test.py \
       configs/pointpillars/pointpillars_toggle_patch_kitti-3d-3class_EVALUATION_V3.py \
       <path/to/poisoned_checkpoint.pth>

Toggle use_patched_velodyne to compare benign (trigger off) and triggered
(trigger on) behavior. Dataset paths (e.g., the patched velodyne directory)
and victim checkpoints are set inside each config.


CITATION
--------
@article{parsons2026mirage,
  title   = {Mirage: a Clean-Label Backdoor against LiDAR 3D Object Detection},
  author  = {Parsons, Ziba and Li, Ang},
  journal = {arXiv preprint arXiv:2606.20752},
  year    = {2026}
}


LICENSE
-------
Released under the GNU Affero General Public License v3.0 (AGPL-3.0).
See the LICENSE file.
