import json
import os
from collections import defaultdict
import numpy as np
import torch

from mmdet3d.registry import METRICS
from mmdet3d.evaluation.metrics.kitti_metric import KittiMetric
from mmdet3d.structures import bbox_overlaps_3d

from mmdet3d.structures import BaseInstance3DBoxes
from mmdet3d.evaluation.functional.kitti_utils.eval import (
    d3_box_overlap,
    clean_data
)

from mmdet3d.structures import CameraInstance3DBoxes, Box3DMode
from collections import defaultdict



@METRICS.register_module()
class KittiMetricASR(KittiMetric):

    def __init__(self,
                 manifest_path=None,
                 baseline_map=None, baseline_car_ap=None, baseline_ped_ap=None, baseline_cyc_ap=None,
                 iou_thresholds=None,
                 map_key="pred_instances_3d/KITTI/Overall_3D_AP11_moderate",
                 **kwargs):
        super().__init__(**kwargs)

        self.manifest_path = manifest_path
        self.baseline_map = baseline_map
        self.map_key = map_key

        # IoU thresholds per class index
        self.iou_thresholds = iou_thresholds or {
            0: 0.5,  # Pedestrian
            1: 0.5,  # Cyclist
            2: 0.7   # Car
        }

        self.patched_objects = []
        if manifest_path is not None:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            self.patched_objects = manifest["deployment"]["patched_objects"]

        print("Total patched objects:", len(self.patched_objects))

        # Build lookup: frame → list of patched entries
        self.frame_to_patched = defaultdict(list)

        # for entry in self.patched_objects:
        #     self.frame_to_patched[entry["frame"]].append(entry)
        temp = defaultdict(list)
        for entry in self.patched_objects:
            temp[entry["frame"]].append(entry)

        for frame, entries in temp.items():
            self.frame_to_patched[frame] = self._dedup_entries(entries)


        self.lidar2cam = np.eye(4)

        self.baseline_ap = {
            'Car': baseline_car_ap,
            'Pedestrian': baseline_ped_ap,
            'Cyclist': baseline_cyc_ap
        } if baseline_car_ap is not None else {}
        
        
        print("Unique patched frames:", len(self.frame_to_patched))

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _box_center_distance(self, boxes, target_box):
        return np.linalg.norm(boxes[:, :3] - target_box[:3], axis=1)

    
    def _compute_iou(self, pred_boxes, gt_box):
        """
        pred_boxes: (N, 7) numpy
        gt_box: (7,) numpy
        returns IoU array (N,)
        """

        if len(pred_boxes) == 0:
            return np.zeros((0,), dtype=np.float32)

        # pred_tensor = torch.tensor(pred_boxes, dtype=torch.float32)
        # gt_tensor = torch.tensor(gt_box[None, :], dtype=torch.float32)
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
    
    def _compute_bev_iou(self, pred_boxes, gt_box):
        """
        Compute BEV IoU (top-down overlap)
        pred_boxes: (N,7) [x,y,z,l,w,h,yaw]
        gt_box: (7,)
        """

        if len(pred_boxes) == 0:
            return np.zeros((0,), dtype=np.float32)

        pred_tensor = torch.from_numpy(pred_boxes).float()
        gt_tensor = torch.from_numpy(gt_box[None, :]).float()

        pred_boxes3d = BaseInstance3DBoxes(pred_tensor)
        gt_boxes3d = BaseInstance3DBoxes(gt_tensor)

        # Use BEV representation
        pred_bev = pred_boxes3d.bev
        gt_bev = gt_boxes3d.bev

        # Compute BEV IoU using 2D box overlap
        from mmcv.ops import box_iou_rotated

        overlaps = box_iou_rotated(
            pred_bev,
            gt_bev
        )

        return overlaps.squeeze(1).cpu().numpy()

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

        # GT
        gt_loc = gt_anno['location'][gt_indices]
        gt_dim = gt_anno['dimensions'][gt_indices]  # (h, w, l)
        gt_rot = gt_anno['rotation_y'][gt_indices]

        gt_boxes = np.concatenate([
            gt_loc,
            gt_dim[:, [2, 1, 0]],  # convert (h,w,l) -> (l,w,h)
            gt_rot[:, None]
        ], axis=1)

        # DT
        dt_loc = dt_anno['location'][dt_indices]
        dt_dim = dt_anno['dimensions'][dt_indices]
        dt_rot = dt_anno['rotation_y'][dt_indices]

        dt_boxes = np.concatenate([
            dt_loc,
            dt_dim[:, [2, 1, 0]],
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

        # thresholds = [0.1, 0.3, 0.5]
        score_thresholds = [0.1, 0.3, 0.5]
        success_radius = 2.0   # meters (match optimization ~1–2m)

        # -----------------------------------
        # ASR counters  (per threshold + class)
        # -----------------------------------
        asr_success = {
            thr: defaultdict(int) for thr in score_thresholds
        }
        asr_total = defaultdict(int)

        # -----------------------------------
        # Target-class FP diagnostics
        # -----------------------------------

        # tcfp_success = defaultdict(int)
        # tcfp_thresholds = [0.0, 0.05, 0.1]

        # -----------------------------------
        # IoU histogram diagnostics
        # -----------------------------------
        # iou_samples = []
        # bev_iou_samples = []
        # distance_iou_table = []
        closest_car_distances = []


        diagnostics = {
            'num_patched_entries': 0,
            'num_with_dt_car': 0,
            # 'num_with_iou_gt_0': 0,
            # 'max_iou_observed': 0.0,
            # 'num_gt_iou_ge_0_1': 0,
            'closest_car_distances' : 0,
            'bev_iou_samples' : 0,
            'best_local_scores' : [],
        }

        debug_limit = 10
        debug_count = 0

        car_label = self.classes.index('Car')

        for idx in range(len(gt_annos)):

            gt_anno = gt_annos[idx]
            dt_anno = dt_annos[idx]

            sample_idx = results[idx]['sample_idx']
            data_info = self.data_infos[sample_idx]
            frame_name = self._get_frame_from_data_info(data_info)
            # frame_name = self._get_frame_name(data_info)
            if frame_name not in self.frame_to_patched:
                continue


            calib = data_info['lidar_points']

            Tr_velo_to_cam = np.array(calib['Tr_velo_to_cam'])

            # Ensure 4×4
            if Tr_velo_to_cam.shape == (3, 4):
                lidar2cam = np.eye(4)
                lidar2cam[:3, :4] = Tr_velo_to_cam
            elif Tr_velo_to_cam.shape == (4, 4):
                lidar2cam = Tr_velo_to_cam
            else:
                raise RuntimeError(f"Unexpected Tr_velo_to_cam shape: {Tr_velo_to_cam.shape}")

            # convert cam → lidar
            cam2lidar = np.linalg.inv(lidar2cam)

            self.lidar2cam = cam2lidar



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
            # ASR
            # -------------------------
            # raw_entries = self.frame_to_patched.get(frame_name, [])
            # frame_entries = self._dedup_entries(raw_entries)
            
            frame_entries = self.frame_to_patched[frame_name]
            # conditional breakpoint
            # any(e.get("class") in ["Pedestrian", "Cyclist"] 
            #     for e in self.frame_to_patched.get(frame_name, []))
            
            # Extract LiDAR predictions directly from raw results
            result = results[idx]
            pred_instances = result['pred_instances_3d']

            pred_boxes = pred_instances['bboxes_3d'].tensor.cpu().numpy()
            pred_labels = pred_instances['labels_3d'].cpu().numpy()
            pred_scores = pred_instances['scores_3d'].cpu().numpy()

            for patched_entry in frame_entries:

                debug_count+=1

                # source_cls = patched_entry.get("class", patched_entry.get("source_class"))
                source_cls = patched_entry["source_class"]
                if source_cls is None:
                    continue

                # Denominator
                asr_total[source_cls] += 1
                diagnostics['num_patched_entries'] += 1
                

                # -------------------------
                # LiDAR-frame ASR
                # -------------------------
                # Filter predictions to Car class
                car_mask = (pred_labels == car_label)

                if not np.any(car_mask):
                    continue

                diagnostics['num_with_dt_car'] += 1

                # ------------------------------------------------
                # Step 1 — Extract Car predictions
                # ------------------------------------------------
                car_boxes = pred_boxes[car_mask]
                car_scores = pred_scores[car_mask]

                # -----------------------------------------
                # Score filtering (recommended for KITTI)
                # -----------------------------------------
                score_thr = 0.1
                score_mask = car_scores > score_thr

                car_boxes = car_boxes[score_mask]
                car_scores = car_scores[score_mask]

                if len(car_boxes) == 0:
                    continue


                # ------------------------------------------------
                # Step 3 — Patch box
                # ------------------------------------------------
                # manifest_box_np = np.array(
                #     patched_entry["patch_reference_box_lidar"],
                #     dtype=np.float32
                # ).reshape(-1)

                # assert manifest_box_np.shape == (7,), \
                #     f"Patch box wrong shape: {manifest_box_np.shape}"


                # # filtering nearby boxes
                # patch_center = manifest_box_np[:3]

                patch_center = np.array(patched_entry["patch_centroid"])

                # ------------------------------------------------
                # Step 4 — Distance-based local detections (NEW)
                # ------------------------------------------------
                dists = np.linalg.norm(car_boxes[:, :2] - patch_center[:2], axis=1)

                if len(dists) == 0:
                    continue

                closest_dist = float(dists.min())
                closest_car_distances.append(closest_dist)

                # strictly local detections (NO fallback)
                near_mask = dists < success_radius

                near_boxes = car_boxes[near_mask]
                near_scores = car_scores[near_mask]

                # If no nearby detections → attack failed
                if len(near_boxes) == 0:
                    continue

                # DEBUG
                # if debug_count < debug_limit:
                #                     print("\n=== PATCH BOX RAW ===")
                #                     print("frame:", frame_name)
                #                     print("patch center:", patch_center)
                #                     print("gt box example:", gt_anno['location'][0], gt_anno['dimensions'][0])

                # ------------------------------------------------
                # Step 5 — Distance + confidence ASR (NEW)
                # ------------------------------------------------
                best_local_score = float(near_scores.max())

                diagnostics['best_local_scores'].append(best_local_score)


                for thr in score_thresholds:
                    if np.any(near_scores >= thr):
                        asr_success[thr][source_cls] += 1

                                

        if len(closest_car_distances) > 0:
            dist_arr = np.array(closest_car_distances)

        print("\n===== ASR DIAGNOSTICS (DISTANCE-BASED) =====")
        print("Total patched entries:", diagnostics['num_patched_entries'])
        print("Frames with any Car prediction:", diagnostics['num_with_dt_car'])
        print("Closest detection distance stats available")
        print("============================================\n")

        if 'best_local_scores' in diagnostics and len(diagnostics['best_local_scores']) > 0:
            score_arr = np.array(diagnostics['best_local_scores'])

            print("\n===== PATCH SCORE DIAGNOSTICS =====")
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
            print("====================================\n")



        # -----------------------------------------
        # Distance vs IoU diagnostic table
        # -----------------------------------------

        # if len(distance_iou_table) > 0:

        #     print("\n===== PATCH DETECTION TABLE =====")

        #     table = sorted(distance_iou_table, key=lambda x: x[0])

        #     for dist, iou, score in table[:25]:
        #         print(
        #             f"dist={dist:6.2f} m | IoU={iou:6.3f} | score={score:5.3f}"
        #         )

        #     print("=================================\n")

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


        asr_metrics = {}
        for thr in score_thresholds:

            ped_total = asr_total['Pedestrian']
            cyc_total = asr_total['Cyclist']

            ped_success = asr_success[thr]['Pedestrian']
            cyc_success = asr_success[thr]['Cyclist']

            asr_ped = ped_success / ped_total if ped_total > 0 else 0.0
            asr_cyc = cyc_success / cyc_total if cyc_total > 0 else 0.0

            total_success = ped_success + cyc_success
            total_total = ped_total + cyc_total

            asr_overall = total_success / total_total if total_total > 0 else 0.0

            asr_metrics[f'ASR@{thr}_Pedestrian'] = asr_ped  # ASR@0.3: probability of triggering a Car detection within 2m with score ≥ 0.3

            asr_metrics[f'ASR@{thr}_Cyclist'] = asr_cyc
            asr_metrics[f'ASR@{thr}_overall'] = asr_overall

        metrics.update({
            'CA_Car': ca_per_class['Car'],
            'CA_Pedestrian': ca_per_class['Pedestrian'],
            'CA_Cyclist': ca_per_class['Cyclist'],
            'CA_avg': ca_avg,
        })

        metrics.update(asr_metrics)




        # delta mAP
        if self.baseline_map is not None:
            if self.map_key not in metrics:
                raise RuntimeError(
                    f"{self.map_key} not found in metrics."
                )
            metrics['delta_mAP'] = (
                metrics[self.map_key] - self.baseline_map
            )

        # -----------------------------------------
        # delta AP per class
        # -----------------------------------------

        ap_keys = {
            'Car': 'pred_instances_3d/KITTI/Car_3D_AP11_moderate_strict',
            'Pedestrian': 'pred_instances_3d/KITTI/Pedestrian_3D_AP11_moderate_strict',
            'Cyclist': 'pred_instances_3d/KITTI/Cyclist_3D_AP11_moderate_strict'
        }

        for cls, key in ap_keys.items():

            if key in metrics and cls in self.baseline_ap:

                metrics[f'deltaAP_{cls}'] = (
                    metrics[key] - self.baseline_ap[cls]
        )
                
        
        # ---------------------------------------
        # Convert custom metrics to percentage
        # ---------------------------------------
        for k in list(metrics.keys()):
            if k.startswith('CA_') or k.startswith('ASR@'):
                metrics[k] *= 100


        return metrics