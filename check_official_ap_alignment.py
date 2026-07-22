#!/usr/bin/env python3
"""Check AP-definition compatibility without evaluating any model predictions."""

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
OFFICIAL_V2X = PROJECT_ROOT.parent / "DAIR-V2X" / "v2x"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(OFFICIAL_V2X))

import evaluate_and_visualize_vehicle_lidar_baseline as current  # noqa: E402
from v2x_utils import eval_utils as official  # noqa: E402


def synthetic_ap_check():
    # Same ranked TP/FP sequence, checked against both implementations.
    flags = [True, False, True]
    detections = [
        {"score": 0.9, "is_tp": flags[0]},
        {"score": 0.8, "is_tp": flags[1]},
        {"score": 0.7, "is_tp": flags[2]},
    ]
    official_rows = [
        {"score": 0.9, "type": "tp"},
        {"score": 0.8, "type": "fp"},
        {"score": 0.7, "type": "tp"},
    ]
    current_ap = current.compute_ap_from_detections(detections, 2)
    official_ap = official.compute_ap(official_rows, 2)
    return {
        "current_ap": float(current_ap),
        "official_ap": float(official_ap),
        "equal": bool(np.isclose(current_ap, official_ap, atol=1e-12)),
    }


def synthetic_geometry_check():
    # The converter's corner order is adapted to the official evaluator's
    # documented order by the same permutations used in eval_utils.py.
    box_a = {
        "center_world": [10.0, 5.0, 1.0],
        "box_world": {"dx": 4.0, "dy": 2.0, "dz": 2.0, "heading_world": 0.35},
    }
    box_b = {
        "center_world": [10.5, 5.1, 1.0],
        "box_world": {"dx": 4.0, "dy": 2.0, "dz": 2.0, "heading_world": 0.35},
    }
    def make_corners(center, dx, dy, dz, yaw):
        hx, hy, hz = dx / 2, dy / 2, dz / 2
        local = np.array([
            [hx, hy, -hz], [hx, -hy, -hz], [-hx, -hy, -hz], [-hx, hy, -hz],
            [hx, hy, hz], [hx, -hy, hz], [-hx, -hy, hz], [-hx, hy, hz],
        ], dtype=float)
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
        return local @ rot.T + np.asarray(center, dtype=float)

    corners_a = make_corners(box_a["center_world"], 4.0, 2.0, 2.0, 0.35)
    corners_b = make_corners(box_b["center_world"], 4.0, 2.0, 2.0, 0.35)
    raw_pred_a = corners_a[[2, 6, 7, 3, 1, 5, 4, 0]]
    raw_pred_b = corners_b[[2, 6, 7, 3, 1, 5, 4, 0]]
    raw_gt_a = corners_a[[3, 0, 1, 2, 7, 4, 5, 6]]
    raw_gt_b = corners_b[[3, 0, 1, 2, 7, 4, 5, 6]]
    official_a = raw_gt_a[official.perm_label]
    official_b = raw_pred_b[official.perm_pred]
    official_3d, official_bev = official.box3d_iou(official_a, official_b)
    current_bev = current.bev_iou(corners_a[:4, :2], corners_b[:4, :2])
    current_3d = current.iou_3d(
        {"center_world": box_a["center_world"], "box_world": {
            "corners_bev_world": corners_a[:4, :2].tolist(),
            "z_min": 0.0, "z_max": 2.0,
        }},
        {"center_world": box_b["center_world"], "box_world": {
            "corners_bev_world": corners_b[:4, :2].tolist(),
            "z_min": 0.0, "z_max": 2.0,
        }},
    )
    return {
        "official_bev_iou": float(official_bev),
        "current_bev_iou": float(current_bev),
        "official_3d_iou": float(official_3d),
        "current_3d_iou": float(current_3d),
        "bev_equal": bool(np.isclose(official_bev, current_bev, atol=1e-9)),
        "3d_equal": bool(np.isclose(official_3d, current_3d, atol=1e-9)),
        "corner_permutation": list(official.perm_pred),
    }


def main():
    official_thresholds = {
        "Car": [0.30, 0.50, 0.70],
        "Pedestrian": [0.25, 0.50],
        "Cyclist": [0.25, 0.50],
    }
    current_thresholds = {
        cls: list(current.AP_IOU_THRESHOLDS) for cls in current.AP_CLASSES
    }
    threshold_match = current_thresholds == official_thresholds
    report = {
        "prediction_evaluation_performed": False,
        "official_source": "DAIR-V2X/v2x/v2x_utils/eval_utils.py",
        "current_source": "dair_v2x_project/evaluate_and_visualize_vehicle_lidar_baseline.py",
        "official_iou_thresholds": official_thresholds,
        "current_iou_thresholds": current_thresholds,
        "thresholds_aligned": threshold_match,
        "matching_rule": {
            "official": "per-GT maximum-IoU prediction matching, then remove matched prediction",
            "current_ap": "per-prediction score-ranked maximum-IoU unmatched-GT matching",
            "aligned": False,
        },
        "geometry": {
            "official": "box3d_iou with polygon BEV intersection and height overlap",
            "current": "polygon BEV intersection and height overlap",
            "requires_numeric_box_corner_check": False,
            **synthetic_geometry_check(),
        },
        "ap_interpolation": synthetic_ap_check(),
        "current_evaluator_aligned": False,
        "official_aligned_evaluator_ready": True,
        "conclusion": "现有诊断 evaluator 仍未完全对齐；官方对齐 evaluator 已单独实现，可用于正式评价。",
    }
    output = PROJECT_ROOT / "outputs" / "diagnostics" / "official_ap_alignment_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"报告已保存: {output}")


if __name__ == "__main__":
    main()
