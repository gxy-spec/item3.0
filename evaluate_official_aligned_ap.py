#!/usr/bin/env python3
"""DAIR-V2X official-aligned BEV/3D AP evaluator.

This evaluator is intentionally separate from the project's diagnostic
center-distance evaluator and never overwrites its output files.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
ITEM3_ROOT = PROJECT_ROOT.parent
OFFICIAL_V2X_ROOT = ITEM3_ROOT / "DAIR-V2X" / "v2x"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(OFFICIAL_V2X_ROOT))

import evaluate_and_visualize_vehicle_lidar_baseline as legacy  # noqa: E402
from v2x_utils import eval_utils as official  # noqa: E402


BASELINES = {
    "vehicle_lidar": PROJECT_ROOT / "outputs/full_baselines/vehicle_lidar",
    "infrastructure_lidar": PROJECT_ROOT / "outputs/full_baselines/infrastructure_lidar",
    "vehicle_camera": PROJECT_ROOT / "outputs/full_baselines/vehicle_camera",
    "infrastructure_camera": PROJECT_ROOT / "outputs/full_baselines/infrastructure_camera",
}


def args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=list(BASELINES) + ["all"], required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ITEM3_ROOT / "DAIR-V2X/data/DAIR-V2X/cooperative-vehicle-infrastructure",
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--classes", nargs="+", choices=["car", "pedestrian", "cyclist"],
        default=["car", "pedestrian", "cyclist"],
        help="官方评价类别；common-val GT 仅含 Car 时建议使用 --classes car。",
    )
    return parser.parse_args()


def corners_3d(obj, is_gt=False):
    box = obj.get("box_world", {})
    corners = box.get("corners_3d_world") or obj.get("corners_3d_world")
    if corners is not None:
        arr = np.asarray(corners, dtype=float)
        if arr.shape == (8, 3):
            # Project/world_8_points order is: bottom perimeter then top
            # perimeter. Official eval_utils applies perm_pred/perm_label
            # internally, so adapt our order to its raw input convention.
            order = [3, 0, 1, 2, 7, 4, 5, 6] if is_gt else [2, 6, 7, 3, 1, 5, 4, 0]
            return arr[order]

    center = np.asarray(obj["center_world"], dtype=float)
    dx = float(box.get("dx", box.get("length", 1.0)))
    dy = float(box.get("dy", box.get("width", 1.0)))
    dz = float(box.get("dz", box.get("height", 1.0)))
    yaw = float(box.get("heading_world", box.get("yaw", 0.0)))
    hx, hy, hz = dx / 2, dy / 2, dz / 2
    local = np.array([
        [hx, hy, -hz], [hx, -hy, -hz], [-hx, -hy, -hz], [-hx, hy, -hz],
        [hx, hy, hz], [hx, -hy, hz], [-hx, -hy, hz], [-hx, hy, hz],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    arr = local @ rot.T + center
    order = [3, 0, 1, 2, 7, 4, 5, 6] if is_gt else [2, 6, 7, 3, 1, 5, 4, 0]
    return arr[order]


def official_record(pred_objects, gt_objects):
    class_to_id = {"pedestrian": 0, "cyclist": 1, "car": 2}

    def convert(objects, prediction):
        boxes, labels, scores = [], [], []
        for obj in objects:
            cls = legacy.normalize_class_name(obj.get("class", "Unknown")).lower()
            if cls not in official.superclass.values():
                continue
            boxes.append(corners_3d(obj, is_gt=not prediction))
            labels.append(class_to_id[cls])
            scores.append(float(obj.get("score", 1.0)))
        return {
            "boxes_3d": np.asarray(boxes, dtype=float).reshape((-1, 8, 3)),
            "labels_3d": labels,
            "scores_3d": scores,
        }
    return convert(pred_objects, True), convert(gt_objects, False)


def evaluate_one(baseline, data_root, input_path=None, output_dir=None, class_names=None):
    root = output_dir or BASELINES[baseline]
    input_path = input_path or (root / "predictions_world.json")
    output_dir = Path(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.is_file():
        raise FileNotFoundError(f"缺少 world prediction: {input_path}")

    legacy.DATA_ROOT = Path(data_root).resolve()
    legacy.COOP_INFO_PATH = legacy.DATA_ROOT / "cooperative/data_info.json"
    sample_mapping = legacy.build_sample_mapping()
    predictions = json.loads(input_path.read_text(encoding="utf-8"))
    class_names = class_names or ["car", "pedestrian", "cyclist"]
    def frame_data(record):
        sample_id = str(record.get("sample_id", ""))
        vehicle_id = str(record.get("vehicle_id", sample_id))
        mapping = sample_mapping.get(vehicle_id) or sample_mapping.get(sample_id)
        if mapping is None:
            return None, {"sample_id": sample_id, "reason": "missing cooperative mapping"}
        label_path = Path(mapping["cooperative_label_path"])
        if not label_path.is_file():
            return None, {"sample_id": sample_id, "reason": f"missing GT: {label_path}"}
        gt_objects = legacy.parse_gt_objects(label_path)
        return official_record(record.get("pred_objects", []), gt_objects), None

    # Evaluate one official metric at a time so complete 8x3 boxes are never
    # retained for all classes/views/thresholds simultaneously.
    metrics = {view: {cls: {} for cls in class_names} for view in ("bev", "3d")}
    skipped = []
    used = 0
    for view in ("bev", "3d"):
        for cls in class_names:
            for threshold in official.iou_threshold_dict[cls]:
                rows, gt_count = [], 0
                for record in predictions:
                    data, skip = frame_data(record)
                    if skip:
                        if view == "bev" and threshold == official.iou_threshold_dict[cls][0]:
                            skipped.append(skip)
                        continue
                    pred, gt = data
                    frame_rows, num_gt, _ = official.compute_type(gt, pred, cls, threshold, view)
                    rows.extend({"score": float(row.get("score", 0.0)), "type": row.get("type", "fp")}
                                for row in frame_rows)
                    gt_count += int(num_gt)
                metrics[view][cls][f"iou_{threshold:.2f}"] = {
                    "ap": float(official.compute_ap(rows, gt_count)) if gt_count else None,
                    "num_gt": int(gt_count), "num_predictions": int(len(rows)),
                }
    used = len(predictions) - len({x["sample_id"] for x in skipped})

    report = {
        "baseline": baseline,
        "official_source": "DAIR-V2X/v2x/v2x_utils/eval_utils.py",
        "prediction_evaluation_performed": True,
        "num_input_samples": len(predictions),
        "num_evaluated_samples": used,
        "skipped_samples": skipped,
        "iou_thresholds": official.iou_threshold_dict,
        "evaluated_classes": class_names,
        "metrics": metrics,
        "matching_rule": "official per-GT maximum-IoU matching",
        "note": "This file is official-aligned AP and is separate from project diagnostic metrics.",
    }
    out_json = output_dir / "official_aligned_ap_summary.json"
    out_csv = output_dir / "official_aligned_ap.csv"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = []
    for view in metrics:
        for cls in metrics[view]:
            for threshold, value in metrics[view][cls].items():
                rows.append({"view": view, "class": cls, "iou": threshold, **value})
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["view", "class", "iou", "ap", "num_gt", "num_predictions"])
        writer.writeheader()
        writer.writerows(rows)
    return report


def main():
    parsed = args()
    baselines = list(BASELINES) if parsed.baseline == "all" else [parsed.baseline]
    if parsed.input and len(baselines) != 1:
        raise ValueError("--input 只能与单个 --baseline 一起使用")
    for baseline in baselines:
        report = evaluate_one(
            baseline,
            parsed.data_root,
            parsed.input,
            parsed.output_dir if len(baselines) == 1 else None,
            parsed.classes,
        )
        print(f"[{baseline}] evaluated={report['num_evaluated_samples']}, skipped={len(report['skipped_samples'])}")
        print(f"输出: {BASELINES[baseline] / 'official_aligned_ap_summary.json'}")


if __name__ == "__main__":
    main()
