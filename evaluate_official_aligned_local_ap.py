#!/usr/bin/env python3
"""Official DAIR-V2X AP evaluation in each sensor's local LiDAR frame."""

import argparse
import csv
import json
from pathlib import Path

import evaluate_official_aligned_ap as official_eval
import evaluate_and_visualize_vehicle_lidar_baseline as legacy
from v2x_utils import eval_utils as official


ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("/mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure")
BASELINES = {
    "vehicle_lidar": ("vehicle-side/label/lidar", "vehicle_lidar"),
    "infrastructure_lidar": ("infrastructure-side/label/virtuallidar", "infrastructure_lidar"),
    "vehicle_camera": ("vehicle-side/label/lidar", "vehicle_lidar"),
    "infrastructure_camera": ("infrastructure-side/label/virtuallidar", "infrastructure_lidar"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", choices=list(BASELINES) + ["all"], required=True)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--classes", nargs="+", choices=["car", "pedestrian", "cyclist"], default=["car"])
    return p.parse_args()


def local_prediction_objects(record):
    objects = []
    for obj in record.get("pred_objects", []):
        box = obj.get("box_lidar", {})
        objects.append({
            "class": obj.get("class", "Unknown"),
            "score": float(obj.get("score", 0.0)),
            "center_world": list(obj["center_lidar"]),
            "box_world": {
                "dx": float(box.get("dx", 0.0)),
                "dy": float(box.get("dy", 0.0)),
                "dz": float(box.get("dz", 0.0)),
                "heading_world": float(box.get("heading", 0.0)),
            },
        })
    return objects


def evaluate_one(baseline, data_root, classes):
    label_rel, sensor = BASELINES[baseline]
    root = ROOT / "outputs/full_baselines" / baseline
    pred_path = root / f"predictions_{baseline}.json"
    if not pred_path.is_file():
        raise FileNotFoundError(pred_path)
    predictions = json.loads(pred_path.read_text(encoding="utf-8"))
    metrics = {view: {cls: {} for cls in classes} for view in ("bev", "3d")}
    skipped = []
    for view in ("bev", "3d"):
        for cls in classes:
            for threshold in official.iou_threshold_dict[cls]:
                rows, gt_count = [], 0
                for record in predictions:
                    source_id = str(record.get("source_id", record.get("sample_id", "")))
                    label_path = data_root / label_rel / f"{source_id}.json"
                    if not label_path.is_file():
                        if source_id not in {x["source_id"] for x in skipped}:
                            skipped.append({"source_id": source_id, "reason": str(label_path)})
                        continue
                    gt_objects = legacy.parse_gt_objects(label_path)
                    pred, gt = official_eval.official_record(local_prediction_objects(record), gt_objects)
                    frame_rows, num_gt, _ = official.compute_type(gt, pred, cls, threshold, view)
                    rows.extend({"score": float(x.get("score", 0.0)), "type": x.get("type", "fp")} for x in frame_rows)
                    gt_count += int(num_gt)
                metrics[view][cls][f"iou_{threshold:.2f}"] = {
                    "ap": float(official.compute_ap(rows, gt_count)) if gt_count else None,
                    "num_gt": gt_count,
                    "num_predictions": len(rows),
                }
    report = {
        "baseline": baseline,
        "coordinate_protocol": sensor,
        "gt_protocol": str(data_root / label_rel),
        "official_source": "DAIR-V2X/v2x/v2x_utils/eval_utils.py",
        "prediction_evaluation_performed": True,
        "num_input_samples": len(predictions),
        "num_evaluated_samples": len(predictions) - len(skipped),
        "skipped_samples": skipped,
        "evaluated_classes": classes,
        "iou_thresholds": official.iou_threshold_dict,
        "metrics": metrics,
        "note": "Official-aligned local-coordinate AP; no world transform or relative_error is applied.",
    }
    out_json = root / "official_local_aligned_ap_summary.json"
    out_csv = root / "official_local_aligned_ap.csv"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["view", "class", "iou", "ap", "num_gt", "num_predictions"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for view in metrics:
            for cls in metrics[view]:
                for iou, value in metrics[view][cls].items():
                    writer.writerow({"view": view, "class": cls, "iou": iou, **value})
    print(f"[{baseline}] local AP output: {out_json}")
    return report


def main():
    a = parse_args()
    baselines = list(BASELINES) if a.baseline == "all" else [a.baseline]
    for baseline in baselines:
        evaluate_one(baseline, a.data_root, a.classes)


if __name__ == "__main__":
    main()
