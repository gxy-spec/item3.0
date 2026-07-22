#!/usr/bin/env python3
"""Aggregate four Full Dataset single-modal evaluation summaries."""

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = PROJECT_ROOT / "outputs" / "full_baselines"
BASELINES = (
    ("vehicle_lidar", "Vehicle LiDAR-only"),
    ("infrastructure_lidar", "Infrastructure LiDAR-only"),
    ("vehicle_camera", "Vehicle Camera-only"),
    ("infrastructure_camera", "Infrastructure Camera-only"),
)
COLUMNS = [
    "baseline", "prediction_type", "num_samples", "num_gt", "num_pred", "num_match", "FN", "FP",
    "mean_precision", "mean_recall", "mean_f1", "classification_accuracy",
    "BEV_mAP@0.50", "3D_mAP@0.50", "mean_loc_error_bev", "mean_loc_error_3d", "status", "error_log",
    "gt_classes_with_instances", "evaluation_scope",
    "center_match_micro_precision", "center_match_micro_recall", "center_match_micro_f1",
    "frame_macro_precision", "frame_macro_recall", "frame_macro_f1",
    "false_negative_rate", "false_positive_per_gt", "prediction_gt_ratio",
    "global_count_bias", "abs_global_count_relative_error",
    "Car_BEV_AP@0.25", "Car_BEV_AP@0.50", "Car_BEV_AP@0.70",
    "Car_3D_AP@0.25", "Car_3D_AP@0.50", "Car_3D_AP@0.70",
]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric(summary, key):
    return summary.get(key, summary.get("mean_metrics", {}).get(key))


def safe_ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def get_class_ap(ap_summary, view, iou, class_name="Car"):
    return (
        ap_summary.get("metrics", {})
        .get(view, {})
        .get(f"iou_{iou:.2f}", {})
        .get(class_name, {})
        .get("ap")
    )


METRIC_DEFINITIONS = {
    "center_match_micro_precision": "total_match / total_pred; global, class-agnostic 5 m BEV center matching.",
    "center_match_micro_recall": "total_match / total_gt; global, class-agnostic 5 m BEV center matching.",
    "center_match_micro_f1": "Harmonic mean of global center-match precision and recall.",
    "frame_macro_precision": "Arithmetic mean of per-frame precision; this is the legacy mean_precision field.",
    "frame_macro_recall": "Arithmetic mean of per-frame recall; this is the legacy mean_recall field.",
    "frame_macro_f1": "Arithmetic mean of per-frame F1; this is the legacy mean_f1 field.",
    "false_negative_rate": "FN / total_gt.",
    "false_positive_per_gt": "FP / total_gt, reported instead of an unnormalised FP count for cross-model comparison.",
    "prediction_gt_ratio": "total_pred / total_gt; values above one indicate global over-prediction.",
    "global_count_bias": "(total_pred - total_gt) / total_gt; signed global count bias.",
    "abs_global_count_relative_error": "abs(total_pred - total_gt) / total_gt.",
    "Car_*_AP": "Score-ranked AP for Car at the stated IoU threshold; classes without GT are excluded from interpretation.",
}


def main():
    parser = argparse.ArgumentParser(description="汇总 Full Dataset 四个真实单模态 baseline")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    rows, warnings = [], []
    for baseline, display_name in BASELINES:
        baseline_root = root / baseline
        eval_path, ap_path = baseline_root / "eval_summary.json", baseline_root / "ap_summary.json"
        row = {column: "N/A" for column in COLUMNS}
        row.update({"baseline": display_name, "status": "missing", "error_log": str(baseline_root / "inference_error.log")})
        if not eval_path.is_file():
            warnings.append(f"{baseline}: 缺少 {eval_path}")
            rows.append(row)
            continue
        summary = load_json(eval_path)
        prediction_types = summary.get("prediction_types", [summary.get("prediction_type", "unknown")])
        row.update({
            "prediction_type": ",".join(prediction_types),
            "num_samples": summary.get("num_samples", "N/A"),
            "num_gt": summary.get("total_gt", "N/A"),
            "num_pred": summary.get("total_pred", "N/A"),
            "num_match": summary.get("total_match", "N/A"),
            "FN": summary.get("total_false_negative", "N/A"),
            "FP": summary.get("total_false_positive", "N/A"),
            "mean_precision": summary.get("mean_precision", "N/A"),
            "mean_recall": summary.get("mean_recall", "N/A"),
            "mean_f1": summary.get("mean_f1", "N/A"),
            "classification_accuracy": summary.get("classification_accuracy", "N/A"),
            "mean_loc_error_bev": metric(summary, "mean_loc_error_bev"),
            "mean_loc_error_3d": metric(summary, "mean_loc_error_3d"),
            "status": "engineering_validation" if summary.get("engineering_validation_warning") else "evaluated",
        })
        total_gt = summary.get("total_gt")
        total_pred = summary.get("total_pred")
        total_match = summary.get("total_match")
        total_fn = summary.get("total_false_negative")
        total_fp = summary.get("total_false_positive")
        global_precision = safe_ratio(total_match, total_pred)
        global_recall = safe_ratio(total_match, total_gt)
        row.update({
            "gt_classes_with_instances": ",".join(
                class_name for class_name, values in summary.get("per_class_metrics", {}).items()
                if values.get("gt_count", 0) > 0
            ) or "N/A",
            "evaluation_scope": summary.get("evaluation_scope_note", "N/A"),
            "center_match_micro_precision": global_precision,
            "center_match_micro_recall": global_recall,
            "center_match_micro_f1": (
                2 * global_precision * global_recall / (global_precision + global_recall)
                if global_precision is not None and global_recall is not None and global_precision + global_recall > 0
                else None
            ),
            "frame_macro_precision": summary.get("mean_precision"),
            "frame_macro_recall": summary.get("mean_recall"),
            "frame_macro_f1": summary.get("mean_f1"),
            "false_negative_rate": safe_ratio(total_fn, total_gt),
            "false_positive_per_gt": safe_ratio(total_fp, total_gt),
            "prediction_gt_ratio": safe_ratio(total_pred, total_gt),
            "global_count_bias": safe_ratio(total_pred - total_gt, total_gt),
            "abs_global_count_relative_error": safe_ratio(abs(total_pred - total_gt), total_gt),
        })
        if ap_path.is_file():
            ap = load_json(ap_path)
            row["BEV_mAP@0.50"] = ap.get("map_bev_iou_0.50", ap.get("BEV_mAP@0.50", "N/A"))
            row["3D_mAP@0.50"] = ap.get("map_3d_iou_0.50", ap.get("3D_mAP@0.50", "N/A"))
            for view, prefix in (("bev", "Car_BEV_AP"), ("3d", "Car_3D_AP")):
                for iou in (0.25, 0.50, 0.70):
                    row[f"{prefix}@{iou:.2f}"] = get_class_ap(ap, view, iou)
        else:
            warnings.append(f"{baseline}: 缺少 AP summary")
        rows.append(row)

    root.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = root / "full_single_modal_summary.csv", root / "full_single_modal_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({
        "rows": rows,
        "warnings": warnings,
        "scope": "Only evaluated real-model outputs may be interpreted as detection results.",
        "metric_definitions": METRIC_DEFINITIONS,
        "interpretation_warning": (
            "Among the standard evaluation classes (Car/Pedestrian/Cyclist), the current full_common_val "
            "contains Car GT only. Non-standard labels, if present, remain in total GT counts but are outside "
            "the three-class AP and classification comparison."
        ),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary csv: {csv_path}")
    print(f"summary json: {json_path}")
    for warning in warnings:
        print(f"[WARN] {warning}")


if __name__ == "__main__":
    main()
