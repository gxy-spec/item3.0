#!/usr/bin/env python3
"""Task 2: inspect local LiDAR frames before and after world conversion."""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
from matplotlib.patches import Polygon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluate_and_visualize_vehicle_lidar_baseline as legacy
from prepare_lidar_diagnosis_task1 import (
    load_json,
    matmul,
    transform_from_calibration,
)


DEFAULT_MANIFEST = PROJECT_ROOT / "outputs/diagnosis/lidar_diagnosis_manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/diagnosis"
SENSORS = {
    "vehicle_lidar": "vehicle",
    "infrastructure_lidar": "infrastructure",
}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def finite_float(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def read_pcd(path):
    cloud = o3d.t.io.read_point_cloud(str(path))
    if "positions" not in cloud.point:
        raise ValueError(f"PCD 缺少 positions: {path}")
    xyz = np.asarray(cloud.point.positions.numpy(), dtype=np.float64)
    return xyz


def transform_points(points, transform):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    ones = np.ones((len(points), 1), dtype=np.float64)
    return (np.concatenate([points, ones], axis=1) @ np.asarray(transform).T)[:, :3]


def transform_point(point, transform):
    return transform_points(np.asarray(point, dtype=np.float64).reshape(1, 3), transform)[0]


def calibration_transform(path, apply_relative_error=False):
    raw = load_json(path)
    transform = transform_from_calibration(path)
    relative_error = raw.get("relative_error", {}) if isinstance(raw, dict) else {}
    delta_x = finite_float(relative_error.get("delta_x"), 0.0) if isinstance(relative_error, dict) else 0.0
    delta_y = finite_float(relative_error.get("delta_y"), 0.0) if isinstance(relative_error, dict) else 0.0
    if apply_relative_error:
        transform[0][3] += delta_x or 0.0
        transform[1][3] += delta_y or 0.0
    return transform, {"delta_x": delta_x or 0.0, "delta_y": delta_y or 0.0, "applied": bool(apply_relative_error)}


def normalize_label_object(raw):
    location = raw.get("3d_location", raw.get("location", raw.get("center")))
    dimensions = raw.get("3d_dimensions", raw.get("dimensions", raw.get("size")))
    if not isinstance(location, dict) or not isinstance(dimensions, dict):
        return None
    try:
        center = [float(location["x"]), float(location["y"]), float(location.get("z", 0.0))]
        dx = float(dimensions.get("l", dimensions.get("length", dimensions.get("dx"))))
        dy = float(dimensions.get("w", dimensions.get("width", dimensions.get("dy"))))
        dz = float(dimensions.get("h", dimensions.get("height", dimensions.get("dz"))))
        rotation = raw.get("rotation", raw.get("rotation_y", raw.get("yaw", 0.0)))
        if isinstance(rotation, dict):
            rotation = rotation.get("z", rotation.get("yaw", 0.0))
        heading = float(rotation)
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(center + [dx, dy, dz, heading]).all() or min(dx, dy, dz) <= 0:
        return None
    corners = build_corners(center, dx, dy, dz, heading)
    return {
        "class": legacy.normalize_class_name(raw.get("type", raw.get("class", "Unknown"))),
        "raw_class": raw.get("type", raw.get("class", "Unknown")),
        "center_world": center,
        "box_world": {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "heading_world": heading,
            "length": dx,
            "width": dy,
            "height": dz,
            "yaw": heading,
            "z_min": center[2] - dz / 2.0,
            "z_max": center[2] + dz / 2.0,
            "corners_bev_world": corners[:4, :2].tolist(),
            "corners_3d_world": corners.tolist(),
        },
    }


def parse_label_file(path):
    objects = []
    invalid = 0
    for raw in load_json(path):
        parsed = normalize_label_object(raw)
        if parsed is None:
            invalid += 1
        else:
            objects.append(parsed)
    return objects, invalid


def normalize_prediction(raw):
    center = raw.get("center_lidar")
    box = raw.get("box_lidar", {})
    try:
        center = [float(value) for value in center]
        dx, dy, dz = float(box["dx"]), float(box["dy"]), float(box["dz"])
        heading = float(box["heading"])
    except (KeyError, TypeError, ValueError):
        return None
    if len(center) != 3 or not np.isfinite(center + [dx, dy, dz, heading]).all() or min(dx, dy, dz) <= 0:
        return None
    corners = build_corners(center, dx, dy, dz, heading)
    return {
        "class": legacy.normalize_class_name(raw.get("class", "Unknown")),
        "raw_class": raw.get("class", "Unknown"),
        "score": finite_float(raw.get("score"), 0.0),
        "center_world": center,
        "box_world": {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "heading_world": heading,
            "z_min": center[2] - dz / 2.0,
            "z_max": center[2] + dz / 2.0,
            "corners_bev_world": corners[:4, :2].tolist(),
            "corners_3d_world": corners.tolist(),
        },
    }


def build_corners(center, dx, dy, dz, heading):
    x_half, y_half, z_half = dx / 2.0, dy / 2.0, dz / 2.0
    corners = np.array([
        [x_half, y_half, -z_half], [x_half, -y_half, -z_half],
        [-x_half, -y_half, -z_half], [-x_half, y_half, -z_half],
        [x_half, y_half, z_half], [x_half, -y_half, z_half],
        [-x_half, -y_half, z_half], [-x_half, y_half, z_half],
    ], dtype=np.float64)
    c, s = math.cos(heading), math.sin(heading)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return corners @ rotation.T + np.asarray(center, dtype=np.float64)


def load_predictions(path):
    data = load_json(path)
    samples = data if isinstance(data, list) else data.get("samples", [])
    return {str(item["sample_id"]): item for item in samples}


def local_eval(predictions, gts):
    matches, missed, false_pred = legacy.match_predictions_to_gt(predictions, gts, distance_threshold=5.0, class_aware=False)
    bev_ious = [legacy.bev_iou(legacy.pred_box_corners(predictions[item["pred_index"]]), legacy.gt_box_corners(gts[item["gt_index"]])) for item in matches]
    ious_3d = [legacy.iou_3d(predictions[item["pred_index"]], gts[item["gt_index"]]) for item in matches]
    num_gt, num_pred, num_match = len(gts), len(predictions), len(matches)
    precision = num_match / num_pred if num_pred else 0.0
    recall = num_match / num_gt if num_gt else 0.0
    return {
        "num_gt": num_gt,
        "num_pred": num_pred,
        "num_match": num_match,
        "TP": num_match,
        "FP": len(false_pred),
        "FN": len(missed),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mean_bev_iou_matched": float(np.mean(bev_ious)) if bev_ious else None,
        "mean_3d_iou_matched": float(np.mean(ious_3d)) if ious_3d else None,
        "mean_center_error_bev": float(np.mean([item["distance_bev"] for item in matches])) if matches else None,
        "mean_center_error_3d": float(np.mean([item["distance_3d"] for item in matches])) if matches else None,
        "class_correct": sum(item["class_correct"] for item in matches),
        "class_wrong": sum(not item["class_correct"] for item in matches),
        "matches": matches,
        "missed_gt_indices": missed,
        "false_pred_indices": false_pred,
    }


def convert_object_to_world(obj, transform):
    center = transform_point(obj["center_world"], transform)
    local_corners = np.asarray(obj["box_world"]["corners_3d_world"], dtype=np.float64)
    world_corners = transform_points(local_corners, transform)
    converted = dict(obj)
    converted["center_world"] = center.tolist()
    converted["box_world"] = dict(obj["box_world"])
    converted["box_world"]["corners_3d_world"] = world_corners.tolist()
    converted["box_world"]["corners_bev_world"] = world_corners[:4, :2].tolist()
    converted["box_world"]["z_min"] = float(world_corners[:, 2].min())
    converted["box_world"]["z_max"] = float(world_corners[:, 2].max())
    return converted


def world_object_from_raw(raw):
    location = raw.get("3d_location", raw.get("location", raw.get("center")))
    dimensions = raw.get("3d_dimensions", raw.get("dimensions", raw.get("size")))
    if not isinstance(location, dict) or not isinstance(dimensions, dict):
        return None
    try:
        center = [float(location["x"]), float(location["y"]), float(location.get("z", 0.0))]
        dx, dy, dz = float(dimensions.get("l", dimensions.get("length"))), float(dimensions.get("w", dimensions.get("width"))), float(dimensions.get("h", dimensions.get("height")))
        rotation = raw.get("rotation", raw.get("rotation_y", raw.get("yaw", 0.0)))
        if isinstance(rotation, dict):
            rotation = rotation.get("z", rotation.get("yaw", 0.0))
        corners = raw.get("world_8_points")
        corners = np.asarray(corners, dtype=np.float64) if corners is not None else build_corners(center, dx, dy, dz, float(rotation))
        if corners.shape != (8, 3):
            corners = build_corners(center, dx, dy, dz, float(rotation))
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "class": legacy.normalize_class_name(raw.get("type", raw.get("class", "Unknown"))),
        "center_world": center,
        "box_world": {
            "dx": dx, "dy": dy, "dz": dz,
            "heading_world": float(rotation), "length": dx, "width": dy, "height": dz,
            "yaw": float(rotation), "z_min": float(corners[:, 2].min()), "z_max": float(corners[:, 2].max()),
            "corners_bev_world": corners[:4, :2].tolist(), "corners_3d_world": corners.tolist(),
        },
    }


def alignment_eval(local_gt, world_gt, transform):
    transformed = [convert_object_to_world(item, transform) for item in local_gt]
    matches, missed, false_pred = legacy.match_predictions_to_gt(
        [{**item, "score": 1.0} for item in transformed], world_gt, distance_threshold=5.0, class_aware=False
    )
    center_bev = [item["distance_bev"] for item in matches]
    center_3d = [item["distance_3d"] for item in matches]
    bev_ious = [legacy.bev_iou(legacy.pred_box_corners(transformed[item["pred_index"]]), legacy.gt_box_corners(world_gt[item["gt_index"]])) for item in matches]
    ious_3d = [legacy.iou_3d(transformed[item["pred_index"]], world_gt[item["gt_index"]]) for item in matches]
    return {
        "num_local_gt": len(local_gt), "num_world_gt": len(world_gt), "num_match": len(matches),
        "unmatched_local_gt": len(false_pred), "unmatched_world_gt": len(missed),
        "mean_center_error_bev": float(np.mean(center_bev)) if center_bev else None,
        "max_center_error_bev": float(np.max(center_bev)) if center_bev else None,
        "mean_center_error_3d": float(np.mean(center_3d)) if center_3d else None,
        "max_center_error_3d": float(np.max(center_3d)) if center_3d else None,
        "mean_bev_iou": float(np.mean(bev_ious)) if bev_ious else None,
        "mean_3d_iou": float(np.mean(ious_3d)) if ious_3d else None,
        "matched_class_correct": sum(item["class_correct"] for item in matches),
        "matches": matches,
    }


def prediction_conversion_eval(local_preds, stored_world, transform):
    transformed = [convert_object_to_world(item, transform) for item in local_preds]
    count = min(len(transformed), len(stored_world))
    center_errors, corner_errors = [], []
    for index in range(count):
        center_errors.append(float(np.linalg.norm(np.asarray(transformed[index]["center_world"]) - np.asarray(stored_world[index]["center_world"]))))
        predicted_corners = np.asarray(transformed[index]["box_world"]["corners_3d_world"])
        stored_corners = np.asarray(stored_world[index].get("corners_3d_world", stored_world[index].get("box_world", {}).get("corners_3d_world", [])))
        if stored_corners.shape == (8, 3):
            corner_errors.append(float(np.max(np.linalg.norm(predicted_corners - stored_corners, axis=1))))
    return {
        "local_prediction_count": len(local_preds), "stored_world_prediction_count": len(stored_world),
        "compared_object_count": count, "count_difference": len(local_preds) - len(stored_world),
        "mean_center_residual_m": float(np.mean(center_errors)) if center_errors else None,
        "max_center_residual_m": float(np.max(center_errors)) if center_errors else None,
        "mean_corner_residual_m": float(np.mean(corner_errors)) if corner_errors else None,
        "max_corner_residual_m": float(np.max(corner_errors)) if corner_errors else None,
    }


def draw_boxes(axis, objects, matched_indices, kind):
    for index, obj in enumerate(objects):
        corners = legacy.pred_box_corners(obj) if kind == "prediction" else legacy.gt_box_corners(obj)
        if kind == "prediction":
            color = "#1683ff" if index in matched_indices else "#ff8c00"
            linestyle = "--"
            label = f"P {obj['class']} {obj.get('score', 0.0):.2f}"
        else:
            color = "#16a34a" if index in matched_indices else "#dc2626"
            linestyle = "-"
            label = f"GT {obj['class']}"
        axis.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=color, linewidth=1.1, linestyle=linestyle))
        axis.text(float(np.mean(corners[:, 0])), float(np.mean(corners[:, 1])), label, color=color, fontsize=5)


def plot_local_sample(points, gts, preds, evaluation, sensor_name, sample_id, output_path, panel_b=False):
    fig, axis = plt.subplots(figsize=(10, 8))
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) > 80000:
        indices = np.linspace(0, len(finite) - 1, 80000, dtype=int)
        finite = finite[indices]
    axis.scatter(finite[:, 0], finite[:, 1], s=0.12, c="#6b7280", alpha=0.55, linewidths=0)
    matched_gt = {item["gt_index"] for item in evaluation["matches"]}
    matched_pred = {item["pred_index"] for item in evaluation["matches"]}
    draw_boxes(axis, gts, matched_gt if panel_b else set(), "gt")
    if panel_b:
        draw_boxes(axis, preds, matched_pred, "prediction")
    axis.scatter([0], [0], c="black", s=18, marker="+", label="LiDAR origin")
    suffix = "points + local GT + raw prediction" if panel_b else "points + local GT"
    axis.set_title(f"{sensor_name} | sample {sample_id} | {suffix}\nGT={evaluation['num_gt']} Pred={evaluation['num_pred']} Match={evaluation['num_match']} P={evaluation['precision']:.3f} R={evaluation['recall']:.3f}")
    axis.set_xlabel("local x (m)")
    axis.set_ylabel("local y (m)")
    axis.axis("equal")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="完成 LiDAR Task 2：原始坐标系检查、local 评价、world 对齐和转换闭环")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = load_json(args.manifest)
    output_root = args.output_root.resolve()
    prediction_maps = {
        sensor: load_predictions(manifest["selected_samples"][0][f"{side}_predictions_json_path"])
        for sensor, side in SENSORS.items()
    }
    world_prediction_maps = {}
    for sensor, side in SENSORS.items():
        world_path = Path(manifest["selected_samples"][0][f"{side}_predictions_json_path"]).parent / "predictions_world.json"
        world_prediction_maps[sensor] = load_predictions(world_path)

    all_local_rows, all_alignment_rows, all_conversion_rows, sensor_reports = [], [], [], {}
    for sensor, side in SENSORS.items():
        local_rows, alignment_rows, conversion_rows = [], [], []
        visualization_dir = output_root / "sensor_frame_visualization" / side
        for selected in manifest["selected_samples"]:
            sample_id = selected["cooperative_sample_id"]
            frame_id = selected[f"{side}_frame_id"]
            label_path = Path(selected[f"{side}_local_label_path"])
            point_path = Path(selected[f"{side}_pointcloud_path"])
            local_gts, invalid_gt = parse_label_file(label_path)
            raw_sample = prediction_maps[sensor].get(selected[f"{side}_prediction_sample_key"])
            raw_preds = raw_sample.get("pred_objects", []) if raw_sample else []
            local_preds = [item for raw in raw_preds if (item := normalize_prediction(raw)) is not None]
            evaluation = local_eval(local_preds, local_gts)
            pointcloud = read_pcd(point_path)
            finite = pointcloud[np.isfinite(pointcloud).all(axis=1)]
            config = load_json(args.manifest).get("baseline_snapshot", {})
            local_row = {
                "sensor": sensor, "cooperative_sample_id": sample_id, "frame_id": frame_id,
                "num_points": len(pointcloud), "num_nonfinite_points": len(pointcloud) - len(finite),
                "x_min": float(np.min(finite[:, 0])) if len(finite) else None,
                "x_max": float(np.max(finite[:, 0])) if len(finite) else None,
                "y_min": float(np.min(finite[:, 1])) if len(finite) else None,
                "y_max": float(np.max(finite[:, 1])) if len(finite) else None,
                "z_min": float(np.min(finite[:, 2])) if len(finite) else None,
                "z_max": float(np.max(finite[:, 2])) if len(finite) else None,
                "gt_invalid_boxes": invalid_gt, "pred_invalid_boxes": len(raw_preds) - len(local_preds),
                "pointcloud_path": str(point_path), "local_label_path": str(label_path),
                **{key: value for key, value in evaluation.items() if key not in {"matches", "missed_gt_indices", "false_pred_indices"}},
            }
            local_rows.append(local_row)
            plot_local_sample(pointcloud, local_gts, local_preds, evaluation, sensor, sample_id, visualization_dir / f"rank_{selected['selection_rank']:02d}_{sample_id}_A_points_local_gt.png", panel_b=False)
            plot_local_sample(pointcloud, local_gts, local_preds, evaluation, sensor, sample_id, visualization_dir / f"rank_{selected['selection_rank']:02d}_{sample_id}_B_points_local_gt_prediction.png", panel_b=True)

            world_gt_raw = load_json(selected["cooperative_label_world_path"])
            world_gts = [item for raw in world_gt_raw if (item := world_object_from_raw(raw)) is not None]
            calib_paths = selected[f"{side}_calib_paths"]
            if sensor == "vehicle_lidar":
                first_transform, first_error = calibration_transform(calib_paths[0])
                second_transform, second_error = calibration_transform(calib_paths[1])
                transform = matmul(second_transform, first_transform)
                corrected_transform = transform
                relative_error = {"delta_x": 0.0, "delta_y": 0.0, "applied": False}
            else:
                transform, relative_error = calibration_transform(calib_paths[0], apply_relative_error=False)
                corrected_transform, relative_error = calibration_transform(calib_paths[0], apply_relative_error=True)
            raw_alignment = alignment_eval(local_gts, world_gts, transform)
            corrected_alignment = alignment_eval(local_gts, world_gts, corrected_transform)
            alignment_rows.append({
                "sensor": sensor, "cooperative_sample_id": sample_id, "frame_id": frame_id,
                "relative_error_delta_x": relative_error["delta_x"],
                "relative_error_delta_y": relative_error["delta_y"],
                **{f"raw_{key}": value for key, value in raw_alignment.items() if key != "matches"},
                **{f"corrected_{key}": value for key, value in corrected_alignment.items() if key != "matches"},
            })
            stored_world_sample = world_prediction_maps[sensor].get(sample_id, {})
            stored_world = [normalize_prediction_world(item) for item in stored_world_sample.get("pred_objects", [])]
            stored_world = [item for item in stored_world if item is not None]
            raw_conversion = prediction_conversion_eval(local_preds, stored_world, transform)
            corrected_conversion = prediction_conversion_eval(local_preds, stored_world, corrected_transform)
            conversion_rows.append({
                "sensor": sensor, "cooperative_sample_id": sample_id, "frame_id": frame_id,
                "relative_error_delta_x": relative_error["delta_x"],
                "relative_error_delta_y": relative_error["delta_y"],
                **{f"raw_{key}": value for key, value in raw_conversion.items()},
                **{f"corrected_{key}": value for key, value in corrected_conversion.items()},
            })

        pd.DataFrame(local_rows).to_csv(output_root / f"{side}_local_frame_eval.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(alignment_rows).to_csv(output_root / f"{side}_local_gt_world_alignment.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(conversion_rows).to_csv(output_root / f"{side}_prediction_world_conversion_check.csv", index=False, encoding="utf-8-sig")
        all_local_rows.extend(local_rows)
        all_alignment_rows.extend(alignment_rows)
        all_conversion_rows.extend(conversion_rows)
        sensor_reports[sensor] = {
            "num_samples": len(local_rows),
            "mean_local_precision": float(pd.DataFrame(local_rows)["precision"].mean()),
            "mean_local_recall": float(pd.DataFrame(local_rows)["recall"].mean()),
            "mean_local_f1": float(pd.DataFrame(local_rows)["f1"].mean()),
            "mean_raw_gt_world_center_error_bev": float(pd.DataFrame(alignment_rows)["raw_mean_center_error_bev"].dropna().mean()) if any(row["raw_mean_center_error_bev"] is not None for row in alignment_rows) else None,
            "max_raw_gt_world_center_error_bev": float(pd.DataFrame(alignment_rows)["raw_max_center_error_bev"].dropna().max()) if any(row["raw_max_center_error_bev"] is not None for row in alignment_rows) else None,
            "mean_raw_gt_world_bev_iou": float(pd.DataFrame(alignment_rows)["raw_mean_bev_iou"].dropna().mean()) if any(row["raw_mean_bev_iou"] is not None for row in alignment_rows) else None,
            "mean_corrected_gt_world_center_error_bev": float(pd.DataFrame(alignment_rows)["corrected_mean_center_error_bev"].dropna().mean()) if any(row["corrected_mean_center_error_bev"] is not None for row in alignment_rows) else None,
            "max_corrected_gt_world_center_error_bev": float(pd.DataFrame(alignment_rows)["corrected_max_center_error_bev"].dropna().max()) if any(row["corrected_max_center_error_bev"] is not None for row in alignment_rows) else None,
            "mean_corrected_gt_world_bev_iou": float(pd.DataFrame(alignment_rows)["corrected_mean_bev_iou"].dropna().mean()) if any(row["corrected_mean_bev_iou"] is not None for row in alignment_rows) else None,
            "max_raw_prediction_center_residual_m": float(pd.DataFrame(conversion_rows)["raw_max_center_residual_m"].dropna().max()) if any(row["raw_max_center_residual_m"] is not None for row in conversion_rows) else None,
            "max_corrected_prediction_center_residual_vs_existing_world_m": float(pd.DataFrame(conversion_rows)["corrected_max_center_residual_m"].dropna().max()) if any(row["corrected_max_center_residual_m"] is not None for row in conversion_rows) else None,
            "local_eval_csv": str(output_root / f"{side}_local_frame_eval.csv"),
            "gt_alignment_csv": str(output_root / f"{side}_local_gt_world_alignment.csv"),
            "prediction_conversion_csv": str(output_root / f"{side}_prediction_world_conversion_check.csv"),
            "visualization_dir": str(visualization_dir),
        }

    max_raw_alignment = max((row["raw_max_center_error_bev"] or 0.0 for row in all_alignment_rows), default=0.0)
    max_corrected_alignment = max((row["corrected_max_center_error_bev"] or 0.0 for row in all_alignment_rows), default=0.0)
    max_conversion = max((row["raw_max_center_residual_m"] or 0.0 for row in all_conversion_rows), default=0.0)
    conclusions = []
    infrastructure_report = sensor_reports["infrastructure_lidar"]
    vehicle_report = sensor_reports["vehicle_lidar"]
    conclusions.append(
        "Infrastructure 的 relative_error 修正使 20 帧平均 GT 对齐误差从 "
        f"{infrastructure_report['mean_raw_gt_world_center_error_bev']:.3f} m 降至 "
        f"{infrastructure_report['mean_corrected_gt_world_center_error_bev']:.3f} m，"
        f"平均 BEV IoU 从 {infrastructure_report['mean_raw_gt_world_bev_iou']:.3f} 提升至 {infrastructure_report['mean_corrected_gt_world_bev_iou']:.3f}；但仍有 "
        f"{infrastructure_report['max_corrected_gt_world_center_error_bev']:.3f} m 的离群帧。"
    )
    conclusions.append(
        "Vehicle 的大多数帧对齐接近数值精度，但仍有少量离群；20 帧最大误差为 "
        f"{vehicle_report['max_corrected_gt_world_center_error_bev']:.3f} m。"
    )
    if max_conversion <= 1e-4:
        conclusions.append(
            "local prediction 经当前脚本转换与现有 predictions_world.json 的中心结果一致；这证明文件内部转换实现一致，但不证明标定绝对正确。Infrastructure 的 relative_error 修正与现有 world prediction 最大相差约 "
            f"{sensor_reports['infrastructure_lidar']['max_corrected_prediction_center_residual_vs_existing_world_m']:.3f} m，说明现有 world 结果未应用该修正。"
        )
    else:
        conclusions.append(f"local prediction 与已保存 world prediction 的最大中心残差为 {max_conversion:.6f} m，应检查 prediction 导出/转换实现。")
    conclusions.append("local_eval 与 world_eval 的差异仍需结合对应 CSV 逐帧比较；若 GT 对齐且转换闭环成立，则低指标更可能来自模型预测质量、点云覆盖或配置，而非 world 转换。")
    summary = {
        "task": "Task 2: inspect raw LiDAR coordinates, local GT and raw predictions",
        "manifest": str(args.manifest.resolve()),
        "distance_match_threshold_m": 5.0,
        "sensor_reports": sensor_reports,
        "conclusions": conclusions,
        "raw_coordinate_alignment_max_bev_center_error_m": max_raw_alignment,
        "corrected_coordinate_alignment_max_bev_center_error_m": max_corrected_alignment,
        "prediction_conversion_max_center_residual_m": max_conversion,
        "interpretation_note": "Task 2 的 local_eval 使用与现有 evaluator 相同的 5 m BEV 中心匹配逻辑；它用于诊断，不替代官方 AP 评价。",
    }
    save_json(output_root / "lidar_diagnosis_task2_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def normalize_prediction_world(raw):
    center = raw.get("center_world")
    box = raw.get("box_world", {})
    if center is None:
        return None
    try:
        center = [float(value) for value in center]
        dx = float(box.get("dx", box.get("length")))
        dy = float(box.get("dy", box.get("width")))
        dz = float(box.get("dz", box.get("height")))
        heading = float(box.get("heading_world", box.get("yaw", 0.0)))
    except (TypeError, ValueError):
        return None
    result = normalize_prediction({"class": raw.get("class"), "score": raw.get("score", 0.0), "center_lidar": center, "box_lidar": {"dx": dx, "dy": dy, "dz": dz, "heading": heading}})
    if result is None:
        return None
    result["center_world"] = center
    if raw.get("corners_3d_world") is not None:
        result["box_world"]["corners_3d_world"] = raw["corners_3d_world"]
    elif box.get("corners_3d_world") is not None:
        result["box_world"]["corners_3d_world"] = box["corners_3d_world"]
    return result


if __name__ == "__main__":
    main()
