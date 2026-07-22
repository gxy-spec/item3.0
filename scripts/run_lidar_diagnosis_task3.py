#!/usr/bin/env python3
"""Task 3: validate LiDAR-to-world transform structure and geometric round trips."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prepare_lidar_diagnosis_task1 import load_json, transform_from_calibration, matmul


DEFAULT_DATA_ROOT = Path("/mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure")
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs/diagnosis/lidar_diagnosis_manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/diagnosis/coordinate_transform_task3_summary.json"


def finite_numbers(value):
    values = np.asarray(value, dtype=float)
    return bool(np.isfinite(values).all())


def transform_with_relative_error(path, apply_relative_error=False):
    raw = load_json(path)
    transform = np.asarray(transform_from_calibration(path), dtype=float)
    relative_error = raw.get("relative_error", {}) if isinstance(raw, dict) else {}
    delta_x = float(relative_error.get("delta_x", 0.0)) if isinstance(relative_error, dict) else 0.0
    delta_y = float(relative_error.get("delta_y", 0.0)) if isinstance(relative_error, dict) else 0.0
    if apply_relative_error:
        transform[:3, 3] += [delta_x, delta_y, 0.0]
    return transform, {"delta_x": delta_x, "delta_y": delta_y, "applied": apply_relative_error}


def inverse_transform(transform):
    return np.linalg.inv(transform)


def transform_points(points, transform):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (homogeneous @ transform.T)[:, :3]


def rotation_report(transform):
    rotation = transform[:3, :3]
    return {
        "finite": finite_numbers(transform),
        "shape": list(transform.shape),
        "det_R": float(np.linalg.det(rotation)),
        "orthogonality_error": float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))),
        "translation_norm_m": float(np.linalg.norm(transform[:3, 3])),
        "bottom_row_is_homogeneous": bool(np.allclose(transform[3], [0, 0, 0, 1], atol=1e-9)),
        "valid_rigid_transform": bool(
            finite_numbers(transform)
            and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            and abs(np.linalg.det(rotation) - 1.0) < 1e-6
            and np.allclose(transform[3], [0, 0, 0, 1], atol=1e-9)
        ),
    }


def corners(center, dx, dy, dz, heading):
    x, y, z = [float(v) for v in center]
    values = np.array([
        [dx / 2, dy / 2, -dz / 2], [dx / 2, -dy / 2, -dz / 2],
        [-dx / 2, -dy / 2, -dz / 2], [-dx / 2, dy / 2, -dz / 2],
        [dx / 2, dy / 2, dz / 2], [dx / 2, -dy / 2, dz / 2],
        [-dx / 2, -dy / 2, dz / 2], [-dx / 2, dy / 2, dz / 2],
    ])
    c, s = math.cos(heading), math.sin(heading)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    return values @ rotation.T + [x, y, z]


def parse_box(raw):
    location = raw.get("3d_location", raw.get("location", raw.get("center")))
    dims = raw.get("3d_dimensions", raw.get("dimensions", raw.get("size")))
    if not isinstance(location, dict) or not isinstance(dims, dict):
        return None
    try:
        center = [float(location[axis]) for axis in ("x", "y", "z")]
        dx = float(dims.get("l", dims.get("length", dims.get("dx"))))
        dy = float(dims.get("w", dims.get("width", dims.get("dy"))))
        dz = float(dims.get("h", dims.get("height", dims.get("dz"))))
        heading = raw.get("rotation", raw.get("rotation_y", raw.get("yaw", 0.0)))
        if isinstance(heading, dict):
            heading = heading.get("z", heading.get("yaw", 0.0))
        heading = float(heading)
    except (KeyError, TypeError, ValueError):
        return None
    if min(dx, dy, dz) <= 0 or not finite_numbers(center + [dx, dy, dz, heading]):
        return None
    return {"center": center, "dims": [dx, dy, dz], "heading": heading, "corners": corners(center, dx, dy, dz, heading)}


def parse_boxes(path):
    boxes = []
    for raw in load_json(path):
        parsed = parse_box(raw)
        if parsed:
            boxes.append(parsed)
    return boxes


def nearest_center_errors(source_boxes, target_boxes, transform, threshold=5.0):
    if not source_boxes or not target_boxes:
        return []
    source_centers = transform_points([box["center"] for box in source_boxes], transform)
    target_centers = np.asarray([box["center"] for box in target_boxes], dtype=float)
    distances = []
    used = set()
    for center in source_centers:
        candidates = sorted(
            (float(np.linalg.norm(target_centers[index] - center)), index)
            for index in range(len(target_centers))
            if index not in used
        )
        if candidates and candidates[0][0] <= threshold:
            distance, index = candidates[0]
            used.add(index)
            distances.append(distance)
    return distances


def validate_sample(sample, data_root):
    vehicle_id = sample["vehicle_frame_id"]
    infrastructure_id = sample["infrastructure_frame_id"]
    vehicle_calibs = sample["vehicle_calib_paths"]
    infra_calib = sample["infrastructure_calib_paths"][0]
    vehicle_first = np.asarray(transform_from_calibration(vehicle_calibs[0]), dtype=float)
    vehicle_second = np.asarray(transform_from_calibration(vehicle_calibs[1]), dtype=float)
    vehicle_transform = vehicle_second @ vehicle_first
    infra_transform, relative_error = transform_with_relative_error(infra_calib, apply_relative_error=True)

    local_gt = {
        "vehicle": parse_boxes(sample["vehicle_local_label_path"]),
        "infrastructure": parse_boxes(sample["infrastructure_local_label_path"]),
    }
    world_gt = parse_boxes(sample["cooperative_label_world_path"])
    result = {"sample_id": sample["cooperative_sample_id"], "vehicle_id": vehicle_id, "infrastructure_id": infrastructure_id,
              "vehicle": {}, "infrastructure": {"relative_error": relative_error}}
    for side, transform in (("vehicle", vehicle_transform), ("infrastructure", infra_transform)):
        inverse = inverse_transform(transform)
        sample_points = np.array([[0, 0, 0], [1, 2, 3], [-5, 4, 1]], dtype=float)
        roundtrip_points = transform_points(transform_points(sample_points, transform), inverse)
        point_errors = np.linalg.norm(roundtrip_points - sample_points, axis=1)
        centers = np.array([box["center"] for box in local_gt[side]], dtype=float)
        if len(centers):
            center_roundtrip = np.linalg.norm(transform_points(transform_points(centers, transform), inverse) - centers, axis=1)
        else:
            center_roundtrip = np.array([])
        corner_errors = []
        dimension_errors = []
        for box in local_gt[side]:
            restored = transform_points(transform_points(box["corners"], transform), inverse)
            corner_errors.append(float(np.max(np.linalg.norm(restored - box["corners"], axis=1))))
            dimension_errors.append(float(np.max(np.abs(np.ptp(restored, axis=0) - np.ptp(box["corners"], axis=0)))))
        alignment = nearest_center_errors(local_gt[side], world_gt, transform)
        result[side].update({
            "matrix": rotation_report(transform),
            "point_roundtrip_max_m": float(np.max(point_errors)),
            "center_roundtrip_max_m": float(np.max(center_roundtrip)) if len(center_roundtrip) else 0.0,
            "corner_roundtrip_max_m": float(np.max(corner_errors)) if corner_errors else 0.0,
            "box_extent_roundtrip_max_m": float(np.max(dimension_errors)) if dimension_errors else 0.0,
            "local_gt_count": len(local_gt[side]),
            "world_gt_count": len(world_gt),
            "local_gt_to_world_matched_center_mean_m": float(np.mean(alignment)) if alignment else None,
            "local_gt_to_world_matched_center_max_m": float(np.max(alignment)) if alignment else None,
        })
    return result


def main():
    parser = argparse.ArgumentParser(description="验证 DAIR-V2X LiDAR 到 world 坐标转换链")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    manifest = load_json(args.manifest)
    samples = manifest.get("selected_samples", [])
    reports = [validate_sample(sample, Path(args.data_root)) for sample in samples]
    sensors = {}
    for side in ("vehicle", "infrastructure"):
        rows = [report[side] for report in reports]
        sensors[side] = {
            "num_samples": len(rows),
            "all_matrices_valid": all(row["matrix"]["valid_rigid_transform"] for row in rows),
            "max_point_roundtrip_error_m": max(row["point_roundtrip_max_m"] for row in rows),
            "max_center_roundtrip_error_m": max(row["center_roundtrip_max_m"] for row in rows),
            "max_corner_roundtrip_error_m": max(row["corner_roundtrip_max_m"] for row in rows),
            "max_box_extent_roundtrip_error_m": max(row["box_extent_roundtrip_max_m"] for row in rows),
            "mean_local_gt_to_world_matched_center_error_m": float(np.mean([row["local_gt_to_world_matched_center_mean_m"] for row in rows if row["local_gt_to_world_matched_center_mean_m"] is not None])),
            "max_local_gt_to_world_matched_center_error_m": max(row["local_gt_to_world_matched_center_max_m"] for row in rows if row["local_gt_to_world_matched_center_max_m"] is not None),
        }
    summary = {"task": "Task 3: validate LiDAR to world coordinate transforms", "manifest": str(Path(args.manifest).resolve()),
               "num_samples": len(reports), "sensors": sensors, "sample_reports": reports,
               "interpretation": {"roundtrip_threshold_m": 1e-9, "alignment_is_not_roundtrip": True,
                                   "note": "roundtrip verifies numerical/inverse consistency; local_gt_to_world verifies absolute alignment against cooperative label_world."}}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "num_samples": len(reports), "sensors": sensors}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
