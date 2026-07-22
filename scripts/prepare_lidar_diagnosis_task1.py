#!/usr/bin/env python3
"""Freeze Full Dataset LiDAR baselines and build an auditable Task 1 diagnosis set."""

import argparse
import ast
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import random
import re
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(
    "/mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure"
)
DEFAULT_FULL_ROOT = PROJECT_ROOT / "outputs" / "full_baselines"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "diagnosis"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sample_key(value):
    value = str(value).strip()
    return str(int(value)) if value.isdigit() else value


def sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_metadata(path, hash_file=True):
    path = Path(path)
    result = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return result
    stat = path.stat()
    result.update({"size_bytes": stat.st_size, "modified_utc": dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat()})
    if hash_file:
        result["sha256"] = sha256(path)
    return result


def command_output(command):
    try:
        return subprocess.check_output(command, cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_snapshot():
    return {
        "commit": command_output(["git", "rev-parse", "HEAD"]),
        "branch": command_output(["git", "branch", "--show-current"]),
        "dirty": bool(command_output(["git", "status", "--porcelain"])),
    }


def flatten_numbers(value):
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(flatten_numbers(item))
        return result
    return []


def transform_from_calibration(path):
    raw = load_json(path)
    raw = raw.get("transform", raw)
    rotation = flatten_numbers(raw.get("rotation", []))
    translation = flatten_numbers(raw.get("translation", []))
    if len(rotation) != 9 or len(translation) != 3:
        raise ValueError(f"invalid calibration transform: {path}")
    return [
        [rotation[0], rotation[1], rotation[2], translation[0]],
        [rotation[3], rotation[4], rotation[5], translation[1]],
        [rotation[6], rotation[7], rotation[8], translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def matmul(left, right):
    return [
        [sum(left[row][index] * right[index][column] for index in range(4)) for column in range(4)]
        for row in range(4)
    ]


def transform_origin(transform):
    return [transform[0][3], transform[1][3], transform[2][3]]


def parse_world_gt_stats(label_path, origin):
    labels = load_json(label_path)
    distances = []
    raw_classes = Counter()
    for label in labels:
        raw_classes[str(label.get("type", label.get("class", "Unknown")))] += 1
        location = label.get("3d_location", label.get("location", label.get("center")))
        if not isinstance(location, dict):
            continue
        try:
            dx = float(location["x"]) - origin[0]
            dy = float(location["y"]) - origin[1]
        except (KeyError, TypeError, ValueError):
            continue
        distances.append((dx * dx + dy * dy) ** 0.5)
    distances.sort()
    return {
        "gt_count_from_label": len(labels),
        "gt_count_with_valid_center": len(distances),
        "gt_raw_class_distribution": dict(sorted(raw_classes.items())),
        "gt_nearest_range_m": distances[0] if distances else None,
        "gt_median_range_m": statistics.median(distances) if distances else None,
        "gt_farthest_range_m": distances[-1] if distances else None,
    }


def parse_local_class_stats(label_path):
    counts = Counter()
    for label in load_json(label_path):
        counts[str(label.get("type", label.get("class", "Unknown")))] += 1
    return dict(sorted(counts.items()))


def source_id_map(data_info_path):
    mapping = {}
    for item in load_json(data_info_path):
        pointcloud = item.get("pointcloud_path", "")
        sample_id = Path(pointcloud).stem
        if sample_id:
            mapping[sample_id] = item
    return mapping


def parse_config_summary(path, visited=None):
    path = Path(path).resolve()
    visited = set() if visited is None else set(visited)
    if path in visited:
        raise ValueError(f"cyclic _BASE_CONFIG_ reference: {path}")
    visited.add(path)
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()

    def yaml_value(key):
        for index, line in enumerate(lines):
            match = re.match(rf"^(\s*){re.escape(key)}:\s*(.*)$", line)
            if not match:
                continue
            inline = match.group(2).strip()
            if inline:
                if inline.startswith("[") or inline.startswith("{"):
                    try:
                        return ast.literal_eval(inline)
                    except (SyntaxError, ValueError):
                        pass
                return inline
            base_indent = len(match.group(1))
            values = []
            for child in lines[index + 1:]:
                stripped = child.strip()
                if not stripped:
                    continue
                child_indent = len(child) - len(child.lstrip())
                if child_indent < base_indent or (child_indent == base_indent and not stripped.startswith("- ")):
                    break
                if stripped.startswith("- "):
                    values.append(stripped[2:].strip())
            return values or None
        return None

    base_config = yaml_value("_BASE_CONFIG_")
    base_summary = None
    if isinstance(base_config, str):
        base_path = Path(base_config)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        if base_path.is_file():
            base_summary = parse_config_summary(base_path, visited)

    output_field = {
        "CLASS_NAMES": "class_names",
        "POINT_CLOUD_RANGE": "point_cloud_range",
        "VOXEL_SIZE": "voxel_size",
        "SCORE_THRESH": "score_threshold",
        "NMS_THRESH": "nms_threshold",
        "NMS_TYPE": "nms_type",
    }

    def resolved(key):
        own_value = yaml_value(key)
        if own_value is not None:
            return own_value, str(path)
        if base_summary and base_summary.get(output_field[key]) is not None:
            return base_summary[output_field[key]], base_summary["value_sources"].get(key)
        return None, None

    class_names, class_names_source = resolved("CLASS_NAMES")
    point_cloud_range, point_cloud_range_source = resolved("POINT_CLOUD_RANGE")
    voxel_size, voxel_size_source = resolved("VOXEL_SIZE")
    score_threshold, score_threshold_source = resolved("SCORE_THRESH")
    nms_threshold, nms_threshold_source = resolved("NMS_THRESH")
    nms_type, nms_type_source = resolved("NMS_TYPE")

    return {
        "config_path": str(path),
        "config_sha256": sha256(path),
        "base_config_path": base_summary["config_path"] if base_summary else None,
        "base_config_sha256": base_summary["config_sha256"] if base_summary else None,
        "class_names": class_names,
        "point_cloud_range": point_cloud_range,
        "voxel_size": voxel_size,
        "score_threshold": score_threshold,
        "nms_threshold": nms_threshold,
        "nms_type": nms_type,
        "value_sources": {
            "CLASS_NAMES": class_names_source,
            "POINT_CLOUD_RANGE": point_cloud_range_source,
            "VOXEL_SIZE": voxel_size_source,
            "SCORE_THRESH": score_threshold_source,
            "NMS_THRESH": nms_threshold_source,
            "NMS_TYPE": nms_type_source,
        },
    }


def checkpoint_from_log(path):
    if not Path(path).is_file():
        return None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"\bckpt\s+(.+)$", line)
        if match:
            return match.group(1).strip()
    return None


def load_prediction_samples(path):
    data = load_json(path)
    return data if isinstance(data, list) else data.get("samples", [])


def eval_row_map(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        return {sample_key(row["sample_id"]): row for row in csv.DictReader(file)}


def number(row, field, default=0.0):
    value = row.get(field)
    try:
        return float(value) if value not in (None, "") else default
    except ValueError:
        return default


def select_samples(candidates, target_count, seed):
    """Deterministic, stratified selection. A sample may have several reasons."""
    selected, reasons = [], {}

    def add(rows, label, count):
        for row in rows:
            sample_id = row["cooperative_sample_id"]
            reasons.setdefault(sample_id, []).append(label)
            if sample_id not in selected:
                selected.append(sample_id)
            if len([item for item in reasons if label in reasons[item]]) >= count:
                break

    def descending(field):
        return sorted(
            candidates,
            key=lambda row: (
                row.get(field) is None,
                -float(row[field]) if row.get(field) is not None else 0.0,
                row["cooperative_sample_id"],
            ),
        )

    def ascending(field):
        return sorted(
            candidates,
            key=lambda row: (
                row.get(field) is None,
                float(row[field]) if row.get(field) is not None else 0.0,
                row["cooperative_sample_id"],
            ),
        )

    rng = random.Random(seed)
    controls = list(candidates)
    rng.shuffle(controls)
    add(controls, "random_control", 3)
    class_frequency = Counter(
        class_name
        for row in candidates
        for class_name in row["diagnostic_classes"]
    )
    rare_score = {
        sample_id: sum(1.0 / class_frequency[class_name] for class_name in row["diagnostic_classes"])
        for sample_id, row in ((item["cooperative_sample_id"], item) for item in candidates)
    }
    add(
        sorted(
            candidates,
            key=lambda row: (-len(row["diagnostic_classes"]), -rare_score[row["cooperative_sample_id"]], rng.random(), row["cooperative_sample_id"]),
        ),
        "class_diversity",
        4,
    )
    for class_name in sorted(class_frequency, key=lambda value: (class_frequency[value], value)):
        add(
            sorted(
                [row for row in candidates if class_name in row["diagnostic_classes"]],
                key=lambda row: (-len(row["diagnostic_classes"]), -row["gt_count"], rng.random(), row["cooperative_sample_id"]),
            ),
            f"class_coverage:{class_name}",
            1,
        )
    add(descending("vehicle_relative_disadvantage"), "vehicle_worse_than_infrastructure", 3)
    add(descending("infrastructure_relative_disadvantage"), "infrastructure_worse_than_vehicle", 3)
    add(descending("max_false_negative_rate"), "high_false_negative_rate", 3)
    add(descending("max_false_positive_per_gt"), "high_false_positive_per_gt", 3)
    add(descending("shared_median_gt_range_m"), "far_range_scene", 2)
    add(ascending("gt_count"), "sparse_gt_scene", 1)
    add(descending("gt_count"), "dense_gt_scene", 1)
    add(descending("abs_time_delta_ms"), "large_cross_sensor_time_delta", 1)

    if len(selected) < target_count:
        for row in sorted(candidates, key=lambda item: item["cooperative_sample_id"]):
            sample_id = row["cooperative_sample_id"]
            if sample_id not in selected:
                selected.append(sample_id)
                reasons[sample_id] = ["deterministic_coverage"]
            if len(selected) >= target_count:
                break
    return selected[:target_count], reasons


def percentiles(values):
    values = sorted(values)
    if not values:
        return {"min": None, "median": None, "p95": None, "max": None}
    at = lambda fraction: values[min(len(values) - 1, round((len(values) - 1) * fraction))]
    return {"min": values[0], "median": statistics.median(values), "p95": at(0.95), "max": values[-1]}


def main():
    parser = argparse.ArgumentParser(description="Task 1: 冻结 Full Dataset 双 LiDAR baseline 并建立可审计诊断样本集")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--full-root", type=Path, default=DEFAULT_FULL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260718, help="类别分层和随机对照抽样种子；更换种子会生成另一组可复现样本。")
    parser.add_argument("--no-hash", action="store_true", help="仅用于快速预览；正式诊断快照应保留 SHA256。")
    args = parser.parse_args()
    if not 12 <= args.sample_count <= 50:
        raise ValueError("--sample-count 应在 12 到 50 之间")

    data_root = args.data_root.resolve()
    full_root = args.full_root.resolve()
    output_root = args.output_root.resolve()
    pair_manifest_path = full_root / "dataset_preparation" / "full_common_val_pair_manifest.json"
    vehicle_root, infrastructure_root = full_root / "vehicle_lidar", full_root / "infrastructure_lidar"
    output_root.mkdir(parents=True, exist_ok=True)

    required = [
        pair_manifest_path,
        vehicle_root / "vehicle_lidar_baseline_index.json",
        infrastructure_root / "infrastructure_lidar_baseline_index.json",
        vehicle_root / "eval.csv", infrastructure_root / "eval.csv",
        vehicle_root / "predictions_vehicle_lidar.json",
        infrastructure_root / "predictions_infrastructure_lidar.json",
        data_root / "cooperative/data_info.json",
        data_root / "vehicle-side/data_info.json",
        data_root / "infrastructure-side/data_info.json",
    ]
    missing_required = [str(path) for path in required if not path.is_file()]
    if missing_required:
        raise FileNotFoundError("Task 1 缺少输入文件:\n" + "\n".join(missing_required))

    pairs = load_json(pair_manifest_path)
    vehicle_index = {sample_key(item["sample_id"]): item for item in load_json(vehicle_root / "vehicle_lidar_baseline_index.json")}
    infrastructure_index = {sample_key(item["sample_id"]): item for item in load_json(infrastructure_root / "infrastructure_lidar_baseline_index.json")}
    vehicle_eval = eval_row_map(vehicle_root / "eval.csv")
    infrastructure_eval = eval_row_map(infrastructure_root / "eval.csv")
    vehicle_predictions = {sample_key(item["sample_id"]): item for item in load_prediction_samples(vehicle_root / "predictions_vehicle_lidar.json")}
    infrastructure_predictions = {sample_key(item["sample_id"]): item for item in load_prediction_samples(infrastructure_root / "predictions_infrastructure_lidar.json")}
    cooperative = load_json(data_root / "cooperative/data_info.json")
    vehicle_info = source_id_map(data_root / "vehicle-side/data_info.json")
    infrastructure_info = source_id_map(data_root / "infrastructure-side/data_info.json")

    candidates, invalid_samples, time_deltas = [], [], []
    seen_vehicle, seen_infrastructure = Counter(), Counter()
    for pair in pairs:
        vehicle_id = str(pair["vehicle_id"])
        infrastructure_id = str(pair["infrastructure_id"])
        key = sample_key(vehicle_id)
        seen_vehicle[vehicle_id] += 1
        seen_infrastructure[infrastructure_id] += 1
        vehicle_item = vehicle_index.get(key)
        infrastructure_item = infrastructure_index.get(key)
        coop_item = cooperative[pair["sample_index"]] if isinstance(pair.get("sample_index"), int) and pair["sample_index"] < len(cooperative) else None
        validation_errors = []
        if vehicle_item is None or infrastructure_item is None:
            validation_errors.append("missing baseline index entry")
        if key not in vehicle_eval or key not in infrastructure_eval:
            validation_errors.append("missing evaluation row")
        if key not in vehicle_predictions or key not in infrastructure_predictions:
            validation_errors.append("missing prediction sample")
        if coop_item is None:
            validation_errors.append("invalid cooperative sample_index")
        elif Path(coop_item.get("vehicle_pointcloud_path", "")).stem != vehicle_id or Path(coop_item.get("infrastructure_pointcloud_path", "")).stem != infrastructure_id:
            validation_errors.append("pair manifest does not match cooperative/data_info")
        if validation_errors:
            invalid_samples.append({"cooperative_sample_id": vehicle_id, "infrastructure_frame_id": infrastructure_id, "errors": validation_errors})
            continue

        vehicle_meta = vehicle_info.get(vehicle_id, {})
        infrastructure_meta = infrastructure_info.get(infrastructure_id, {})
        file_paths = {
            "vehicle_pointcloud_path": vehicle_item["vehicle_pointcloud"],
            "infrastructure_pointcloud_path": infrastructure_item["infrastructure_pointcloud_path"],
            "vehicle_local_label_path": str(data_root / "vehicle-side/label/lidar" / f"{vehicle_id}.json"),
            "infrastructure_local_label_path": infrastructure_item["infrastructure_label_path"],
            "cooperative_label_world_path": vehicle_item["cooperative_label_world"],
            "vehicle_lidar_to_novatel_path": vehicle_item["vehicle_calib"]["lidar_to_novatel"],
            "vehicle_novatel_to_world_path": vehicle_item["vehicle_calib"]["novatel_to_world"],
            "infrastructure_virtuallidar_to_world_path": infrastructure_item["infrastructure_calib_path"],
        }
        missing_files = [name for name, path in file_paths.items() if not Path(path).is_file()]
        if missing_files:
            invalid_samples.append({"cooperative_sample_id": vehicle_id, "infrastructure_frame_id": infrastructure_id, "errors": ["missing referenced files"], "missing_files": missing_files})
            continue
        try:
            vehicle_transform = matmul(
                transform_from_calibration(file_paths["vehicle_novatel_to_world_path"]),
                transform_from_calibration(file_paths["vehicle_lidar_to_novatel_path"]),
            )
            infrastructure_transform = transform_from_calibration(file_paths["infrastructure_virtuallidar_to_world_path"])
            vehicle_gt = parse_world_gt_stats(file_paths["cooperative_label_world_path"], transform_origin(vehicle_transform))
            infrastructure_gt = parse_world_gt_stats(file_paths["cooperative_label_world_path"], transform_origin(infrastructure_transform))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            invalid_samples.append({"cooperative_sample_id": vehicle_id, "infrastructure_frame_id": infrastructure_id, "errors": [f"calibration or GT parse error: {error}"]})
            continue

        vehicle_timestamp = vehicle_meta.get("pointcloud_timestamp")
        infrastructure_timestamp = infrastructure_meta.get("pointcloud_timestamp")
        try:
            time_delta_ms = (int(infrastructure_timestamp) - int(vehicle_timestamp)) / 1000.0
        except (TypeError, ValueError):
            time_delta_ms = None
        if time_delta_ms is not None:
            time_deltas.append(abs(time_delta_ms))

        vehicle_metrics, infrastructure_metrics = vehicle_eval[key], infrastructure_eval[key]
        vehicle_local_class_distribution = parse_local_class_stats(file_paths["vehicle_local_label_path"])
        infrastructure_local_class_distribution = parse_local_class_stats(file_paths["infrastructure_local_label_path"])
        diagnostic_classes = sorted(set(vehicle_local_class_distribution) | set(infrastructure_local_class_distribution))
        gt_count = int(number(vehicle_metrics, "num_gt"))
        vehicle_f1, infrastructure_f1 = number(vehicle_metrics, "f1"), number(infrastructure_metrics, "f1")
        vehicle_fn_rate = number(vehicle_metrics, "false_negative") / gt_count if gt_count else None
        infrastructure_fn_rate = number(infrastructure_metrics, "false_negative") / gt_count if gt_count else None
        vehicle_fp_per_gt = number(vehicle_metrics, "false_positive") / gt_count if gt_count else None
        infrastructure_fp_per_gt = number(infrastructure_metrics, "false_positive") / gt_count if gt_count else None
        median_ranges = [value for value in (vehicle_gt["gt_median_range_m"], infrastructure_gt["gt_median_range_m"]) if value is not None]

        candidates.append({
            "cooperative_sample_id": vehicle_id,
            "vehicle_frame_id": vehicle_id,
            "infrastructure_frame_id": infrastructure_id,
            "pair_manifest_sample_index": pair["sample_index"],
            "vehicle_pointcloud_path": file_paths["vehicle_pointcloud_path"],
            "infrastructure_pointcloud_path": file_paths["infrastructure_pointcloud_path"],
            "vehicle_local_label_path": file_paths["vehicle_local_label_path"],
            "infrastructure_local_label_path": file_paths["infrastructure_local_label_path"],
            "cooperative_label_world_path": file_paths["cooperative_label_world_path"],
            "vehicle_calib_paths": [file_paths["vehicle_lidar_to_novatel_path"], file_paths["vehicle_novatel_to_world_path"]],
            "infrastructure_calib_paths": [file_paths["infrastructure_virtuallidar_to_world_path"]],
            "vehicle_predictions_json_path": str(vehicle_root / "predictions_vehicle_lidar.json"),
            "vehicle_prediction_sample_key": str(vehicle_predictions[key]["sample_id"]),
            "infrastructure_predictions_json_path": str(infrastructure_root / "predictions_infrastructure_lidar.json"),
            "infrastructure_prediction_sample_key": str(infrastructure_predictions[key]["sample_id"]),
            "vehicle_pointcloud_timestamp": vehicle_timestamp,
            "infrastructure_pointcloud_timestamp": infrastructure_timestamp,
            "time_delta_ms_infrastructure_minus_vehicle": time_delta_ms,
            "abs_time_delta_ms": abs(time_delta_ms) if time_delta_ms is not None else None,
            "gt_count": gt_count,
            "vehicle_local_class_distribution": vehicle_local_class_distribution,
            "infrastructure_local_class_distribution": infrastructure_local_class_distribution,
            "diagnostic_classes": diagnostic_classes,
            "vehicle_gt_median_range_m": vehicle_gt["gt_median_range_m"],
            "infrastructure_gt_median_range_m": infrastructure_gt["gt_median_range_m"],
            "shared_median_gt_range_m": min(median_ranges) if len(median_ranges) == 2 else None,
            "vehicle_f1": vehicle_f1,
            "infrastructure_f1": infrastructure_f1,
            "vehicle_relative_disadvantage": infrastructure_f1 - vehicle_f1,
            "infrastructure_relative_disadvantage": vehicle_f1 - infrastructure_f1,
            "vehicle_false_negative_rate": vehicle_fn_rate,
            "infrastructure_false_negative_rate": infrastructure_fn_rate,
            "max_false_negative_rate": max(item for item in (vehicle_fn_rate, infrastructure_fn_rate) if item is not None),
            "vehicle_false_positive_per_gt": vehicle_fp_per_gt,
            "infrastructure_false_positive_per_gt": infrastructure_fp_per_gt,
            "max_false_positive_per_gt": max(item for item in (vehicle_fp_per_gt, infrastructure_fp_per_gt) if item is not None),
            "vehicle_mean_loc_error_bev": number(vehicle_metrics, "mean_loc_error_bev", None),
            "vehicle_mean_loc_error_3d": number(vehicle_metrics, "mean_loc_error_3d", None),
            "infrastructure_mean_loc_error_bev": number(infrastructure_metrics, "mean_loc_error_bev", None),
            "infrastructure_mean_loc_error_3d": number(infrastructure_metrics, "mean_loc_error_3d", None),
            "file_validation": {"all_required_files_exist": True, "missing_files": []},
            "pairing_validation": {"status": "valid", "note": "vehicle_frame_id and infrastructure_frame_id are intentionally different IDs; pairing is validated through cooperative/data_info and the fixed pair manifest."},
            "vehicle_gt_range_stats": vehicle_gt,
            "infrastructure_gt_range_stats": infrastructure_gt,
        })

    selected_ids, selection_reasons = select_samples(candidates, args.sample_count, args.seed)
    selected = []
    candidate_by_id = {row["cooperative_sample_id"]: row for row in candidates}
    for rank, sample_id in enumerate(selected_ids, start=1):
        record = dict(candidate_by_id[sample_id])
        record["selection_rank"] = rank
        record["selection_reasons"] = selection_reasons[sample_id]
        selected.append(record)

    baselines = {}
    for name, root, prediction_file in (
        ("vehicle_lidar", vehicle_root, "predictions_vehicle_lidar.json"),
        ("infrastructure_lidar", infrastructure_root, "predictions_infrastructure_lidar.json"),
    ):
        config_path = next((root / "runtime_configs").glob("*.yaml"), None)
        inference_log = root / "inference.log"
        prediction_summary = load_json(root / prediction_file.replace(".json", "_summary.json"))
        prediction_samples = load_prediction_samples(root / prediction_file)
        checkpoint = checkpoint_from_log(inference_log)
        result_pkl = (
            prediction_summary.get("input_result_pkl")
            or prediction_summary.get("input")
            or (prediction_samples[0].get("source_file") if prediction_samples else None)
        )
        prediction_type = (
            prediction_summary.get("prediction_type")
            or (prediction_summary.get("prediction_types") or [None])[0]
            or (prediction_samples[0].get("prediction_type") if prediction_samples else None)
        )
        artifacts = {
            "runtime_config": file_metadata(config_path, not args.no_hash) if config_path else None,
            "inference_log": file_metadata(inference_log, not args.no_hash),
            "checkpoint": file_metadata(checkpoint, not args.no_hash) if checkpoint else None,
            "result_pkl": file_metadata(result_pkl, not args.no_hash) if result_pkl else None,
            "predictions_local": file_metadata(root / prediction_file, not args.no_hash),
            "predictions_world": file_metadata(root / "predictions_world.json", not args.no_hash),
            "prediction_summary": file_metadata(root / prediction_file.replace(".json", "_summary.json"), not args.no_hash),
            "world_summary": file_metadata(root / "predictions_world_summary.json", not args.no_hash),
            "eval_csv": file_metadata(root / "eval.csv", not args.no_hash),
            "eval_summary": file_metadata(root / "eval_summary.json", not args.no_hash),
            "ap_summary": file_metadata(root / "ap_summary.json", not args.no_hash),
        }
        baselines[name] = {
            "prediction_type": prediction_type,
            "prediction_summary": prediction_summary,
            "runtime_config": parse_config_summary(config_path) if config_path else None,
            "checkpoint_path_from_inference_log": checkpoint,
            "artifacts": artifacts,
        }

    snapshot = {
        "task": "Task 1: freeze LiDAR baselines and establish an auditable diagnosis sample set",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "data_root": str(data_root),
        "full_baselines_root": str(full_root),
        "pair_manifest": file_metadata(pair_manifest_path, not args.no_hash),
        "git": git_snapshot(),
        "runtime": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "conda_environment": os.environ.get("CONDA_DEFAULT_ENV")},
        "evaluation_protocol": {
            "split": "full_common_val",
            "center_match_distance_threshold_meter": 5.0,
            "ap_iou_thresholds": [0.25, 0.50, 0.70],
            "note": "NMS IoU, AP IoU and 5 m center matching are distinct controls and are recorded separately.",
        },
        "baselines": baselines,
    }
    save_json(output_root / "baseline_snapshot.json", snapshot)

    manifest = {
        "task": snapshot["task"],
        "selection_policy": {
            "target_sample_count": args.sample_count,
            "random_seed": args.seed,
            "strata": ["random_control", "class_diversity", "class_coverage:<raw_class>", "vehicle_worse_than_infrastructure", "infrastructure_worse_than_vehicle", "high_false_negative_rate", "high_false_positive_per_gt", "far_range_scene", "sparse_gt_scene", "dense_gt_scene", "large_cross_sensor_time_delta"],
            "selection_is_deterministic": True,
        },
        "validation": {
            "num_pair_manifest_records": len(pairs),
            "num_valid_candidates": len(candidates),
            "num_invalid_samples": len(invalid_samples),
            "invalid_samples": invalid_samples,
            "duplicate_vehicle_ids": [key for key, count in seen_vehicle.items() if count > 1],
            "duplicate_infrastructure_ids": [key for key, count in seen_infrastructure.items() if count > 1],
            "abs_cross_sensor_time_delta_ms": percentiles(time_deltas),
            "pairing_note": "Different vehicle and infrastructure frame IDs are expected. Validity is determined by the fixed pair manifest and cooperative/data_info path consistency, not by numeric ID equality.",
        },
        "selected_samples": selected,
        "class_distribution_scope": "diagnostic_classes are the raw local-label classes from both sensors; they are not the restricted Car/Pedestrian/Cyclist AP taxonomy.",
    }
    save_json(output_root / "lidar_diagnosis_manifest.json", manifest)

    csv_columns = [
        "selection_rank", "selection_reasons", "cooperative_sample_id", "vehicle_frame_id", "infrastructure_frame_id",
        "pair_manifest_sample_index", "vehicle_pointcloud_timestamp", "infrastructure_pointcloud_timestamp",
        "time_delta_ms_infrastructure_minus_vehicle", "gt_count", "vehicle_gt_median_range_m",
        "infrastructure_gt_median_range_m", "shared_median_gt_range_m", "vehicle_f1", "infrastructure_f1",
        "vehicle_false_negative_rate", "infrastructure_false_negative_rate", "vehicle_false_positive_per_gt",
        "infrastructure_false_positive_per_gt", "vehicle_mean_loc_error_bev", "vehicle_mean_loc_error_3d",
        "infrastructure_mean_loc_error_bev", "infrastructure_mean_loc_error_3d", "vehicle_pointcloud_path",
        "infrastructure_pointcloud_path", "vehicle_local_label_path", "infrastructure_local_label_path",
        "cooperative_label_world_path", "vehicle_predictions_json_path", "vehicle_prediction_sample_key",
        "infrastructure_predictions_json_path", "infrastructure_prediction_sample_key",
        "diagnostic_classes", "vehicle_local_class_distribution", "infrastructure_local_class_distribution",
    ]
    with (output_root / "lidar_sample_pairing.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        for item in selected:
            row = dict(item)
            row["selection_reasons"] = ";".join(item["selection_reasons"])
            writer.writerow(row)

    print(json.dumps({
        "baseline_snapshot": str(output_root / "baseline_snapshot.json"),
        "diagnosis_manifest": str(output_root / "lidar_diagnosis_manifest.json"),
        "sample_pairing_csv": str(output_root / "lidar_sample_pairing.csv"),
        "total_pairs": len(pairs),
        "valid_candidates": len(candidates),
        "invalid_samples": len(invalid_samples),
        "selected_samples": len(selected),
        "abs_cross_sensor_time_delta_ms": percentiles(time_deltas),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
