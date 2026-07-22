#!/usr/bin/env python3
"""Validate DAIR-V2X-C Full Dataset files and data_info path references."""

import argparse
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(
    "/mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "full_baselines" / "full_dataset_readiness_summary.json"
MAX_EXAMPLES = 20


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def summarize_directory(path):
    path = Path(path)
    if not path.is_dir():
        return {"exists": False, "file_count": 0, "empty_file_count": 0, "examples": []}
    files = [item for item in path.rglob("*") if item.is_file()]
    empty = [item for item in files if item.stat().st_size == 0]
    return {
        "exists": True,
        "is_symlink": path.is_symlink(),
        "resolved_path": str(path.resolve()),
        "file_count": len(files),
        "empty_file_count": len(empty),
        "empty_examples": [str(item) for item in empty[:MAX_EXAMPLES]],
    }


def validate_path_references(data_root, side, entries, fields):
    missing, empty, invalid = [], [], []
    for index, entry in enumerate(entries):
        for field in fields:
            relative = entry.get(field)
            if not relative:
                invalid.append({"index": index, "field": field, "reason": "missing field"})
                continue
            path = data_root / side / relative
            if not path.is_file():
                missing.append({"index": index, "field": field, "path": str(path)})
            elif path.stat().st_size == 0:
                empty.append({"index": index, "field": field, "path": str(path)})
    return {
        "entries": len(entries),
        "fields": fields,
        "missing_reference_count": len(missing),
        "empty_reference_count": len(empty),
        "invalid_reference_count": len(invalid),
        "missing_reference_examples": missing[:MAX_EXAMPLES],
        "empty_reference_examples": empty[:MAX_EXAMPLES],
        "invalid_reference_examples": invalid[:MAX_EXAMPLES],
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="检查 DAIR-V2X-C Full Dataset 是否可用于四类单模态 baseline")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data_root = args.data_root.resolve()

    required_dirs = {
        "vehicle_image": data_root / "vehicle-side/image",
        "vehicle_velodyne": data_root / "vehicle-side/velodyne",
        "vehicle_calib": data_root / "vehicle-side/calib",
        "vehicle_label": data_root / "vehicle-side/label",
        "infrastructure_image": data_root / "infrastructure-side/image",
        "infrastructure_velodyne": data_root / "infrastructure-side/velodyne",
        "infrastructure_calib": data_root / "infrastructure-side/calib",
        "infrastructure_label": data_root / "infrastructure-side/label",
        "cooperative_label_world": data_root / "cooperative/label_world",
    }
    data_info_paths = {
        "vehicle": data_root / "vehicle-side/data_info.json",
        "infrastructure": data_root / "infrastructure-side/data_info.json",
        "cooperative": data_root / "cooperative/data_info.json",
    }
    missing_data_info = [name for name, path in data_info_paths.items() if not path.is_file()]
    data_infos = {name: load_json(path) for name, path in data_info_paths.items() if path.is_file()}

    references = {}
    if "vehicle" in data_infos:
        references["vehicle"] = validate_path_references(
            data_root, "vehicle-side", data_infos["vehicle"], [
                "image_path", "pointcloud_path", "calib_novatel_to_world_path",
                "calib_lidar_to_novatel_path", "calib_lidar_to_camera_path",
                "calib_camera_intrinsic_path", "label_lidar_std_path", "label_camera_std_path",
            ]
        )
    if "infrastructure" in data_infos:
        references["infrastructure"] = validate_path_references(
            data_root, "infrastructure-side", data_infos["infrastructure"], [
                "image_path", "pointcloud_path", "calib_camera_intrinsic_path",
                "calib_virtuallidar_to_world_path", "calib_virtuallidar_to_camera_path",
                "label_lidar_std_path", "label_camera_std_path",
            ]
        )
    if "cooperative" in data_infos:
        references["cooperative"] = validate_path_references(
            data_root, "", data_infos["cooperative"], [
                "vehicle_image_path", "vehicle_pointcloud_path", "infrastructure_image_path",
                "infrastructure_pointcloud_path", "cooperative_label_path",
            ]
        )

    vehicle_ids = [Path(item.get("vehicle_pointcloud_path", "")).stem for item in data_infos.get("cooperative", [])]
    infrastructure_ids = [Path(item.get("infrastructure_pointcloud_path", "")).stem for item in data_infos.get("cooperative", [])]
    duplicate_vehicle_ids = [key for key, count in Counter(vehicle_ids).items() if key and count > 1]
    duplicate_infrastructure_ids = [key for key, count in Counter(infrastructure_ids).items() if key and count > 1]
    directory_summary = {name: summarize_directory(path) for name, path in required_dirs.items()}
    reference_error_count = sum(
        section["missing_reference_count"] + section["empty_reference_count"] + section["invalid_reference_count"]
        for section in references.values()
    )
    summary = {
        "data_root": str(data_root),
        "data_info_paths": {name: str(path) for name, path in data_info_paths.items()},
        "missing_data_info": missing_data_info,
        "directories": directory_summary,
        "data_info_entry_counts": {name: len(entries) for name, entries in data_infos.items()},
        "path_references": references,
        "cooperative_unique_vehicle_ids": len(set(filter(None, vehicle_ids))),
        "cooperative_unique_infrastructure_ids": len(set(filter(None, infrastructure_ids))),
        "duplicate_vehicle_id_examples": duplicate_vehicle_ids[:MAX_EXAMPLES],
        "duplicate_infrastructure_id_examples": duplicate_infrastructure_ids[:MAX_EXAMPLES],
        "ready": not missing_data_info and all(item["exists"] for item in directory_summary.values()) and reference_error_count == 0,
        "reference_error_count": reference_error_count,
    }
    write_json(args.output.resolve(), summary)
    print(json.dumps({
        "data_root": summary["data_root"],
        "ready": summary["ready"],
        "data_info_entry_counts": summary["data_info_entry_counts"],
        "reference_error_count": reference_error_count,
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
