#!/usr/bin/env python3
"""Build Full Dataset val indexes compatible with existing LiDAR export scripts."""

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(
    "/mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure"
)
DEFAULT_PAIR_MANIFEST = PROJECT_ROOT / "outputs" / "full_baselines" / "dataset_preparation" / "full_train_val_pair_manifest.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "full_baselines"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def exists(path):
    return Path(path).is_file()


def main():
    parser = argparse.ArgumentParser(description="构建 Full Dataset 的 Vehicle/Infrastructure LiDAR val 索引")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pair-manifest", type=Path, default=DEFAULT_PAIR_MANIFEST)
    parser.add_argument("--split", default="val", help="pair manifest 的 split 名，例如 val 或 full_common_val")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    data_root, output_root = args.data_root.resolve(), args.output_root.resolve()
    pairs = [record for record in load_json(args.pair_manifest) if record.get("split") == args.split]
    if not pairs:
        raise ValueError(f"pair manifest 中没有 {args.split} 样本")

    vehicle_records, infrastructure_records, invalid = [], [], []
    for record in pairs:
        vehicle_id, infrastructure_id = record["vehicle_id"], record["infrastructure_id"]
        vehicle_paths = {
            "vehicle_pointcloud": data_root / "vehicle-side/velodyne" / f"{vehicle_id}.pcd",
            "cooperative_label_world": data_root / "cooperative/label_world" / f"{vehicle_id}.json",
            "lidar_to_novatel": data_root / "vehicle-side/calib/lidar_to_novatel" / f"{vehicle_id}.json",
            "novatel_to_world": data_root / "vehicle-side/calib/novatel_to_world" / f"{vehicle_id}.json",
        }
        infra_paths = {
            "infrastructure_pointcloud_path": data_root / "infrastructure-side/velodyne" / f"{infrastructure_id}.pcd",
            "infrastructure_label_path": data_root / "infrastructure-side/label/virtuallidar" / f"{infrastructure_id}.json",
            "infrastructure_calib_path": data_root / "infrastructure-side/calib/virtuallidar_to_world" / f"{infrastructure_id}.json",
            "cooperative_label_world_path": vehicle_paths["cooperative_label_world"],
        }
        missing = [str(path) for path in [*vehicle_paths.values(), *infra_paths.values()] if not exists(path)]
        if missing:
            invalid.append({"sample_id": vehicle_id, "infrastructure_id": infrastructure_id, "missing_files": missing})
            continue
        vehicle_records.append({
            "sample_id": vehicle_id, "vehicle_id": vehicle_id, "infrastructure_id": infrastructure_id,
            "split": args.split, "vehicle_pointcloud": str(vehicle_paths["vehicle_pointcloud"]),
            "cooperative_label_world": str(vehicle_paths["cooperative_label_world"]),
            "vehicle_calib": {key: str(value) for key, value in vehicle_paths.items() if key not in {"vehicle_pointcloud", "cooperative_label_world"}},
            "all_required_files_exist": True,
        })
        infrastructure_records.append({
            "sample_id": vehicle_id, "vehicle_id": vehicle_id, "infrastructure_id": infrastructure_id,
            "split": args.split, **{key: str(value) for key, value in infra_paths.items()},
            "all_required_files_exist": True, "missing_files": {},
        })
    if invalid:
        save_json(output_root / "dataset_preparation" / f"full_{args.split}_lidar_index_invalid_samples.json", invalid)
        raise RuntimeError(f"{len(invalid)} 个 Full Dataset 样本缺少 LiDAR 所需文件，未写入不完整索引")
    vehicle_path = output_root / "vehicle_lidar" / "vehicle_lidar_baseline_index.json"
    infrastructure_path = output_root / "infrastructure_lidar" / "infrastructure_lidar_baseline_index.json"
    save_json(vehicle_path, vehicle_records)
    save_json(infrastructure_path, infrastructure_records)
    summary = {
        "data_root": str(data_root), "split": args.split, "num_samples": len(vehicle_records),
        "vehicle_index_path": str(vehicle_path), "infrastructure_index_path": str(infrastructure_path),
        "invalid_samples": 0,
    }
    summary_path = output_root / "dataset_preparation" / f"full_{args.split}_lidar_index_summary.json"
    save_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
