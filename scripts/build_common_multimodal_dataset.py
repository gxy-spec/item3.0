#!/usr/bin/env python3
"""Build the maximal four-modality common subset without modifying source data."""

import argparse
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure")
DEFAULT_PAIR_MANIFEST = PROJECT_ROOT / "outputs/full_baselines/dataset_preparation/full_train_val_pair_manifest.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/common_multimodal_dataset"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def path_map(root, row):
    v, i = row["vehicle_id"], row["infrastructure_id"]
    return {
        "vehicle_image": root / "vehicle-side/image" / f"{v}.jpg",
        "vehicle_lidar": root / "vehicle-side/velodyne" / f"{v}.pcd",
        "vehicle_label": root / "vehicle-side/label/lidar" / f"{v}.json",
        "vehicle_calib_lidar_to_novatel": root / "vehicle-side/calib/lidar_to_novatel" / f"{v}.json",
        "vehicle_calib_novatel_to_world": root / "vehicle-side/calib/novatel_to_world" / f"{v}.json",
        "vehicle_calib_camera_intrinsic": root / "vehicle-side/calib/camera_intrinsic" / f"{v}.json",
        "vehicle_calib_lidar_to_camera": root / "vehicle-side/calib/lidar_to_camera" / f"{v}.json",
        "infrastructure_image": root / "infrastructure-side/image" / f"{i}.jpg",
        "infrastructure_lidar": root / "infrastructure-side/velodyne" / f"{i}.pcd",
        "infrastructure_label": root / "infrastructure-side/label/virtuallidar" / f"{i}.json",
        "infrastructure_calib_virtuallidar_to_world": root / "infrastructure-side/calib/virtuallidar_to_world" / f"{i}.json",
        "infrastructure_calib_camera_intrinsic": root / "infrastructure-side/calib/camera_intrinsic" / f"{i}.json",
        "infrastructure_calib_virtuallidar_to_camera": root / "infrastructure-side/calib/virtuallidar_to_camera" / f"{i}.json",
        "cooperative_label_world": root / "cooperative/label_world" / f"{v}.json",
    }


def main():
    parser = argparse.ArgumentParser(description="构建四模态最大公共子集索引")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pair-manifest", type=Path, default=DEFAULT_PAIR_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    root, output = args.data_root.resolve(), args.output_root.resolve()
    rows = load(args.pair_manifest)
    all_valid, summary = [], {"data_root": str(root), "pair_manifest": str(args.pair_manifest.resolve()), "splits": {}}
    for split in sorted({str(row["split"]) for row in rows}):
        split_rows = [row for row in rows if str(row["split"]) == split]
        valid, excluded, reasons = [], [], Counter()
        for row in split_rows:
            paths = path_map(root, row)
            missing = sorted(name for name, path in paths.items() if not path.is_file())
            record = {"sample_id": row["vehicle_id"], "vehicle_id": row["vehicle_id"], "infrastructure_id": row["infrastructure_id"], "split": split,
                      "paths": {name: str(path) for name, path in paths.items()}, "missing_files": missing}
            if missing:
                excluded.append(record)
                reasons.update(missing)
            else:
                valid.append(record)
                all_valid.append(record)
        split_dir = output / split
        for sensor, key in (("vehicle_camera", "vehicle_image"), ("vehicle_lidar", "vehicle_lidar"), ("infrastructure_camera", "infrastructure_image"), ("infrastructure_lidar", "infrastructure_lidar")):
            ids = [r["vehicle_id"] if sensor.startswith("vehicle") else r["infrastructure_id"] for r in valid]
            (split_dir / sensor / "ImageSets.txt").parent.mkdir(parents=True, exist_ok=True)
            (split_dir / sensor / "ImageSets.txt").write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
        write(split_dir / "common_manifest.json", valid)
        write(split_dir / "excluded_samples.json", excluded)
        summary["splits"][split] = {"requested": len(split_rows), "valid": len(valid), "excluded": len(excluded), "missing_file_counts": dict(reasons)}
    write(output / "common_multimodal_manifest.json", all_valid)
    pair_manifest = [
        {"split": row["split"], "vehicle_id": row["vehicle_id"], "infrastructure_id": row["infrastructure_id"]}
        for row in all_valid
    ]
    write(output / "common_pair_manifest.json", pair_manifest)
    write(output / "common_multimodal_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
