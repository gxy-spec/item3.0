#!/usr/bin/env python3
"""Prepare selected Full Dataset LiDAR samples for real OpenPCDet inference.

The script writes a separate CustomDataset directory supplied by ``--target-root``.
It never writes into the source DAIR-V2X-C tree or the existing small-sample
``custom_vehicle`` / ``custom_infrastructure`` directories.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(
    "/mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure"
)
DEFAULT_PAIR_MANIFEST = PROJECT_ROOT / "outputs" / "full_baselines" / "dataset_preparation" / "full_train_val_pair_manifest.json"

CLASS_MAPPING = {
    "car": "Car", "van": "Car", "truck": "Car", "bus": "Car",
    "pedestrian": "Pedestrian", "person": "Pedestrian",
    "cyclist": "Cyclist", "motorcyclist": "Cyclist", "motorcycle": "Cyclist",
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_pcd(path):
    try:
        import open3d as o3d
    except ModuleNotFoundError as exc:
        raise RuntimeError("需要 Open3D 解码 binary_compressed PCD；请在 torch-gpu-test 环境运行。") from exc
    cloud = o3d.t.io.read_point_cloud(str(path))
    if "positions" not in cloud.point:
        raise ValueError(f"PCD 缺少 positions: {path}")
    xyz = np.asarray(cloud.point.positions.numpy(), dtype=np.float32)
    if "intensity" in cloud.point:
        intensity = np.asarray(cloud.point.intensity.numpy(), dtype=np.float32).reshape(-1, 1)
    else:
        intensity = np.zeros((len(xyz), 1), dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(intensity) != len(xyz):
        raise ValueError(f"PCD 点格式异常: {path}")
    return np.concatenate([xyz, intensity], axis=1)


def label_line(label):
    class_name = CLASS_MAPPING.get(str(label.get("type", "")).lower())
    location = label.get("3d_location")
    dimensions = label.get("3d_dimensions")
    if class_name is None or not isinstance(location, dict) or not isinstance(dimensions, dict):
        return None
    try:
        x, y, z = float(location["x"]), float(location["y"]), float(location["z"])
        dx, dy, dz = float(dimensions["l"]), float(dimensions["w"]), float(dimensions["h"])
        heading = float(label["rotation"])
    except (KeyError, TypeError, ValueError):
        return None
    if min(dx, dy, dz) <= 0 or not np.isfinite([x, y, z, dx, dy, dz, heading]).all():
        return None
    return f"{x:.6f} {y:.6f} {z:.6f} {dx:.6f} {dy:.6f} {dz:.6f} {heading:.6f} {class_name}", class_name


def sensor_paths(data_root, sensor, source_id):
    if sensor == "vehicle_lidar":
        return (
            data_root / "vehicle-side" / "velodyne" / f"{source_id}.pcd",
            data_root / "vehicle-side" / "label" / "lidar" / f"{source_id}.json",
        )
    return (
        data_root / "infrastructure-side" / "velodyne" / f"{source_id}.pcd",
        data_root / "infrastructure-side" / "label" / "virtuallidar" / f"{source_id}.json",
    )


def main():
    parser = argparse.ArgumentParser(description="为 Full Dataset 的真实 PointPillars 推理准备 OpenPCDet CustomDataset")
    parser.add_argument("--sensor", choices=["vehicle_lidar", "infrastructure_lidar"], required=True)
    parser.add_argument("--split", default="val", help="pair manifest 的 split 名，例如 val 或 full_common_val")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pair-manifest", type=Path, default=DEFAULT_PAIR_MANIFEST)
    parser.add_argument("--target-root", type=Path, required=True, help="独立的 OpenPCDet CustomDataset 目录")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖同名派生 npy/label 文件")
    parser.add_argument("--max-samples", type=int, default=0, help="仅用于真实链路试跑；0 表示该 split 全部样本")
    args = parser.parse_args()

    data_root, target_root = args.data_root.resolve(), args.target_root.resolve()
    records = [item for item in load_json(args.pair_manifest) if item.get("split") == args.split]
    if args.max_samples:
        records = records[:args.max_samples]
    if not records:
        raise ValueError(f"pair manifest 中没有 {args.split} 样本: {args.pair_manifest}")
    points_dir, labels_dir, imagesets_dir, meta_dir = (
        target_root / "points", target_root / "labels", target_root / "ImageSets", target_root / "meta"
    )
    for path in (points_dir, labels_dir, imagesets_dir, meta_dir):
        path.mkdir(parents=True, exist_ok=True)

    output_records, class_counts, failures = [], Counter(), []
    ids = []
    for index, record in enumerate(records, start=1):
        source_id = record["vehicle_id"] if args.sensor == "vehicle_lidar" else record["infrastructure_id"]
        point_path, label_path = sensor_paths(data_root, args.sensor, source_id)
        point_output, label_output = points_dir / f"{source_id}.npy", labels_dir / f"{source_id}.txt"
        item = {"sample_id": record["vehicle_id"], "source_id": source_id, "status": "ok"}
        try:
            if point_output.exists() and not args.overwrite:
                points = np.load(point_output, mmap_mode="r")
            else:
                points = read_pcd(point_path)
                np.save(point_output, points)
            labels = load_json(label_path)
            lines = []
            for label in labels if isinstance(labels, list) else []:
                converted = label_line(label) if isinstance(label, dict) else None
                if converted is not None:
                    line, class_name = converted
                    lines.append(line)
                    class_counts[class_name] += 1
            if not label_output.exists() or args.overwrite:
                label_output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            item.update({"num_points": int(points.shape[0]), "num_labels": len(lines), "pointcloud_path": str(point_path), "label_path": str(label_path)})
            ids.append(source_id)
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
            item.update({"status": "error", "error": str(exc)})
            failures.append(item)
        output_records.append(item)
        print(f"[{index}/{len(records)}] {args.sensor} source_id={source_id}: {item['status']}")

    if failures:
        write_json(meta_dir / f"prepare_{args.split}_failure_log.json", failures)
        raise RuntimeError(f"{len(failures)} 个样本准备失败；失败日志已写入 {meta_dir}")
    (imagesets_dir / f"{args.split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    summary = {
        "sensor": args.sensor,
        "split": args.split,
        "data_root": str(data_root),
        "pair_manifest": str(args.pair_manifest.resolve()),
        "target_root": str(target_root),
        "num_samples": len(output_records),
        "num_failures": 0,
        "class_distribution": dict(class_counts),
        "source_is_real_dair_lidar": True,
        "prediction_generation": "not performed; this script only prepares real model inputs",
        "records": output_records,
    }
    summary_path = meta_dir / f"prepare_{args.split}_summary.json"
    write_json(summary_path, summary)
    print(f"[DONE] {summary_path}")


if __name__ == "__main__":
    main()
