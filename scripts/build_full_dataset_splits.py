#!/usr/bin/env python3
"""Build reproducible train/val ImageSets for the available DAIR-V2X-C Full data."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ITEM3_ROOT = PROJECT_ROOT.parent
DEFAULT_DATA_ROOT = Path(
    "/mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure"
)
DEFAULT_SPLIT = ITEM3_ROOT / "DAIR-V2X" / "data" / "split_datas" / "cooperative-split-data.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "full_baselines" / "dataset_preparation" / "full_split_summary.json"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_ids(path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="从官方 cooperative split 构建 Full Dataset 的共同 train/val ImageSets")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--duplicate-policy", choices=["first", "last", "fail"], default="first",
        help="同一 vehicle_id 对应多个路侧帧时的固定映射策略；默认 first 并记录警告。",
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    cooperative = load_json(data_root / "cooperative" / "data_info.json")
    split = load_json(args.split_path)["cooperative_split"]
    if not isinstance(split.get("train"), list) or not isinstance(split.get("val"), list):
        raise ValueError(f"split 缺少 cooperative_split.train/val: {args.split_path}")

    pairs_by_vehicle = defaultdict(list)
    for index, item in enumerate(cooperative):
        vehicle_id = Path(item.get("vehicle_pointcloud_path", "")).stem
        infrastructure_id = Path(item.get("infrastructure_pointcloud_path", "")).stem
        if vehicle_id and infrastructure_id:
            pairs_by_vehicle[vehicle_id].append({
                "index": index,
                "vehicle_id": vehicle_id,
                "infrastructure_id": infrastructure_id,
                "item": item,
            })

    selected = {}
    ambiguous = []
    missing = []
    for split_name in ("train", "val"):
        records = []
        for vehicle_id in map(str, split[split_name]):
            candidates = pairs_by_vehicle.get(vehicle_id, [])
            if not candidates:
                missing.append({"split": split_name, "vehicle_id": vehicle_id})
                continue
            if len(candidates) > 1:
                ambiguous.append({
                    "split": split_name,
                    "vehicle_id": vehicle_id,
                    "candidate_infrastructure_ids": [item["infrastructure_id"] for item in candidates],
                    "selected_infrastructure_id": None,
                })
                if args.duplicate_policy == "fail":
                    continue
                candidate = candidates[0] if args.duplicate_policy == "first" else candidates[-1]
                ambiguous[-1]["selected_infrastructure_id"] = candidate["infrastructure_id"]
            else:
                candidate = candidates[0]
            records.append(candidate)
        selected[split_name] = records

    if missing:
        raise RuntimeError(f"官方 split 有 {len(missing)} 个样本不在当前 cooperative/data_info.json；见 summary: {args.summary}")
    if args.duplicate_policy == "fail" and ambiguous:
        raise RuntimeError(f"发现 {len(ambiguous)} 个一车多路侧歧义映射；请指定 first 或 last 策略")

    for split_name, records in selected.items():
        vehicle_ids = [record["vehicle_id"] for record in records]
        infrastructure_ids = [record["infrastructure_id"] for record in records]
        write_ids(data_root / "vehicle-side" / "ImageSets" / f"{split_name}.txt", vehicle_ids)
        write_ids(data_root / "infrastructure-side" / "ImageSets" / f"{split_name}.txt", infrastructure_ids)

    # Four-way comparison requires both image and LiDAR files for both endpoints.
    common_val, excluded_common_val = [], []
    for record in selected["val"]:
        required = {
            "vehicle_image": data_root / "vehicle-side" / "image" / f"{record['vehicle_id']}.jpg",
            "vehicle_lidar": data_root / "vehicle-side" / "velodyne" / f"{record['vehicle_id']}.pcd",
            "infrastructure_image": data_root / "infrastructure-side" / "image" / f"{record['infrastructure_id']}.jpg",
            "infrastructure_lidar": data_root / "infrastructure-side" / "velodyne" / f"{record['infrastructure_id']}.pcd",
        }
        missing_modalities = [name for name, path in required.items() if not path.is_file()]
        if missing_modalities:
            excluded_common_val.append({
                "vehicle_id": record["vehicle_id"],
                "infrastructure_id": record["infrastructure_id"],
                "missing_modalities": missing_modalities,
            })
        else:
            common_val.append(record)
    write_ids(data_root / "vehicle-side" / "ImageSets" / "full_common_val.txt", [item["vehicle_id"] for item in common_val])
    write_ids(data_root / "infrastructure-side" / "ImageSets" / "full_common_val.txt", [item["infrastructure_id"] for item in common_val])

    pair_manifest = [
        {"split": split_name, "sample_index": record["index"], "vehicle_id": record["vehicle_id"], "infrastructure_id": record["infrastructure_id"]}
        for split_name, records in selected.items() for record in records
    ]
    pair_manifest_path = args.summary.resolve().with_name("full_train_val_pair_manifest.json")
    write_json(pair_manifest_path, pair_manifest)
    common_manifest_path = args.summary.resolve().with_name("full_common_val_pair_manifest.json")
    write_json(common_manifest_path, [
        {"split": "full_common_val", "sample_index": record["index"], "vehicle_id": record["vehicle_id"], "infrastructure_id": record["infrastructure_id"]}
        for record in common_val
    ])
    summary = {
        "data_root": str(data_root),
        "official_split_path": str(args.split_path.resolve()),
        "scope": "official cooperative train/val only; current data package has no cooperative test samples",
        "duplicate_policy": args.duplicate_policy,
        "num_train": len(selected["train"]),
        "num_val": len(selected["val"]),
        "num_full_common_val": len(common_val),
        "num_excluded_from_full_common_val": len(excluded_common_val),
        "excluded_full_common_val_examples": excluded_common_val[:20],
        "num_missing": len(missing),
        "num_ambiguous_vehicle_to_infrastructure": len(ambiguous),
        "ambiguous_examples": ambiguous[:20],
        "pair_manifest_path": str(pair_manifest_path),
        "full_common_val_pair_manifest_path": str(common_manifest_path),
        "vehicle_imagesets": str(data_root / "vehicle-side" / "ImageSets"),
        "infrastructure_imagesets": str(data_root / "infrastructure-side" / "ImageSets"),
    }
    write_json(args.summary.resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
