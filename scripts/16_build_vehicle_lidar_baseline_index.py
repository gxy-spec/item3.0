import json
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **kwargs):
        return iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import DATA_ROOT, PREPROCESS_DIR


OPENPCDET_CUSTOM_ROOT = PROJECT_ROOT.parent / "OpenPCDet" / "data" / "custom_vehicle"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_split_ids(split_name):
    candidates = [
        DATA_ROOT / "vehicle-side" / "ImageSets" / f"{split_name}.txt",
        OPENPCDET_CUSTOM_ROOT / "ImageSets" / f"{split_name}.txt",
    ]

    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]

    return []


def count_labels(label_path):
    path = Path(label_path)
    if not path.exists():
        return 0

    labels = load_json(path)
    return len(labels) if isinstance(labels, list) else 0


def path_exists(path):
    return Path(path).exists()


def main():
    pair_index_path = PREPROCESS_DIR / "dair_v2x_pair_index.json"
    world_gt_path = PREPROCESS_DIR / "world_gt_objects.json"

    if not pair_index_path.exists():
        raise FileNotFoundError(
            f"找不到 pair index: {pair_index_path}\n"
            "请先运行: python dair_v2x_project/scripts/07_build_pair_index.py"
        )

    pair_index = load_json(pair_index_path)
    world_gt_records = load_json(world_gt_path) if world_gt_path.exists() else []

    world_gt_by_vehicle_id = {
        item["vehicle_id"]: item
        for item in world_gt_records
        if item.get("vehicle_id")
    }

    split_by_sample = {}
    for split_name in ["train", "val", "test"]:
        for sample_id in read_split_ids(split_name):
            split_by_sample[sample_id] = split_name

    records = []
    for item in tqdm(pair_index, desc="Building Vehicle LiDAR-only baseline index"):
        vehicle_id = item["vehicle_id"]
        infrastructure_id = item["infrastructure_id"]
        world_gt = world_gt_by_vehicle_id.get(vehicle_id)
        vehicle_calib = {
            "camera_intrinsic": str(
                DATA_ROOT / "vehicle-side" / "calib" / "camera_intrinsic" / f"{vehicle_id}.json"
            ),
            "lidar_to_camera": str(
                DATA_ROOT / "vehicle-side" / "calib" / "lidar_to_camera" / f"{vehicle_id}.json"
            ),
            "lidar_to_novatel": str(
                DATA_ROOT / "vehicle-side" / "calib" / "lidar_to_novatel" / f"{vehicle_id}.json"
            ),
            "novatel_to_world": str(
                DATA_ROOT / "vehicle-side" / "calib" / "novatel_to_world" / f"{vehicle_id}.json"
            ),
        }
        vehicle_pointcloud = str(DATA_ROOT / "vehicle-side" / "velodyne" / f"{vehicle_id}.pcd")
        cooperative_label_world = str(DATA_ROOT / "cooperative" / "label_world" / f"{vehicle_id}.json")

        record = {
            "sample_index": item["index"],
            "sample_id": vehicle_id,
            "vehicle_id": vehicle_id,
            "infrastructure_id": infrastructure_id,
            "split": split_by_sample.get(vehicle_id, "unknown"),
            "coordinate_system": {
                "input": "vehicle_lidar",
                "evaluation": "world",
            },
            "vehicle_pointcloud": vehicle_pointcloud,
            "vehicle_training_pointcloud_bin": str(
                DATA_ROOT / "vehicle-side" / "training" / "velodyne" / f"{vehicle_id}.bin"
            ),
            "vehicle_training_label_kitti": str(
                DATA_ROOT / "vehicle-side" / "training" / "label_2" / f"{vehicle_id}.txt"
            ),
            "opencdet_custom_points": str(OPENPCDET_CUSTOM_ROOT / "points" / f"{vehicle_id}.npy"),
            "opencdet_custom_label": str(OPENPCDET_CUSTOM_ROOT / "labels" / f"{vehicle_id}.txt"),
            "cooperative_label_world": cooperative_label_world,
            "num_cooperative_world_gt": (
                int(world_gt["num_objects"])
                if world_gt is not None and "num_objects" in world_gt
                else count_labels(cooperative_label_world)
            ),
            "vehicle_calib": vehicle_calib,
            "transform_chain": [
                "vehicle_lidar",
                "vehicle_novatel",
                "world",
            ],
            "all_required_files_exist": all(
                path_exists(path)
                for path in [
                    vehicle_pointcloud,
                    cooperative_label_world,
                    vehicle_calib["lidar_to_novatel"],
                    vehicle_calib["novatel_to_world"],
                ]
            ),
            "openpcdet_custom_ready": all(
                path_exists(path)
                for path in [
                    OPENPCDET_CUSTOM_ROOT / "points" / f"{vehicle_id}.npy",
                    OPENPCDET_CUSTOM_ROOT / "labels" / f"{vehicle_id}.txt",
                ]
            ),
        }
        records.append(record)

    save_path = PREPROCESS_DIR / "vehicle_lidar_baseline_index.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    summary = {
        "data_root": str(DATA_ROOT),
        "num_samples": len(records),
        "num_train": sum(1 for item in records if item["split"] == "train"),
        "num_val": sum(1 for item in records if item["split"] == "val"),
        "num_test": sum(1 for item in records if item["split"] == "test"),
        "num_unknown_split": sum(1 for item in records if item["split"] == "unknown"),
        "num_missing_required_files": sum(1 for item in records if not item["all_required_files_exist"]),
        "num_openpcdet_custom_ready": sum(1 for item in records if item["openpcdet_custom_ready"]),
        "output_path": str(save_path),
    }

    summary_path = PREPROCESS_DIR / "vehicle_lidar_baseline_index_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"Vehicle LiDAR-only baseline index 保存完成: {save_path}")
    print(f"Summary: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
