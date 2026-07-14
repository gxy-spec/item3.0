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


COOPERATIVE_INFO_PATH = DATA_ROOT / "cooperative" / "data_info.json"
INFRASTRUCTURE_INFO_PATH = DATA_ROOT / "infrastructure-side" / "data_info.json"

OUTPUT_PATH = PREPROCESS_DIR / "infrastructure_lidar_baseline_index.json"
SUMMARY_PATH = PREPROCESS_DIR / "infrastructure_lidar_baseline_index_summary.json"
SPLIT_SOURCE = DATA_ROOT / "vehicle-side" / "ImageSets"


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def resolve_dataset_path(side, rel_path):
    if not rel_path:
        return None
    return str(DATA_ROOT / side / rel_path)


def path_exists(path):
    return bool(path) and Path(path).exists()


def read_split_ids(split_name):
    split_path = SPLIT_SOURCE / f"{split_name}.txt"
    if not split_path.exists():
        return []

    with open(split_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_split_by_vehicle_id():
    split_by_sample = {}
    for split_name in ["train", "val", "test"]:
        for sample_id in read_split_ids(split_name):
            split_by_sample[sample_id] = split_name
    return split_by_sample


def build_infrastructure_info_by_id():
    infrastructure_info = load_json(INFRASTRUCTURE_INFO_PATH)
    info_by_id = {}

    for item in infrastructure_info:
        infrastructure_id = Path(item.get("pointcloud_path", "")).stem
        if infrastructure_id:
            info_by_id[infrastructure_id] = item

    return info_by_id


def count_labels(label_path):
    if not path_exists(label_path):
        return 0

    labels = load_json(label_path)
    return len(labels) if isinstance(labels, list) else 0


def collect_missing_files(paths_by_type):
    missing_files = {}

    for missing_type, paths in paths_by_type.items():
        missing = [path for path in paths if not path_exists(path)]
        if missing:
            missing_files[missing_type] = missing

    return missing_files


def build_record(sample_index, coop_item, infrastructure_info, split_by_sample):
    vehicle_id = Path(coop_item.get("vehicle_pointcloud_path", "")).stem
    infrastructure_id = Path(coop_item.get("infrastructure_pointcloud_path", "")).stem

    if not vehicle_id:
        raise ValueError(f"cooperative sample[{sample_index}] 缺少 vehicle_pointcloud_path")
    if not infrastructure_id:
        raise ValueError(f"cooperative sample[{sample_index}] 缺少 infrastructure_pointcloud_path")

    infra_item = infrastructure_info.get(infrastructure_id, {})

    infrastructure_pointcloud_path = resolve_dataset_path(
        "infrastructure-side",
        infra_item.get(
            "pointcloud_path",
            f"velodyne/{infrastructure_id}.pcd",
        ),
    )
    infrastructure_label_path = resolve_dataset_path(
        "infrastructure-side",
        infra_item.get(
            "label_lidar_std_path",
            f"label/virtuallidar/{infrastructure_id}.json",
        ),
    )
    infrastructure_calib_path = resolve_dataset_path(
        "infrastructure-side",
        infra_item.get(
            "calib_virtuallidar_to_world_path",
            f"calib/virtuallidar_to_world/{infrastructure_id}.json",
        ),
    )
    cooperative_label_world_path = str(
        DATA_ROOT
        / coop_item.get(
            "cooperative_label_path",
            f"cooperative/label_world/{vehicle_id}.json",
        )
    )

    missing_files = collect_missing_files({
        "pointcloud": [infrastructure_pointcloud_path],
        "label": [infrastructure_label_path],
        "calib": [infrastructure_calib_path],
        "cooperative_label_world": [cooperative_label_world_path],
    })

    record = {
        "sample_index": sample_index,
        "sample_id": vehicle_id,
        "vehicle_id": vehicle_id,
        "infrastructure_id": infrastructure_id,
        "split": split_by_sample.get(vehicle_id, "unknown"),
        "split_source": str(SPLIT_SOURCE),
        "coordinate_system": {
            "input": "infrastructure_lidar",
            "evaluation": "world",
        },
        "infrastructure_pointcloud_path": infrastructure_pointcloud_path,
        "infrastructure_label_path": infrastructure_label_path,
        "infrastructure_calib_path": infrastructure_calib_path,
        "cooperative_label_world_path": cooperative_label_world_path,
        "num_infrastructure_lidar_labels": count_labels(infrastructure_label_path),
        "num_cooperative_world_gt": count_labels(cooperative_label_world_path),
        "transform_chain": [
            "infrastructure_lidar",
            "world",
        ],
        "all_required_files_exist": not missing_files,
        "missing_files": missing_files,
        "raw_cooperative_sample": coop_item,
        "raw_infrastructure_info": infra_item,
    }

    return record


def summarize(records):
    invalid_records = [record for record in records if not record["all_required_files_exist"]]

    summary = {
        "data_root": str(DATA_ROOT),
        "cooperative_info_path": str(COOPERATIVE_INFO_PATH),
        "infrastructure_info_path": str(INFRASTRUCTURE_INFO_PATH),
        "output_path": str(OUTPUT_PATH),
        "split_source": str(SPLIT_SOURCE),
        "total_samples": len(records),
        "valid_samples": sum(1 for record in records if record["all_required_files_exist"]),
        "invalid_samples": len(invalid_records),
        "missing_pointcloud": sum(1 for record in records if "pointcloud" in record["missing_files"]),
        "missing_label": sum(1 for record in records if "label" in record["missing_files"]),
        "missing_calib": sum(1 for record in records if "calib" in record["missing_files"]),
        "missing_cooperative_label_world": sum(
            1 for record in records if "cooperative_label_world" in record["missing_files"]
        ),
        "num_train": sum(1 for record in records if record["split"] == "train"),
        "num_val": sum(1 for record in records if record["split"] == "val"),
        "num_test": sum(1 for record in records if record["split"] == "test"),
        "num_unknown_split": sum(1 for record in records if record["split"] == "unknown"),
        "invalid_sample_ids": [record["sample_id"] for record in invalid_records],
        "invalid_samples_detail": [
            {
                "sample_id": record["sample_id"],
                "infrastructure_id": record["infrastructure_id"],
                "missing_files": record["missing_files"],
            }
            for record in invalid_records
        ],
    }

    return summary


def main():
    cooperative_info = load_json(COOPERATIVE_INFO_PATH)
    infrastructure_info = build_infrastructure_info_by_id()
    split_by_sample = build_split_by_vehicle_id()

    records = []
    for sample_index, coop_item in enumerate(
        tqdm(cooperative_info, desc="Building Infrastructure LiDAR-only baseline index")
    ):
        records.append(
            build_record(
                sample_index=sample_index,
                coop_item=coop_item,
                infrastructure_info=infrastructure_info,
                split_by_sample=split_by_sample,
            )
        )

    save_json(records, OUTPUT_PATH)

    summary = summarize(records)
    save_json(summary, SUMMARY_PATH)

    print("=" * 80)
    print(f"Infrastructure LiDAR-only baseline index 保存完成: {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"total_samples: {summary['total_samples']}")
    print(f"valid_samples: {summary['valid_samples']}")
    print(f"invalid_samples: {summary['invalid_samples']}")
    print(f"missing_pointcloud: {summary['missing_pointcloud']}")
    print(f"missing_label: {summary['missing_label']}")
    print(f"missing_calib: {summary['missing_calib']}")
    print(f"missing_cooperative_label_world: {summary['missing_cooperative_label_world']}")
    print("=" * 80)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
