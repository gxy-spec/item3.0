import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ITEM3_ROOT = PROJECT_ROOT.parent

CUSTOM_VEHICLE_ROOT = (
    ITEM3_ROOT
    / "OpenPCDet"
    / "data"
    / "custom_vehicle"
)

IMAGESETS_ROOT = CUSTOM_VEHICLE_ROOT / "ImageSets"
LABELS_ROOT = CUSTOM_VEHICLE_ROOT / "labels"

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "baselines" / "vehicle_lidar"
OUTPUT_PATH = OUTPUT_ROOT / "predictions_vehicle_lidar.json"


def read_ids(split_name):
    split_path = IMAGESETS_ROOT / f"{split_name}.txt"

    if not split_path.exists():
        raise FileNotFoundError(f"找不到 split 文件: {split_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f.readlines() if line.strip()]

    return ids


def parse_custom_label_line(line):
    """
    custom_vehicle label 格式：
    x y z dx dy dz heading class_name
    """
    parts = line.strip().split()

    if len(parts) != 8:
        return None

    try:
        x = float(parts[0])
        y = float(parts[1])
        z = float(parts[2])
        dx = float(parts[3])
        dy = float(parts[4])
        dz = float(parts[5])
        heading = float(parts[6])
        cls = parts[7]
    except Exception:
        return None

    if dx <= 0 or dy <= 0 or dz <= 0:
        return None

    if cls not in ["Car", "Pedestrian", "Cyclist"]:
        return None

    return {
        "class": cls,
        "score": 1.0,
        "center_lidar": [x, y, z],
        "box_lidar": {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "heading": heading
        },
        "source": "vehicle_lidar_label_oracle"
    }


def load_label_as_predictions(sample_id):
    label_path = LABELS_ROOT / f"{sample_id}.txt"

    if not label_path.exists():
        print(f"[WARN] 找不到 label 文件: {label_path}")
        return []

    objects = []

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = parse_custom_label_line(line)
            if obj is not None:
                objects.append(obj)

    return objects


def main():
    print("=" * 80)
    print("Export vehicle LiDAR label-oracle predictions")
    print("=" * 80)

    if not CUSTOM_VEHICLE_ROOT.exists():
        raise FileNotFoundError(f"找不到 custom_vehicle 数据目录: {CUSTOM_VEHICLE_ROOT}")

    if not LABELS_ROOT.exists():
        raise FileNotFoundError(f"找不到 labels 目录: {LABELS_ROOT}")

    # 这里优先导出 val split，用于后续评价和可视化。
    # 如果 val 不存在，就退回 train。
    if (IMAGESETS_ROOT / "val.txt").exists():
        sample_ids = read_ids("val")
        split_name = "val"
    else:
        sample_ids = read_ids("train")
        split_name = "train"

    records = []

    for idx, sample_id in enumerate(sample_ids):
        objects = load_label_as_predictions(sample_id)

        records.append({
            "sample_id": sample_id,
            "coordinate_system": "vehicle_lidar",
            "prediction_type": "vehicle_lidar_label_oracle",
            "sensor": "vehicle_lidar",
            "source_id": sample_id,
            "note": "This file uses vehicle-side LiDAR labels as detector-like outputs for engineering validation.",
            "pred_objects": objects
        })

        print(
            f"[{idx + 1}/{len(sample_ids)}] "
            f"{sample_id}: pred_objects={len(objects)}"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    summary = {
        "split": split_name,
        "num_samples": len(records),
        "num_objects": sum(len(r["pred_objects"]) for r in records),
        "output_path": str(OUTPUT_PATH),
        "prediction_type": "vehicle_lidar_label_oracle"
    }

    summary_path = OUTPUT_ROOT / "predictions_vehicle_lidar_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("导出完成")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Summary: {summary_path}")
    print("=" * 80)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
