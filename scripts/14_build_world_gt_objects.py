import sys
import json
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import PREPROCESS_DIR


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_label_object(label):
    cls_name = label.get("type", label.get("class", "Unknown"))

    loc = label.get("3d_location", label.get("location", None))
    dim = label.get("3d_dimensions", label.get("dimensions", None))
    rot = label.get("rotation", label.get("rotation_y", 0.0))

    if loc is None or dim is None:
        return None

    center_world = [
        float(loc["x"]),
        float(loc["y"]),
        float(loc.get("z", 0.0))
    ]

    length = float(dim.get("l", dim.get("length", 0.0)))
    width = float(dim.get("w", dim.get("width", 0.0)))
    height = float(dim.get("h", dim.get("height", 0.0)))

    if isinstance(rot, dict):
        yaw = float(rot.get("z", 0.0))
    else:
        yaw = float(rot)

    return {
        "class": cls_name,
        "center_world": center_world,
        "box_world": {
            "length": length,
            "width": width,
            "height": height,
            "yaw": yaw
        },
        "raw_label": label
    }


def main():
    pair_index_path = PREPROCESS_DIR / "dair_v2x_pair_index.json"
    pair_index = load_json(pair_index_path)

    world_gt_records = []

    for item in tqdm(pair_index, desc="Building world GT objects"):
        label_path = item["cooperative_label"]

        if label_path is None or not Path(label_path).exists():
            continue

        labels = load_json(label_path)

        objects = []

        for label in labels:
            obj = parse_label_object(label)
            if obj is not None:
                objects.append(obj)

        record = {
            "sample_index": item["index"],
            "vehicle_id": item["vehicle_id"],
            "infrastructure_id": item["infrastructure_id"],
            "coordinate_system": "world",
            "num_objects": len(objects),
            "objects": objects,
            "label_path": label_path
        }

        world_gt_records.append(record)

    save_path = PREPROCESS_DIR / "world_gt_objects.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(world_gt_records, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"world GT 目标集合保存完成：{save_path}")
    print(f"样本数量：{len(world_gt_records)}")

    if len(world_gt_records) > 0:
        print("第一条样例：")
        print(json.dumps(world_gt_records[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()