import sys
import json
from pathlib import Path
from collections import Counter

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import PREPROCESS_DIR


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_class_name(label):
    return label.get("type", label.get("class", "Unknown"))


def main():
    index_path = PREPROCESS_DIR / "dair_v2x_pair_index.json"
    pair_index = load_json(index_path)

    frame_records = []
    object_records = []
    class_counter = Counter()

    for item in tqdm(pair_index, desc="Reading labels"):
        label_path = item["cooperative_label"]

        if label_path is None or not Path(label_path).exists():
            continue

        labels = load_json(label_path)

        frame_records.append({
            "index": item["index"],
            "vehicle_id": item["vehicle_id"],
            "infrastructure_id": item["infrastructure_id"],
            "label_path": label_path,
            "num_objects": len(labels),
        })

        for obj_idx, label in enumerate(labels):
            cls = get_class_name(label)
            class_counter[cls] += 1

            object_records.append({
                "index": item["index"],
                "object_index": obj_idx,
                "class": cls,
            })

    frame_df = pd.DataFrame(frame_records)
    object_df = pd.DataFrame(object_records)

    frame_csv = PREPROCESS_DIR / "label_frame_statistics.csv"
    object_csv = PREPROCESS_DIR / "label_object_statistics.csv"
    class_csv = PREPROCESS_DIR / "class_distribution.csv"

    frame_df.to_csv(frame_csv, index=False, encoding="utf-8-sig")
    object_df.to_csv(object_csv, index=False, encoding="utf-8-sig")

    class_df = pd.DataFrame([
        {"class": k, "count": v}
        for k, v in class_counter.most_common()
    ])
    class_df.to_csv(class_csv, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("统计完成")
    print(f"逐帧目标数量统计：{frame_csv}")
    print(f"逐目标类别统计：{object_csv}")
    print(f"类别分布统计：{class_csv}")
    print("=" * 80)

    print("每帧目标数量统计：")
    print(frame_df["num_objects"].describe())

    print("=" * 80)
    print("类别分布：")
    print(class_df)


if __name__ == "__main__":
    main()