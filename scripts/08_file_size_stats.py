import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import json
import os
import pandas as pd
from tqdm import tqdm
from config import PREPROCESS_DIR

index_path = PREPROCESS_DIR / "dair_v2x_pair_index.json"

with open(index_path, "r", encoding="utf-8") as f:
    pair_index = json.load(f)

records = []

modalities = [
    ("vehicle", "camera", "vehicle_image"),
    ("vehicle", "lidar", "vehicle_pointcloud"),
    ("infrastructure", "camera", "infrastructure_image"),
    ("infrastructure", "lidar", "infrastructure_pointcloud"),
]

for item in tqdm(pair_index, desc="Calculating file sizes"):
    for device, modality, key in modalities:
        path = item.get(key, None)

        if path is None or not os.path.exists(path):
            continue

        size_bytes = os.path.getsize(path)
        records.append({
            "index": item["index"],
            "device": device,
            "modality": modality,
            "file_path": path,
            "size_bytes": size_bytes,
            "size_mb": size_bytes / 1024 / 1024
        })

df = pd.DataFrame(records)

save_csv = PREPROCESS_DIR / "file_size_stats.csv"
df.to_csv(save_csv, index=False, encoding="utf-8-sig")

print(f"保存完成：{save_csv}")
print("=" * 80)

summary = df.groupby(["device", "modality"])["size_mb"].agg(["count", "mean", "min", "max"])
print(summary)

summary_csv = PREPROCESS_DIR / "file_size_summary.csv"
summary.to_csv(summary_csv, encoding="utf-8-sig")

print(f"统计摘要保存完成：{summary_csv}")