import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import json
from pathlib import Path
from tqdm import tqdm
from config import DATA_ROOT, PREPROCESS_DIR

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_get(sample, key):
    value = sample.get(key, None)
    if value is None:
        return None
    return str(DATA_ROOT / value)

coop_info = load_json(DATA_ROOT / "cooperative" / "data_info.json")

pair_index = []

for idx, sample in enumerate(tqdm(coop_info, desc="Building pair index")):
    item = {
        "index": idx,

        # 原始 sample 信息保存下来，防止字段遗漏
        "raw_sample": sample,

        # 常用数据路径
        "vehicle_image": safe_get(sample, "vehicle_image_path"),
        "vehicle_pointcloud": safe_get(sample, "vehicle_pointcloud_path"),
        "infrastructure_image": safe_get(sample, "infrastructure_image_path"),
        "infrastructure_pointcloud": safe_get(sample, "infrastructure_pointcloud_path"),
        "cooperative_label": safe_get(sample, "cooperative_label_path"),
    }

    pair_index.append(item)

save_path = PREPROCESS_DIR / "dair_v2x_pair_index.json"

with open(save_path, "w", encoding="utf-8") as f:
    json.dump(pair_index, f, indent=2, ensure_ascii=False)

print(f"保存完成：{save_path}")
print(f"样本数量：{len(pair_index)}")
print("第一条索引：")
print(pair_index[0])