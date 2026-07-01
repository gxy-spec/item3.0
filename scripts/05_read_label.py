import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import json
from config import DATA_ROOT

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

coop_info = load_json(DATA_ROOT / "cooperative" / "data_info.json")
sample = coop_info[0]

print("第一条 cooperative sample:")
print(sample)

label_rel = sample.get("cooperative_label_path", None)

# 有些版本可能没有 cooperative_label_path 字段，需要根据 sample_id 拼接
if label_rel is None:
    # 尝试根据 system_error_offset 或 vehicle_frame 等字段不安全
    # 所以这里直接提醒你打印字段
    raise KeyError(
        "没有找到 cooperative_label_path 字段。"
        "请运行 02_inspect_data_info.py，把第一条 sample 发给我，我帮你确定 label_world 路径。"
    )

label_path = DATA_ROOT / label_rel

print("cooperative label path:", label_path)

labels = load_json(label_path)

print("=" * 80)
print("目标数量:", len(labels))
print("=" * 80)

if len(labels) > 0:
    print("第一个目标标注:")
    print(labels[0])
    print("=" * 80)
    print("第一个目标包含字段:")
    print(list(labels[0].keys()))
else:
    print("该样本没有目标。")