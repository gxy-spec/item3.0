import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_ROOT

required_paths = [
    DATA_ROOT / "vehicle-side",
    DATA_ROOT / "infrastructure-side",
    DATA_ROOT / "cooperative",

    DATA_ROOT / "vehicle-side" / "data_info.json",
    DATA_ROOT / "infrastructure-side" / "data_info.json",
    DATA_ROOT / "cooperative" / "data_info.json",

    DATA_ROOT / "vehicle-side" / "image",
    DATA_ROOT / "vehicle-side" / "velodyne",
    DATA_ROOT / "vehicle-side" / "label",
    DATA_ROOT / "vehicle-side" / "calib",

    DATA_ROOT / "infrastructure-side" / "image",
    DATA_ROOT / "infrastructure-side" / "velodyne",
    DATA_ROOT / "infrastructure-side" / "label",
    DATA_ROOT / "infrastructure-side" / "calib",

    DATA_ROOT / "cooperative" / "label_world",
]

print("=" * 80)
print("DAIR-V2X-C 数据集路径：")
print(DATA_ROOT)
print("=" * 80)

all_ok = True

for p in required_paths:
    exists = p.exists()
    print(f"{'OK' if exists else 'MISSING'} | {p}")
    if not exists:
        all_ok = False

print("=" * 80)

if all_ok:
    print("目录结构检查通过。")
else:
    print("有文件或文件夹缺失，请先检查数据集是否解压完整，或者 DATA_ROOT 是否写错。")