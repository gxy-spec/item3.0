import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_ROOT

coop_info_path = DATA_ROOT / "cooperative" / "data_info.json"
veh_info_path = DATA_ROOT / "vehicle-side" / "data_info.json"
inf_info_path = DATA_ROOT / "infrastructure-side" / "data_info.json"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

coop_info = load_json(coop_info_path)
veh_info = load_json(veh_info_path)
inf_info = load_json(inf_info_path)

print("=" * 80)
print("cooperative samples:", len(coop_info))
print("vehicle-side samples:", len(veh_info))
print("infrastructure-side samples:", len(inf_info))
print("=" * 80)

print("cooperative 第一条数据：")
print(coop_info[0])
print("=" * 80)

print("cooperative 第一条数据包含的字段：")
print(list(coop_info[0].keys()))
print("=" * 80)

print("vehicle-side 第一条数据字段：")
print(list(veh_info[0].keys()))
print(veh_info[0])
print("=" * 80)

print("infrastructure-side 第一条数据字段：")
print(list(inf_info[0].keys()))
print(inf_info[0])