import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import json
from pathlib import Path
from config import DATA_ROOT


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def try_load_calib(name, path):
    print("=" * 80)
    print(name)
    print("path:", path)

    if not path.exists():
        print("MISSING：该标定文件不存在，请检查目录结构或文件名。")
        return None

    data = load_json(path)
    print("OK：成功读取")
    print(data)
    return data


coop_info = load_json(DATA_ROOT / "cooperative" / "data_info.json")
sample = coop_info[0]

print("第一条 cooperative sample:")
print(sample)

# 从路径中提取样本编号
vehicle_id = Path(sample["vehicle_image_path"]).stem
infra_id = Path(sample["infrastructure_image_path"]).stem

print("=" * 80)
print("vehicle_id:", vehicle_id)
print("infrastructure_id:", infra_id)

# 车辆端标定文件候选路径
vehicle_calibs = {
    "vehicle_camera_intrinsic": DATA_ROOT / "vehicle-side" / "calib" / "camera_intrinsic" / f"{vehicle_id}.json",
    "vehicle_lidar_to_camera": DATA_ROOT / "vehicle-side" / "calib" / "lidar_to_camera" / f"{vehicle_id}.json",
    "vehicle_lidar_to_novatel": DATA_ROOT / "vehicle-side" / "calib" / "lidar_to_novatel" / f"{vehicle_id}.json",
    "vehicle_novatel_to_world": DATA_ROOT / "vehicle-side" / "calib" / "novatel_to_world" / f"{vehicle_id}.json",
}

# 路侧端标定文件候选路径
infra_calibs = {
    "infrastructure_camera_intrinsic": DATA_ROOT / "infrastructure-side" / "calib" / "camera_intrinsic" / f"{infra_id}.json",
    "infrastructure_virtuallidar_to_camera": DATA_ROOT / "infrastructure-side" / "calib" / "virtuallidar_to_camera" / f"{infra_id}.json",
    "infrastructure_virtuallidar_to_world": DATA_ROOT / "infrastructure-side" / "calib" / "virtuallidar_to_world" / f"{infra_id}.json",
}

print("\n开始读取车辆端标定文件：")
for name, path in vehicle_calibs.items():
    try_load_calib(name, path)

print("\n开始读取路侧端标定文件：")
for name, path in infra_calibs.items():
    try_load_calib(name, path)