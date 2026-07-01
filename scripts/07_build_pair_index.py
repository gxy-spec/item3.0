import sys
import json
from pathlib import Path
from tqdm import tqdm

# 保证无论你从项目根目录还是 scripts 目录运行，都能找到 config.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import DATA_ROOT, PREPROCESS_DIR


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_abs_path(rel_path):
    if rel_path is None:
        return None
    return str(DATA_ROOT / rel_path)


def build_calib_paths(vehicle_id, infra_id):
    vehicle_calib = {
        "camera_intrinsic": str(DATA_ROOT / "vehicle-side" / "calib" / "camera_intrinsic" / f"{vehicle_id}.json"),
        "lidar_to_camera": str(DATA_ROOT / "vehicle-side" / "calib" / "lidar_to_camera" / f"{vehicle_id}.json"),
        "lidar_to_novatel": str(DATA_ROOT / "vehicle-side" / "calib" / "lidar_to_novatel" / f"{vehicle_id}.json"),
        "novatel_to_world": str(DATA_ROOT / "vehicle-side" / "calib" / "novatel_to_world" / f"{vehicle_id}.json"),
    }

    infrastructure_calib = {
        "camera_intrinsic": str(DATA_ROOT / "infrastructure-side" / "calib" / "camera_intrinsic" / f"{infra_id}.json"),
        "virtuallidar_to_camera": str(DATA_ROOT / "infrastructure-side" / "calib" / "virtuallidar_to_camera" / f"{infra_id}.json"),
        "virtuallidar_to_world": str(DATA_ROOT / "infrastructure-side" / "calib" / "virtuallidar_to_world" / f"{infra_id}.json"),
    }

    return vehicle_calib, infrastructure_calib


def check_exists(path_str):
    if path_str is None:
        return False
    return Path(path_str).exists()


def main():
    coop_info_path = DATA_ROOT / "cooperative" / "data_info.json"
    coop_info = load_json(coop_info_path)

    pair_index = []
    missing_count = 0

    for idx, sample in enumerate(tqdm(coop_info, desc="Building DAIR-V2X pair index")):
        vehicle_img_rel = sample.get("vehicle_image_path")
        infra_img_rel = sample.get("infrastructure_image_path")

        if vehicle_img_rel is None or infra_img_rel is None:
            missing_count += 1
            continue

        vehicle_id = Path(vehicle_img_rel).stem
        infra_id = Path(infra_img_rel).stem

        vehicle_calib, infrastructure_calib = build_calib_paths(vehicle_id, infra_id)

        item = {
            "index": idx,
            "vehicle_id": vehicle_id,
            "infrastructure_id": infra_id,

            "vehicle_image": to_abs_path(sample.get("vehicle_image_path")),
            "vehicle_pointcloud": to_abs_path(sample.get("vehicle_pointcloud_path")),
            "infrastructure_image": to_abs_path(sample.get("infrastructure_image_path")),
            "infrastructure_pointcloud": to_abs_path(sample.get("infrastructure_pointcloud_path")),
            "cooperative_label": to_abs_path(sample.get("cooperative_label_path")),

            "system_error_offset": sample.get("system_error_offset", None),

            "vehicle_calib": vehicle_calib,
            "infrastructure_calib": infrastructure_calib,

            "raw_sample": sample,
        }

        # 检查关键文件是否存在
        key_paths = [
            item["vehicle_image"],
            item["vehicle_pointcloud"],
            item["infrastructure_image"],
            item["infrastructure_pointcloud"],
            item["cooperative_label"],
            item["vehicle_calib"]["lidar_to_novatel"],
            item["vehicle_calib"]["novatel_to_world"],
            item["infrastructure_calib"]["virtuallidar_to_world"],
        ]

        item["all_key_files_exist"] = all(check_exists(p) for p in key_paths)

        if not item["all_key_files_exist"]:
            missing_count += 1

        pair_index.append(item)

    save_path = PREPROCESS_DIR / "dair_v2x_pair_index.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(pair_index, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"索引保存完成：{save_path}")
    print(f"样本数量：{len(pair_index)}")
    print(f"存在缺失关键文件的样本数量：{missing_count}")
    print("=" * 80)

    if len(pair_index) > 0:
        print("第一条索引样例：")
        print(json.dumps(pair_index[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()