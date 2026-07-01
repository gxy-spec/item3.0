import sys
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import PREPROCESS_DIR


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_transform(rotation, translation):
    T = np.eye(4)
    T[:3, :3] = np.array(rotation, dtype=float)
    T[:3, 3] = np.array(translation, dtype=float).reshape(3)
    return T


def extract_rt(calib_data):
    if "rotation" in calib_data and "translation" in calib_data:
        return calib_data["rotation"], calib_data["translation"]

    if "transform" in calib_data:
        transform = calib_data["transform"]
        if "rotation" in transform and "translation" in transform:
            return transform["rotation"], transform["translation"]

    raise KeyError(f"无法解析标定文件，字段为：{list(calib_data.keys())}")


def get_vehicle_lidar_to_world(item):
    lidar_to_novatel = load_json(item["vehicle_calib"]["lidar_to_novatel"])
    novatel_to_world = load_json(item["vehicle_calib"]["novatel_to_world"])

    R1, t1 = extract_rt(lidar_to_novatel)
    R2, t2 = extract_rt(novatel_to_world)

    T_lidar_to_novatel = make_transform(R1, t1)
    T_novatel_to_world = make_transform(R2, t2)

    return T_novatel_to_world @ T_lidar_to_novatel


def get_infra_lidar_to_world(item):
    virtuallidar_to_world = load_json(item["infrastructure_calib"]["virtuallidar_to_world"])
    R, t = extract_rt(virtuallidar_to_world)
    return make_transform(R, t)


def transform_to_list(T):
    return T.tolist()


def sensor_position_world(T):
    """
    传感器局部坐标原点 [0,0,0] 变换到 world 后的位置。
    """
    p = np.array([0, 0, 0, 1], dtype=float)
    pw = T @ p
    return pw[:3].tolist()


def main():
    pair_index_path = PREPROCESS_DIR / "dair_v2x_pair_index.json"
    pair_index = load_json(pair_index_path)

    gt_path = PREPROCESS_DIR / "world_gt_objects.json"
    gt_records = load_json(gt_path)

    gt_map = {
        item["sample_index"]: item
        for item in gt_records
    }

    system_index = []

    for item in tqdm(pair_index, desc="Building device-modality index"):
        sample_index = item["index"]

        T_vehicle = get_vehicle_lidar_to_world(item)
        T_infra = get_infra_lidar_to_world(item)

        sample = {
            "sample_index": sample_index,
            "vehicle_id": item["vehicle_id"],
            "infrastructure_id": item["infrastructure_id"],
            "coordinate_system": "world",

            "devices": {
                "vehicle": {
                    "device_type": "vehicle",
                    "local_coordinate": "vehicle_lidar",
                    "to_world_transform": transform_to_list(T_vehicle),
                    "sensor_position_world": sensor_position_world(T_vehicle),

                    "modalities": {
                        "camera": {
                            "available": item["vehicle_image"] is not None,
                            "data_path": item["vehicle_image"],
                            "modality_type": "image"
                        },
                        "lidar": {
                            "available": item["vehicle_pointcloud"] is not None,
                            "data_path": item["vehicle_pointcloud"],
                            "modality_type": "pointcloud"
                        }
                    },

                    "calib": item["vehicle_calib"]
                },

                "infrastructure": {
                    "device_type": "infrastructure",
                    "local_coordinate": "infrastructure_virtuallidar",
                    "to_world_transform": transform_to_list(T_infra),
                    "sensor_position_world": sensor_position_world(T_infra),

                    "modalities": {
                        "camera": {
                            "available": item["infrastructure_image"] is not None,
                            "data_path": item["infrastructure_image"],
                            "modality_type": "image"
                        },
                        "lidar": {
                            "available": item["infrastructure_pointcloud"] is not None,
                            "data_path": item["infrastructure_pointcloud"],
                            "modality_type": "pointcloud"
                        }
                    },

                    "calib": item["infrastructure_calib"]
                }
            },

            "gt_world": gt_map.get(sample_index, None),
            "system_error_offset": item.get("system_error_offset", None)
        }

        system_index.append(sample)

    save_path = PREPROCESS_DIR / "device_modality_index.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(system_index, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"设备—模态抽象索引保存完成：{save_path}")
    print(f"样本数量：{len(system_index)}")
    print("=" * 80)

    if len(system_index) > 0:
        print("第一条样例：")
        print(json.dumps(system_index[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()