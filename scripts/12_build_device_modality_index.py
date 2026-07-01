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


def main():
    pair_index_path = PREPROCESS_DIR / "dair_v2x_pair_index.json"
    pair_index = load_json(pair_index_path)

    system_index = []

    for item in tqdm(pair_index, desc="Building device-modality index"):
        sample = {
            "sample_index": item["index"],
            "vehicle_id": item["vehicle_id"],
            "infrastructure_id": item["infrastructure_id"],

            "devices": {
                "vehicle": {
                    "device_type": "vehicle",
                    "modalities": {
                        "camera": {
                            "available": item["vehicle_image"] is not None,
                            "data_path": item["vehicle_image"],
                            "modality_type": "image",
                        },
                        "lidar": {
                            "available": item["vehicle_pointcloud"] is not None,
                            "data_path": item["vehicle_pointcloud"],
                            "modality_type": "pointcloud",
                        }
                    },
                    "calib": item["vehicle_calib"],
                },

                "infrastructure": {
                    "device_type": "infrastructure",
                    "modalities": {
                        "camera": {
                            "available": item["infrastructure_image"] is not None,
                            "data_path": item["infrastructure_image"],
                            "modality_type": "image",
                        },
                        "lidar": {
                            "available": item["infrastructure_pointcloud"] is not None,
                            "data_path": item["infrastructure_pointcloud"],
                            "modality_type": "pointcloud",
                        }
                    },
                    "calib": item["infrastructure_calib"],
                }
            },

            "label": {
                "cooperative_label": item["cooperative_label"],
            },

            "system_error_offset": item.get("system_error_offset", None),
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