#!/usr/bin/env python3
"""Build minimal MMDetection3D 1.x inference infos for DAIR-V2X-C cameras."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_ROOT


SENSORS = {
    "vehicle_camera": "vehicle-side",
    "infrastructure_camera": "infrastructure-side",
}


def read_calib(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        name, content = line.split(":", 1)
        values[name.strip()] = [float(value) for value in content.split()]

    if "P2" not in values or "Tr_velo_to_cam" not in values:
        raise ValueError(f"KITTI 标定缺少 P2 或 Tr_velo_to_cam: {path}")

    p2 = np.asarray(values["P2"], dtype=np.float32).reshape(3, 4)
    lidar2cam = np.eye(4, dtype=np.float32)
    lidar2cam[:3, :] = np.asarray(values["Tr_velo_to_cam"], dtype=np.float32).reshape(3, 4)
    return p2, lidar2cam


def build_data_info(sensor_root, frame_id):
    image_path = sensor_root / "training" / "image_2" / f"{frame_id}.jpg"
    calib_path = sensor_root / "training" / "calib" / f"{frame_id}.txt"
    if not image_path.is_file():
        raise FileNotFoundError(f"缺少 Camera 图像: {image_path}")
    if not calib_path.is_file():
        raise FileNotFoundError(f"缺少 Camera 标定: {calib_path}")

    cam2img, lidar2cam = read_calib(calib_path)
    lidar2img = cam2img @ lidar2cam
    return {
        "sample_idx": str(frame_id),
        "lidar_points": {"lidar_path": f"training/velodyne/{frame_id}.bin"},
        "images": {
            "CAM2": {
                "img_path": str(image_path),
                "cam2img": cam2img,
                "lidar2cam": lidar2cam,
                "lidar2img": lidar2img,
            }
        },
        "instances": [],
    }


def dump_pickle(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)


def build_infos(sensor, data_root):
    sensor_root = data_root / SENSORS[sensor]
    imagesets_root = sensor_root / "ImageSets"
    if not imagesets_root.is_dir():
        raise FileNotFoundError(f"缺少 ImageSets: {imagesets_root}")

    result = {}
    for split in ("train", "val"):
        split_path = imagesets_root / f"{split}.txt"
        if not split_path.is_file():
            raise FileNotFoundError(f"缺少 split 文件: {split_path}")
        frame_ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        data_list = [build_data_info(sensor_root, frame_id) for frame_id in frame_ids]
        output_path = sensor_root / f"mmdet3d_camera_infos_{split}.pkl"
        dump_pickle(
            {
                "metainfo": {"dataset": "DAIR-V2X-C", "classes": ["Car"]},
                "data_list": data_list,
            },
            output_path,
        )
        result[split] = {"num_samples": len(data_list), "output_path": str(output_path)}
        print(f"[{sensor}] {split}: {len(data_list)} -> {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="生成 Camera ImVoxelNet 推理 info 文件")
    parser.add_argument("--sensor", choices=sorted(SENSORS), required=True)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    build_infos(args.sensor, args.data_root.resolve())


if __name__ == "__main__":
    main()
