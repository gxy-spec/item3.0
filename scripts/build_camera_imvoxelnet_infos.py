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


def make_transform(calibration):
    if "transform" in calibration:
        calibration = calibration["transform"]
    rotation = np.asarray(calibration["rotation"], dtype=np.float32).reshape(3, 3)
    translation = np.asarray(calibration["translation"], dtype=np.float32).reshape(3)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def read_raw_dair_calib(sensor, sensor_root, frame_id):
    intrinsic_path = sensor_root / "calib" / "camera_intrinsic" / f"{frame_id}.json"
    transform_name = "lidar_to_camera" if sensor == "vehicle_camera" else "virtuallidar_to_camera"
    transform_path = sensor_root / "calib" / transform_name / f"{frame_id}.json"
    if not intrinsic_path.is_file() or not transform_path.is_file():
        raise FileNotFoundError(f"缺少原始 DAIR 相机标定: {intrinsic_path}, {transform_path}")
    intrinsic = json.loads(intrinsic_path.read_text(encoding="utf-8"))
    cam2img = np.zeros((3, 4), dtype=np.float32)
    cam2img[:, :3] = np.asarray(intrinsic["cam_K"], dtype=np.float32).reshape(3, 3)
    return cam2img, make_transform(json.loads(transform_path.read_text(encoding="utf-8")))


def build_data_info(sensor, sensor_root, frame_id, input_format):
    if input_format == "kitti":
        image_path = sensor_root / "training" / "image_2" / f"{frame_id}.jpg"
        calib_path = sensor_root / "training" / "calib" / f"{frame_id}.txt"
        if not image_path.is_file():
            raise FileNotFoundError(f"缺少 Camera 图像: {image_path}")
        if not calib_path.is_file():
            raise FileNotFoundError(f"缺少 Camera 标定: {calib_path}")
        cam2img, lidar2cam = read_calib(calib_path)
    else:
        image_path = sensor_root / "image" / f"{frame_id}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(f"缺少原始 Camera 图像: {image_path}")
        cam2img, lidar2cam = read_raw_dair_calib(sensor, sensor_root, frame_id)

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


def normalize_class(raw_type):
    name = str(raw_type).lower()
    if name in {"car", "van", "truck", "bus"}:
        return "Car"
    return None


def build_instances(sensor, data_root, frame_id, lidar2cam):
    label_dir = "lidar" if sensor == "vehicle_camera" else "virtuallidar"
    label_path = data_root / ("vehicle-side" if sensor == "vehicle_camera" else "infrastructure-side") / "label" / label_dir / f"{frame_id}.json"
    if not label_path.is_file():
        raise FileNotFoundError(f"缺少 Camera 训练标注: {label_path}")
    labels = json.loads(label_path.read_text(encoding="utf-8"))
    rotation = np.asarray(lidar2cam, dtype=np.float32)[:3, :3]
    translation = np.asarray(lidar2cam, dtype=np.float32)[:3, 3]
    instances = []
    for label in labels if isinstance(labels, list) else []:
        class_name = normalize_class(label.get("type", label.get("class", "")))
        location = label.get("3d_location", {})
        dimensions = label.get("3d_dimensions", {})
        if class_name is None or not isinstance(location, dict) or not isinstance(dimensions, dict):
            continue
        try:
            center_lidar = np.array([float(location["x"]), float(location["y"]), float(location["z"])], dtype=np.float32)
            length = float(dimensions["l"])
            width = float(dimensions["w"])
            height = float(dimensions["h"])
            yaw_lidar = float(label["rotation"])
        except (KeyError, TypeError, ValueError):
            continue
        if min(length, width, height) <= 0:
            continue
        # KITTI/MMDetection3D camera boxes use bottom-center [x, y, z,
        # length, height, width, rotation_y]. Convert the LiDAR box's
        # bottom center and forward direction through lidar2cam.
        bottom_lidar = center_lidar.copy()
        bottom_lidar[2] -= height / 2.0
        center_cam = rotation @ bottom_lidar + translation
        forward_lidar = center_lidar + np.array([np.cos(yaw_lidar), np.sin(yaw_lidar), 0], dtype=np.float32)
        forward_cam = rotation @ forward_lidar + translation
        rotation_y = float(np.arctan2(-(forward_cam[2] - center_cam[2]), forward_cam[0] - center_cam[0]))
        instances.append({
            "bbox_3d": [float(center_cam[0]), float(center_cam[1]), float(center_cam[2]), length, height, width, rotation_y],
            # KittiDataset's canonical label order is Pedestrian=0,
            # Cyclist=1, Car=2.  The project intentionally evaluates the
            # official DAIR-V2X camera baseline in the Car-only setting.
            "bbox_label_3d": 2,
        })
    return instances


def dump_pickle(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)


def build_infos(sensor, data_root, splits=("train", "val"), input_format="kitti", with_labels=False, split_id_dir=None):
    sensor_root = data_root / SENSORS[sensor]
    imagesets_root = sensor_root / "ImageSets"
    if not imagesets_root.is_dir():
        raise FileNotFoundError(f"缺少 ImageSets: {imagesets_root}")

    result = {}
    for split in splits:
        split_path = (
            Path(split_id_dir) / ("infrastructure_camera" if sensor == "infrastructure_camera" else "vehicle_camera") / f"{split}.txt"
            if split_id_dir
            else (
            data_root / "vehicle-side" / "ImageSets" / f"{split}.txt"
            if sensor == "infrastructure_camera"
            else imagesets_root / f"{split}.txt"
            )
        )
        if not split_path.is_file():
            raise FileNotFoundError(f"缺少 split 文件: {split_path}")
        frame_ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if sensor == "infrastructure_camera" and not split_id_dir:
            # Full Dataset cooperative splits are vehicle-sample keyed. Resolve
            # the paired infrastructure frame through cooperative/data_info.
            coop_info = json.loads((data_root / "cooperative" / "data_info.json").read_text(encoding="utf-8"))
            vehicle_to_infrastructure = {
                Path(item.get("vehicle_pointcloud_path", "")).stem: Path(item.get("infrastructure_pointcloud_path", "")).stem
                for item in coop_info
                if item.get("vehicle_pointcloud_path") and item.get("infrastructure_pointcloud_path")
            }
            frame_ids = [vehicle_to_infrastructure.get(frame_id, frame_id) for frame_id in frame_ids]
        data_list = [build_data_info(sensor, sensor_root, frame_id, input_format) for frame_id in frame_ids]
        if with_labels:
            for entry in data_list:
                frame_id = entry["sample_idx"]
                entry["instances"] = build_instances(sensor, data_root, frame_id, entry["images"]["CAM2"]["lidar2cam"])
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
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="ImageSets 文件名，例如 val 或 full_common_val")
    parser.add_argument(
        "--input-format", choices=["kitti", "raw_dair"], default="kitti",
        help="raw_dair 直接读取标准 image/calib，适合 Full Dataset。",
    )
    parser.add_argument("--with-labels", action="store_true", help="将本地 3D 标注写入 instances，供训练使用。")
    parser.add_argument("--split-id-dir", type=Path, help="公共集 split 根目录，内含 vehicle_camera/train.txt 等文件。")
    args = parser.parse_args()
    build_infos(args.sensor, args.data_root.resolve(), tuple(args.splits), args.input_format, args.with_labels, args.split_id_dir)


if __name__ == "__main__":
    main()
