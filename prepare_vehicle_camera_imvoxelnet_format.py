#!/usr/bin/env python3
"""Replace legacy placeholder Vehicle Camera calibration with official KITTI calibration.

Only ``vehicle-side/training/calib`` is rewritten. The raw DAIR-V2X image,
annotation, calibration JSON and data_info files remain read-only inputs.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_ROOT, PREPROCESS_DIR
from prepare_infrastructure_camera_imvoxelnet_format import (
    DEFAULT_SPLIT_PATH,
    DAIR_REPO_ROOT,
    load_official_converter_functions,
    make_relative_symlink,
    read_json,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_vehicle_ids(split_path):
    split_data = read_json(split_path)
    vehicle_split = split_data.get("vehicle_split", {})
    train_ids = vehicle_split.get("train", [])
    val_ids = vehicle_split.get("val", [])
    if not isinstance(train_ids, list) or not isinstance(val_ids, list):
        raise ValueError(f"split 文件缺少 vehicle_split.train/val: {split_path}")
    frame_ids = train_ids + val_ids
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("vehicle example split 存在重复 frame id")
    return frame_ids


def prepare_calibration(data_root, split_path, dair_repo_root, summary_path):
    vehicle_root = data_root / "vehicle-side"
    output_dir = vehicle_root / "training" / "calib"
    frame_ids = load_vehicle_ids(split_path)
    required = {
        "vehicle-side/data_info.json": vehicle_root / "data_info.json",
        "vehicle-side/calib/camera_intrinsic": vehicle_root / "calib" / "camera_intrinsic",
        "vehicle-side/calib/lidar_to_camera": vehicle_root / "calib" / "lidar_to_camera",
    }
    missing_roots = [name for name, path in required.items() if not path.exists()]
    if missing_roots:
        raise FileNotFoundError("缺少 Vehicle Camera 原始输入: " + ", ".join(missing_roots))

    missing_calibs = []
    for frame_id in frame_ids:
        intrinsic = vehicle_root / "calib" / "camera_intrinsic" / f"{frame_id}.json"
        lidar_to_camera = vehicle_root / "calib" / "lidar_to_camera" / f"{frame_id}.json"
        if not intrinsic.is_file() or not lidar_to_camera.is_file():
            missing_calibs.append(frame_id)
    if missing_calibs:
        raise FileNotFoundError("以下样本缺少 Vehicle Camera 标定: " + ", ".join(missing_calibs))

    output_dir.mkdir(parents=True, exist_ok=True)
    official = load_official_converter_functions(dair_repo_root)
    with tempfile.TemporaryDirectory(prefix="dair_veh_imvoxelnet_calib_") as temp_dir:
        source_root = Path(temp_dir) / "source"
        intrinsic_root = source_root / "calib" / "camera_intrinsic"
        lidar_to_camera_root = source_root / "calib" / "lidar_to_camera"
        for frame_id in frame_ids:
            make_relative_symlink(
                vehicle_root / "calib" / "camera_intrinsic" / f"{frame_id}.json",
                intrinsic_root / f"{frame_id}.json",
            )
            make_relative_symlink(
                vehicle_root / "calib" / "lidar_to_camera" / f"{frame_id}.json",
                lidar_to_camera_root / f"{frame_id}.json",
            )
        official["gen_calib2kitti"](
            str(intrinsic_root), str(lidar_to_camera_root), str(output_dir)
        )

    generated = [frame_id for frame_id in frame_ids if (output_dir / f"{frame_id}.txt").is_file()]
    if len(generated) != len(frame_ids):
        raise RuntimeError(f"Vehicle KITTI 标定生成不完整: expected={len(frame_ids)}, actual={len(generated)}")
    summary = {
        "sensor": "vehicle_camera",
        "num_calibs": len(generated),
        "missing_calibs": [],
        "output_dir": str(output_dir),
        "split_path": str(split_path),
        "raw_inputs_modified": False,
        "official_converter_sources": official["source_files"],
    }
    write_json(summary_path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="使用官方标定重建 Vehicle Camera ImVoxelNet 的 KITTI calib")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--dair-repo-root", type=Path, default=DAIR_REPO_ROOT)
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=PREPROCESS_DIR / "vehicle_camera_imvoxelnet_calib_summary.json",
    )
    args = parser.parse_args()
    if args.split_path.name != "example-cooperative-split-data.json":
        parser.error("当前 Camera 小样本流程仅允许 example-cooperative-split-data.json")
    summary = prepare_calibration(
        args.data_root.resolve(),
        args.split_path.resolve(),
        args.dair_repo_root.resolve(),
        args.summary_path.resolve(),
    )
    print(f"Vehicle Camera calib: {summary['num_calibs']} -> {summary['output_dir']}")
    print(f"summary: {args.summary_path.resolve()}")


if __name__ == "__main__":
    main()
