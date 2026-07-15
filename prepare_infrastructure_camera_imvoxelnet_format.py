#!/usr/bin/env python3
"""Prepare DAIR-V2X-C infrastructure camera data for MMDetection3D ImVoxelNet.

Only derived KITTI-style files are created under ``infrastructure-side/training``
and ``infrastructure-side/ImageSets``. Raw images, annotations, calibration JSON
files, and data_info.json are read-only inputs.
"""

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path

from config import DATA_ROOT, PREPROCESS_DIR


PROJECT_ROOT = Path(__file__).resolve().parent
ITEM3_ROOT = PROJECT_ROOT.parent
DAIR_REPO_ROOT = ITEM3_ROOT / "DAIR-V2X"
DEFAULT_SPLIT_PATH = DAIR_REPO_ROOT / "data" / "split_datas" / "example-cooperative-split-data.json"
DEFAULT_SUMMARY_PATH = PREPROCESS_DIR / "infrastructure_camera_imvoxelnet_format_summary.json"


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


def mkdir_p(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_files_path(directory, extension=".json"):
    return [str(path) for path in Path(directory).rglob(f"*{extension}") if path.is_file()]


def write_txt(path, content):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content, encoding="utf-8")


def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载官方转换模块: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_official_converter_functions(dair_repo_root):
    """Load official functions without importing its unrelated open3d PCD helper.

    Official gen_kitti modules import ``tools.dataset_converter.utils``. That
    utility imports open3d even for camera-only conversion. Injecting the four
    filesystem helpers these modules require preserves the official conversion
    implementation while avoiding an unnecessary point-cloud dependency.
    """
    converter_root = dair_repo_root / "tools" / "dataset_converter"
    gen_kitti_root = converter_root / "gen_kitti"
    required_files = [
        gen_kitti_root / "label_lidarcoord_to_cameracoord.py",
        gen_kitti_root / "label_json2kitti.py",
        gen_kitti_root / "gen_calib2kitti.py",
        gen_kitti_root / "gen_ImageSets_from_split_data.py",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少 DAIR-V2X 官方转换脚本:\n" + "\n".join(missing))

    compat_utils = types.ModuleType("tools.dataset_converter.utils")
    compat_utils.mkdir_p = mkdir_p
    compat_utils.read_json = read_json
    compat_utils.get_files_path = get_files_path
    compat_utils.write_txt = write_txt
    sys.modules["tools.dataset_converter.utils"] = compat_utils

    label_transform = load_module("official_label_lidarcoord_to_cameracoord", required_files[0])
    label_writer = load_module("official_label_json2kitti", required_files[1])
    calib_writer = load_module("official_gen_calib2kitti", required_files[2])
    imageset_writer = load_module("official_gen_imagesets", required_files[3])
    return {
        "gen_lidar2cam": label_transform.gen_lidar2cam,
        "json2kitti": label_writer.json2kitti,
        "rewrite_label": label_writer.rewrite_label,
        "label_filter": label_writer.label_filter,
        "gen_calib2kitti": calib_writer.gen_calib2kitti,
        "gen_imagesets": imageset_writer.gen_ImageSet_from_split_data,
        "source_files": [str(path) for path in required_files],
    }


def parse_example_split(split_path):
    split_data = read_json(split_path)
    infrastructure_split = split_data.get("infrastructure_split")
    if not isinstance(infrastructure_split, dict):
        raise ValueError(f"split 文件缺少 infrastructure_split: {split_path}")

    train_ids = infrastructure_split.get("train")
    val_ids = infrastructure_split.get("val")
    if not isinstance(train_ids, list) or not isinstance(val_ids, list):
        raise ValueError(f"infrastructure_split 必须同时包含 train 和 val: {split_path}")

    all_ids = train_ids + val_ids
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("example split 中 infrastructure train/val 存在重复 ID")
    return train_ids, val_ids, all_ids


def validate_inputs(data_root, split_path, all_ids):
    infrastructure_root = data_root / "infrastructure-side"
    required_paths = {
        "infrastructure-side/image": infrastructure_root / "image",
        "infrastructure-side/label": infrastructure_root / "label",
        "infrastructure-side/calib": infrastructure_root / "calib",
        "cooperative/data_info.json": data_root / "cooperative" / "data_info.json",
        "example-cooperative-split-data.json": split_path,
        "infrastructure-side/label/camera": infrastructure_root / "label" / "camera",
        "infrastructure-side/calib/camera_intrinsic": infrastructure_root / "calib" / "camera_intrinsic",
        "infrastructure-side/calib/virtuallidar_to_camera": infrastructure_root
        / "calib"
        / "virtuallidar_to_camera",
        "infrastructure-side/data_info.json": infrastructure_root / "data_info.json",
    }
    missing_paths = [f"{name}: {path}" for name, path in required_paths.items() if not path.exists()]
    if missing_paths:
        raise FileNotFoundError("生成前检查失败，缺少输入路径:\n" + "\n".join(missing_paths))

    cooperative_info = read_json(required_paths["cooperative/data_info.json"])
    cooperative_infra_ids = {
        Path(item["infrastructure_image_path"]).stem
        for item in cooperative_info
        if item.get("infrastructure_image_path")
    }
    missing_cooperative_ids = sorted(set(all_ids) - cooperative_infra_ids)
    if missing_cooperative_ids:
        raise ValueError(
            "example infrastructure split 中以下 ID 不在 cooperative/data_info.json: "
            + ", ".join(missing_cooperative_ids)
        )

    missing_images = []
    missing_labels = []
    missing_calibs = []
    for frame_id in all_ids:
        if not (infrastructure_root / "image" / f"{frame_id}.jpg").is_file():
            missing_images.append(frame_id)
        if not (infrastructure_root / "label" / "camera" / f"{frame_id}.json").is_file():
            missing_labels.append(frame_id)
        intrinsic = infrastructure_root / "calib" / "camera_intrinsic" / f"{frame_id}.json"
        lidar_to_camera = infrastructure_root / "calib" / "virtuallidar_to_camera" / f"{frame_id}.json"
        if not intrinsic.is_file() or not lidar_to_camera.is_file():
            missing_calibs.append(frame_id)

    return infrastructure_root, missing_images, missing_labels, missing_calibs


def make_relative_symlink(source, destination):
    source = Path(source).resolve()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        destination.unlink()
    elif destination.exists():
        # Existing regular files can come from a previous official conversion.
        # Keep them only when they are already an exact copy of the raw image.
        if source.is_file() and destination.is_file() and source.stat().st_size == destination.stat().st_size:
            return
        raise FileExistsError(f"拒绝覆盖已有派生文件: {destination}")

    relative_source = os.path.relpath(source, start=destination.parent)
    destination.symlink_to(relative_source)


def build_subset_source(temp_root, infrastructure_root, selected_ids):
    """Create a temporary subset source accepted by official converter functions."""
    temp_root = Path(temp_root)
    source_info = read_json(infrastructure_root / "data_info.json")
    selected_set = set(selected_ids)
    subset_info = [item for item in source_info if Path(item["image_path"]).stem in selected_set]
    if len(subset_info) != len(selected_ids):
        found_ids = {Path(item["image_path"]).stem for item in subset_info}
        absent_ids = sorted(selected_set - found_ids)
        raise ValueError("infrastructure-side/data_info.json 缺少 split 样本: " + ", ".join(absent_ids))
    write_json(temp_root / "data_info.json", subset_info)

    for frame_id in selected_ids:
        for relative_path in [
            Path("label/camera") / f"{frame_id}.json",
            Path("calib/camera_intrinsic") / f"{frame_id}.json",
            Path("calib/virtuallidar_to_camera") / f"{frame_id}.json",
        ]:
            make_relative_symlink(infrastructure_root / relative_path, temp_root / relative_path)


def remove_stale_files(directory, expected_names, suffix):
    directory = Path(directory)
    expected_names = set(expected_names)
    for path in directory.glob(f"*{suffix}"):
        if path.name not in expected_names:
            path.unlink()


def normalize_official_rotation_fields(label_root):
    """Adapt temporary labels to the official writer's ``eval(rotation)`` API."""
    for label_path in Path(label_root).glob("*.json"):
        labels = read_json(label_path)
        changed = False
        for label in labels:
            rotation = label.get("rotation")
            if isinstance(rotation, (int, float)):
                label["rotation"] = str(rotation)
                changed = True
        if changed:
            write_json(label_path, labels)


def prepare_format(data_root, split_path, summary_path, dair_repo_root):
    train_ids, val_ids, all_ids = parse_example_split(split_path)
    infrastructure_root, missing_images, missing_labels, missing_calibs = validate_inputs(
        data_root, split_path, all_ids
    )

    summary = {
        "data_root": str(data_root),
        "split_path": str(split_path),
        "split_name": "example-cooperative-split-data.json",
        "sensor": "infrastructure_camera",
        "format": "MMDetection3D / KITTI-style for ImVoxelNet",
        "num_images": 0,
        "num_labels": 0,
        "num_calibs": 0,
        "num_imagesets": 0,
        "missing_images": missing_images,
        "missing_labels": missing_labels,
        "missing_calibs": missing_calibs,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "raw_inputs_modified": False,
    }
    if missing_images or missing_labels or missing_calibs:
        write_json(summary_path, summary)
        raise RuntimeError(
            "example split 存在缺失输入，未生成不完整格式。"
            f" images={len(missing_images)}, labels={len(missing_labels)}, calibs={len(missing_calibs)}"
        )

    training_root = infrastructure_root / "training"
    image_output = training_root / "image_2"
    label_output = training_root / "label_2"
    calib_output = training_root / "calib"
    imagesets_output = infrastructure_root / "ImageSets"
    for path in [image_output, label_output, calib_output, imagesets_output]:
        path.mkdir(parents=True, exist_ok=True)

    expected_image_names = [f"{frame_id}.jpg" for frame_id in all_ids]
    expected_label_names = [f"{frame_id}.txt" for frame_id in all_ids]
    expected_calib_names = [f"{frame_id}.txt" for frame_id in all_ids]
    remove_stale_files(image_output, expected_image_names, ".jpg")
    remove_stale_files(label_output, expected_label_names, ".txt")
    remove_stale_files(calib_output, expected_calib_names, ".txt")

    for frame_id in all_ids:
        make_relative_symlink(
            infrastructure_root / "image" / f"{frame_id}.jpg",
            image_output / f"{frame_id}.jpg",
        )

    official = load_official_converter_functions(dair_repo_root)
    with tempfile.TemporaryDirectory(prefix="dair_inf_imvoxelnet_") as temp_dir:
        temp_root = Path(temp_dir)
        official_source = temp_root / "source"
        official_labels = temp_root / "camera_labels"
        build_subset_source(official_source, infrastructure_root, all_ids)

        official["gen_lidar2cam"](str(official_source), str(official_labels), label_type="camera")
        normalize_official_rotation_fields(official_labels / "label" / "camera")
        official["json2kitti"](str(official_labels / "label" / "camera"), str(label_output))
        official["rewrite_label"](str(label_output))
        official["label_filter"](str(label_output))
        official["gen_calib2kitti"](
            str(official_source / "calib" / "camera_intrinsic"),
            str(official_source / "calib" / "virtuallidar_to_camera"),
            str(calib_output),
        )

    # This directly reuses the official parser for infrastructure_split.
    official["gen_imagesets"](str(imagesets_output), str(split_path), sensor_view="infrastructure")

    summary.update(
        {
            "num_images": len(list(image_output.glob("*.jpg"))),
            "num_labels": len(list(label_output.glob("*.txt"))),
            "num_calibs": len(list(calib_output.glob("*.txt"))),
            "num_imagesets": len(list(imagesets_output.glob("*.txt"))),
            "training_root": str(training_root),
            "image_2_path": str(image_output),
            "label_2_path": str(label_output),
            "calib_path": str(calib_output),
            "imagesets_path": str(imagesets_output),
            "official_converter_sources": official["source_files"],
        }
    )
    write_json(summary_path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="补齐 Infrastructure Camera ImVoxelNet 的 KITTI-style 目录")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="cooperative-vehicle-infrastructure 根目录")
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH, help="必须使用 example cooperative split")
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH, help="生成摘要 JSON 路径")
    parser.add_argument("--dair-repo-root", type=Path, default=DAIR_REPO_ROOT, help="DAIR-V2X 官方仓库根目录")
    args = parser.parse_args()

    if args.split_path.name != "example-cooperative-split-data.json":
        parser.error("Day2 Step1 仅允许使用 example-cooperative-split-data.json")

    summary = prepare_format(
        args.data_root.resolve(),
        args.split_path.resolve(),
        args.summary_path.resolve(),
        args.dair_repo_root.resolve(),
    )
    print("=" * 80)
    print("Infrastructure Camera ImVoxelNet KITTI-style 格式生成完成")
    for key in ["num_images", "num_labels", "num_calibs", "num_imagesets"]:
        print(f"{key}: {summary[key]}")
    for key in ["missing_images", "missing_labels", "missing_calibs"]:
        print(f"{key}: {len(summary[key])} {summary[key]}")
    print(f"training_root: {summary['training_root']}")
    print(f"summary_path: {args.summary_path.resolve()}")


if __name__ == "__main__":
    main()
