#!/usr/bin/env python3
"""Train one DAIR-V2X Camera-only ImVoxelNet baseline with MMEngine."""

import argparse
import os
import pickle
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = Path(
    "/mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full/"
    "cooperative-vehicle-infrastructure"
)
CONFIGS = {
    "vehicle_camera": PROJECT_ROOT / "outputs/common_multimodal_dataset/configs/"
    "imvoxelnet_vehicle_common_train.py",
    "infrastructure_camera": PROJECT_ROOT / "outputs/common_multimodal_dataset/configs/"
    "imvoxelnet_infrastructure_common_train.py",
}
SENSOR_DIRS = {
    "vehicle_camera": "vehicle-side",
    "infrastructure_camera": "infrastructure-side",
}


def check_training_annotations(sensor_root: Path) -> None:
    """Reject inference-only info files before allocating GPU time."""
    failures = []
    for filename in ("mmdet3d_camera_infos_train.pkl",
                     "mmdet3d_camera_infos_full_common_val.pkl"):
        path = sensor_root / filename
        try:
            payload = pickle.loads(path.read_bytes())
            data_list = payload.get("data_list", payload)
            total_instances = sum(
                len(item.get("instances", [])) for item in data_list
            )
        except Exception as exc:
            failures.append(f"{path}: 无法读取 ({exc})")
            continue
        if filename.endswith("train.pkl") and total_instances == 0:
            failures.append(
                f"{path}: instances 总数为 0；这是推理用 info，不是训练用 info"
            )
        print(f"[Camera training preflight] {filename}: samples={len(data_list)}, "
              f"instances={total_instances}", flush=True)
    if failures:
        detail = "\n".join(f"  - {item}" for item in failures)
        raise RuntimeError(
            "Camera 训练已阻止：训练标注为空或 info 无法读取。\n"
            f"{detail}\n"
            "请先用 --with-labels 重新生成训练 info，再启动训练。"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, choices=CONFIGS)
    parser.add_argument("--data-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-train-samples", type=int,
        help="仅用于 smoke test；限制训练样本数，不用于正式训练。"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sensor_root = args.data_root / SENSOR_DIRS[args.baseline]
    config_path = CONFIGS[args.baseline]
    work_dir = args.work_dir or (PROJECT_ROOT / "outputs/full_baselines" / args.baseline / "work_dir")

    for path in (sensor_root, config_path):
        if not path.exists():
            raise FileNotFoundError(f"Required path does not exist: {path}")
    for filename in ("mmdet3d_camera_infos_train.pkl", "mmdet3d_camera_infos_full_common_val.pkl"):
        path = sensor_root / filename
        if not path.exists():
            raise FileNotFoundError(f"Required camera info file does not exist: {path}")
    check_training_annotations(sensor_root)

    os.environ["DAIR_CAMERA_DATA_ROOT"] = str(sensor_root) + "/"
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    # scipy/mmdet3d extensions must use the Conda C++ runtime, not the older
    # system libstdc++ shipped by the WSL image.
    conda_lib = Path(os.environ.get("CONDA_PREFIX", "")) / "lib"
    if (conda_lib / "libstdc++.so.6").exists():
        old_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:{old_ld}" if old_ld else str(conda_lib)

    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner

    cfg = Config.fromfile(str(config_path))
    cfg.work_dir = str(work_dir)
    cfg.launcher = "none"
    cfg.env_cfg = dict(
        cudnn_benchmark=False,
        mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
        dist_cfg=dict(backend="nccl"),
    )
    cfg.device = args.device
    if args.max_epochs is not None:
        if args.max_epochs <= 0:
            raise ValueError("--max-epochs must be positive")
        cfg.train_cfg.max_epochs = args.max_epochs
    if args.max_train_samples is not None:
        if args.max_train_samples <= 0:
            raise ValueError("--max-train-samples must be positive")
        cfg.train_dataloader.dataset.indices = list(range(args.max_train_samples))
    # KittiMetric needs the same annotation file as the validation dataset.
    val_ann_file = sensor_root / "mmdet3d_camera_infos_full_common_val.pkl"
    cfg.val_evaluator = dict(
        type="KittiMetric", ann_file=str(val_ann_file), metric="bbox"
    )
    cfg.test_evaluator = cfg.val_evaluator
    # The generated DAIR-V2X info PKL uses MMDetection3D ``classes`` metadata,
    # while this installed KittiMetric version expects KITTI ``categories``.
    # Disable the incompatible in-training evaluator; project evaluation runs
    # after training against the unified world-coordinate predictions.
    cfg.val_dataloader = None
    cfg.val_cfg = None
    cfg.val_evaluator = None
    cfg.test_dataloader = None
    cfg.test_cfg = None
    cfg.test_evaluator = None
    if args.resume:
        cfg.resume = True
    work_dir.mkdir(parents=True, exist_ok=True)

    init_default_scope(cfg.get("default_scope", "mmdet3d"))
    print(f"[Camera training] baseline={args.baseline}")
    print(f"[Camera training] config={config_path}")
    print(f"[Camera training] data_root={sensor_root}")
    print(f"[Camera training] work_dir={work_dir}")
    print("[Camera training] progress is printed below by MMEngine.", flush=True)
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
