#!/usr/bin/env python3
"""Build OpenPCDet info PKL and a runtime config for one Full Dataset LiDAR split."""

import argparse
import pickle
import sys
from pathlib import Path

import yaml
from easydict import EasyDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENPCDET_ROOT = PROJECT_ROOT.parent / "OpenPCDet"


def sensor_defaults(sensor, opencdet_root):
    if sensor == "vehicle_lidar":
        return (
            opencdet_root / "tools" / "cfgs" / "dataset_configs" / "custom_dataset.yaml",
            opencdet_root / "tools" / "cfgs" / "custom_models" / "pointpillar_custom_vehicle.yaml",
        )
    return (
        opencdet_root / "tools" / "cfgs" / "dataset_configs" / "custom_infrastructure_dataset.yaml",
        opencdet_root / "tools" / "cfgs" / "custom_models" / "pointpillar_custom_infrastructure.yaml",
    )


def main():
    parser = argparse.ArgumentParser(description="生成 Full Dataset OpenPCDet infos 与不覆盖原配置的运行时 YAML")
    parser.add_argument("--sensor", choices=["vehicle_lidar", "infrastructure_lidar"], required=True)
    parser.add_argument("--data-path", type=Path, required=True, help="prepare_full_opencdet_lidar.py 的 target-root")
    parser.add_argument("--split", default="val", help="CustomDataset ImageSets 名，例如 val 或 full_common_val")
    parser.add_argument("--opencdet-root", type=Path, default=DEFAULT_OPENPCDET_ROOT)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument(
        "--for-training", action="store_true",
        help="生成训练配置：train 使用当前 split info，test 使用 --val-info-path 指定的验证 info。",
    )
    parser.add_argument("--val-info-path", type=Path, help="训练配置使用的验证 info pkl。")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 OpenPCDet OPTIMIZATION.NUM_EPOCHS。")
    parser.add_argument("--val-split", default="full_common_val", help="训练配置使用的验证 split 名称。")
    args = parser.parse_args()
    opencdet_root, data_path = args.opencdet_root.resolve(), args.data_path.resolve()
    dataset_cfg_path, model_cfg_path = sensor_defaults(args.sensor, opencdet_root)
    if not (data_path / "ImageSets" / f"{args.split}.txt").is_file():
        raise FileNotFoundError(f"缺少 CustomDataset split: {data_path / 'ImageSets' / f'{args.split}.txt'}")
    sys.path.insert(0, str(opencdet_root))
    from pcdet.datasets.custom.custom_dataset import CustomDataset
    from pcdet.utils import common_utils

    dataset_cfg = EasyDict(yaml.safe_load(dataset_cfg_path.read_text(encoding="utf-8")))
    dataset_cfg.DATA_PATH = str(data_path)
    dataset_cfg.DATA_SPLIT = {"train": "train", "test": args.split}
    dataset_cfg.INFO_PATH = {"train": [], "test": []}
    logger = common_utils.create_logger()
    dataset = CustomDataset(
        dataset_cfg=dataset_cfg,
        class_names=list(dataset_cfg.CLASS_NAMES),
        root_path=data_path,
        training=False,
        logger=logger,
    )
    dataset.set_split(args.split)
    infos = dataset.get_infos(
        class_names=list(dataset_cfg.CLASS_NAMES),
        num_workers=4,
        has_label=True,
        num_features=len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list),
    )
    info_path = data_path / f"custom_infos_{args.split}.pkl"
    with info_path.open("wb") as file:
        pickle.dump(infos, file, protocol=pickle.HIGHEST_PROTOCOL)

    model_cfg = yaml.safe_load(model_cfg_path.read_text(encoding="utf-8"))
    data_cfg = model_cfg.setdefault("DATA_CONFIG", {})
    data_cfg["_BASE_CONFIG_"] = str(dataset_cfg_path)
    data_cfg["DATA_PATH"] = str(data_path)
    data_cfg["DATA_SPLIT"] = {"train": "train", "test": args.split}
    if args.for_training:
        if not args.val_info_path:
            raise ValueError("--for-training 必须同时指定 --val-info-path")
        val_info_path = args.val_info_path.resolve()
        if not val_info_path.is_file():
            raise FileNotFoundError(f"缺少验证 info: {val_info_path}")
        data_cfg["INFO_PATH"] = {"train": [info_path.name], "test": [str(val_info_path)]}
        data_cfg["DATA_SPLIT"] = {"train": args.split, "test": args.val_split}
    else:
        data_cfg["INFO_PATH"] = {"train": [], "test": [info_path.name]}
    if args.epochs is not None:
        model_cfg.setdefault("OPTIMIZATION", {})["NUM_EPOCHS"] = args.epochs
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(yaml.safe_dump(model_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"infos: {info_path} ({len(infos)} samples)")
    print(f"runtime config: {args.output_config.resolve()}")


if __name__ == "__main__":
    main()
