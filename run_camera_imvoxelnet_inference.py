#!/usr/bin/env python3
"""Run DAIR-V2X ImVoxelNet checkpoints with MMDetection3D 1.x on one camera."""

import argparse
import json
import os
import pickle
import tempfile
from collections import Counter
from pathlib import Path

from config import DATA_ROOT, PROJECT_ROOT


ITEM3_ROOT = PROJECT_ROOT.parent
DAIR_ROOT = ITEM3_ROOT / "DAIR-V2X"
CONFIG_PATH = PROJECT_ROOT / "configs" / "imvoxelnet_mmdet3d_1x_camera.py"
COOPERATIVE_INFO_PATH = DATA_ROOT / "cooperative" / "data_info.json"

SENSORS = {
    "vehicle_camera": {
        "data_dir": "vehicle-side",
        "checkpoint": DAIR_ROOT / "configs" / "vic3d" / "late-fusion-image" / "imvoxelnet" / "vic3d_latefusion_veh_imvoxelnet_9d0ad4d4930c41d62839d45c06f86326.pth",
        "output_dir": PROJECT_ROOT / "outputs" / "baselines" / "vehicle_camera",
        "prediction_file": "predictions_vehicle_camera.json",
        "source_coordinate_system": "vehicle_lidar",
    },
    "infrastructure_camera": {
        "data_dir": "infrastructure-side",
        "checkpoint": DAIR_ROOT / "configs" / "vic3d" / "late-fusion-image" / "imvoxelnet" / "vic3d_latefusion_inf_imvoxelnet_973cefc0b2c14fee1b8775aa996ac779.pth",
        "output_dir": PROJECT_ROOT / "outputs" / "baselines" / "infrastructure_camera",
        "prediction_file": "predictions_infrastructure_camera.json",
        "source_coordinate_system": "infrastructure_lidar",
    },
}


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


def load_info(path):
    with Path(path).open("rb") as file:
        value = pickle.load(file)
    if not isinstance(value, dict) or not isinstance(value.get("data_list"), list):
        raise ValueError(f"MMD3D info 格式错误: {path}")
    return value


def make_cooperative_mappings():
    by_infrastructure = {}
    for item in load_json(COOPERATIVE_INFO_PATH):
        vehicle_id = Path(item.get("vehicle_pointcloud_path", "")).stem
        infrastructure_id = Path(item.get("infrastructure_pointcloud_path", "")).stem
        if vehicle_id and infrastructure_id:
            by_infrastructure[infrastructure_id] = vehicle_id
    return by_infrastructure


def write_single_info(info, path):
    with Path(path).open("wb") as file:
        pickle.dump({"metainfo": info.get("metainfo", {}), "data_list": [info["entry"]]}, file)


def prediction_record(sensor, frame_id, instances, class_names, infra_to_vehicle):
    config = SENSORS[sensor]
    if sensor == "vehicle_camera":
        sample_id = frame_id
        source_id = frame_id
        identifiers = {"vehicle_id": frame_id}
    else:
        if frame_id not in infra_to_vehicle:
            raise KeyError(f"infrastructure_id 无法匹配 cooperative/data_info.json: {frame_id}")
        sample_id = infra_to_vehicle[frame_id]
        source_id = frame_id
        identifiers = {"vehicle_id": sample_id, "infrastructure_id": frame_id}

    boxes = instances.bboxes_3d.tensor.detach().cpu().tolist()
    scores = instances.scores_3d.detach().cpu().tolist()
    labels = instances.labels_3d.detach().cpu().tolist()
    pred_objects = []
    for box, score, label in zip(boxes, scores, labels):
        if len(box) < 7:
            raise ValueError(f"{frame_id}: ImVoxelNet 预测框维度小于 7: {box}")
        label = int(label)
        class_name = class_names[label] if 0 <= label < len(class_names) else f"unknown_{label}"
        pred_objects.append(
            {
                "class": class_name,
                "score": float(score),
                "center_lidar": [float(value) for value in box[:3]],
                "box_lidar": {
                    "dx": float(box[3]),
                    "dy": float(box[4]),
                    "dz": float(box[5]),
                    "heading": float(box[6]),
                },
                "source": "mmdet3d_1x_imvoxelnet_camera_only",
            }
        )

    return {
        "sample_id": sample_id,
        "sensor": sensor,
        "source_id": source_id,
        **identifiers,
        "coordinate_system": config["source_coordinate_system"],
        "box_coordinate_system": config["source_coordinate_system"],
        "prediction_type": f"mmdet3d_1x_imvoxelnet_{sensor}",
        "pred_objects": pred_objects,
    }


def run(args):
    config = SENSORS[args.sensor]
    sensor_root = DATA_ROOT / config["data_dir"]
    info_path = sensor_root / f"mmdet3d_camera_infos_{args.split}.pkl"
    checkpoint = Path(args.checkpoint) if args.checkpoint else config["checkpoint"]
    output_path = Path(args.output) if args.output else config["output_dir"] / config["prediction_file"]
    summary_path = output_path.with_name(output_path.stem + "_summary.json")

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"缺少 MMD3D 配置: {CONFIG_PATH}")
    if not info_path.is_file():
        raise FileNotFoundError(
            f"缺少推理 info: {info_path}\n"
            f"请先运行: python scripts/build_camera_imvoxelnet_infos.py --sensor {args.sensor}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"缺少 ImVoxelNet checkpoint: {checkpoint}")

    info = load_info(info_path)
    entries = info["data_list"][: args.max_samples] if args.max_samples else info["data_list"]
    if not entries:
        raise ValueError(f"{info_path} 中没有可推理样本")

    os.environ["DAIR_CAMERA_DATA_ROOT"] = str(sensor_root)
    try:
        from mmdet3d.apis import inference_mono_3d_detector, init_model
    except ImportError as exc:
        raise RuntimeError(
            "无法导入 MMDetection3D。请激活 dair-camera-5060 环境后重试。"
        ) from exc

    device = args.device
    model = init_model(str(CONFIG_PATH), str(checkpoint), device=device)
    class_names = list(model.dataset_meta.get("classes", ["Car"]))
    infra_to_vehicle = make_cooperative_mappings()
    records = []

    with tempfile.TemporaryDirectory(prefix="dair_camera_imvoxelnet_") as temp_dir:
        single_info_path = Path(temp_dir) / "single_info.pkl"
        for index, entry in enumerate(entries, start=1):
            frame_id = str(entry.get("sample_idx", ""))
            image_path = Path(entry["images"]["CAM2"]["img_path"])
            if not frame_id or not image_path.is_file():
                raise FileNotFoundError(f"样本 image 或 frame_id 无效: {entry}")
            write_single_info({"metainfo": info.get("metainfo", {}), "entry": entry}, single_info_path)
            result = inference_mono_3d_detector(
                model, str(image_path), str(single_info_path), cam_type="CAM2"
            )
            instances = result.pred_instances_3d
            keep = instances.scores_3d >= args.score_threshold
            instances = instances[keep]
            record = prediction_record(args.sensor, frame_id, instances, class_names, infra_to_vehicle)
            records.append(record)
            print(f"[{index}/{len(entries)}] frame_id={frame_id}, predictions={len(record['pred_objects'])}")

    class_distribution = Counter(
        obj["class"] for record in records for obj in record["pred_objects"]
    )
    save_json(records, output_path)
    save_json(
        {
            "sensor": args.sensor,
            "prediction_type": records[0]["prediction_type"],
            "checkpoint": str(checkpoint),
            "config": str(CONFIG_PATH),
            "info_path": str(info_path),
            "num_samples": len(records),
            "total_pred_objects": sum(len(record["pred_objects"]) for record in records),
            "empty_samples": [record["sample_id"] for record in records if not record["pred_objects"]],
            "class_distribution": dict(sorted(class_distribution.items())),
            "score_threshold": args.score_threshold,
            "device": device,
            "output_path": str(output_path),
        },
        summary_path,
    )
    print(f"predictions: {output_path}")
    print(f"summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="运行 Vehicle/Infrastructure Camera-only ImVoxelNet 小样本推理")
    parser.add_argument("--sensor", choices=sorted(SENSORS), required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--max-samples", type=int, default=0, help="0 表示运行该 split 的全部样本")
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", help="可选，自定义 checkpoint 路径")
    parser.add_argument("--output", help="可选，自定义 predictions JSON 路径")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
