import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from config import DATA_ROOT, PROJECT_ROOT


BASELINE_ROOT = PROJECT_ROOT / "outputs" / "baselines"
COOP_INFO_PATH = DATA_ROOT / "cooperative" / "data_info.json"

SENSOR_CONFIGS = {
    "vehicle_lidar": {
        "input_name": "predictions_vehicle_lidar.json",
        "output_dir": BASELINE_ROOT / "vehicle_lidar",
        "source_coordinate_system": "vehicle_lidar",
        "coordinate_transform": (
            "vehicle_lidar -> lidar_to_novatel -> novatel_to_world -> world"
        ),
        "center_fields": ["center_lidar", "center_vehicle_lidar", "center_sensor"],
        "box_fields": ["box_lidar", "box_vehicle_lidar", "box_sensor"],
        "corner_fields": [
            "corners_3d_lidar",
            "corners_3d_vehicle_lidar",
            "corners_3d_sensor",
        ],
    },
    "infrastructure_lidar": {
        "input_name": "predictions_infrastructure_lidar.json",
        "output_dir": BASELINE_ROOT / "infrastructure_lidar",
        "source_coordinate_system": "infrastructure_lidar",
        "coordinate_transform": (
            "infrastructure_lidar/virtuallidar -> virtuallidar_to_world -> world"
        ),
        "center_fields": [
            "center_lidar",
            "center_infrastructure_lidar",
            "center_virtuallidar",
            "center_sensor",
        ],
        "box_fields": [
            "box_lidar",
            "box_infrastructure_lidar",
            "box_virtuallidar",
            "box_sensor",
        ],
        "corner_fields": [
            "corners_3d_lidar",
            "corners_3d_infrastructure_lidar",
            "corners_3d_virtuallidar",
            "corners_3d_sensor",
        ],
    },
    # ImVoxelNet consumes an image but predicts boxes in the sensor LiDAR frame.
    # Reuse the verified LiDAR-to-world calibration chains for those boxes.
    "vehicle_camera": {
        "input_name": "predictions_vehicle_camera.json",
        "output_dir": BASELINE_ROOT / "vehicle_camera",
        "source_coordinate_system": "vehicle_lidar",
        "coordinate_transform": (
            "vehicle_camera model output (vehicle_lidar) -> lidar_to_novatel -> novatel_to_world -> world"
        ),
        "center_fields": ["center_lidar", "center_vehicle_lidar", "center_sensor"],
        "box_fields": ["box_lidar", "box_vehicle_lidar", "box_sensor"],
        "corner_fields": ["corners_3d_lidar", "corners_3d_vehicle_lidar", "corners_3d_sensor"],
    },
    "infrastructure_camera": {
        "input_name": "predictions_infrastructure_camera.json",
        "output_dir": BASELINE_ROOT / "infrastructure_camera",
        "source_coordinate_system": "infrastructure_lidar",
        "coordinate_transform": (
            "infrastructure_camera model output (infrastructure_lidar/virtuallidar) -> virtuallidar_to_world -> world"
        ),
        "center_fields": [
            "center_lidar",
            "center_infrastructure_lidar",
            "center_virtuallidar",
            "center_sensor",
        ],
        "box_fields": [
            "box_lidar",
            "box_infrastructure_lidar",
            "box_virtuallidar",
            "box_sensor",
        ],
        "corner_fields": [
            "corners_3d_lidar",
            "corners_3d_infrastructure_lidar",
            "corners_3d_virtuallidar",
            "corners_3d_sensor",
        ],
    },
}


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_predictions(path):
    predictions = load_json(path)
    if not isinstance(predictions, list):
        raise ValueError(f"预测文件顶层必须是 list: {path}")
    return predictions


def normalize_rotation(rotation):
    R = np.array(rotation, dtype=float)

    if R.shape == (3, 3):
        return R

    if R.size == 9:
        return R.reshape(3, 3)

    raise ValueError(f"rotation 格式异常: {rotation}")


def normalize_translation(translation):
    t = np.array(translation, dtype=float).reshape(-1)

    if t.size != 3:
        raise ValueError(f"translation 格式异常: {translation}")

    return t


def make_transform(calib):
    if "rotation" in calib and "translation" in calib:
        R = normalize_rotation(calib["rotation"])
        t = normalize_translation(calib["translation"])
    elif "transform" in calib:
        transform = calib["transform"]
        R = normalize_rotation(transform["rotation"])
        t = normalize_translation(transform["translation"])
    else:
        raise KeyError(f"无法解析标定字段: {list(calib.keys())}")

    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def require_calib_file(path, description):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"缺少{description}标定文件: {path}")
    return path


def build_sample_mappings():
    coop_info = load_json(COOP_INFO_PATH)
    by_vehicle_id = {}
    by_infrastructure_id = {}
    by_any_id = {}

    for item in coop_info:
        vehicle_id = Path(item.get("vehicle_pointcloud_path", "")).stem
        infrastructure_id = Path(item.get("infrastructure_pointcloud_path", "")).stem

        if not vehicle_id or not infrastructure_id:
            continue

        mapped = {
            "vehicle_id": vehicle_id,
            "infrastructure_id": infrastructure_id,
            "cooperative_label_path": str(
                DATA_ROOT / item.get("cooperative_label_path", "")
            ),
            "raw_info": item,
        }

        by_vehicle_id[vehicle_id] = mapped
        by_infrastructure_id[infrastructure_id] = mapped
        by_any_id[vehicle_id] = mapped
        by_any_id[infrastructure_id] = mapped

    return {
        "vehicle": by_vehicle_id,
        "infrastructure": by_infrastructure_id,
        "any": by_any_id,
    }


def resolve_sample_info(record, sensor, mappings):
    sample_id = str(record.get("sample_id", ""))
    source_id = str(record.get("source_id", ""))

    if not sample_id:
        raise KeyError("预测记录缺少 sample_id")

    if sensor in {"vehicle_lidar", "vehicle_camera"}:
        candidates = [sample_id, source_id, str(record.get("vehicle_id", ""))]
        for candidate in candidates:
            if candidate in mappings["vehicle"]:
                return mappings["vehicle"][candidate]

        raise KeyError(
            "sample_id 无法匹配到 cooperative/data_info.json 中的车辆端样本: "
            f"sample_id={sample_id}, source_id={source_id}"
        )

    if sensor in {"infrastructure_lidar", "infrastructure_camera"}:
        candidates = [
            source_id,
            str(record.get("infrastructure_id", "")),
            sample_id,
            str(record.get("vehicle_id", "")),
        ]
        for candidate in candidates:
            if candidate in mappings["infrastructure"]:
                return mappings["infrastructure"][candidate]
            if candidate in mappings["any"]:
                return mappings["any"][candidate]

        raise KeyError(
            "sample_id/source_id 无法匹配到 cooperative/data_info.json 中的路侧样本: "
            f"sample_id={sample_id}, source_id={source_id}"
        )

    raise ValueError(f"不支持的 sensor: {sensor}")


def get_vehicle_lidar_to_world(vehicle_id):
    lidar_to_novatel_path = require_calib_file(
        DATA_ROOT
        / "vehicle-side"
        / "calib"
        / "lidar_to_novatel"
        / f"{vehicle_id}.json",
        "vehicle_lidar -> novatel",
    )
    novatel_to_world_path = require_calib_file(
        DATA_ROOT
        / "vehicle-side"
        / "calib"
        / "novatel_to_world"
        / f"{vehicle_id}.json",
        "novatel -> world",
    )

    T_lidar_to_novatel = make_transform(load_json(lidar_to_novatel_path))
    T_novatel_to_world = make_transform(load_json(novatel_to_world_path))
    return T_novatel_to_world @ T_lidar_to_novatel


def get_infrastructure_lidar_to_world(infrastructure_id):
    calib_path = require_calib_file(
        DATA_ROOT
        / "infrastructure-side"
        / "calib"
        / "virtuallidar_to_world"
        / f"{infrastructure_id}.json",
        "infrastructure_lidar/virtuallidar -> world",
    )
    return make_transform(load_json(calib_path))


def get_lidar_to_world_transform(sensor, sample_info):
    if sensor in {"vehicle_lidar", "vehicle_camera"}:
        return get_vehicle_lidar_to_world(sample_info["vehicle_id"])

    if sensor in {"infrastructure_lidar", "infrastructure_camera"}:
        return get_infrastructure_lidar_to_world(sample_info["infrastructure_id"])

    raise ValueError(f"不支持的 sensor: {sensor}")


def transform_center(center_xyz, T_lidar_to_world):
    point = np.array([center_xyz[0], center_xyz[1], center_xyz[2], 1.0], dtype=float)
    out = T_lidar_to_world @ point
    return out[:3].tolist()


def transform_points(points_xyz, T_lidar_to_world):
    points_xyz = np.asarray(points_xyz, dtype=float)
    if points_xyz.size == 0:
        return np.zeros((0, 3), dtype=float)

    points_xyz = points_xyz.reshape((-1, 3))
    ones = np.ones((points_xyz.shape[0], 1), dtype=float)
    points_h = np.concatenate([points_xyz, ones], axis=1)
    points_w = points_h @ T_lidar_to_world.T
    return points_w[:, :3]


def build_lidar_box_bev_corners(center_lidar, dx, dy, heading):
    x, y, z = center_lidar

    x_half = dx / 2.0
    y_half = dy / 2.0

    corners = np.array([
        [x_half, y_half, z],
        [x_half, -y_half, z],
        [-x_half, -y_half, z],
        [-x_half, y_half, z],
    ], dtype=float)

    c = math.cos(heading)
    s = math.sin(heading)
    R = np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)

    rotated = corners @ R.T
    rotated[:, 0] += x
    rotated[:, 1] += y
    return rotated


def build_lidar_box_corners_3d(center_lidar, dx, dy, dz, heading):
    x, y, z = center_lidar

    x_half = dx / 2.0
    y_half = dy / 2.0
    z_half = dz / 2.0

    corners = np.array([
        [x_half, y_half, -z_half],
        [x_half, -y_half, -z_half],
        [-x_half, -y_half, -z_half],
        [-x_half, y_half, -z_half],
        [x_half, y_half, z_half],
        [x_half, -y_half, z_half],
        [-x_half, -y_half, z_half],
        [-x_half, y_half, z_half],
    ], dtype=float)

    c = math.cos(heading)
    s = math.sin(heading)
    R = np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)

    rotated = corners @ R.T
    rotated[:, 0] += x
    rotated[:, 1] += y
    rotated[:, 2] += z
    return rotated


def estimate_world_heading(center_lidar, heading_lidar, T_lidar_to_world):
    cx, cy, cz = center_lidar
    forward_lidar = [
        cx + math.cos(heading_lidar),
        cy + math.sin(heading_lidar),
        cz,
    ]

    center_world = np.array(transform_center(center_lidar, T_lidar_to_world), dtype=float)
    forward_world = np.array(
        transform_center(forward_lidar, T_lidar_to_world), dtype=float
    )
    direction = forward_world - center_world
    return float(math.atan2(direction[1], direction[0]))


def first_present(obj, fields):
    for field in fields:
        if field in obj:
            return field, obj[field]
    return None, None


def read_box_dimension(box, names, field_label):
    for name in names:
        if name in box:
            return float(box[name])
    raise KeyError(f"box 缺少 {field_label} 字段，候选字段为 {names}")


def read_box_heading(box):
    for name in ["heading", "heading_lidar", "yaw", "rotation_z", "rotation_y"]:
        if name in box:
            value = box[name]
            if isinstance(value, dict):
                return float(value.get("z", 0.0))
            return float(value)
    return 0.0


def extract_lidar_box(obj, sensor_config, sample_id, obj_idx):
    center_field, center = first_present(obj, sensor_config["center_fields"])
    box_field, box = first_present(obj, sensor_config["box_fields"])

    if center is None:
        raise KeyError(
            f"{sample_id}: pred_objects[{obj_idx}] 缺少中心点字段，"
            f"候选字段为 {sensor_config['center_fields']}"
        )
    if box is None:
        raise KeyError(
            f"{sample_id}: pred_objects[{obj_idx}] 缺少 box 字段，"
            f"候选字段为 {sensor_config['box_fields']}"
        )
    if not isinstance(box, dict):
        raise TypeError(f"{sample_id}: pred_objects[{obj_idx}].{box_field} 必须是 dict")

    center = np.asarray(center, dtype=float).reshape(-1)
    if center.size != 3:
        raise ValueError(
            f"{sample_id}: pred_objects[{obj_idx}].{center_field} 必须是长度为 3 的列表"
        )

    dx = read_box_dimension(box, ["dx", "length", "l"], "dx/length")
    dy = read_box_dimension(box, ["dy", "width", "w"], "dy/width")
    dz = read_box_dimension(box, ["dz", "height", "h"], "dz/height")
    heading = read_box_heading(box)

    return center.tolist(), dx, dy, dz, heading


def extract_or_build_3d_corners(obj, sensor_config, center, dx, dy, dz, heading):
    _, corners = first_present(obj, sensor_config["corner_fields"])
    if corners is not None:
        corners = np.asarray(corners, dtype=float)
        if corners.shape == (8, 3):
            return corners

    return build_lidar_box_corners_3d(center, dx, dy, dz, heading)


def transform_box_to_world(obj_lidar, sensor_config, sample_id, obj_idx, T_lidar_to_world):
    center_lidar, dx, dy, dz, heading_lidar = extract_lidar_box(
        obj=obj_lidar,
        sensor_config=sensor_config,
        sample_id=sample_id,
        obj_idx=obj_idx,
    )

    center_world = transform_center(center_lidar, T_lidar_to_world)

    corners_3d_lidar = extract_or_build_3d_corners(
        obj=obj_lidar,
        sensor_config=sensor_config,
        center=center_lidar,
        dx=dx,
        dy=dy,
        dz=dz,
        heading=heading_lidar,
    )
    corners_3d_world = transform_points(corners_3d_lidar, T_lidar_to_world)

    corners_bev_lidar = build_lidar_box_bev_corners(
        center_lidar=center_lidar,
        dx=dx,
        dy=dy,
        heading=heading_lidar,
    )
    corners_bev_world_3d = transform_points(corners_bev_lidar, T_lidar_to_world)
    corners_bev_world = corners_bev_world_3d[:, :2].tolist()

    heading_world = estimate_world_heading(
        center_lidar=center_lidar,
        heading_lidar=heading_lidar,
        T_lidar_to_world=T_lidar_to_world,
    )

    box_world = {
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "heading_world": heading_world,
        "corners_bev_world": corners_bev_world,
        "corners_3d_world": corners_3d_world.tolist(),
        "z_min": float(corners_3d_world[:, 2].min()),
        "z_max": float(corners_3d_world[:, 2].max()),
    }

    return center_world, box_world, corners_bev_world


def convert_prediction_object(obj_lidar, sensor_config, sample_id, obj_idx, T_lidar_to_world):
    center_world, box_world, corners_bev_world = transform_box_to_world(
        obj_lidar=obj_lidar,
        sensor_config=sensor_config,
        sample_id=sample_id,
        obj_idx=obj_idx,
        T_lidar_to_world=T_lidar_to_world,
    )

    return {
        "class": obj_lidar["class"],
        "score": float(obj_lidar.get("score", 1.0)),
        "center_world": center_world,
        "box_world": box_world,
        "corners_bev_world": corners_bev_world,
        "corners_3d_world": box_world["corners_3d_world"],
        "z_min": box_world["z_min"],
        "z_max": box_world["z_max"],
        "source": obj_lidar.get("source", "unknown"),
    }


def save_world_predictions(world_records, output_path, summary, summary_path):
    save_json(world_records, output_path)
    save_json(summary, summary_path)


def build_world_record(record, sample_info, sensor, config, pred_objects_world):
    world_record = {
        "sample_id": str(record.get("sample_id", "")),
        "vehicle_id": sample_info["vehicle_id"],
        "infrastructure_id": sample_info["infrastructure_id"],
        "coordinate_system": "world",
        "source_coordinate_system": record.get(
            "coordinate_system", config["source_coordinate_system"]
        ),
        "prediction_type": record.get("prediction_type", "unknown"),
        "sensor": record.get("sensor", sensor),
        "source_id": record.get(
            "source_id",
            sample_info["vehicle_id"]
            if sensor in {"vehicle_lidar", "vehicle_camera"}
            else sample_info["infrastructure_id"],
        ),
        "pred_objects": pred_objects_world,
    }

    if "source_file" in record:
        world_record["source_file"] = record["source_file"]

    return world_record


def default_paths(sensor):
    config = SENSOR_CONFIGS[sensor]
    output_dir = config["output_dir"]
    return {
        "input": output_dir / config["input_name"],
        "output": output_dir / "predictions_world.json",
        "summary": output_dir / "predictions_world_summary.json",
    }


def convert_lidar_predictions_to_world(sensor, input_path=None, output_path=None, summary_path=None):
    if sensor not in SENSOR_CONFIGS:
        raise ValueError(f"不支持的 sensor: {sensor}")

    config = SENSOR_CONFIGS[sensor]
    paths = default_paths(sensor)
    input_path = Path(input_path) if input_path else paths["input"]
    output_path = Path(output_path) if output_path else paths["output"]
    summary_path = Path(summary_path) if summary_path else paths["summary"]

    print("=" * 80)
    print(f"Convert {sensor} predictions to world coordinate")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print("=" * 80)

    predictions_lidar = load_predictions(input_path)
    mappings = build_sample_mappings()

    world_records = []
    total_pred_objects = 0
    converted_objects = 0
    failed_objects = 0
    missing_calib_samples = []

    for idx, record in enumerate(predictions_lidar):
        sample_id = str(record.get("sample_id", ""))
        sample_info = resolve_sample_info(record, sensor, mappings)

        pred_objects_lidar = record.get("pred_objects", [])
        if not isinstance(pred_objects_lidar, list):
            raise TypeError(f"{sample_id}: pred_objects 必须是 list")
        total_pred_objects += len(pred_objects_lidar)

        try:
            T_lidar_to_world = get_lidar_to_world_transform(sensor, sample_info)
        except FileNotFoundError as exc:
            failed_objects += len(pred_objects_lidar)
            missing_calib_record = {
                "sample_id": sample_id,
                "vehicle_id": sample_info["vehicle_id"],
                "infrastructure_id": sample_info["infrastructure_id"],
                "error": str(exc),
            }
            missing_calib_samples.append(missing_calib_record)
            world_record = build_world_record(
                record=record,
                sample_info=sample_info,
                sensor=sensor,
                config=config,
                pred_objects_world=[],
            )
            world_record["conversion_error"] = str(exc)
            world_records.append(world_record)
            print(
                f"[WARN] [{idx + 1}/{len(predictions_lidar)}] "
                f"sample_id={sample_id} 缺少标定，跳过 {len(pred_objects_lidar)} 个预测框: {exc}"
            )
            continue

        pred_objects_world = []
        for obj_idx, obj_lidar in enumerate(pred_objects_lidar):
            try:
                world_object = convert_prediction_object(
                    obj_lidar=obj_lidar,
                    sensor_config=config,
                    sample_id=sample_id,
                    obj_idx=obj_idx,
                    T_lidar_to_world=T_lidar_to_world,
                )
            except (KeyError, TypeError, ValueError) as exc:
                failed_objects += 1
                print(
                    f"[WARN] sample_id={sample_id}, pred_objects[{obj_idx}] 转换失败，"
                    f"已跳过: {exc}"
                )
                continue
            pred_objects_world.append(world_object)
            converted_objects += 1

        world_record = build_world_record(
            record=record,
            sample_info=sample_info,
            sensor=sensor,
            config=config,
            pred_objects_world=pred_objects_world,
        )

        world_records.append(world_record)

        print(
            f"[{idx + 1}/{len(predictions_lidar)}] "
            f"sample_id={sample_id}, vehicle_id={sample_info['vehicle_id']}, "
            f"infrastructure_id={sample_info['infrastructure_id']}, "
            f"objects={len(pred_objects_world)}"
        )

    summary = {
        "sensor": sensor,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "num_samples": len(predictions_lidar),
        "total_pred_objects": total_pred_objects,
        "converted_objects": converted_objects,
        "failed_objects": failed_objects,
        "missing_calib_samples": missing_calib_samples,
        "num_input_samples": len(predictions_lidar),
        "num_output_samples": len(world_records),
        "num_world_objects": converted_objects,
        "coordinate_transform": config["coordinate_transform"],
    }

    save_world_predictions(
        world_records=world_records,
        output_path=output_path,
        summary=summary,
        summary_path=summary_path,
    )

    print("=" * 80)
    print("转换完成")
    print(f"World predictions: {output_path}")
    print(f"Summary: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser(
        description="将 vehicle/infrastructure LiDAR baseline 预测框转换到 world 坐标系。"
    )
    parser.add_argument(
        "--sensor",
        required=True,
        choices=sorted(SENSOR_CONFIGS.keys()),
        help="选择输入预测所属传感器。",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="可选：覆盖默认输入 predictions_*.json 路径。",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="可选：覆盖默认输出 predictions_world.json 路径。",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="可选：覆盖默认输出 predictions_world_summary.json 路径。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        convert_lidar_predictions_to_world(
            sensor=args.sensor,
            input_path=args.input,
            output_path=args.output,
            summary_path=args.summary,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
