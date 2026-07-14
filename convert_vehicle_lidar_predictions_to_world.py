import json
import math
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
ITEM3_ROOT = PROJECT_ROOT.parent

DATA_ROOT = (
    ITEM3_ROOT
    / "DAIR-V2X"
    / "data"
    / "DAIR-V2X"
    / "cooperative-vehicle-infrastructure"
)

BASELINE_ROOT = PROJECT_ROOT / "outputs" / "baselines" / "vehicle_lidar"

PRED_LIDAR_PATH = BASELINE_ROOT / "predictions_vehicle_lidar.json"
PRED_WORLD_PATH = BASELINE_ROOT / "predictions_world.json"
SUMMARY_PATH = BASELINE_ROOT / "predictions_world_summary.json"

COOP_INFO_PATH = DATA_ROOT / "cooperative" / "data_info.json"


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


def normalize_rotation(rotation):
    """
    兼容 DAIR-V2X 标定文件中不同 rotation 写法：
    1. [[...], [...], [...]]
    2. 一维 9 个数
    """
    R = np.array(rotation, dtype=float)

    if R.shape == (3, 3):
        return R

    if R.size == 9:
        return R.reshape(3, 3)

    raise ValueError(f"rotation 格式异常: {rotation}")


def normalize_translation(translation):
    """
    兼容 translation 为：
    1. [x, y, z]
    2. [[x], [y], [z]]
    """
    t = np.array(translation, dtype=float).reshape(-1)

    if t.size != 3:
        raise ValueError(f"translation 格式异常: {translation}")

    return t


def make_transform(calib):
    """
    把标定 json 转成 4x4 齐次变换矩阵。
    """
    if "rotation" in calib and "translation" in calib:
        R = normalize_rotation(calib["rotation"])
        t = normalize_translation(calib["translation"])

    elif "transform" in calib:
        trans = calib["transform"]
        R = normalize_rotation(trans["rotation"])
        t = normalize_translation(trans["translation"])

    else:
        raise KeyError(f"无法解析标定文件字段: {calib.keys()}")

    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def get_vehicle_lidar_to_world(vehicle_id):
    """
    vehicle LiDAR -> NovAtel -> world
    """
    lidar_to_novatel_path = (
        DATA_ROOT
        / "vehicle-side"
        / "calib"
        / "lidar_to_novatel"
        / f"{vehicle_id}.json"
    )

    novatel_to_world_path = (
        DATA_ROOT
        / "vehicle-side"
        / "calib"
        / "novatel_to_world"
        / f"{vehicle_id}.json"
    )

    T_lidar_to_novatel = make_transform(load_json(lidar_to_novatel_path))
    T_novatel_to_world = make_transform(load_json(novatel_to_world_path))

    T_lidar_to_world = T_novatel_to_world @ T_lidar_to_novatel

    return T_lidar_to_world


def transform_point(point_xyz, T):
    p = np.array([point_xyz[0], point_xyz[1], point_xyz[2], 1.0], dtype=float)
    out = T @ p
    return out[:3].tolist()


def make_lidar_box_bev_corners(center_lidar, dx, dy, heading):
    """
    根据 LiDAR 坐标系下的 3D box，生成 BEV 俯视图四角点。
    这里只生成底面中心高度处的四个 xy 角点，用于 world BEV 可视化。
    """
    x, y, z = center_lidar

    x_half = dx / 2.0
    y_half = dy / 2.0

    corners = np.array([
        [ x_half,  y_half, z],
        [ x_half, -y_half, z],
        [-x_half, -y_half, z],
        [-x_half,  y_half, z],
    ], dtype=float)

    c = math.cos(heading)
    s = math.sin(heading)

    R = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)

    rotated = corners @ R.T
    rotated[:, 0] += x
    rotated[:, 1] += y

    return rotated


def make_lidar_box_corners_3d(center_lidar, dx, dy, dz, heading):
    """
    生成 vehicle LiDAR 坐标系下的 8 个 3D 角点。
    z 使用 box center +/- dz/2，和当前 custom/OpenPCDet box 定义保持一致。
    """
    x, y, z = center_lidar

    x_half = dx / 2.0
    y_half = dy / 2.0
    z_half = dz / 2.0

    corners = np.array([
        [ x_half,  y_half, -z_half],
        [ x_half, -y_half, -z_half],
        [-x_half, -y_half, -z_half],
        [-x_half,  y_half, -z_half],
        [ x_half,  y_half,  z_half],
        [ x_half, -y_half,  z_half],
        [-x_half, -y_half,  z_half],
        [-x_half,  y_half,  z_half],
    ], dtype=float)

    c = math.cos(heading)
    s = math.sin(heading)

    R = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)

    rotated = corners @ R.T
    rotated[:, 0] += x
    rotated[:, 1] += y
    rotated[:, 2] += z

    return rotated


def transform_points(points_xyz, T):
    if len(points_xyz) == 0:
        return np.zeros((0, 3), dtype=float)

    points_xyz = np.asarray(points_xyz, dtype=float)

    ones = np.ones((points_xyz.shape[0], 1), dtype=float)
    points_h = np.concatenate([points_xyz[:, :3], ones], axis=1)

    points_w = points_h @ T.T

    return points_w[:, :3]


def estimate_world_heading_from_box(center_lidar, heading_lidar, T):
    """
    用中心点 + 朝向方向点估计 world 坐标系下 yaw。
    做法：
    1. 在 LiDAR 坐标中取中心点 c
    2. 沿 heading 方向取一个前向点 f
    3. 分别转 world
    4. 用 atan2 得到 world heading
    """
    cx, cy, cz = center_lidar

    forward_lidar = [
        cx + math.cos(heading_lidar),
        cy + math.sin(heading_lidar),
        cz
    ]

    center_world = np.array(transform_point(center_lidar, T), dtype=float)
    forward_world = np.array(transform_point(forward_lidar, T), dtype=float)

    direction = forward_world - center_world

    heading_world = math.atan2(direction[1], direction[0])

    return float(heading_world)


def build_sample_mapping():
    """
    从 cooperative/data_info.json 建立：
    vehicle_id -> cooperative 信息
    """
    coop_info = load_json(COOP_INFO_PATH)
    mapping = {}

    for item in coop_info:
        vehicle_path = item.get("vehicle_pointcloud_path", "")
        vehicle_id = Path(vehicle_path).stem

        if not vehicle_id:
            continue

        mapping[vehicle_id] = {
            "vehicle_id": vehicle_id,
            "cooperative_label_path": str(DATA_ROOT / item.get("cooperative_label_path", "")),
            "raw_info": item
        }

    return mapping


def convert_one_object(obj_lidar, T_lidar_to_world):
    center_lidar = obj_lidar["center_lidar"]
    box_lidar = obj_lidar["box_lidar"]

    dx = float(box_lidar["dx"])
    dy = float(box_lidar["dy"])
    dz = float(box_lidar["dz"])
    heading_lidar = float(box_lidar["heading"])

    center_world = transform_point(center_lidar, T_lidar_to_world)

    corners_3d_lidar = obj_lidar.get("corners_3d_lidar", None)
    if corners_3d_lidar is not None:
        corners_3d_lidar = np.asarray(corners_3d_lidar, dtype=float)
        if corners_3d_lidar.shape != (8, 3):
            corners_3d_lidar = make_lidar_box_corners_3d(
                center_lidar=center_lidar,
                dx=dx,
                dy=dy,
                dz=dz,
                heading=heading_lidar
            )
    else:
        corners_3d_lidar = make_lidar_box_corners_3d(
            center_lidar=center_lidar,
            dx=dx,
            dy=dy,
            dz=dz,
            heading=heading_lidar
        )

    corners_3d_world = transform_points(corners_3d_lidar, T_lidar_to_world)

    corners_lidar = make_lidar_box_bev_corners(
        center_lidar=center_lidar,
        dx=dx,
        dy=dy,
        heading=heading_lidar
    )

    corners_world = transform_points(corners_lidar, T_lidar_to_world)
    corners_bev_world = corners_world[:, :2].tolist()

    heading_world = estimate_world_heading_from_box(
        center_lidar=center_lidar,
        heading_lidar=heading_lidar,
        T=T_lidar_to_world
    )

    return {
        "class": obj_lidar["class"],
        "score": float(obj_lidar.get("score", 1.0)),
        "center_world": center_world,
        "box_world": {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "heading_world": heading_world,
            "corners_bev_world": corners_bev_world,
            "corners_3d_world": corners_3d_world.tolist(),
            "z_min": float(corners_3d_world[:, 2].min()),
            "z_max": float(corners_3d_world[:, 2].max())
        },
        "source": obj_lidar.get("source", "unknown")
    }


def main():
    print("=" * 80)
    print("Convert vehicle LiDAR predictions to world coordinate")
    print("=" * 80)

    predictions_lidar = load_json(PRED_LIDAR_PATH)
    sample_mapping = build_sample_mapping()

    world_records = []
    skipped = []

    total_objects = 0

    for idx, record in enumerate(predictions_lidar):
        sample_id = record["sample_id"]

        if sample_id not in sample_mapping:
            skipped.append(sample_id)
            print(f"[WARN] sample_id 不在 cooperative/data_info.json 中: {sample_id}")
            continue

        vehicle_id = sample_mapping[sample_id]["vehicle_id"]

        T_lidar_to_world = get_vehicle_lidar_to_world(vehicle_id)

        pred_objects_lidar = record.get("pred_objects", [])
        pred_objects_world = []

        for obj in pred_objects_lidar:
            pred_objects_world.append(
                convert_one_object(obj, T_lidar_to_world)
            )

        total_objects += len(pred_objects_world)

        world_records.append({
            "sample_id": sample_id,
            "vehicle_id": vehicle_id,
            "coordinate_system": "world",
            "source_coordinate_system": record.get("coordinate_system", "vehicle_lidar"),
            "prediction_type": record.get("prediction_type", "unknown"),
            "sensor": record.get("sensor", "vehicle_lidar"),
            "source_id": record.get("source_id", sample_id),
            "pred_objects": pred_objects_world
        })

        print(
            f"[{idx + 1}/{len(predictions_lidar)}] "
            f"{sample_id}: objects={len(pred_objects_world)}"
        )

    save_json(world_records, PRED_WORLD_PATH)

    summary = {
        "input_path": str(PRED_LIDAR_PATH),
        "output_path": str(PRED_WORLD_PATH),
        "num_input_samples": len(predictions_lidar),
        "num_output_samples": len(world_records),
        "num_skipped_samples": len(skipped),
        "skipped_samples": skipped,
        "num_world_objects": total_objects,
        "coordinate_transform": "vehicle_lidar -> lidar_to_novatel -> novatel_to_world -> world"
    }

    save_json(summary, SUMMARY_PATH)

    print("=" * 80)
    print("转换完成")
    print(f"World predictions: {PRED_WORLD_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
