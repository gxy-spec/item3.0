import sys
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import PREPROCESS_DIR, VISUAL_DIR


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_pcd(path):
    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.points)


def make_transform(rotation, translation):
    """
    构造 4x4 齐次变换矩阵。
    rotation: 3x3
    translation: 3 或 3x1
    """
    T = np.eye(4)
    T[:3, :3] = np.array(rotation, dtype=float)
    T[:3, 3] = np.array(translation, dtype=float).reshape(3)
    return T


def extract_rt(calib_data):
    """
    尽量兼容 DAIR-V2X 标定 json 的不同写法。
    常见情况：
    1. {"rotation": [...], "translation": [...]}
    2. {"transform": {"rotation": [...], "translation": [...]}}
    """
    if "rotation" in calib_data and "translation" in calib_data:
        return calib_data["rotation"], calib_data["translation"]

    if "transform" in calib_data:
        trans = calib_data["transform"]
        if "rotation" in trans and "translation" in trans:
            return trans["rotation"], trans["translation"]

    raise KeyError(f"无法从标定文件中解析 rotation 和 translation，当前字段为：{list(calib_data.keys())}")


def transform_points(points, T):
    """
    points: N x 3
    T: 4 x 4
    """
    if points.shape[0] == 0:
        return points

    ones = np.ones((points.shape[0], 1))
    pts_h = np.concatenate([points[:, :3], ones], axis=1)
    pts_trans = pts_h @ T.T
    return pts_trans[:, :3]


def get_vehicle_lidar_to_world(item):
    lidar_to_novatel = load_json(item["vehicle_calib"]["lidar_to_novatel"])
    novatel_to_world = load_json(item["vehicle_calib"]["novatel_to_world"])

    R1, t1 = extract_rt(lidar_to_novatel)
    R2, t2 = extract_rt(novatel_to_world)

    T_lidar_to_novatel = make_transform(R1, t1)
    T_novatel_to_world = make_transform(R2, t2)

    T_lidar_to_world = T_novatel_to_world @ T_lidar_to_novatel
    return T_lidar_to_world


def get_infra_lidar_to_world(item):
    virtuallidar_to_world = load_json(item["infrastructure_calib"]["virtuallidar_to_world"])
    R, t = extract_rt(virtuallidar_to_world)
    T = make_transform(R, t)
    return T


def parse_box(label):
    """
    解析 label_world 里的 3D 框。
    不同版本字段可能略有差异，这里做常见兼容。
    """
    cls_name = label.get("type", label.get("class", "Unknown"))

    loc = label.get("3d_location", label.get("location", None))
    dim = label.get("3d_dimensions", label.get("dimensions", None))
    rot = label.get("rotation", label.get("rotation_y", 0.0))

    if loc is None or dim is None:
        return None

    x = float(loc["x"])
    y = float(loc["y"])
    z = float(loc.get("z", 0.0))

    # DAIR 常见字段：h, w, l
    # BEV 画框主要用 length 和 width
    length = float(dim.get("l", dim.get("length", 0.0)))
    width = float(dim.get("w", dim.get("width", 0.0)))
    height = float(dim.get("h", dim.get("height", 0.0)))

    if isinstance(rot, dict):
        yaw = float(rot.get("z", 0.0))
    else:
        yaw = float(rot)

    return {
        "class": cls_name,
        "center": np.array([x, y, z], dtype=float),
        "length": length,
        "width": width,
        "height": height,
        "yaw": yaw,
    }


def bev_corners(center_xy, length, width, yaw):
    """
    根据中心点、长宽、朝向角，计算 BEV 四个角点。
    """
    l2 = length / 2.0
    w2 = width / 2.0

    corners = np.array([
        [ l2,  w2],
        [ l2, -w2],
        [-l2, -w2],
        [-l2,  w2],
    ])

    c = np.cos(yaw)
    s = np.sin(yaw)

    R = np.array([
        [c, -s],
        [s,  c],
    ])

    rotated = corners @ R.T
    return rotated + center_xy.reshape(1, 2)


def plot_world_bev(points_world, boxes, title, save_path, max_points=60000):
    """
    将 world 坐标下的点云和 label_world 标注框画在一张 BEV 图上。
    为了避免 world 坐标数字太大，绘图时减去一个 origin。
    """
    if points_world.shape[0] > max_points:
        idx = np.random.choice(points_world.shape[0], max_points, replace=False)
        points_world = points_world[idx]

    # 优先用点云中心作为可视化原点
    origin = np.mean(points_world[:, :2], axis=0)

    pts_xy = points_world[:, :2] - origin

    plt.figure(figsize=(8, 8))
    plt.scatter(pts_xy[:, 0], pts_xy[:, 1], s=0.1)

    for box in boxes:
        corners = bev_corners(
            box["center"][:2],
            box["length"],
            box["width"],
            box["yaw"]
        )
        corners_plot = corners - origin

        poly = Polygon(corners_plot, closed=True, fill=False, linewidth=1.2)
        plt.gca().add_patch(poly)

        center_plot = box["center"][:2] - origin
        plt.text(center_plot[0], center_plot[1], box["class"], fontsize=6)

    plt.title(title)
    plt.xlabel("x in world coordinate, shifted")
    plt.ylabel("y in world coordinate, shifted")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def main():
    index_path = PREPROCESS_DIR / "dair_v2x_pair_index.json"
    pair_index = load_json(index_path)

    num_vis = min(20, len(pair_index))

    for i in range(num_vis):
        item = pair_index[i]

        vehicle_points = read_pcd(item["vehicle_pointcloud"])
        infra_points = read_pcd(item["infrastructure_pointcloud"])

        T_vehicle = get_vehicle_lidar_to_world(item)
        T_infra = get_infra_lidar_to_world(item)

        vehicle_world = transform_points(vehicle_points, T_vehicle)
        infra_world = transform_points(infra_points, T_infra)

        labels = load_json(item["cooperative_label"])
        boxes = []

        for label in labels:
            box = parse_box(label)
            if box is not None:
                boxes.append(box)

        print(f"sample {i}: labels={len(labels)}, parsed_boxes={len(boxes)}")

        save_vehicle = VISUAL_DIR / f"sample_{i:03d}_vehicle_world_bev_boxes.png"
        save_infra = VISUAL_DIR / f"sample_{i:03d}_infra_world_bev_boxes.png"

        plot_world_bev(
            vehicle_world,
            boxes,
            f"Vehicle LiDAR in World + Label Boxes | sample {i}",
            save_vehicle
        )

        plot_world_bev(
            infra_world,
            boxes,
            f"Infrastructure LiDAR in World + Label Boxes | sample {i}",
            save_infra
        )

    print(f"完成 {num_vis} 组 BEV + 3D框 可视化，结果保存在：{VISUAL_DIR}")


if __name__ == "__main__":
    main()