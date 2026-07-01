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
    pts = np.asarray(pcd.points)
    return pts


def make_transform(rotation, translation):
    T = np.eye(4)
    T[:3, :3] = np.array(rotation, dtype=float)
    T[:3, 3] = np.array(translation, dtype=float).reshape(3)
    return T


def extract_rt(calib_data):
    """
    兼容 DAIR-V2X 中常见标定格式。
    """
    if "rotation" in calib_data and "translation" in calib_data:
        return calib_data["rotation"], calib_data["translation"]

    if "transform" in calib_data:
        transform = calib_data["transform"]
        if "rotation" in transform and "translation" in transform:
            return transform["rotation"], transform["translation"]

    raise KeyError(f"无法解析标定文件，字段为：{list(calib_data.keys())}")


def transform_points(points, T):
    if points.shape[0] == 0:
        return points

    ones = np.ones((points.shape[0], 1))
    points_h = np.concatenate([points[:, :3], ones], axis=1)
    points_world = points_h @ T.T
    return points_world[:, :3]


def get_vehicle_lidar_to_world(item):
    lidar_to_novatel = load_json(item["vehicle_calib"]["lidar_to_novatel"])
    novatel_to_world = load_json(item["vehicle_calib"]["novatel_to_world"])

    R1, t1 = extract_rt(lidar_to_novatel)
    R2, t2 = extract_rt(novatel_to_world)

    T_lidar_to_novatel = make_transform(R1, t1)
    T_novatel_to_world = make_transform(R2, t2)

    return T_novatel_to_world @ T_lidar_to_novatel


def get_infra_lidar_to_world(item):
    virtuallidar_to_world = load_json(item["infrastructure_calib"]["virtuallidar_to_world"])
    R, t = extract_rt(virtuallidar_to_world)
    return make_transform(R, t)


def parse_box(label):
    cls_name = label.get("type", label.get("class", "Unknown"))

    loc = label.get("3d_location", label.get("location", None))
    dim = label.get("3d_dimensions", label.get("dimensions", None))
    rot = label.get("rotation", label.get("rotation_y", 0.0))

    if loc is None or dim is None:
        return None

    center = np.array([
        float(loc["x"]),
        float(loc["y"]),
        float(loc.get("z", 0.0))
    ], dtype=float)

    length = float(dim.get("l", dim.get("length", 0.0)))
    width = float(dim.get("w", dim.get("width", 0.0)))
    height = float(dim.get("h", dim.get("height", 0.0)))

    if isinstance(rot, dict):
        yaw = float(rot.get("z", 0.0))
    else:
        yaw = float(rot)

    return {
        "class": cls_name,
        "center": center,
        "length": length,
        "width": width,
        "height": height,
        "yaw": yaw
    }


def bev_corners(center_xy, length, width, yaw):
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

    return corners @ R.T + center_xy.reshape(1, 2)


def sample_points(points, max_points=60000):
    if points.shape[0] <= max_points:
        return points

    idx = np.random.choice(points.shape[0], max_points, replace=False)
    return points[idx]


def plot_fused_world_bev(vehicle_world, infra_world, boxes, save_path, title):
    vehicle_world = sample_points(vehicle_world)
    infra_world = sample_points(infra_world)

    all_points = np.concatenate([vehicle_world[:, :2], infra_world[:, :2]], axis=0)
    origin = np.mean(all_points, axis=0)

    veh_xy = vehicle_world[:, :2] - origin
    inf_xy = infra_world[:, :2] - origin

    plt.figure(figsize=(9, 9))

    plt.scatter(veh_xy[:, 0], veh_xy[:, 1], s=0.1, label="Vehicle LiDAR")
    plt.scatter(inf_xy[:, 0], inf_xy[:, 1], s=0.1, label="Infrastructure LiDAR")

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
    plt.legend(markerscale=10)
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

        save_path = VISUAL_DIR / f"sample_{i:03d}_fused_world_bev_boxes.png"

        print(f"sample {i}: vehicle_points={len(vehicle_world)}, infra_points={len(infra_world)}, boxes={len(boxes)}")
        plot_fused_world_bev(
            vehicle_world,
            infra_world,
            boxes,
            save_path,
            title=f"Fused World BEV + Label Boxes | sample {i}"
        )

    print("=" * 80)
    print(f"融合 world BEV 可视化完成，结果保存在：{VISUAL_DIR}")


if __name__ == "__main__":
    main()