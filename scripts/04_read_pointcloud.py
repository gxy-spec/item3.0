import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from config import DATA_ROOT, VISUAL_DIR

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_pcd(path):
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points)
    return pts

def plot_bev(points, title, save_path):
    if points.shape[0] == 0:
        raise ValueError(f"{title} 点云为空，请检查 pcd 文件。")

    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1], s=0.1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()

coop_info = load_json(DATA_ROOT / "cooperative" / "data_info.json")
sample = coop_info[0]

print("第一条 cooperative sample:")
print(sample)

vehicle_pcd_rel = sample.get("vehicle_pointcloud_path", None)
infra_pcd_rel = sample.get("infrastructure_pointcloud_path", None)

if vehicle_pcd_rel is None or infra_pcd_rel is None:
    raise KeyError(
        "没有找到 vehicle_pointcloud_path 或 infrastructure_pointcloud_path。"
        "请先运行 02_inspect_data_info.py，把第一条 sample 发给我，我帮你改字段名。"
    )

vehicle_pcd_path = DATA_ROOT / vehicle_pcd_rel
infra_pcd_path = DATA_ROOT / infra_pcd_rel

print("vehicle point cloud:", vehicle_pcd_path)
print("infrastructure point cloud:", infra_pcd_path)

veh_pts = read_pcd(vehicle_pcd_path)
inf_pts = read_pcd(infra_pcd_path)

print("vehicle points shape:", veh_pts.shape)
print("infrastructure points shape:", inf_pts.shape)

plot_bev(
    veh_pts,
    "Vehicle-side LiDAR BEV",
    VISUAL_DIR / "sample_000_vehicle_lidar_bev.png"
)

plot_bev(
    inf_pts,
    "Infrastructure-side LiDAR BEV",
    VISUAL_DIR / "sample_000_infrastructure_lidar_bev.png"
)

print("点云 BEV 检查图已保存到：", VISUAL_DIR)