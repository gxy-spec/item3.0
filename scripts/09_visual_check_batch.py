import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import json
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from config import PREPROCESS_DIR, VISUAL_DIR

index_path = PREPROCESS_DIR / "dair_v2x_pair_index.json"

with open(index_path, "r", encoding="utf-8") as f:
    pair_index = json.load(f)

def read_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def read_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points)

def plot_one_sample(item, save_path):
    veh_img = read_image(item["vehicle_image"]) if item["vehicle_image"] else None
    inf_img = read_image(item["infrastructure_image"]) if item["infrastructure_image"] else None

    veh_pts = read_pcd(item["vehicle_pointcloud"]) if item["vehicle_pointcloud"] else None
    inf_pts = read_pcd(item["infrastructure_pointcloud"]) if item["infrastructure_pointcloud"] else None

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    if veh_img is not None:
        plt.imshow(veh_img)
    plt.title("Vehicle Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    if inf_img is not None:
        plt.imshow(inf_img)
    plt.title("Infrastructure Image")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    if veh_pts is not None and len(veh_pts) > 0:
        plt.scatter(veh_pts[:, 0], veh_pts[:, 1], s=0.1)
        plt.axis("equal")
    plt.title("Vehicle LiDAR BEV")

    plt.subplot(2, 2, 4)
    if inf_pts is not None and len(inf_pts) > 0:
        plt.scatter(inf_pts[:, 0], inf_pts[:, 1], s=0.1)
        plt.axis("equal")
    plt.title("Infrastructure LiDAR BEV")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

num_check = min(20, len(pair_index))

for i in range(num_check):
    item = pair_index[i]
    save_path = VISUAL_DIR / f"sample_{i:03d}_check.png"
    print(f"Saving {save_path}")
    plot_one_sample(item, save_path)

print(f"完成 {num_check} 组样本可视化，结果保存在：{VISUAL_DIR}")