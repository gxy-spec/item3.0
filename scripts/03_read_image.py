import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import json
import cv2
import matplotlib.pyplot as plt
from config import DATA_ROOT, VISUAL_DIR

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

coop_info = load_json(DATA_ROOT / "cooperative" / "data_info.json")
sample = coop_info[0]

print("第一条 cooperative sample:")
print(sample)

# 尝试从 sample 里自动找 image 路径字段
vehicle_img_rel = sample.get("vehicle_image_path", None)
infra_img_rel = sample.get("infrastructure_image_path", None)

if vehicle_img_rel is None or infra_img_rel is None:
    raise KeyError(
        "没有找到 vehicle_image_path 或 infrastructure_image_path。"
        "请先运行 02_inspect_data_info.py，把第一条 sample 发给我，我帮你改字段名。"
    )

vehicle_img_path = DATA_ROOT / vehicle_img_rel
infra_img_path = DATA_ROOT / infra_img_rel

print("vehicle image:", vehicle_img_path)
print("infrastructure image:", infra_img_path)

veh_img = cv2.imread(str(vehicle_img_path))
inf_img = cv2.imread(str(infra_img_path))

if veh_img is None:
    raise FileNotFoundError(f"读取车端图像失败：{vehicle_img_path}")

if inf_img is None:
    raise FileNotFoundError(f"读取路侧图像失败：{infra_img_path}")

veh_img = cv2.cvtColor(veh_img, cv2.COLOR_BGR2RGB)
inf_img = cv2.cvtColor(inf_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(veh_img)
plt.title("Vehicle-side image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(inf_img)
plt.title("Infrastructure-side image")
plt.axis("off")

save_path = VISUAL_DIR / "sample_000_image_pair.png"
plt.tight_layout()
plt.savefig(save_path, dpi=200)
plt.show()

print(f"图像检查结果已保存到：{save_path}")