from pathlib import Path

# 改成你自己的 DAIR-V2X-C 数据集路径
DATA_ROOT = Path(r"D:/Python/study/item3.0/datasets/DAIR-V2X-C/cooperative-vehicle-infrastructure")

# 输出目录
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
VISUAL_DIR = OUTPUT_ROOT / "visual_check"
PREPROCESS_DIR = OUTPUT_ROOT / "preprocessed"
LOG_DIR = OUTPUT_ROOT / "logs"

# 自动创建输出文件夹
for p in [OUTPUT_ROOT, VISUAL_DIR, PREPROCESS_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# Coordinate setting
# ============================================================
# 本项目统一使用 DAIR-V2X 的 world 坐标系作为协同感知统一坐标系
COORDINATE_SYSTEM = "world"

# 任务损失权重，后续可以调整
LAMBDA_LOC = 1.0
LAMBDA_COUNT = 1.0