import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ITEM3_ROOT = PROJECT_ROOT.parent


def resolve_data_root():
    """
    优先使用环境变量，随后自动匹配当前仓库内的 DAIR-V2X-C 数据目录。
    保留 Windows 路径作为最后兜底，方便同一套脚本在原 item3.0 环境中继续使用。
    """
    env_value = os.environ.get("DAIR_V2X_C_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()

    local_root = (
        ITEM3_ROOT
        / "DAIR-V2X"
        / "data"
        / "DAIR-V2X"
        / "cooperative-vehicle-infrastructure"
    )
    if local_root.exists():
        return local_root

    return Path(r"D:/Python/study/item3.0/datasets/DAIR-V2X-C/cooperative-vehicle-infrastructure")


# DAIR-V2X-C cooperative-vehicle-infrastructure 数据集路径
DATA_ROOT = resolve_data_root()

# 输出目录
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
