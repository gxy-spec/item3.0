import json
from pathlib import Path

import numpy as np


PRED_WORLD_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "baselines"
    / "vehicle_lidar"
    / "predictions_world.json"
)


def main():
    if not PRED_WORLD_PATH.exists():
        raise FileNotFoundError(f"找不到文件: {PRED_WORLD_PATH}")

    with open(PRED_WORLD_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    print("=" * 80)
    print("Check predictions_world.json")
    print("path:", PRED_WORLD_PATH)
    print("num records:", len(records))

    total_objects = 0
    all_centers = []

    for record in records:
        pred_objects = record.get("pred_objects", [])
        total_objects += len(pred_objects)

        for obj in pred_objects:
            center = obj["center_world"]
            all_centers.append(center)

            arr = np.array(center, dtype=float)
            if not np.isfinite(arr).all():
                raise ValueError(f"center_world 含 NaN/Inf: {record['sample_id']} {center}")

            corners = np.array(obj["box_world"]["corners_bev_world"], dtype=float)
            if corners.shape != (4, 2):
                raise ValueError(
                    f"corners_bev_world 形状异常: {record['sample_id']} {corners.shape}"
                )

            if not np.isfinite(corners).all():
                raise ValueError(
                    f"corners_bev_world 含 NaN/Inf: {record['sample_id']}"
                )

            corners_3d = obj["box_world"].get("corners_3d_world", None)
            if corners_3d is not None:
                corners_3d = np.array(corners_3d, dtype=float)
                if corners_3d.shape != (8, 3):
                    raise ValueError(
                        f"corners_3d_world 形状异常: {record['sample_id']} {corners_3d.shape}"
                    )

                if not np.isfinite(corners_3d).all():
                    raise ValueError(
                        f"corners_3d_world 含 NaN/Inf: {record['sample_id']}"
                    )

    print("num objects:", total_objects)

    if len(all_centers) > 0:
        centers = np.array(all_centers, dtype=float)

        print("center_world range:")
        print("  x:", float(centers[:, 0].min()), "to", float(centers[:, 0].max()))
        print("  y:", float(centers[:, 1].min()), "to", float(centers[:, 1].max()))
        print("  z:", float(centers[:, 2].min()), "to", float(centers[:, 2].max()))

    print("=" * 80)
    print("predictions_world check passed")


if __name__ == "__main__":
    main()
