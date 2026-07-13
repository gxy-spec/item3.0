import argparse
import importlib
import json
import math
import pickle
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
ITEM3_ROOT = PROJECT_ROOT.parent

BASELINE_ROOT = PROJECT_ROOT / "outputs" / "baselines" / "vehicle_lidar"
DEFAULT_OUTPUT_PATH = BASELINE_ROOT / "predictions_vehicle_lidar.json"

OFFICIAL_LABEL_ID_TO_CLASS = {
    0: "Pedestrian",
    1: "Cyclist",
    2: "Car",
}


class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core"):]
        try:
            importlib.import_module(module)
        except ModuleNotFoundError:
            pass
        return super().find_class(module, name)


def load_pickle(path):
    with open(path, "rb") as f:
        return NumpyCompatUnpickler(f).load()


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def to_numpy(value, dtype=float):
    if value is None:
        return None
    return np.asarray(value, dtype=dtype)


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    return float(angle)


def make_box_corners_3d(center, dx, dy, dz, heading):
    x, y, z = [float(v) for v in center]
    hx = float(dx) / 2.0
    hy = float(dy) / 2.0
    hz = float(dz) / 2.0

    corners = np.array(
        [
            [hx, hy, -hz],
            [hx, -hy, -hz],
            [-hx, -hy, -hz],
            [-hx, hy, -hz],
            [hx, hy, hz],
            [hx, -hy, hz],
            [-hx, -hy, hz],
            [-hx, hy, hz],
        ],
        dtype=float,
    )

    c = math.cos(heading)
    s = math.sin(heading)
    rotation = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    corners = corners @ rotation.T
    corners[:, 0] += x
    corners[:, 1] += y
    corners[:, 2] += z
    return corners


def estimate_heading_from_corners(corners):
    xy = corners[:, :2]
    centered = xy - xy.mean(axis=0, keepdims=True)

    if not np.isfinite(centered).all() or np.allclose(centered, 0.0):
        return 0.0

    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    return normalize_angle(math.atan2(axis[1], axis[0]))


def corners_to_lidar_box(corners, arrow=None):
    corners = np.asarray(corners, dtype=float)
    if corners.shape != (8, 3):
        raise ValueError(f"boxes_3d corner shape should be (8, 3), got {corners.shape}")

    center = corners.mean(axis=0)

    if arrow is not None:
        arrow = np.asarray(arrow, dtype=float)
        if arrow.shape == (2, 3):
            direction = arrow[1, :2] - arrow[0, :2]
            if np.linalg.norm(direction) > 1e-6:
                heading = math.atan2(direction[1], direction[0])
            else:
                heading = estimate_heading_from_corners(corners)
        else:
            heading = estimate_heading_from_corners(corners)
    else:
        heading = estimate_heading_from_corners(corners)

    heading = normalize_angle(heading)
    axis_x = np.array([math.cos(heading), math.sin(heading)], dtype=float)
    axis_y = np.array([-math.sin(heading), math.cos(heading)], dtype=float)
    offsets = corners[:, :2] - center[:2].reshape(1, 2)

    proj_x = offsets @ axis_x
    proj_y = offsets @ axis_y

    dx = float(proj_x.max() - proj_x.min())
    dy = float(proj_y.max() - proj_y.min())
    dz = float(corners[:, 2].max() - corners[:, 2].min())

    return center.tolist(), dx, dy, dz, heading


def is_degenerate_prediction(score, class_name, center, dx, dy, dz, corners=None):
    if class_name is None:
        return True
    if score <= 0.0:
        if corners is None:
            return True
        if np.allclose(corners, 0.0):
            return True
    if dx <= 0.0 or dy <= 0.0 or dz <= 0.0:
        return True
    if not np.isfinite(np.array(center + [dx, dy, dz], dtype=float)).all():
        return True
    return False


def make_prediction_object(class_name, score, center, dx, dy, dz, heading, source, corners=None):
    obj = {
        "class": class_name,
        "score": float(score),
        "center_lidar": [float(v) for v in center],
        "box_lidar": {
            "dx": float(dx),
            "dy": float(dy),
            "dz": float(dz),
            "heading": normalize_angle(float(heading)),
        },
        "source": source,
    }

    if corners is not None:
        obj["corners_3d_lidar"] = np.asarray(corners, dtype=float).tolist()

    return obj


def official_class_name(label_id):
    try:
        label_id = int(round(float(label_id)))
    except Exception:
        return None
    return OFFICIAL_LABEL_ID_TO_CLASS.get(label_id)


def convert_official_prediction_dict(pred_dict, sample_id, score_threshold, source_name):
    boxes = to_numpy(pred_dict.get("boxes_3d"))
    scores = to_numpy(pred_dict.get("scores_3d"))
    labels = to_numpy(pred_dict.get("labels_3d"))
    arrows = to_numpy(pred_dict.get("arrows"))

    if boxes is None:
        return []

    boxes = boxes.reshape((-1, 8, 3)) if boxes.size else np.zeros((0, 8, 3), dtype=float)
    if scores is None:
        scores = np.ones((boxes.shape[0],), dtype=float)
    if labels is None:
        labels = np.full((boxes.shape[0],), 2, dtype=float)

    objects = []
    for idx in range(boxes.shape[0]):
        score = float(scores[idx]) if idx < len(scores) else 1.0
        if score < score_threshold:
            continue

        class_name = official_class_name(labels[idx] if idx < len(labels) else 2)
        arrow = arrows[idx] if arrows is not None and arrows.ndim == 3 and idx < arrows.shape[0] else None

        center, dx, dy, dz, heading = corners_to_lidar_box(boxes[idx], arrow=arrow)
        if is_degenerate_prediction(score, class_name, center, dx, dy, dz, corners=boxes[idx]):
            continue

        objects.append(
            make_prediction_object(
                class_name=class_name,
                score=score,
                center=center,
                dx=dx,
                dy=dy,
                dz=dz,
                heading=heading,
                source=source_name,
                corners=boxes[idx],
            )
        )

    return objects


def discover_official_prediction_files(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]

    candidates = [
        input_path / "veh" / "lidar",
        input_path / "result",
        input_path / "preds",
        input_path,
    ]

    for directory in candidates:
        if directory.exists() and directory.is_dir():
            files = sorted(directory.glob("*.pkl"))
            if files:
                return files

    raise FileNotFoundError(f"没有在 {input_path} 下找到官方预测 pkl 文件")


def export_official_predictions(input_path, score_threshold):
    records = []
    for path in discover_official_prediction_files(input_path):
        pred_dict = load_pickle(path)
        sample_id = str(pred_dict.get("veh_id", pred_dict.get("info", path.stem)))
        objects = convert_official_prediction_dict(
            pred_dict=pred_dict,
            sample_id=sample_id,
            score_threshold=score_threshold,
            source_name="dair_v2x_official_pointpillars",
        )

        records.append(
            {
                "sample_id": sample_id,
                "coordinate_system": "vehicle_lidar",
                "prediction_type": "dair_v2x_official_vehicle_lidar_pointpillars",
                "source_file": str(path),
                "pred_objects": objects,
            }
        )

    return records


def export_openpcdet_predictions(input_path, score_threshold):
    result = load_pickle(input_path)
    if not isinstance(result, list):
        raise ValueError("OpenPCDet result.pkl 应该是 det_annos list")

    records = []
    for anno in result:
        sample_id = str(anno.get("frame_id", anno.get("sample_id", "")))
        if not sample_id:
            raise ValueError(f"OpenPCDet 预测缺少 frame_id: {anno.keys()}")

        names = anno.get("name", [])
        scores = np.asarray(anno.get("score", []), dtype=float)
        boxes = np.asarray(anno.get("boxes_lidar", []), dtype=float)

        if boxes.size == 0:
            boxes = np.zeros((0, 7), dtype=float)
        boxes = boxes.reshape((-1, boxes.shape[-1]))

        objects = []
        for idx in range(boxes.shape[0]):
            score = float(scores[idx]) if idx < len(scores) else 1.0
            if score < score_threshold:
                continue

            class_name = str(names[idx]) if idx < len(names) else "Unknown"
            x, y, z, dx, dy, dz, heading = [float(v) for v in boxes[idx, :7]]
            center = [x, y, z]
            corners = make_box_corners_3d(center, dx, dy, dz, heading)

            if is_degenerate_prediction(score, class_name, center, dx, dy, dz, corners=corners):
                continue

            objects.append(
                make_prediction_object(
                    class_name=class_name,
                    score=score,
                    center=center,
                    dx=dx,
                    dy=dy,
                    dz=dz,
                    heading=heading,
                    source="openpcdet_pointpillars",
                    corners=corners,
                )
            )

        records.append(
            {
                "sample_id": sample_id,
                "coordinate_system": "vehicle_lidar",
                "prediction_type": "openpcdet_vehicle_lidar_pointpillars",
                "source_file": str(input_path),
                "pred_objects": objects,
            }
        )

    return records


def detect_source_format(input_path):
    input_path = Path(input_path)
    if input_path.is_dir():
        return "dair-official-pkl"

    obj = load_pickle(input_path)
    if isinstance(obj, list):
        return "openpcdet-result-pkl"
    if isinstance(obj, dict) and "boxes_3d" in obj:
        return "dair-official-pkl"

    raise ValueError(f"无法自动识别预测格式: {input_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Vehicle LiDAR-only detector predictions to item3.0 JSON format."
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "DAIR-V2X 官方输出目录/单帧 pkl，或 OpenPCDet tools/test.py 生成的 result.pkl。"
        ),
    )
    parser.add_argument(
        "--source-format",
        choices=["auto", "dair-official-pkl", "openpcdet-result-pkl"],
        default="auto",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="统一 predictions_vehicle_lidar.json 输出路径。",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="导出前按置信度过滤预测框；默认保留所有非空预测。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"找不到预测输入: {input_path}")

    source_format = args.source_format
    if source_format == "auto":
        source_format = detect_source_format(input_path)

    if source_format == "dair-official-pkl":
        records = export_official_predictions(input_path, args.score_threshold)
    elif source_format == "openpcdet-result-pkl":
        records = export_openpcdet_predictions(input_path, args.score_threshold)
    else:
        raise ValueError(f"不支持的 source format: {source_format}")

    output_path = Path(args.output)
    save_json(records, output_path)

    summary = {
        "input": str(input_path),
        "source_format": source_format,
        "score_threshold": args.score_threshold,
        "output_path": str(output_path),
        "num_samples": len(records),
        "num_objects": sum(len(record["pred_objects"]) for record in records),
        "prediction_types": sorted({record["prediction_type"] for record in records}),
    }
    summary_path = output_path.with_name("predictions_vehicle_lidar_summary.json")
    save_json(summary, summary_path)

    print("=" * 80)
    print("Vehicle LiDAR predictions export finished")
    print(f"Output: {output_path}")
    print(f"Summary: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
