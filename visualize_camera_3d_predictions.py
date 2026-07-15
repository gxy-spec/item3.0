#!/usr/bin/env python3
"""Project Camera-only 3D detections and cooperative world GT onto source images."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT.parent / "DAIR-V2X" / "data" / "DAIR-V2X" / "cooperative-vehicle-infrastructure"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# This parser establishes the GT ordering used by the world-coordinate evaluator.
import evaluate_and_visualize_vehicle_lidar_baseline as legacy  # noqa: E402


BASELINES = {
    "vehicle_camera": {
        "name": "Vehicle Camera-only",
        "side": "vehicle-side",
        "prediction_path": PROJECT_ROOT / "outputs/baselines/vehicle_camera/predictions_vehicle_camera.json",
        "details_path": PROJECT_ROOT / "outputs/baselines/vehicle_camera/vehicle_camera_eval_details.json",
        "output_root": PROJECT_ROOT / "outputs/baselines/vehicle_camera/visualization",
    },
    "infrastructure_camera": {
        "name": "Infrastructure Camera-only",
        "side": "infrastructure-side",
        "prediction_path": PROJECT_ROOT / "outputs/baselines/infrastructure_camera/predictions_infrastructure_camera.json",
        "details_path": PROJECT_ROOT / "outputs/baselines/infrastructure_camera/infrastructure_camera_eval_details.json",
        "output_root": PROJECT_ROOT / "outputs/baselines/infrastructure_camera/visualization",
    },
}

COLORS = {
    "gt_matched": (0, 220, 0),
    "gt_missed": (0, 0, 255),
    "pred_matched": (255, 255, 0),
    "pred_false": (0, 165, 255),
    "pred_class_wrong": (255, 0, 255),
}
EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def make_transform(calibration):
    if "transform" in calibration:
        calibration = calibration["transform"]
    if "rotation" not in calibration or "translation" not in calibration:
        raise ValueError("calibration must contain rotation and translation")
    rotation = np.asarray(calibration["rotation"], dtype=np.float64)
    translation = np.asarray(calibration["translation"], dtype=np.float64)
    if rotation.size != 9 or translation.size != 3:
        raise ValueError("invalid calibration matrix shape")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation.reshape(3, 3)
    transform[:3, 3] = translation.reshape(3)
    return transform


def transform_points(points, transform):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (transform @ homogeneous.T).T[:, :3]


def build_lidar_corners(center, dx, dy, dz, heading):
    x_offsets = np.array([dx / 2, dx / 2, -dx / 2, -dx / 2])
    y_offsets = np.array([dy / 2, -dy / 2, -dy / 2, dy / 2])
    rotation = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
    bev = rotation @ np.vstack([x_offsets, y_offsets])
    bottom = np.column_stack([bev[0] + center[0], bev[1] + center[1], np.full(4, center[2] - dz / 2)])
    top = np.column_stack([bev[0] + center[0], bev[1] + center[1], np.full(4, center[2] + dz / 2)])
    return np.vstack([bottom, top])


def build_world_gt_corners(gt_object):
    box = gt_object["box_world"]
    corners = np.asarray(box.get("corners_3d_world", []), dtype=np.float64)
    if corners.shape == (8, 3):
        return corners
    center = box["center_world"]
    return build_lidar_corners(
        np.array([center["x"], center["y"], (float(box["z_min"]) + float(box["z_max"])) / 2]),
        float(box["length"]),
        float(box["width"]),
        float(box["z_max"]) - float(box["z_min"]),
        float(box["yaw"]),
    )


def source_id_for(baseline, sample):
    if baseline == "vehicle_camera":
        return str(sample.get("vehicle_id") or sample.get("source_id") or sample["sample_id"])
    return str(sample.get("infrastructure_id") or sample.get("source_id") or sample["sample_id"])


def parse_center_lidar(center):
    """Accept both the shared-template object form and current Camera list form."""
    if isinstance(center, dict):
        values = [center.get("x"), center.get("y"), center.get("z")]
    else:
        values = np.asarray(center, dtype=np.float64).reshape(-1).tolist()
    if len(values) != 3 or any(value is None for value in values):
        raise ValueError("center_lidar must contain x, y, z")
    return np.asarray(values, dtype=np.float64)


def load_projection_calibration(baseline, sample):
    source_id = source_id_for(baseline, sample)
    side_root = DATA_ROOT / BASELINES[baseline]["side"]
    image_path = side_root / "image" / f"{source_id}.jpg"
    intrinsic_path = side_root / "calib/camera_intrinsic" / f"{source_id}.json"
    if baseline == "vehicle_camera":
        lidar_to_camera_path = side_root / "calib/lidar_to_camera" / f"{source_id}.json"
        lidar_to_novatel_path = side_root / "calib/lidar_to_novatel" / f"{source_id}.json"
        novatel_to_world_path = side_root / "calib/novatel_to_world" / f"{source_id}.json"
        required = [image_path, intrinsic_path, lidar_to_camera_path, lidar_to_novatel_path, novatel_to_world_path]
    else:
        lidar_to_camera_path = side_root / "calib/virtuallidar_to_camera" / f"{source_id}.json"
        lidar_to_world_path = side_root / "calib/virtuallidar_to_world" / f"{source_id}.json"
        required = [image_path, intrinsic_path, lidar_to_camera_path, lidar_to_world_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))

    camera_matrix = np.asarray(load_json(intrinsic_path)["cam_K"], dtype=np.float64).reshape(3, 3)
    lidar_to_camera = make_transform(load_json(lidar_to_camera_path))
    if baseline == "vehicle_camera":
        lidar_to_world = make_transform(load_json(novatel_to_world_path)) @ make_transform(load_json(lidar_to_novatel_path))
    else:
        lidar_to_world = make_transform(load_json(lidar_to_world_path))
    return image_path, camera_matrix, lidar_to_camera, lidar_to_world


def draw_label(image, text, point, color):
    x = max(0, min(image.shape[1] - 1, int(point[0])))
    y = max(18, min(image.shape[0] - 4, int(point[1])))
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)
    cv2.rectangle(image, (x, y - height - baseline - 3), (min(image.shape[1] - 1, x + width + 4), y + 2), (24, 24, 24), -1)
    cv2.putText(image, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)


def draw_projected_box(image, corners_lidar, lidar_to_camera, camera_matrix, color, label):
    camera_points = transform_points(corners_lidar, lidar_to_camera)
    valid = camera_points[:, 2] > 0.05
    pixels = np.full((8, 2), np.nan)
    if valid.any():
        projected = (camera_matrix @ camera_points[valid].T).T
        pixels[valid] = projected[:, :2] / projected[:, 2:3]
    edges_drawn = 0
    rectangle = (0, 0, image.shape[1], image.shape[0])
    for start, end in EDGES:
        if not (valid[start] and valid[end]):
            continue
        point_a = tuple(np.rint(pixels[start]).astype(int))
        point_b = tuple(np.rint(pixels[end]).astype(int))
        visible, clipped_a, clipped_b = cv2.clipLine(rectangle, point_a, point_b)
        if visible:
            cv2.line(image, clipped_a, clipped_b, color, 2, cv2.LINE_AA)
            edges_drawn += 1
    if edges_drawn:
        draw_label(image, label, np.nanmin(pixels[valid], axis=0), color)
    return edges_drawn > 0


def draw_legend(image, title, detail):
    overlay = image.copy()
    cv2.rectangle(overlay, (8, 8), (438, 126), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.78, image, 0.22, 0, image)
    cv2.putText(image, title, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
    items = [
        ("GT matched", COLORS["gt_matched"]),
        ("GT missed (FN)", COLORS["gt_missed"]),
        ("Prediction matched", COLORS["pred_matched"]),
        ("Prediction false positive", COLORS["pred_false"]),
        ("Prediction class wrong", COLORS["pred_class_wrong"]),
    ]
    for index, (text, color) in enumerate(items):
        column, row = index % 2, index // 2
        x, y = 18 + column * 215, 53 + row * 22
        cv2.line(image, (x, y - 4), (x + 20, y - 4), color, 3, cv2.LINE_AA)
        cv2.putText(image, text, (x + 28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (240, 240, 240), 1, cv2.LINE_AA)
    metrics = detail.get("metrics", {})
    text = "GT={} Pred={} Match={} P={:.3f} R={:.3f}".format(
        metrics.get("num_gt", 0), metrics.get("num_pred", 0), metrics.get("num_match", 0),
        float(metrics.get("precision", 0.0)), float(metrics.get("recall", 0.0)),
    )
    cv2.putText(image, text, (18, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (240, 240, 240), 1, cv2.LINE_AA)


def render_sample(baseline, sample, detail, output_path):
    image_path, camera_matrix, lidar_to_camera, lidar_to_world = load_projection_calibration(baseline, sample)
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"cannot read {image_path}")
    gt_objects = legacy.parse_gt_objects(Path(detail["label_world_path"]))
    matched_pred = {int(item["pred_index"]) for item in detail.get("matches", [])}
    matched_gt = {int(item["gt_index"]) for item in detail.get("matches", [])}
    class_wrong = {int(item["pred_index"]) for item in detail.get("matches", []) if not item.get("class_correct", False)}
    missed_gt = {int(index) for index in detail.get("missed_gt_indices", [])}
    counts = Counter()

    world_to_lidar = np.linalg.inv(lidar_to_world)
    for gt_index, gt_object in enumerate(gt_objects):
        is_missed = gt_index in missed_gt or gt_index not in matched_gt
        state = "gt_missed" if is_missed else "gt_matched"
        label = f"GT-FN {gt_object['class']}" if is_missed else f"GT {gt_object['class']}"
        corners_lidar = transform_points(build_world_gt_corners(gt_object), world_to_lidar)
        key = f"drawn_{state}" if draw_projected_box(image, corners_lidar, lidar_to_camera, camera_matrix, COLORS[state], label) else f"hidden_{state}"
        counts[key] += 1

    for pred_index, pred_object in enumerate(sample.get("pred_objects", [])):
        center, box = pred_object.get("center_lidar"), pred_object.get("box_lidar", {})
        if center is None or any(key not in box for key in ("dx", "dy", "dz", "heading")):
            counts["invalid_prediction_boxes"] += 1
            continue
        try:
            corners_lidar = build_lidar_corners(
                parse_center_lidar(center), box["dx"], box["dy"], box["dz"], box["heading"]
            )
        except (TypeError, ValueError):
            counts["invalid_prediction_boxes"] += 1
            continue
        if pred_index in class_wrong:
            state, prefix = "pred_class_wrong", "P-CLS"
        elif pred_index in matched_pred:
            state, prefix = "pred_matched", "P-TP"
        else:
            state, prefix = "pred_false", "P-FP"
        label = f"{prefix} {pred_object.get('class', 'Unknown')} {float(pred_object.get('score', 0.0)):.2f}"
        key = f"drawn_{state}" if draw_projected_box(image, corners_lidar, lidar_to_camera, camera_matrix, COLORS[state], label) else f"hidden_{state}"
        counts[key] += 1

    draw_legend(image, f"{BASELINES[baseline]['name']} | sample {sample['sample_id']}", detail)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"cannot write {output_path}")
    return counts


def create_contact_sheet(paths, output_path, columns):
    if not paths:
        return None
    tile_width, tile_height, margin = 480, 270, 8
    rows = int(np.ceil(len(paths) / columns))
    canvas = np.full((rows * tile_height + (rows + 1) * margin, columns * tile_width + (columns + 1) * margin, 3), 22, dtype=np.uint8)
    for index, path in enumerate(paths):
        image = cv2.imread(str(path))
        if image is None:
            continue
        row, column = divmod(index, columns)
        y, x = margin + row * (tile_height + margin), margin + column * (tile_width + margin)
        canvas[y:y + tile_height, x:x + tile_width] = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"cannot write {output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate source-image 3D box visualizations for Camera-only baselines.")
    parser.add_argument("--baseline", choices=sorted(BASELINES), required=True)
    parser.add_argument("--max-samples", type=int, default=0, help="only process the first N samples; 0 means all")
    parser.add_argument("--contact-sheet-columns", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = BASELINES[args.baseline]
    if args.contact_sheet_columns <= 0:
        raise ValueError("--contact-sheet-columns must be positive")
    if not config["prediction_path"].exists() or not config["details_path"].exists():
        raise FileNotFoundError(
            "missing predictions or evaluator details. Run camera inference, world conversion, and "
            f"evaluate_and_visualize_lidar_baseline.py --baseline {args.baseline} first."
        )
    prediction_data = load_json(config["prediction_path"])
    if isinstance(prediction_data, list):
        samples = prediction_data
    elif isinstance(prediction_data, dict):
        samples = prediction_data.get("samples", [])
    else:
        samples = []
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    if not samples:
        raise ValueError(f"no samples in {config['prediction_path']}")
    details_by_sample = {str(item["sample_id"]): item for item in load_json(config["details_path"])}
    output_root = args.output_dir or config["output_root"]
    image_dir, summary_dir = output_root / "image_3d_detection", output_root / "summary"
    written, skipped, counts = [], [], Counter()
    processed_details = []
    for index, sample in enumerate(samples, start=1):
        sample_id = str(sample.get("sample_id", ""))
        detail = details_by_sample.get(sample_id)
        if detail is None:
            message = "sample_id is absent from evaluator details"
            print(f"[WARNING] {sample_id}: {message}")
            skipped.append({"sample_id": sample_id, "reason": message})
            continue
        source_id = source_id_for(args.baseline, sample)
        stem = sample_id if source_id == sample_id else f"{sample_id}_{source_id}"
        output_path = image_dir / f"{stem}_camera_3d.jpg"
        try:
            counts.update(render_sample(args.baseline, sample, detail, output_path))
        except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
            print(f"[WARNING] skip {sample_id}: {error}")
            skipped.append({"sample_id": sample_id, "reason": str(error)})
            continue
        processed_details.append(detail)
        written.append(output_path)
        print(f"[{index}/{len(samples)}] wrote {output_path}")
    contact_sheet = create_contact_sheet(written, summary_dir / "camera_3d_detection_contact_sheet.png", args.contact_sheet_columns)
    summary = {
        "baseline": args.baseline,
        "baseline_name": config["name"],
        "prediction_path": str(config["prediction_path"]),
        "evaluation_details_path": str(config["details_path"]),
        "num_samples_requested": len(samples),
        "num_images_written": len(written),
        "num_gt": sum(item.get("metrics", {}).get("num_gt", 0) for item in processed_details),
        "num_pred": sum(item.get("metrics", {}).get("num_pred", 0) for item in processed_details),
        "num_match": sum(item.get("metrics", {}).get("num_match", 0) for item in processed_details),
        "false_negative": sum(item.get("metrics", {}).get("false_negative", 0) for item in processed_details),
        "false_positive": sum(item.get("metrics", {}).get("false_positive", 0) for item in processed_details),
        "render_counts": dict(counts),
        "skipped_samples": skipped,
        "image_output_dir": str(image_dir),
        "contact_sheet_path": str(contact_sheet) if contact_sheet else None,
        "state_source": "States are read from existing world-coordinate evaluator details.",
        "color_legend_bgr": COLORS,
    }
    summary_path = summary_dir / "camera_3d_visualization_summary.json"
    save_json(summary_path, summary)
    print(f"[DONE] images: {image_dir}")
    print(f"[DONE] contact sheet: {contact_sheet}")
    print(f"[DONE] summary: {summary_path}")


if __name__ == "__main__":
    main()
