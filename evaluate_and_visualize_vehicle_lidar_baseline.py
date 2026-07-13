import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch


PROJECT_ROOT = Path(__file__).resolve().parent
ITEM3_ROOT = PROJECT_ROOT.parent

DATA_ROOT = (
    ITEM3_ROOT
    / "DAIR-V2X"
    / "data"
    / "DAIR-V2X"
    / "cooperative-vehicle-infrastructure"
)

BASELINE_ROOT = PROJECT_ROOT / "outputs" / "baselines" / "vehicle_lidar"

PRED_WORLD_PATH = BASELINE_ROOT / "predictions_world.json"
PRED_LIDAR_PATH = BASELINE_ROOT / "predictions_vehicle_lidar.json"

EVAL_CSV_PATH = BASELINE_ROOT / "vehicle_lidar_only_eval.csv"
EVAL_SUMMARY_PATH = BASELINE_ROOT / "vehicle_lidar_only_eval_summary.json"
AP_SUMMARY_PATH = BASELINE_ROOT / "vehicle_lidar_only_ap_summary.json"
MATCHED_PAIRS_CSV_PATH = BASELINE_ROOT / "matched_prediction_gt_pairs.csv"
PER_CLASS_METRICS_CSV_PATH = BASELINE_ROOT / "per_class_metrics.csv"
CONFUSION_MATRIX_CSV_PATH = BASELINE_ROOT / "class_confusion_matrix.csv"

VIS_ROOT = BASELINE_ROOT / "visualization"
WORLD_VIS_DIR = VIS_ROOT / "world_gt_pred_compare"
SUMMARY_VIS_DIR = VIS_ROOT / "summary"

COOP_INFO_PATH = DATA_ROOT / "cooperative" / "data_info.json"

AP_CLASSES = ["Car", "Pedestrian", "Cyclist"]
AP_IOU_THRESHOLDS = [0.25, 0.5, 0.7]


def load_json(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_class_name(name):
    """
    DAIR-V2X 中类别名称可能更细，这里统一到三类：
    Car / Pedestrian / Cyclist
    """
    if name is None:
        return "Unknown"

    n = str(name).lower()

    if n in ["car", "van", "truck", "bus"]:
        return "Car"

    if n in ["pedestrian", "person"]:
        return "Pedestrian"

    if n in ["cyclist", "bicycle", "motorcyclist", "motorcycle"]:
        return "Cyclist"

    return str(name)


def build_sample_mapping():
    """
    建立：
    vehicle_id / sample_id -> cooperative label_world 路径
    """
    coop_info = load_json(COOP_INFO_PATH)

    mapping = {}

    for item in coop_info:
        vehicle_pointcloud_path = item.get("vehicle_pointcloud_path", "")
        vehicle_id = Path(vehicle_pointcloud_path).stem

        coop_label_rel = item.get("cooperative_label_path", "")
        coop_label_path = DATA_ROOT / coop_label_rel

        if vehicle_id:
            mapping[vehicle_id] = {
                "vehicle_id": vehicle_id,
                "cooperative_label_path": str(coop_label_path),
                "raw_info": item
            }

    return mapping


def parse_rotation_yaw(rot):
    """
    兼容 rotation 的不同写法：
    1. float
    2. {"x":..., "y":..., "z":...}
    3. {"roll":..., "pitch":..., "yaw":...}
    """
    if rot is None:
        return 0.0

    if isinstance(rot, (int, float)):
        return float(rot)

    if isinstance(rot, dict):
        if "z" in rot:
            return float(rot["z"])
        if "yaw" in rot:
            return float(rot["yaw"])
        if "rotation_y" in rot:
            return float(rot["rotation_y"])

    try:
        return float(rot)
    except Exception:
        return 0.0


def parse_dim_value(dim, keys, default=0.0):
    for k in keys:
        if k in dim:
            return float(dim[k])
    return float(default)


def parse_gt_objects(label_world_path):
    """
    读取 cooperative label_world。
    输出统一格式：
    {
      class,
      center_world,
      box_world: length, width, height, yaw, corners_bev_world
    }
    """
    labels = load_json(label_world_path)
    objects = []

    for obj in labels:
        raw_cls = obj.get("type", obj.get("class", obj.get("name", "Unknown")))
        cls = normalize_class_name(raw_cls)

        loc = obj.get("3d_location", obj.get("location", obj.get("center", None)))
        dim = obj.get("3d_dimensions", obj.get("dimensions", obj.get("size", None)))
        rot = obj.get("rotation", obj.get("rotation_y", obj.get("yaw", 0.0)))

        if loc is None or dim is None:
            continue

        try:
            center = [
                float(loc["x"]),
                float(loc["y"]),
                float(loc.get("z", 0.0))
            ]
        except Exception:
            continue

        length = parse_dim_value(dim, ["l", "length", "dx"])
        width = parse_dim_value(dim, ["w", "width", "dy"])
        height = parse_dim_value(dim, ["h", "height", "dz"])

        if length <= 0 or width <= 0 or height <= 0:
            continue

        yaw = parse_rotation_yaw(rot)

        world_8_points = obj.get("world_8_points", None)
        corners_3d_world = None
        z_min = center[2] - height / 2.0
        z_max = center[2] + height / 2.0

        if world_8_points is not None:
            try:
                corners_3d_world = np.array(world_8_points, dtype=float)
                if corners_3d_world.shape == (8, 3):
                    z_min = float(corners_3d_world[:, 2].min())
                    z_max = float(corners_3d_world[:, 2].max())
                else:
                    corners_3d_world = None
            except Exception:
                corners_3d_world = None

        if corners_3d_world is not None:
            corners = corners_3d_world[:4, :2]
        else:
            corners = make_box_corners_bev(
                center_xy=np.array(center[:2], dtype=float),
                dx=length,
                dy=width,
                heading=yaw
            )

        box_world = {
            "length": length,
            "width": width,
            "height": height,
            "yaw": yaw,
            "corners_bev_world": corners.tolist(),
            "z_min": z_min,
            "z_max": z_max
        }

        if corners_3d_world is not None:
            box_world["corners_3d_world"] = corners_3d_world.tolist()

        objects.append({
            "class": cls,
            "raw_class": raw_cls,
            "center_world": center,
            "box_world": box_world
        })

    return objects


def make_box_corners_bev(center_xy, dx, dy, heading):
    """
    生成 BEV 四角点。
    dx 沿 heading 方向，dy 为横向宽度。
    """
    hx = dx / 2.0
    hy = dy / 2.0

    corners = np.array([
        [ hx,  hy],
        [ hx, -hy],
        [-hx, -hy],
        [-hx,  hy],
    ], dtype=float)

    c = math.cos(heading)
    s = math.sin(heading)

    R = np.array([
        [c, -s],
        [s,  c],
    ], dtype=float)

    return corners @ R.T + center_xy.reshape(1, 2)


def pred_box_corners(pred_obj):
    """
    prediction_world.json 中已经有 corners_bev_world，优先使用。
    """
    box_world = pred_obj.get("box_world", {})
    corners = box_world.get("corners_bev_world", None)

    if corners is not None:
        arr = np.array(corners, dtype=float)
        if arr.shape == (4, 2):
            return arr

    center = np.array(pred_obj["center_world"][:2], dtype=float)
    dx = float(box_world.get("dx", 1.0))
    dy = float(box_world.get("dy", 1.0))
    heading = float(box_world.get("heading_world", 0.0))

    return make_box_corners_bev(center, dx, dy, heading)


def gt_box_corners(gt_obj):
    corners = gt_obj.get("box_world", {}).get("corners_bev_world", None)

    if corners is not None:
        arr = np.array(corners, dtype=float)
        if arr.shape == (4, 2):
            return arr

    center = np.array(gt_obj["center_world"][:2], dtype=float)
    box = gt_obj["box_world"]

    return make_box_corners_bev(
        center_xy=center,
        dx=float(box["length"]),
        dy=float(box["width"]),
        heading=float(box["yaw"])
    )


def pred_z_range(pred_obj):
    box = pred_obj.get("box_world", {})
    if "z_min" in box and "z_max" in box:
        return float(box["z_min"]), float(box["z_max"])

    center_z = float(pred_obj["center_world"][2])
    dz = float(box.get("dz", box.get("height", 0.0)))
    return center_z - dz / 2.0, center_z + dz / 2.0


def gt_z_range(gt_obj):
    box = gt_obj.get("box_world", {})
    if "z_min" in box and "z_max" in box:
        return float(box["z_min"]), float(box["z_max"])

    center_z = float(gt_obj["center_world"][2])
    height = float(box.get("height", box.get("dz", 0.0)))
    return center_z - height / 2.0, center_z + height / 2.0


def polygon_signed_area(poly):
    poly = np.asarray(poly, dtype=float)
    if poly.shape[0] < 3:
        return 0.0
    return float(
        0.5 * (
            np.dot(poly[:, 0], np.roll(poly[:, 1], -1))
            - np.dot(poly[:, 1], np.roll(poly[:, 0], -1))
        )
    )


def polygon_area(poly):
    return abs(polygon_signed_area(poly))


def ensure_ccw(poly):
    poly = np.asarray(poly, dtype=float)
    if polygon_signed_area(poly) < 0:
        return poly[::-1]
    return poly


def line_intersection(p1, p2, q1, q2):
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)

    r = p2 - p1
    s = q2 - q1
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-9:
        return p2

    qp = q1 - p1
    t = (qp[0] * s[1] - qp[1] * s[0]) / denom
    return p1 + t * r


def polygon_clip(subject_polygon, clip_polygon):
    subject = ensure_ccw(subject_polygon).tolist()
    clip = ensure_ccw(clip_polygon)

    if len(subject) == 0:
        return np.zeros((0, 2), dtype=float)

    def inside(point, edge_start, edge_end):
        point = np.asarray(point, dtype=float)
        edge_start = np.asarray(edge_start, dtype=float)
        edge_end = np.asarray(edge_end, dtype=float)
        edge = edge_end - edge_start
        rel = point - edge_start
        return edge[0] * rel[1] - edge[1] * rel[0] >= -1e-9

    output = subject
    for idx in range(len(clip)):
        edge_start = clip[idx]
        edge_end = clip[(idx + 1) % len(clip)]
        input_list = output
        output = []
        if len(input_list) == 0:
            break

        previous = input_list[-1]
        for current in input_list:
            current_inside = inside(current, edge_start, edge_end)
            previous_inside = inside(previous, edge_start, edge_end)

            if current_inside:
                if not previous_inside:
                    output.append(line_intersection(previous, current, edge_start, edge_end).tolist())
                output.append(current)
            elif previous_inside:
                output.append(line_intersection(previous, current, edge_start, edge_end).tolist())

            previous = current

    return np.asarray(output, dtype=float)


def bev_iou(corners_a, corners_b):
    poly_a = ensure_ccw(np.asarray(corners_a, dtype=float))
    poly_b = ensure_ccw(np.asarray(corners_b, dtype=float))

    area_a = polygon_area(poly_a)
    area_b = polygon_area(poly_b)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0

    inter_poly = polygon_clip(poly_a, poly_b)
    inter_area = polygon_area(inter_poly)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return float(inter_area / union)


def iou_3d(pred_obj, gt_obj):
    pred_corners = pred_box_corners(pred_obj)
    gt_corners = gt_box_corners(gt_obj)

    area_pred = polygon_area(pred_corners)
    area_gt = polygon_area(gt_corners)
    if area_pred <= 0.0 or area_gt <= 0.0:
        return 0.0

    inter_poly = polygon_clip(pred_corners, gt_corners)
    inter_area = polygon_area(inter_poly)
    if inter_area <= 0.0:
        return 0.0

    pred_z_min, pred_z_max = pred_z_range(pred_obj)
    gt_z_min, gt_z_max = gt_z_range(gt_obj)
    z_overlap = max(0.0, min(pred_z_max, gt_z_max) - max(pred_z_min, gt_z_min))
    if z_overlap <= 0.0:
        return 0.0

    pred_height = max(0.0, pred_z_max - pred_z_min)
    gt_height = max(0.0, gt_z_max - gt_z_min)
    pred_volume = area_pred * pred_height
    gt_volume = area_gt * gt_height
    inter_volume = inter_area * z_overlap
    union = pred_volume + gt_volume - inter_volume
    if union <= 0.0:
        return 0.0
    return float(inter_volume / union)


def compute_ap_from_detections(detections, num_gt):
    if num_gt <= 0:
        return None
    if len(detections) == 0:
        return 0.0

    detections = sorted(detections, key=lambda item: item["score"], reverse=True)
    tp = np.array([1.0 if item["is_tp"] else 0.0 for item in detections], dtype=float)
    fp = 1.0 - tp

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall = tp_cum / max(float(num_gt), 1.0)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[0.0], precision, [0.0]])

    for idx in range(len(precision) - 1, 0, -1):
        precision[idx - 1] = max(precision[idx - 1], precision[idx])

    changing_points = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum(
        (recall[changing_points + 1] - recall[changing_points])
        * precision[changing_points + 1]
    )
    return float(ap)


def evaluate_ap_for_view(samples, class_name, iou_threshold, view):
    gt_by_sample = {}
    for sample in samples:
        sample_id = sample["sample_id"]
        gt_by_sample[sample_id] = [
            obj for obj in sample["gt_objects"]
            if normalize_class_name(obj.get("class")) == class_name
        ]

    num_gt = sum(len(items) for items in gt_by_sample.values())

    pred_items = []
    for sample in samples:
        sample_id = sample["sample_id"]
        for pred in sample["pred_objects"]:
            if normalize_class_name(pred.get("class")) != class_name:
                continue
            pred_items.append({
                "sample_id": sample_id,
                "pred": pred,
                "score": float(pred.get("score", 0.0))
            })

    pred_items.sort(key=lambda item: item["score"], reverse=True)
    used_gt = {sample_id: set() for sample_id in gt_by_sample}
    detections = []

    for item in pred_items:
        sample_id = item["sample_id"]
        pred = item["pred"]
        gts = gt_by_sample.get(sample_id, [])

        best_iou = 0.0
        best_gt_idx = None

        for gt_idx, gt in enumerate(gts):
            if gt_idx in used_gt[sample_id]:
                continue

            if view == "bev":
                cur_iou = bev_iou(pred_box_corners(pred), gt_box_corners(gt))
            elif view == "3d":
                cur_iou = iou_3d(pred, gt)
            else:
                raise ValueError(f"未知 AP view: {view}")

            if cur_iou > best_iou:
                best_iou = cur_iou
                best_gt_idx = gt_idx

        is_tp = best_gt_idx is not None and best_iou >= iou_threshold
        if is_tp:
            used_gt[sample_id].add(best_gt_idx)

        detections.append({
            "score": item["score"],
            "is_tp": is_tp,
            "best_iou": float(best_iou)
        })

    ap = compute_ap_from_detections(detections, num_gt)
    num_tp = sum(1 for item in detections if item["is_tp"])

    return {
        "class": class_name,
        "view": view,
        "iou_threshold": float(iou_threshold),
        "num_gt": int(num_gt),
        "num_pred": int(len(pred_items)),
        "num_tp": int(num_tp),
        "ap": ap
    }


def evaluate_average_precision(samples):
    results = {
        "classes": AP_CLASSES,
        "iou_thresholds": AP_IOU_THRESHOLDS,
        "metrics": {
            "bev": {},
            "3d": {}
        },
        "map": {
            "bev": {},
            "3d": {}
        }
    }

    for view in ["bev", "3d"]:
        for threshold in AP_IOU_THRESHOLDS:
            threshold_key = f"iou_{threshold:.2f}"
            class_metrics = {}
            ap_values = []

            for class_name in AP_CLASSES:
                metric = evaluate_ap_for_view(samples, class_name, threshold, view)
                class_metrics[class_name] = metric
                if metric["ap"] is not None:
                    ap_values.append(metric["ap"])

            results["metrics"][view][threshold_key] = class_metrics
            results["map"][view][threshold_key] = (
                float(np.mean(ap_values)) if len(ap_values) > 0 else None
            )

    return results


def center_distance_3d(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return float(np.linalg.norm(a - b))


def center_distance_bev(a, b):
    a = np.array(a[:2], dtype=float)
    b = np.array(b[:2], dtype=float)
    return float(np.linalg.norm(a - b))


def match_predictions_to_gt(preds, gts, distance_threshold=5.0, class_aware=False):
    """
    贪心匹配：
    1. 每个 prediction 找最近的未匹配 GT
    2. 距离小于阈值则匹配
    3. class_aware=False 时先按空间匹配，再在匹配对上统计类别是否正确
    """
    used_gt = set()
    matches = []

    # 置信度高的预测优先匹配
    pred_indices = list(range(len(preds)))
    pred_indices.sort(key=lambda i: float(preds[i].get("score", 0.0)), reverse=True)

    for pi in pred_indices:
        pred = preds[pi]
        pred_cls = normalize_class_name(pred.get("class", "Unknown"))

        best_gi = None
        best_dist = 1e18

        for gi, gt in enumerate(gts):
            if gi in used_gt:
                continue

            gt_cls = normalize_class_name(gt.get("class", "Unknown"))

            if class_aware and pred_cls != gt_cls:
                continue

            dist = center_distance_bev(pred["center_world"], gt["center_world"])

            if dist < best_dist:
                best_dist = dist
                best_gi = gi

        if best_gi is not None and best_dist <= distance_threshold:
            used_gt.add(best_gi)
            gt = gts[best_gi]
            gt_cls = normalize_class_name(gt.get("class", "Unknown"))

            matches.append({
                "pred_index": pi,
                "gt_index": best_gi,
                "pred_class": pred_cls,
                "gt_class": gt_cls,
                "pred_score": float(pred.get("score", 0.0)),
                "distance_bev": float(best_dist),
                "distance_3d": float(center_distance_3d(
                    pred["center_world"],
                    gts[best_gi]["center_world"]
                )),
                "class": pred_cls,
                "class_correct": bool(pred_cls == gt_cls)
            })

    matched_pred = {m["pred_index"] for m in matches}
    matched_gt = {m["gt_index"] for m in matches}

    missed_gt = [i for i in range(len(gts)) if i not in matched_gt]
    false_pred = [i for i in range(len(preds)) if i not in matched_pred]

    return matches, missed_gt, false_pred


def compute_class_counts(objects):
    counts = {
        "Car": 0,
        "Pedestrian": 0,
        "Cyclist": 0,
        "Other": 0
    }

    for obj in objects:
        cls = normalize_class_name(obj.get("class", "Unknown"))
        if cls in counts:
            counts[cls] += 1
        else:
            counts["Other"] += 1

    return counts


def safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def f1_score(precision, recall):
    if precision is None or recall is None:
        return None
    denom = precision + recall
    if denom <= 0:
        return 0.0
    return float(2.0 * precision * recall / denom)


def build_match_pair_rows(sample_id, pred_objects, gt_objects, matches):
    rows = []
    for match in matches:
        pred = pred_objects[match["pred_index"]]
        gt = gt_objects[match["gt_index"]]
        pred_cls = normalize_class_name(pred.get("class", "Unknown"))
        gt_cls = normalize_class_name(gt.get("class", "Unknown"))
        rows.append({
            "sample_id": sample_id,
            "pred_index": int(match["pred_index"]),
            "gt_index": int(match["gt_index"]),
            "pred_class": pred_cls,
            "gt_class": gt_cls,
            "pred_score": float(pred.get("score", 0.0)),
            "center_distance_bev": float(match["distance_bev"]),
            "center_distance_3d": float(match["distance_3d"]),
            "class_correct": bool(pred_cls == gt_cls)
        })
    return rows


def compute_per_class_metrics(samples, matched_pair_rows):
    gt_counts = {class_name: 0 for class_name in AP_CLASSES}
    pred_counts = {class_name: 0 for class_name in AP_CLASSES}
    true_positive = {class_name: 0 for class_name in AP_CLASSES}

    for sample in samples:
        for gt in sample["gt_objects"]:
            cls = normalize_class_name(gt.get("class", "Unknown"))
            if cls in gt_counts:
                gt_counts[cls] += 1
        for pred in sample["pred_objects"]:
            cls = normalize_class_name(pred.get("class", "Unknown"))
            if cls in pred_counts:
                pred_counts[cls] += 1

    for row in matched_pair_rows:
        pred_cls = row["pred_class"]
        gt_cls = row["gt_class"]
        if pred_cls == gt_cls and gt_cls in true_positive:
            true_positive[gt_cls] += 1

    rows = []
    for class_name in AP_CLASSES:
        tp = int(true_positive[class_name])
        fp = int(pred_counts[class_name] - tp)
        fn = int(gt_counts[class_name] - tp)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        rows.append({
            "class": class_name,
            "gt_count": int(gt_counts[class_name]),
            "pred_count": int(pred_counts[class_name]),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1_score(precision, recall)
        })
    return rows


def build_confusion_matrix(matched_pair_rows):
    matrix = pd.DataFrame(0, index=AP_CLASSES, columns=AP_CLASSES, dtype=int)
    for row in matched_pair_rows:
        gt_cls = row["gt_class"]
        pred_cls = row["pred_class"]
        if gt_cls in matrix.index and pred_cls in matrix.columns:
            matrix.loc[gt_cls, pred_cls] += 1
    matrix.index.name = "gt_class"
    matrix.columns.name = "pred_class"
    return matrix


def plot_confusion_matrix(confusion_df, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    values = confusion_df.values.astype(float)
    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    im = ax.imshow(values, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(confusion_df.columns)))
    ax.set_yticks(np.arange(len(confusion_df.index)))
    ax.set_xticklabels(confusion_df.columns)
    ax.set_yticklabels(confusion_df.index)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("GT class")
    ax.set_title("Matched-object class confusion matrix")

    max_value = values.max() if values.size > 0 else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if max_value > 0 and values[i, j] > max_value / 2.0 else "black"
            ax.text(j, i, str(int(values[i, j])), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_per_class_metrics(per_class_df, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    classes = per_class_df["class"].tolist()
    x = np.arange(len(classes))
    width = 0.25

    precision = per_class_df["precision"].fillna(0.0).to_numpy(dtype=float)
    recall = per_class_df["recall"].fillna(0.0).to_numpy(dtype=float)
    f1 = per_class_df["f1"].fillna(0.0).to_numpy(dtype=float)

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, precision, width, label="Precision")
    plt.bar(x, recall, width, label="Recall")
    plt.bar(x + width, f1, width, label="F1")
    plt.xticks(x, classes)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score")
    plt.title("Per-class precision / recall / F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_class_distribution(per_class_df, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    classes = per_class_df["class"].tolist()
    x = np.arange(len(classes))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2.0, per_class_df["gt_count"], width, label="GT")
    plt.bar(x + width / 2.0, per_class_df["pred_count"], width, label="Prediction")
    plt.xticks(x, classes)
    plt.ylabel("Object count")
    plt.title("GT vs prediction class distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_score_distribution(matched_pair_rows, samples, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    matched_scores = [float(row["pred_score"]) for row in matched_pair_rows]
    matched_pred_keys = {
        (row["sample_id"], int(row["pred_index"]))
        for row in matched_pair_rows
    }

    false_scores = []
    for sample in samples:
        sample_id = sample["sample_id"]
        for pred_idx, pred in enumerate(sample["pred_objects"]):
            if (sample_id, pred_idx) not in matched_pred_keys:
                false_scores.append(float(pred.get("score", 0.0)))

    bins = np.linspace(0.0, 1.0, 21)
    plt.figure(figsize=(8, 5))
    if matched_scores:
        plt.hist(matched_scores, bins=bins, alpha=0.65, label="Matched predictions")
    if false_scores:
        plt.hist(false_scores, bins=bins, alpha=0.65, label="False predictions")
    if not matched_scores and not false_scores:
        plt.hist([], bins=bins)
    plt.xlabel("Prediction score")
    plt.ylabel("Count")
    plt.title("Score distribution by match status")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_world_compare(
    sample_id,
    gt_objects,
    pred_objects,
    matches,
    missed_gt,
    false_pred,
    save_path,
    classification_accuracy=None,
    precision=None,
    recall=None
):
    """
    一张图展示：
    绿色实线：匹配 GT
    红色实线：漏检 GT
    蓝色虚线：匹配预测
    橙色虚线：误检预测
    灰色虚线：匹配中心连线
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    matched_gt = {m["gt_index"] for m in matches}
    matched_pred = {m["pred_index"] for m in matches}

    all_xy = []

    for gt in gt_objects:
        all_xy.append(np.array(gt["center_world"][:2], dtype=float))

    for pred in pred_objects:
        all_xy.append(np.array(pred["center_world"][:2], dtype=float))

    if len(all_xy) == 0:
        origin = np.array([0.0, 0.0], dtype=float)
    else:
        origin = np.mean(np.stack(all_xy, axis=0), axis=0)

    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # GT
    for gi, gt in enumerate(gt_objects):
        corners = gt_box_corners(gt) - origin
        center = np.array(gt["center_world"][:2], dtype=float) - origin

        if gi in matched_gt:
            edge_color = "green"
            label_text = "GT matched"
        else:
            edge_color = "red"
            label_text = "GT missed"

        poly = Polygon(
            corners,
            closed=True,
            fill=False,
            edgecolor=edge_color,
            linewidth=1.5
        )
        ax.add_patch(poly)

        plt.scatter(center[0], center[1], c=edge_color, s=15)

        plt.text(
            center[0],
            center[1],
            f"GT-{normalize_class_name(gt['class'])}",
            fontsize=6,
            color=edge_color
        )

    # Predictions
    for pi, pred in enumerate(pred_objects):
        corners = pred_box_corners(pred) - origin
        center = np.array(pred["center_world"][:2], dtype=float) - origin

        if pi in matched_pred:
            edge_color = "blue"
            label_text = "Pred matched"
        else:
            edge_color = "orange"
            label_text = "Pred false"

        poly = Polygon(
            corners,
            closed=True,
            fill=False,
            edgecolor=edge_color,
            linewidth=1.3,
            linestyle="--"
        )
        ax.add_patch(poly)

        plt.scatter(center[0], center[1], c=edge_color, s=12)

        plt.text(
            center[0],
            center[1],
            f"P-{normalize_class_name(pred['class'])} {float(pred.get('score', 0.0)):.2f}",
            fontsize=6,
            color=edge_color
        )

    # Match lines
    for m in matches:
        pred = pred_objects[m["pred_index"]]
        gt = gt_objects[m["gt_index"]]

        p = np.array(pred["center_world"][:2], dtype=float) - origin
        g = np.array(gt["center_world"][:2], dtype=float) - origin

        plt.plot(
            [p[0], g[0]],
            [p[1], g[1]],
            color="gray",
            linewidth=0.8,
            linestyle=":"
        )

    cls_acc_text = "N/A" if classification_accuracy is None else f"{classification_accuracy:.3f}"
    precision_text = "N/A" if precision is None else f"{precision:.3f}"
    recall_text = "N/A" if recall is None else f"{recall:.3f}"

    plt.title(
        f"Vehicle LiDAR-only vs cooperative label_world | {sample_id}\n"
        f"GT={len(gt_objects)}, Pred={len(pred_objects)}, "
        f"Match={len(matches)}, Missed={len(missed_gt)}, False Pred={len(false_pred)}\n"
        f"Matched classification accuracy={cls_acc_text}, "
        f"Precision={precision_text}, Recall={recall_text}"
    )
    plt.xlabel("World x, shifted")
    plt.ylabel("World y, shifted")
    plt.axis("equal")
    plt.grid(True, linestyle="--", linewidth=0.3, alpha=0.4)

    legend_items = [
        Patch(edgecolor="green", facecolor="none", label="Matched GT"),
        Patch(edgecolor="red", facecolor="none", label="Missed GT"),
        Patch(edgecolor="blue", facecolor="none", label="Matched prediction"),
        Patch(edgecolor="orange", facecolor="none", label="False prediction"),
    ]

    plt.legend(handles=legend_items, loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=240)
    plt.close()


def plot_summary(eval_df):
    SUMMARY_VIS_DIR.mkdir(parents=True, exist_ok=True)

    metric_names = [
        "false_negative",
        "false_positive",
        "count_error",
        "mean_loc_error_bev",
        "recall",
        "precision",
        "f1",
        "classification_accuracy"
    ]

    metric_values = []

    for name in metric_names:
        if name not in eval_df.columns:
            metric_values.append(0.0)
            continue

        values = eval_df[name].dropna()

        if len(values) == 0:
            metric_values.append(0.0)
        else:
            metric_values.append(float(values.mean()))

    # 平均指标柱状图
    plt.figure(figsize=(9, 5))
    plt.bar(metric_names, metric_values)
    plt.ylabel("Average value")
    plt.title("Vehicle LiDAR-only baseline summary")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(SUMMARY_VIS_DIR / "metric_summary_bar.png", dpi=220)
    plt.close()

    # 每帧数量对比图
    plt.figure(figsize=(10, 5))
    plt.plot(eval_df["sample_id"], eval_df["num_gt"], marker="o", label="GT count")
    plt.plot(eval_df["sample_id"], eval_df["num_pred"], marker="o", label="Prediction count")
    plt.plot(eval_df["sample_id"], eval_df["num_match"], marker="o", label="Matched count")
    plt.xticks(rotation=60)
    plt.ylabel("Count")
    plt.title("Per-sample GT / prediction / matched counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SUMMARY_VIS_DIR / "per_sample_count_curve.png", dpi=220)
    plt.close()

    # 每帧误差图
    plt.figure(figsize=(10, 5))
    plt.plot(eval_df["sample_id"], eval_df["false_negative"], marker="o", label="False negative")
    plt.plot(eval_df["sample_id"], eval_df["false_positive"], marker="o", label="False positive")
    plt.plot(eval_df["sample_id"], eval_df["count_error"], marker="o", label="Count error")
    plt.xticks(rotation=60)
    plt.ylabel("Count")
    plt.title("Per-sample error curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SUMMARY_VIS_DIR / "per_sample_error_curve.png", dpi=220)
    plt.close()

    # 定位误差图
    if "mean_loc_error_bev" in eval_df.columns:
        loc_df = eval_df.dropna(subset=["mean_loc_error_bev"])
        if len(loc_df) > 0:
            plt.figure(figsize=(10, 5))
            plt.plot(
                loc_df["sample_id"],
                loc_df["mean_loc_error_bev"],
                marker="o",
                label="Mean BEV localization error"
            )
            plt.xticks(rotation=60)
            plt.ylabel("Meters")
            plt.title("Per-sample localization error")
            plt.legend()
            plt.tight_layout()
            plt.savefig(SUMMARY_VIS_DIR / "per_sample_localization_error.png", dpi=220)
            plt.close()


def main():
    print("=" * 80)
    print("Evaluate and visualize Vehicle LiDAR-only baseline")
    print("=" * 80)

    WORLD_VIS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_VIS_DIR.mkdir(parents=True, exist_ok=True)

    predictions_world = load_json(PRED_WORLD_PATH)
    sample_mapping = build_sample_mapping()

    eval_rows = []
    detailed_records = []
    ap_samples = []
    all_matched_pair_rows = []

    for idx, pred_record in enumerate(predictions_world):
        sample_id = pred_record["sample_id"]
        vehicle_id = pred_record.get("vehicle_id", sample_id)

        if vehicle_id not in sample_mapping:
            print(f"[WARN] 找不到 cooperative label_world 映射: {vehicle_id}")
            continue

        label_world_path = sample_mapping[vehicle_id]["cooperative_label_path"]

        gt_objects = parse_gt_objects(label_world_path)
        pred_objects = pred_record.get("pred_objects", [])
        ap_samples.append({
            "sample_id": sample_id,
            "gt_objects": gt_objects,
            "pred_objects": pred_objects
        })

        matches, missed_gt, false_pred = match_predictions_to_gt(
            preds=pred_objects,
            gts=gt_objects,
            distance_threshold=5.0,
            class_aware=False
        )

        num_gt = len(gt_objects)
        num_pred = len(pred_objects)
        num_match = len(matches)

        false_negative = len(missed_gt)
        false_positive = len(false_pred)
        count_error = abs(num_pred - num_gt)

        if num_match > 0:
            mean_loc_error_bev = float(np.mean([m["distance_bev"] for m in matches]))
            mean_loc_error_3d = float(np.mean([m["distance_3d"] for m in matches]))
        else:
            mean_loc_error_bev = None
            mean_loc_error_3d = None

        recall = num_match / num_gt if num_gt > 0 else None
        precision = num_match / num_pred if num_pred > 0 else None
        f1 = f1_score(precision, recall)

        matched_cls_correct = sum(1 for m in matches if m.get("class_correct", False))
        matched_cls_wrong = num_match - matched_cls_correct
        classification_accuracy = (
            matched_cls_correct / num_match if num_match > 0 else None
        )

        gt_counts = compute_class_counts(gt_objects)
        pred_counts = compute_class_counts(pred_objects)
        matched_pair_rows = build_match_pair_rows(
            sample_id=sample_id,
            pred_objects=pred_objects,
            gt_objects=gt_objects,
            matches=matches
        )
        all_matched_pair_rows.extend(matched_pair_rows)

        row = {
            "sample_id": sample_id,
            "vehicle_id": vehicle_id,
            "num_gt": num_gt,
            "num_pred": num_pred,
            "num_match": num_match,
            "false_negative": false_negative,
            "false_positive": false_positive,
            "count_error": count_error,
            "mean_loc_error_bev": mean_loc_error_bev,
            "mean_loc_error_3d": mean_loc_error_3d,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "matched_cls_correct": matched_cls_correct,
            "matched_cls_wrong": matched_cls_wrong,
            "classification_accuracy": classification_accuracy,
            "gt_car": gt_counts["Car"],
            "gt_pedestrian": gt_counts["Pedestrian"],
            "gt_cyclist": gt_counts["Cyclist"],
            "pred_car": pred_counts["Car"],
            "pred_pedestrian": pred_counts["Pedestrian"],
            "pred_cyclist": pred_counts["Cyclist"]
        }

        eval_rows.append(row)

        detailed_records.append({
            "sample_id": sample_id,
            "vehicle_id": vehicle_id,
            "label_world_path": label_world_path,
            "matches": matches,
            "matched_pairs": matched_pair_rows,
            "missed_gt_indices": missed_gt,
            "false_pred_indices": false_pred,
            "metrics": row
        })

        save_path = WORLD_VIS_DIR / f"sample_{idx:03d}_{sample_id}_world_compare.png"

        plot_world_compare(
            sample_id=sample_id,
            gt_objects=gt_objects,
            pred_objects=pred_objects,
            matches=matches,
            missed_gt=missed_gt,
            false_pred=false_pred,
            save_path=save_path,
            classification_accuracy=classification_accuracy,
            precision=precision,
            recall=recall
        )

        print(
            f"[{idx + 1}/{len(predictions_world)}] {sample_id}: "
            f"GT={num_gt}, Pred={num_pred}, Match={num_match}, "
            f"FN={false_negative}, FP={false_positive}, "
            f"CountErr={count_error}, "
            f"LocErrBEV={mean_loc_error_bev}, "
            f"Recall={recall}, Precision={precision}, "
            f"ClsAcc={classification_accuracy}"
        )

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(EVAL_CSV_PATH, index=False, encoding="utf-8-sig")

    save_json(detailed_records, BASELINE_ROOT / "vehicle_lidar_only_eval_details.json")
    matched_pairs_df = pd.DataFrame(all_matched_pair_rows)
    matched_pairs_df.to_csv(MATCHED_PAIRS_CSV_PATH, index=False, encoding="utf-8-sig")

    per_class_rows = compute_per_class_metrics(ap_samples, all_matched_pair_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df.to_csv(PER_CLASS_METRICS_CSV_PATH, index=False, encoding="utf-8-sig")

    confusion_df = build_confusion_matrix(all_matched_pair_rows)
    confusion_df.to_csv(CONFUSION_MATRIX_CSV_PATH, encoding="utf-8-sig")

    if len(eval_df) > 0:
        plot_summary(eval_df)
        plot_confusion_matrix(
            confusion_df,
            SUMMARY_VIS_DIR / "class_confusion_matrix.png"
        )
        plot_per_class_metrics(
            per_class_df,
            SUMMARY_VIS_DIR / "per_class_precision_recall_f1.png"
        )
        plot_class_distribution(
            per_class_df,
            SUMMARY_VIS_DIR / "gt_pred_class_distribution.png"
        )
        plot_score_distribution(
            all_matched_pair_rows,
            ap_samples,
            SUMMARY_VIS_DIR / "score_distribution_by_match.png"
        )

    ap_summary = evaluate_average_precision(ap_samples)
    save_json(ap_summary, AP_SUMMARY_PATH)

    pred_class_set = sorted({
        normalize_class_name(pred.get("class", "Unknown"))
        for sample in ap_samples
        for pred in sample["pred_objects"]
    })
    gt_class_set = sorted({
        normalize_class_name(gt.get("class", "Unknown"))
        for sample in ap_samples
        for gt in sample["gt_objects"]
    })

    if pred_class_set == ["Car"]:
        evaluation_scope_note = (
            "Current result is Car-only detection evaluation because all predictions are Car. "
            "Do not interpret classification_accuracy as full multi-class classification accuracy. "
            "If the detector/config supports it, run multi-class prediction with Car/Pedestrian/Cyclist "
            "and then recompute the full classification analysis."
        )
    else:
        evaluation_scope_note = (
            "Current result contains multi-class predictions; matched-object classification metrics "
            "are computed over Car/Pedestrian/Cyclist where available."
        )

    total_match = int(len(all_matched_pair_rows))
    matched_cls_correct = int(sum(1 for row in all_matched_pair_rows if row["class_correct"]))
    matched_cls_wrong = int(total_match - matched_cls_correct)
    overall_classification_accuracy = (
        matched_cls_correct / total_match if total_match > 0 else None
    )

    per_class_metrics = {
        row["class"]: {
            "gt_count": int(row["gt_count"]),
            "pred_count": int(row["pred_count"]),
            "TP": int(row["TP"]),
            "FP": int(row["FP"]),
            "FN": int(row["FN"]),
            "precision": None if pd.isna(row["precision"]) else float(row["precision"]),
            "recall": None if pd.isna(row["recall"]) else float(row["recall"]),
            "f1": None if pd.isna(row["f1"]) else float(row["f1"])
        }
        for _, row in per_class_df.iterrows()
    }

    summary = {
        "num_samples": int(len(eval_df)),
        "output_csv": str(EVAL_CSV_PATH),
        "details_json": str(BASELINE_ROOT / "vehicle_lidar_only_eval_details.json"),
        "matched_pairs_csv": str(MATCHED_PAIRS_CSV_PATH),
        "per_class_metrics_path": str(PER_CLASS_METRICS_CSV_PATH),
        "confusion_matrix_path": str(CONFUSION_MATRIX_CSV_PATH),
        "ap_summary_json": str(AP_SUMMARY_PATH),
        "visualization_world_dir": str(WORLD_VIS_DIR),
        "visualization_summary_dir": str(SUMMARY_VIS_DIR),
        "class_confusion_matrix_png": str(SUMMARY_VIS_DIR / "class_confusion_matrix.png"),
        "per_class_metrics_png": str(SUMMARY_VIS_DIR / "per_class_precision_recall_f1.png"),
        "gt_pred_class_distribution_png": str(SUMMARY_VIS_DIR / "gt_pred_class_distribution.png"),
        "score_distribution_by_match_png": str(SUMMARY_VIS_DIR / "score_distribution_by_match.png"),
        "matching_method": "class-agnostic greedy center matching in BEV, then class correctness analysis",
        "distance_threshold_meter": 5.0,
        "ap_method": "class-aware IoU PR matching in world coordinate",
        "ap_iou_thresholds": AP_IOU_THRESHOLDS,
        "gt_classes_present": gt_class_set,
        "pred_classes_present": pred_class_set,
        "matched_cls_correct": matched_cls_correct,
        "matched_cls_wrong": matched_cls_wrong,
        "classification_accuracy": overall_classification_accuracy,
        "per_class_metrics": per_class_metrics,
        "evaluation_scope_note": evaluation_scope_note,
        "note": "Current predictions may be label-oracle engineering outputs if prediction_type is vehicle_lidar_label_oracle."
    }

    if len(eval_df) > 0:
        numeric_means = eval_df.mean(numeric_only=True).to_dict()
        summary["mean_metrics"] = {
            k: float(v) for k, v in numeric_means.items()
        }
        summary["mean_precision"] = (
            None if pd.isna(eval_df["precision"].mean()) else float(eval_df["precision"].mean())
        )
        summary["mean_recall"] = (
            None if pd.isna(eval_df["recall"].mean()) else float(eval_df["recall"].mean())
        )
        summary["mean_f1"] = (
            None if pd.isna(eval_df["f1"].mean()) else float(eval_df["f1"].mean())
        )
        summary["map_bev_iou_0.50"] = ap_summary["map"]["bev"]["iou_0.50"]
        summary["map_3d_iou_0.50"] = ap_summary["map"]["3d"]["iou_0.50"]
        summary["ap_bev_iou_0.50_by_class"] = {
            cls: ap_summary["metrics"]["bev"]["iou_0.50"][cls]["ap"]
            for cls in AP_CLASSES
        }
        summary["ap_3d_iou_0.50_by_class"] = {
            cls: ap_summary["metrics"]["3d"]["iou_0.50"][cls]["ap"]
            for cls in AP_CLASSES
        }

    save_json(summary, EVAL_SUMMARY_PATH)

    print("=" * 80)
    print("评价与可视化完成")
    print(f"Eval CSV: {EVAL_CSV_PATH}")
    print(f"Eval summary: {EVAL_SUMMARY_PATH}")
    print(f"AP summary: {AP_SUMMARY_PATH}")
    print(f"World visualization dir: {WORLD_VIS_DIR}")
    print(f"Summary visualization dir: {SUMMARY_VIS_DIR}")

    if len(eval_df) > 0:
        print("=" * 80)
        print("平均指标:")
        print(eval_df[
            [
                "num_gt",
                "num_pred",
                "num_match",
                "false_negative",
                "false_positive",
                "count_error",
                "mean_loc_error_bev",
                "mean_loc_error_3d",
                "recall",
                "precision",
                "f1",
                "classification_accuracy"
            ]
        ].mean(numeric_only=True))
        print("Matched classification correct/wrong:",
              matched_cls_correct, "/", matched_cls_wrong)
        print("Overall matched classification accuracy:",
              overall_classification_accuracy)
        print("Evaluation scope:", evaluation_scope_note)
        print("=" * 80)
        print("AP / mAP @ IoU=0.50:")
        print("BEV mAP:", ap_summary["map"]["bev"]["iou_0.50"])
        print("3D mAP:", ap_summary["map"]["3d"]["iou_0.50"])
        print("BEV AP by class:", summary["ap_bev_iou_0.50_by_class"])
        print("3D AP by class:", summary["ap_3d_iou_0.50_by_class"])


if __name__ == "__main__":
    main()
