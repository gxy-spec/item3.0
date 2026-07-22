#!/usr/bin/env python3
"""Create paper-ready analysis figures for four DAIR-V2X single-modal baselines.

AP and PR use DAIR-V2X's local-coordinate ``eval_utils.py`` protocol.  TP/FP/
FN and F1 retain the project's world-coordinate 5 m centre-match diagnostic
protocol, and every relevant figure labels that distinction explicitly.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
ITEM3_ROOT = ROOT.parent
OFFICIAL_ROOT = ITEM3_ROOT / "DAIR-V2X" / "v2x"
for path in (ROOT, OFFICIAL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_and_visualize_vehicle_lidar_baseline as legacy  # noqa: E402
import evaluate_official_aligned_local_ap as local_eval  # noqa: E402
from v2x_utils import eval_utils as official  # noqa: E402


DEFAULT_OUTPUT = ROOT / "results" / "visualization" / "baseline_analysis"
DEFAULT_DATA_ROOT = Path(
    "/mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full/"
    "cooperative-vehicle-infrastructure"
)

STYLE = {
    "font.family": "DejaVu Serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
COLORS = ["#4C78A8", "#59A14F", "#B07AA1", "#E17C05"]


def default_config():
    root = ROOT / "outputs" / "full_baselines"
    return {
        "data_root": str(DEFAULT_DATA_ROOT),
        "evaluation_class": "car",
        "iou_threshold": 0.50,
        "distance_bins_m": [[0, 30], [30, 50], [50, 100]],
        "baselines": [
            {
                "key": "vehicle_lidar",
                "label": "Vehicle LiDAR-only",
                "modality": "LiDAR",
                "device": "Vehicle",
                "model": "PointPillars",
                "epochs": 20,
                "label_rel": "vehicle-side/label/lidar",
                "root": str(root / "vehicle_lidar"),
                "prediction_file": "predictions_vehicle_lidar.json",
            },
            {
                "key": "infrastructure_lidar",
                "label": "Infrastructure LiDAR-only",
                "modality": "LiDAR",
                "device": "Infrastructure",
                "model": "PointPillars",
                "epochs": 20,
                "label_rel": "infrastructure-side/label/virtuallidar",
                "root": str(root / "infrastructure_lidar"),
                "prediction_file": "predictions_infrastructure_lidar.json",
            },
            {
                "key": "vehicle_camera",
                "label": "Vehicle Camera-only",
                "modality": "Camera",
                "device": "Vehicle",
                "model": "ImVoxelNet",
                "epochs": 1,
                "label_rel": "vehicle-side/label/lidar",
                "root": str(root / "vehicle_camera"),
                "prediction_file": "predictions_vehicle_camera.json",
            },
            {
                "key": "infrastructure_camera",
                "label": "Infrastructure Camera-only",
                "modality": "Camera",
                "device": "Infrastructure",
                "model": "ImVoxelNet",
                "epochs": 1,
                "label_rel": "infrastructure-side/label/virtuallidar",
                "root": str(root / "infrastructure_camera"),
                "prediction_file": "predictions_infrastructure_camera.json",
            },
        ],
    }


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_or_create_config(path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(path, default_config())
        print(f"[INFO] Created data interface config: {path}")
    return load_json(path)


def resolve_input_paths(item):
    root = Path(item["root"])
    return {
        "root": root,
        "prediction": root / item["prediction_file"],
        "official": root / "official_local_aligned_ap_summary.json",
        "diagnostic": root / "eval_summary.json",
    }


def in_distance_bin(obj, lower, upper):
    center = obj.get("center_world", obj.get("center_lidar", []))
    if len(center) < 2:
        return False
    distance = float(np.hypot(center[0], center[1]))
    return lower <= distance < upper


def official_rows_for_scope(predictions, item, data_root, lower=None, upper=None):
    rows, total_gt, skipped = [], 0, []
    for record in predictions:
        source_id = str(record.get("source_id", record.get("sample_id", "")))
        label_path = data_root / item["label_rel"] / f"{source_id}.json"
        if not label_path.is_file():
            skipped.append(source_id)
            continue
        gt_objects = legacy.parse_gt_objects(label_path)
        pred_objects = local_eval.local_prediction_objects(record)
        if lower is not None:
            gt_objects = [obj for obj in gt_objects if in_distance_bin(obj, lower, upper)]
            pred_objects = [obj for obj in pred_objects if in_distance_bin(obj, lower, upper)]
        pred, gt = local_eval.official_eval.official_record(pred_objects, gt_objects)
        frame_rows, count, _ = official.compute_type(
            gt, pred, "car", 0.50, "3d"
        )
        rows.extend({"score": float(row["score"]), "type": row["type"]} for row in frame_rows)
        total_gt += int(count)
    return rows, total_gt, skipped


def pr_curve(rows, total_gt):
    if not rows or not total_gt:
        return [0.0], [1.0]
    sorted_rows = sorted(rows, key=lambda row: row["score"], reverse=True)
    tp = np.cumsum([row["type"] == "tp" for row in sorted_rows], dtype=float)
    precision = tp / np.arange(1, len(sorted_rows) + 1)
    recall = tp / float(total_gt)
    # Precision envelope is conventional for visualizing AP-style PR curves.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return np.concatenate(([0.0], recall)), np.concatenate(([precision[0]], precision))


def gather_metrics(config, output_dir, recompute=False):
    cache_path = output_dir / "baseline_analysis_metrics.json"
    if cache_path.is_file() and not recompute:
        return load_json(cache_path)

    data_root = Path(config["data_root"])
    records = []
    for item in config["baselines"]:
        paths = resolve_input_paths(item)
        status = "ready"
        missing = [str(path) for path in paths.values() if path.name != "eval_summary.json" and not path.exists()]
        official_data = load_json(paths["official"]) if paths["official"].is_file() else None
        diagnostic = load_json(paths["diagnostic"]) if paths["diagnostic"].is_file() else None
        if missing:
            status = "missing_input"
        metrics = {"item": item, "status": status, "missing": missing}
        if official_data:
            metrics["ap3d"] = 100.0 * official_data["metrics"]["3d"]["car"]["iou_0.50"]["ap"]
            metrics["apbev"] = 100.0 * official_data["metrics"]["bev"]["car"]["iou_0.50"]["ap"]
        else:
            metrics["ap3d"] = metrics["apbev"] = None
        if diagnostic:
            metrics.update({
                "f1": 100.0 * float(diagnostic.get("mean_f1", 0.0)),
                "tp": int(diagnostic.get("total_match", 0)),
                "fp": int(diagnostic.get("total_false_positive", 0)),
                "fn": int(diagnostic.get("total_false_negative", 0)),
            })
        else:
            metrics.update({"f1": None, "tp": None, "fp": None, "fn": None})

        if status == "ready":
            predictions = load_json(paths["prediction"])
            rows, total_gt, skipped = official_rows_for_scope(predictions, item, data_root)
            recall, precision = pr_curve(rows, total_gt)
            metrics["pr_curve"] = {"recall": recall.tolist(), "precision": precision.tolist()}
            distance = []
            for lower, upper in config["distance_bins_m"]:
                bin_rows, bin_gt, bin_skipped = official_rows_for_scope(
                    predictions, item, data_root, lower, upper
                )
                # Compute both AP views with the same local-coordinate filtering.
                ap3d = 100.0 * official.compute_ap(bin_rows, bin_gt) if bin_gt else None
                bev_rows, bev_gt, _ = official_rows_for_scope_bev(
                    predictions, item, data_root, lower, upper
                )
                apbev = 100.0 * official.compute_ap(bev_rows, bev_gt) if bev_gt else None
                distance.append({
                    "range": f"{lower}-{upper}m", "ap3d": ap3d, "apbev": apbev,
                    "num_gt": bin_gt, "skipped": sorted(set(bin_skipped)),
                })
            metrics["distance"] = distance
            metrics["skipped_samples"] = sorted(set(skipped))
        records.append(metrics)
    result = {
        "protocol": {
            "ap": "DAIR-V2X official eval_utils.py, local sensor coordinate, Car, IoU=0.50",
            "diagnostic": "project world-coordinate 5 m BEV centre matching",
            "distance_bins": "AP is recomputed after filtering local GT and predictions by radial distance.",
        },
        "records": records,
    }
    save_json(cache_path, result)
    return result


def official_rows_for_scope_bev(predictions, item, data_root, lower, upper):
    rows, total_gt, skipped = [], 0, []
    for record in predictions:
        source_id = str(record.get("source_id", record.get("sample_id", "")))
        label_path = data_root / item["label_rel"] / f"{source_id}.json"
        if not label_path.is_file():
            skipped.append(source_id)
            continue
        gt_objects = [obj for obj in legacy.parse_gt_objects(label_path) if in_distance_bin(obj, lower, upper)]
        pred_objects = [obj for obj in local_eval.local_prediction_objects(record) if in_distance_bin(obj, lower, upper)]
        pred, gt = local_eval.official_eval.official_record(pred_objects, gt_objects)
        frame_rows, count, _ = official.compute_type(gt, pred, "car", 0.50, "bev")
        rows.extend({"score": float(row["score"]), "type": row["type"]} for row in frame_rows)
        total_gt += int(count)
    return rows, total_gt, skipped


def save_figure(fig, output_dir, stem):
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def label_bars(ax, bars):
    for bar in bars:
        value = bar.get_height()
        ax.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)


def fig1(records, output_dir):
    labels = [r["item"]["label"].replace("-only", "") for r in records]
    x = np.arange(len(records)); width = 0.24
    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    for offset, key, label, color in [(-width, "ap3d", "AP3D@0.50", "#4C78A8"),
                                      (0, "apbev", "APBEV@0.50", "#59A14F"),
                                      (width, "f1", "World diagnostic F1", "#B07AA1")]:
        values = [r.get(key) or 0.0 for r in records]
        bars = ax.bar(x + offset, values, width, label=label, color=color, edgecolor="white", linewidth=0.5)
        label_bars(ax, bars)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 105)
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(ncol=3, frameon=False, loc="upper center")
    ax.set_title("Fig. 1. Single-modal baseline performance summary")
    save_figure(fig, output_dir, "fig1_comprehensive_performance")


def fig2(records, output_dir):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for record, color in zip(records, COLORS):
        curve = record.get("pr_curve")
        if not curve:
            continue
        ax.plot(curve["recall"], curve["precision"], color=color, linewidth=1.6,
                label=f"{record['item']['label']} (AP3D {record.get('ap3d', 0):.2f})")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Fig. 2. Official local-coordinate PR curves (Car, 3D IoU=0.50)")
    save_figure(fig, output_dir, "fig2_precision_recall")


def fig3(records, output_dir):
    labels = [r["item"]["label"].replace("-only", "") for r in records]
    x = np.arange(len(records)); width = 0.24
    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    for offset, key, label, color in [(-width, "tp", "TP", "#59A14F"),
                                      (0, "fp", "FP", "#E17C05"),
                                      (width, "fn", "FN", "#C44E52")]:
        values = [r.get(key) or 0 for r in records]
        bars = ax.bar(x + offset, values, width, label=label, color=color, edgecolor="white", linewidth=0.5)
        label_bars(ax, bars)
    ax.set_ylabel("Object count")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, ncol=3)
    ax.set_title("Fig. 3. World-coordinate diagnostic error analysis (5 m centre match)")
    save_figure(fig, output_dir, "fig3_tp_fp_fn")


def fig4(records, config, output_dir):
    names = [f"{a}-{b}m" for a, b in config["distance_bins_m"]]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharex=True, sharey=True)
    for ax, key, title in zip(axes, ("ap3d", "apbev"), ("AP3D@0.50", "APBEV@0.50")):
        for record, color in zip(records, COLORS):
            distance = record.get("distance", [])
            values = [entry.get(key) if entry.get(key) is not None else np.nan for entry in distance]
            ax.plot(x, values, marker="o", markersize=4, linewidth=1.5, color=color,
                    label=record["item"]["label"].replace("-only", ""))
            for xi, value in zip(x, values):
                if np.isfinite(value):
                    ax.annotate(f"{value:.2f}", (xi, value), xytext=(0, 4), textcoords="offset points",
                                ha="center", fontsize=6.5, color=color)
        ax.set_title(title)
        ax.set_xticks(x, names)
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("AP")
    axes[1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Fig. 4. Local-coordinate detection performance by distance range", y=1.02, fontsize=10)
    save_figure(fig, output_dir, "fig4_distance_range_performance")


def fig5(records, output_dir):
    matrix = np.full((2, 2), np.nan)
    annotations = [["" for _ in range(2)] for _ in range(2)]
    device_index = {"Vehicle": 0, "Infrastructure": 1}
    modality_index = {"LiDAR": 0, "Camera": 1}
    for record in records:
        item = record["item"]
        row, col = device_index[item["device"]], modality_index[item["modality"]]
        matrix[row, col] = record.get("ap3d", np.nan)
        annotations[row][col] = f"AP3D {record.get('ap3d', 0):.2f}\nAPBEV {record.get('apbev', 0):.2f}"
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, np.nanmax(matrix)))
    ax.set_xticks([0, 1], ["LiDAR", "Camera"])
    ax.set_yticks([0, 1], ["Vehicle", "Infrastructure"])
    for row in range(2):
        for col in range(2):
            value = matrix[row, col]
            color = "white" if np.isfinite(value) and value > np.nanmax(matrix) * 0.55 else "black"
            ax.text(col, row, annotations[row][col], ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("AP3D@0.50")
    ax.set_title("Fig. 5. Device-modality performance matrix\n(cell colour: AP3D; annotation: AP3D / APBEV)")
    save_figure(fig, output_dir, "fig5_modality_device_matrix")


def write_table(records, output_dir):
    fields = ["baseline", "model", "epochs", "ap3d", "apbev", "world_f1", "tp", "fp", "fn", "status"]
    with (output_dir / "baseline_analysis_table.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            item = record["item"]
            writer.writerow({
                "baseline": item["label"], "model": item["model"], "epochs": item["epochs"],
                "ap3d": f"{record.get('ap3d', 0):.2f}", "apbev": f"{record.get('apbev', 0):.2f}",
                "world_f1": f"{record.get('f1', 0):.2f}", "tp": record.get("tp"),
                "fp": record.get("fp"), "fn": record.get("fn"), "status": record["status"],
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--recompute", action="store_true", help="Recompute PR and distance-bin AP cache.")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.config or output_dir / "baseline_analysis_config.json"
    config = load_or_create_config(config_path)
    plt.rcParams.update(STYLE)
    metrics = gather_metrics(config, output_dir, args.recompute)
    records = metrics["records"]
    save_json(output_dir / "baseline_analysis_metrics.json", metrics)
    write_table(records, output_dir)
    fig1(records, output_dir)
    fig2(records, output_dir)
    fig3(records, output_dir)
    fig4(records, config, output_dir)
    fig5(records, output_dir)
    print(json.dumps({
        "output_dir": str(output_dir),
        "figures": [f"fig{i}_{name}.{suffix}" for i, name in [
            (1, "comprehensive_performance"), (2, "precision_recall"),
            (3, "tp_fp_fn"), (4, "distance_range_performance"),
            (5, "modality_device_matrix"),
        ] for suffix in ("png", "pdf")],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
