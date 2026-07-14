import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
BASELINES_ROOT = PROJECT_ROOT / "outputs" / "baselines"
OUTPUT_CSV = BASELINES_ROOT / "day1_lidar_baseline_summary.csv"
OUTPUT_JSON = BASELINES_ROOT / "day1_lidar_baseline_summary.json"
VISUALIZATION_DIR = BASELINES_ROOT / "day1_lidar_summary_visualization"

BASELINE_CONFIGS = [
    {
        "key": "vehicle_lidar",
        "name": "Vehicle LiDAR-only",
        "summary_path": BASELINES_ROOT / "vehicle_lidar" / "vehicle_lidar_eval_summary.json",
        "ap_path": BASELINES_ROOT / "vehicle_lidar" / "vehicle_lidar_ap_summary.json",
    },
    {
        "key": "infrastructure_lidar",
        "name": "Infrastructure LiDAR-only",
        "summary_path": (
            BASELINES_ROOT
            / "infrastructure_lidar"
            / "infrastructure_lidar_eval_summary.json"
        ),
        "ap_path": (
            BASELINES_ROOT
            / "infrastructure_lidar"
            / "infrastructure_lidar_ap_summary.json"
        ),
    },
]


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到评价 summary 文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


def nested_get(mapping, keys, default="N/A"):
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return default if value is None else value


def value_or_na(value):
    if value is None:
        return "N/A"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return value


def numeric_values(rows, column):
    values = []
    for row in rows:
        value = row.get(column)
        if isinstance(value, (int, float, np.integer, np.floating)):
            values.append(float(value))
        else:
            values.append(np.nan)
    return np.asarray(values, dtype=float)


def add_value_labels(ax, bars, values):
    for bar, value in zip(bars, values):
        label = "N/A" if not np.isfinite(value) else f"{value:.3f}"
        y = bar.get_height() if np.isfinite(value) else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def baseline_labels(rows):
    return [row["baseline"] for row in rows]


def plot_grouped_metric(rows, columns, labels, title, ylabel, output_path, ylim=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(rows))
    width = 0.8 / len(columns)
    fig, ax = plt.subplots(figsize=(9, 5))

    for index, (column, label) in enumerate(zip(columns, labels)):
        values = numeric_values(rows, column)
        shown_values = np.nan_to_num(values, nan=0.0)
        offset = (index - (len(columns) - 1) / 2.0) * width
        bars = ax.bar(x + offset, shown_values, width, label=label)
        add_value_labels(ax, bars, values)

    ax.set_xticks(x)
    ax.set_xticklabels(baseline_labels(rows))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_single_metric(rows, column, title, ylabel, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = numeric_values(rows, column)
    shown_values = np.nan_to_num(values, nan=0.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(np.arange(len(rows)), shown_values)
    add_value_labels(ax, bars, values)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(baseline_labels(rows))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def extract_ap_metrics(ap_path):
    ap_path = Path(ap_path)
    if not ap_path.exists():
        return "N/A", "N/A", f"AP summary 不存在: {ap_path}"

    try:
        ap_summary = load_json(ap_path)
    except (OSError, json.JSONDecodeError) as exc:
        return "N/A", "N/A", f"AP summary 读取失败: {ap_path}: {exc}"

    bev_map = value_or_na(nested_get(ap_summary, ["map", "bev", "iou_0.50"]))
    map_3d = value_or_na(nested_get(ap_summary, ["map", "3d", "iou_0.50"]))
    return bev_map, map_3d, None


def build_summary_row(config):
    evaluation = load_json(config["summary_path"])
    mean_metrics = evaluation.get("mean_metrics", {})
    bev_map, map_3d, ap_warning = extract_ap_metrics(config["ap_path"])

    row = {
        "baseline": config["name"],
        "prediction_type": evaluation.get("prediction_type", "unknown"),
        "num_samples": evaluation.get("num_samples", "N/A"),
        "num_gt": evaluation.get("total_gt", "N/A"),
        "num_pred": evaluation.get("total_pred", "N/A"),
        "num_match": evaluation.get("total_match", "N/A"),
        "FN": evaluation.get("total_false_negative", "N/A"),
        "FP": evaluation.get("total_false_positive", "N/A"),
        "mean_precision": value_or_na(evaluation.get("mean_precision")),
        "mean_recall": value_or_na(evaluation.get("mean_recall")),
        "mean_f1": value_or_na(evaluation.get("mean_f1")),
        "classification_accuracy": value_or_na(evaluation.get("classification_accuracy")),
        "BEV_mAP@0.50": bev_map,
        "3D_mAP@0.50": map_3d,
        "mean_loc_error_bev": value_or_na(mean_metrics.get("mean_loc_error_bev")),
        "mean_loc_error_3d": value_or_na(mean_metrics.get("mean_loc_error_3d")),
        "mean_count_error": value_or_na(mean_metrics.get("count_error")),
        "engineering_validation_warning": bool(
            evaluation.get("engineering_validation_warning", False)
            or "label_oracle_engineering_validation"
            in str(evaluation.get("prediction_type", ""))
        ),
        "evaluation_summary_path": str(config["summary_path"]),
        "ap_summary_path": str(config["ap_path"]),
    }
    return row, ap_warning


def main():
    rows = []
    warnings = []
    for config in BASELINE_CONFIGS:
        row, ap_warning = build_summary_row(config)
        rows.append(row)
        if ap_warning:
            warnings.append({"baseline": config["name"], "warning": ap_warning})

    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = {
        "precision_recall_f1": VISUALIZATION_DIR / "day1_lidar_precision_recall_f1.png",
        "fn_fp": VISUALIZATION_DIR / "day1_lidar_fn_fp_comparison.png",
        "count_error": VISUALIZATION_DIR / "day1_lidar_count_error_comparison.png",
        "loc_error": VISUALIZATION_DIR / "day1_lidar_loc_error_comparison.png",
        "ap": VISUALIZATION_DIR / "day1_lidar_ap_comparison.png",
    }
    plot_grouped_metric(
        rows,
        ["mean_precision", "mean_recall", "mean_f1"],
        ["Precision", "Recall", "F1"],
        "Day1 LiDAR Baseline: Precision / Recall / F1",
        "Score",
        image_paths["precision_recall_f1"],
        ylim=(0.0, 1.05),
    )
    plot_grouped_metric(
        rows,
        ["FN", "FP"],
        ["False negative", "False positive"],
        "Day1 LiDAR Baseline: False Negatives / False Positives",
        "Object count",
        image_paths["fn_fp"],
    )
    plot_single_metric(
        rows,
        "mean_count_error",
        "Day1 LiDAR Baseline: Mean Per-sample Count Error",
        "Mean absolute count error",
        image_paths["count_error"],
    )
    plot_grouped_metric(
        rows,
        ["mean_loc_error_bev", "mean_loc_error_3d"],
        ["BEV localization error", "3D localization error"],
        "Day1 LiDAR Baseline: Localization Error",
        "Meters",
        image_paths["loc_error"],
    )
    plot_grouped_metric(
        rows,
        ["BEV_mAP@0.50", "3D_mAP@0.50"],
        ["BEV mAP@0.50", "3D mAP@0.50"],
        "Day1 LiDAR Baseline: AP / mAP Comparison",
        "mAP",
        image_paths["ap"],
        ylim=(0.0, 1.05),
    )

    columns = [
        "baseline", "prediction_type", "num_samples", "num_gt", "num_pred",
        "num_match", "FN", "FP", "mean_precision", "mean_recall", "mean_f1",
        "classification_accuracy", "BEV_mAP@0.50", "3D_mAP@0.50",
        "mean_loc_error_bev", "mean_loc_error_3d", "mean_count_error",
        "engineering_validation_warning",
    ]
    summary_df = pd.DataFrame(rows)[columns]
    summary_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    output = {
        "summary_csv": str(OUTPUT_CSV),
        "visualization_dir": str(VISUALIZATION_DIR),
        "rows": rows,
        "warnings": warnings,
        "image_paths": {name: str(path) for name, path in image_paths.items()},
    }
    save_json(output, OUTPUT_JSON)

    print("=" * 80)
    print("Day1 双 LiDAR baseline 对比汇总完成")
    print(summary_df.to_string(index=False))
    print(f"Summary CSV: {OUTPUT_CSV}")
    print(f"Summary JSON: {OUTPUT_JSON}")
    for name, path in image_paths.items():
        print(f"{name}: {path}")
    for warning in warnings:
        print(f"[WARN] {warning['baseline']}: {warning['warning']}")


if __name__ == "__main__":
    main()
