#!/usr/bin/env python3
"""Create summary charts and auditable representative world views for Full Dataset baselines."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluate_and_visualize_vehicle_lidar_baseline as legacy  # noqa: E402


DEFAULT_ROOT = PROJECT_ROOT / "outputs" / "full_baselines"
BASELINES = (
    ("vehicle_lidar", "Vehicle LiDAR-only"),
    ("infrastructure_lidar", "Infrastructure LiDAR-only"),
    ("vehicle_camera", "Vehicle Camera-only"),
    ("infrastructure_camera", "Infrastructure Camera-only"),
)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_number(value):
    if value is None or pd.isna(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def sample_key(sample_id):
    """Compare numeric IDs independently of CSV's loss of leading zeroes."""
    value = str(sample_id).strip()
    return str(int(value)) if value.isdigit() else value


def select_representatives(eval_df, max_samples):
    """Select error modes and strong cases without using model-specific assumptions."""
    if eval_df.empty:
        return pd.DataFrame(columns=["sample_id", "selection_reasons"])
    rules = (
        ("highest_false_negative", "false_negative", False),
        ("highest_false_positive", "false_positive", False),
        ("highest_loc_error_3d", "mean_loc_error_3d", False),
        ("lowest_recall", "recall", True),
        ("highest_f1", "f1", False),
    )
    per_rule = max(1, int(np.ceil(max_samples / len(rules))))
    reasons = defaultdict(list)
    for label, column, ascending in rules:
        if column not in eval_df.columns:
            continue
        ranked = eval_df.dropna(subset=[column]).sort_values(
            column, ascending=ascending, kind="stable"
        )
        for _, row in ranked.head(per_rule).iterrows():
            reasons[str(row["sample_id"])].append(label)
    if len(reasons) < max_samples:
        for _, row in eval_df.sort_values("sample_id", kind="stable").iterrows():
            reasons.setdefault(str(row["sample_id"]), ["deterministic_coverage"])
            if len(reasons) >= max_samples:
                break
    selected_ids = list(reasons)[:max_samples]
    selected = eval_df[eval_df["sample_id"].astype(str).isin(selected_ids)].copy()
    selected["selection_reasons"] = selected["sample_id"].astype(str).map(
        lambda sample_id: ";".join(reasons[sample_id])
    )
    selected["selection_rank"] = selected["sample_id"].astype(str).map(
        {sample_id: index + 1 for index, sample_id in enumerate(selected_ids)}
    )
    return selected.sort_values("selection_rank", kind="stable")


def plot_grouped(rows, columns, title, ylabel, output_path):
    labels = [row["display_name"] for row in rows]
    x = np.arange(len(rows))
    width = 0.8 / len(columns)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for index, (column, label) in enumerate(columns):
        values = [clean_number(row.get(column)) for row in rows]
        ax.bar(x - 0.4 + width / 2 + index * width, values, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def get_ap(ap_summary, view, iou, class_name="Car"):
    return (
        ap_summary.get("metrics", {})
        .get(view, {})
        .get(f"iou_{iou:.2f}", {})
        .get(class_name, {})
        .get("ap")
    )


def safe_ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def add_academic_metrics(summary, ap_summary, baseline, display_name):
    total_gt = summary.get("total_gt")
    total_pred = summary.get("total_pred")
    total_match = summary.get("total_match")
    total_fn = summary.get("total_false_negative")
    total_fp = summary.get("total_false_positive")
    micro_precision = safe_ratio(total_match, total_pred)
    micro_recall = safe_ratio(total_match, total_gt)
    gt_classes = [
        class_name for class_name, values in summary.get("per_class_metrics", {}).items()
        if values.get("gt_count", 0) > 0
    ]
    return {
        "baseline": baseline,
        "display_name": display_name,
        "prediction_type": summary.get("prediction_type"),
        "num_samples": summary.get("num_samples"),
        "num_gt": total_gt,
        "num_pred": total_pred,
        "num_match": total_match,
        "FN": total_fn,
        "FP": total_fp,
        "gt_classes_with_instances": ",".join(gt_classes) or "N/A",
        "all_gt_classes_present": ",".join(summary.get("gt_classes_present", [])) or "N/A",
        "center_match_micro_precision": micro_precision,
        "center_match_micro_recall": micro_recall,
        "center_match_micro_f1": (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if micro_precision is not None and micro_recall is not None and micro_precision + micro_recall > 0
            else None
        ),
        "frame_macro_precision": summary.get("mean_precision"),
        "frame_macro_recall": summary.get("mean_recall"),
        "frame_macro_f1": summary.get("mean_f1"),
        "false_negative_rate": safe_ratio(total_fn, total_gt),
        "false_positive_per_gt": safe_ratio(total_fp, total_gt),
        "prediction_gt_ratio": safe_ratio(total_pred, total_gt),
        "global_count_bias": safe_ratio(total_pred - total_gt, total_gt),
        "abs_global_count_relative_error": safe_ratio(abs(total_pred - total_gt), total_gt),
        "classification_accuracy_matched_only": summary.get("classification_accuracy"),
        "mean_loc_error_bev": summary.get("mean_metrics", {}).get("mean_loc_error_bev"),
        "mean_loc_error_3d": summary.get("mean_metrics", {}).get("mean_loc_error_3d"),
        "Car_BEV_AP@0.25": get_ap(ap_summary, "bev", 0.25),
        "Car_BEV_AP@0.50": get_ap(ap_summary, "bev", 0.50),
        "Car_BEV_AP@0.70": get_ap(ap_summary, "bev", 0.70),
        "Car_3D_AP@0.25": get_ap(ap_summary, "3d", 0.25),
        "Car_3D_AP@0.50": get_ap(ap_summary, "3d", 0.50),
        "Car_3D_AP@0.70": get_ap(ap_summary, "3d", 0.70),
        "evaluation_scope": summary.get("evaluation_scope_note"),
    }


def plot_pr_tradeoff(rows, output_path):
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    for row in rows:
        recall = clean_number(row["center_match_micro_recall"])
        precision = clean_number(row["center_match_micro_precision"])
        ax.scatter(recall, precision, s=100)
        ax.annotate(row["display_name"], (recall, precision), xytext=(6, 6), textcoords="offset points")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Global center-match recall")
    ax.set_ylabel("Global center-match precision")
    ax.set_title("Global Detection Precision-Recall Trade-off")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_ap_iou_sweep(rows, output_path):
    ious = (0.25, 0.50, 0.70)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
    for axis, view, title in zip(axes, ("BEV", "3D"), ("Car BEV AP across IoU thresholds", "Car 3D AP across IoU thresholds")):
        for row in rows:
            values = [clean_number(row[f"Car_{view}_AP@{iou:.2f}"]) for iou in ious]
            axis.plot(ious, values, marker="o", linewidth=2, label=row["display_name"])
        axis.set_title(title)
        axis.set_xlabel("IoU threshold")
        axis.set_ylabel("AP")
        axis.set_xticks(ious)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_academic_protocol(summary_dir, rows):
    scope = rows[0].get("gt_classes_with_instances", "N/A") if rows else "N/A"
    all_gt_classes = rows[0].get("all_gt_classes_present", "N/A") if rows else "N/A"
    lines = [
        "# Full Dataset 单模态 Baseline 比较口径",
        "",
        "## 数据与范围",
        f"- 四个 baseline 使用同一 `full_common_val` 验证集，样本数为 {rows[0].get('num_samples', 'N/A') if rows else 'N/A'}。",
        f"- 当前标准评测类 Car/Pedestrian/Cyclist 中实际出现的 GT 类别：`{scope}`。",
        f"- 原始 GT 标签还包含：`{all_gt_classes}`。非标准类别会保留在总 GT 计数中，但不进入三类 AP 或分类矩阵。",
        "- 因此当前结果是 Car 检测比较；Pedestrian/Cyclist 没有 GT 实例时，不能将 classification accuracy 或 mAP 解读为完整多类别性能。",
        "",
        "## 匹配指标",
        "- Global micro Precision/Recall/F1：先在所有帧汇总 5 m BEV 中心距离匹配产生的 TP、FP、FN，再计算指标；用于整体检测能力比较。",
        "- Frame macro Precision/Recall/F1：逐帧计算后等权平均；用于描述场景间稳定性，不能替代全局 micro 指标。",
        "- Classification accuracy：仅对已匹配目标计数；在当前 Car-only GT 条件下信息量有限，不能单独作为模型优劣依据。",
        "",
        "## AP 与误差",
        "- Car BEV/3D AP 分别在 IoU=0.25、0.50、0.70 计算，展示从宽松到严格几何条件下的性能衰减。",
        "- FN rate=FN/GT，FP per GT=FP/GT，Prediction/GT=Pred/GT；均为归一化量，避免样本规模影响。",
        "- 定位误差是已匹配目标的中心距离均值，必须与 Recall 和 AP 一起解释，因为它不惩罚漏检。",
        "",
        "## 限制",
        "- 当前中心匹配和 AP 使用不同阈值定义，二者回答不同问题，不应直接混为同一个 mAP 指标。",
        "- 这些模型均是当前 checkpoint 的推理结果。性能差异可能来自训练数据、模型配置、传感器视角、坐标标定、分数阈值和检测后处理；仅凭单个图或单个指标不能归因。",
    ]
    (summary_dir / "full_single_modal_evaluation_protocol.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_representative_world_views(baseline, display_name, root, selected):
    details_path = root / "eval_details.json"
    predictions_path = root / "predictions_world.json"
    output_dir = root / "visualization" / "representative_world"
    if not details_path.is_file() or not predictions_path.is_file():
        return {"rendered": 0, "skipped": ["missing eval_details.json or predictions_world.json"]}
    details = {sample_key(item["sample_id"]): item for item in load_json(details_path)}
    predictions = {
        sample_key(item["sample_id"]): item for item in load_json(predictions_path)
    }
    skipped, rendered = [], []
    output_dir.mkdir(parents=True, exist_ok=True)
    for _, selected_row in selected.iterrows():
        selected_sample_id = str(selected_row["sample_id"])
        detail, prediction = (
            details.get(sample_key(selected_sample_id)),
            predictions.get(sample_key(selected_sample_id)),
        )
        if detail is None or prediction is None:
            skipped.append({"sample_id": selected_sample_id, "reason": "missing detail or world prediction"})
            continue
        sample_id = str(prediction["sample_id"])
        try:
            gt_objects = legacy.parse_gt_objects(Path(detail["label_world_path"]))
            matches = detail.get("matches", [])
            legacy.plot_world_compare(
                sample_id=sample_id,
                gt_objects=gt_objects,
                pred_objects=prediction.get("pred_objects", []),
                matches=matches,
                missed_gt=detail.get("missed_gt_indices", []),
                false_pred=detail.get("false_pred_indices", []),
                save_path=output_dir / f"rank_{int(selected_row['selection_rank']):02d}_{sample_id}_world_compare.png",
                classification_accuracy=clean_number(selected_row.get("classification_accuracy")),
                precision=clean_number(selected_row.get("precision")),
                recall=clean_number(selected_row.get("recall")),
                baseline_label=display_name,
            )
            rendered.append(sample_id)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            skipped.append({"sample_id": sample_id, "reason": str(error)})
    return {"rendered": len(rendered), "rendered_sample_ids": rendered, "skipped": skipped}


def main():
    parser = argparse.ArgumentParser(description="生成 Full Dataset 四个 baseline 的汇总图与代表帧 world 可视化")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--max-representatives", type=int, default=12)
    args = parser.parse_args()
    if args.max_representatives <= 0:
        raise ValueError("--max-representatives must be positive")

    root = args.root.resolve()
    legacy.DATA_ROOT = args.data_root.resolve()
    legacy.COOP_INFO_PATH = legacy.DATA_ROOT / "cooperative" / "data_info.json"
    summary_dir = root / "visualization" / "summary"
    rows, academic_rows, report = [], [], {"root": str(root), "baselines": {}}

    for baseline, display_name in BASELINES:
        baseline_root = root / baseline
        eval_path, summary_path = baseline_root / "eval.csv", baseline_root / "eval_summary.json"
        item_report = {"status": "missing"}
        if not eval_path.is_file() or not summary_path.is_file():
            item_report["reason"] = "missing eval.csv or eval_summary.json"
            report["baselines"][baseline] = item_report
            continue
        eval_df = pd.read_csv(eval_path, dtype={"sample_id": str})
        summary = load_json(summary_path)
        ap_path = baseline_root / "ap_summary.json"
        ap_summary = load_json(ap_path) if ap_path.is_file() else {}
        selected = select_representatives(eval_df, args.max_representatives)
        vis_root = baseline_root / "visualization"
        selected.to_csv(vis_root / "representative_samples.csv", index=False, encoding="utf-8-sig")
        (vis_root / "representative_sample_ids.txt").write_text(
            "\n".join(selected["sample_id"].astype(str)) + "\n", encoding="utf-8"
        )
        render_result = render_representative_world_views(baseline, display_name, baseline_root, selected)
        item_report.update({
            "status": "visualized",
            "num_evaluated_samples": int(len(eval_df)),
            "representative_csv": str(vis_root / "representative_samples.csv"),
            "representative_sample_ids": str(vis_root / "representative_sample_ids.txt"),
            "world_render": render_result,
        })
        report["baselines"][baseline] = item_report
        rows.append({
            "baseline": baseline,
            "display_name": display_name,
            "mean_precision": summary.get("mean_precision"),
            "mean_recall": summary.get("mean_recall"),
            "mean_f1": summary.get("mean_f1"),
            "total_false_negative": summary.get("total_false_negative"),
            "total_false_positive": summary.get("total_false_positive"),
            "map_bev_iou_0.50": summary.get("map_bev_iou_0.50"),
            "map_3d_iou_0.50": summary.get("map_3d_iou_0.50"),
            "mean_loc_error_bev": summary.get("mean_metrics", {}).get("mean_loc_error_bev"),
            "mean_loc_error_3d": summary.get("mean_metrics", {}).get("mean_loc_error_3d"),
        })
        academic_rows.append(add_academic_metrics(summary, ap_summary, baseline, display_name))

    if rows:
        plot_grouped(rows, [("mean_precision", "Precision"), ("mean_recall", "Recall"), ("mean_f1", "F1")], "Full Dataset Single-Modal Metrics", "Score", summary_dir / "full_precision_recall_f1.png")
        plot_grouped(rows, [("total_false_negative", "FN"), ("total_false_positive", "FP")], "Full Dataset False Negatives and Positives", "Object count", summary_dir / "full_fn_fp.png")
        plot_grouped(rows, [("map_bev_iou_0.50", "BEV mAP@0.50"), ("map_3d_iou_0.50", "3D mAP@0.50")], "Full Dataset Detection AP", "mAP", summary_dir / "full_ap.png")
        plot_grouped(rows, [("mean_loc_error_bev", "BEV center error"), ("mean_loc_error_3d", "3D center error")], "Full Dataset Matched Localization Error", "Meters", summary_dir / "full_localization_error.png")
        plot_grouped(academic_rows, [("center_match_micro_precision", "Micro precision"), ("center_match_micro_recall", "Micro recall"), ("center_match_micro_f1", "Micro F1")], "Global Center-Match Detection Quality", "Score", summary_dir / "full_global_micro_precision_recall_f1.png")
        plot_grouped(academic_rows, [("frame_macro_precision", "Frame macro precision"), ("frame_macro_recall", "Frame macro recall"), ("frame_macro_f1", "Frame macro F1")], "Per-Frame Macro Detection Quality", "Score", summary_dir / "full_frame_macro_precision_recall_f1.png")
        plot_grouped(academic_rows, [("false_negative_rate", "FN / GT"), ("false_positive_per_gt", "FP / GT"), ("abs_global_count_relative_error", "|Pred-GT| / GT")], "Normalised Detection and Count Errors", "Normalised error", summary_dir / "full_normalized_detection_errors.png")
        plot_grouped(academic_rows, [("prediction_gt_ratio", "Pred / GT"), ("global_count_bias", "(Pred-GT) / GT")], "Global Prediction Count Bias", "Ratio", summary_dir / "full_prediction_count_bias.png")
        plot_pr_tradeoff(academic_rows, summary_dir / "full_global_precision_recall_tradeoff.png")
        plot_ap_iou_sweep(academic_rows, summary_dir / "full_car_ap_iou_sweep.png")
        academic_df = pd.DataFrame(academic_rows)
        academic_df.to_csv(summary_dir / "full_single_modal_academic_comparison.csv", index=False, encoding="utf-8-sig")
        save_json(summary_dir / "full_single_modal_academic_comparison.json", academic_rows)
        write_academic_protocol(summary_dir, academic_rows)
    report["summary_charts_dir"] = str(summary_dir)
    report["num_visualized_baselines"] = len(rows)
    save_json(root / "full_baseline_visualization_summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
