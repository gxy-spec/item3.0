import argparse
import json
from pathlib import Path

import pandas as pd

import evaluate_and_visualize_vehicle_lidar_baseline as legacy


PROJECT_ROOT = Path(__file__).resolve().parent
BASELINES = {
    "vehicle_lidar": {
        "display_name": "Vehicle LiDAR-only",
        "root": PROJECT_ROOT / "outputs" / "baselines" / "vehicle_lidar",
    },
    "infrastructure_lidar": {
        "display_name": "Infrastructure LiDAR-only",
        "root": PROJECT_ROOT / "outputs" / "baselines" / "infrastructure_lidar",
    },
    "vehicle_camera": {
        "display_name": "Vehicle Camera-only",
        "root": PROJECT_ROOT / "outputs" / "baselines" / "vehicle_camera",
    },
    "infrastructure_camera": {
        "display_name": "Infrastructure Camera-only",
        "root": PROJECT_ROOT / "outputs" / "baselines" / "infrastructure_camera",
    },
}


def get_paths(baseline):
    root = BASELINES[baseline]["root"]
    visualization_root = root / "visualization"
    return {
        "root": root,
        "predictions_world": root / "predictions_world.json",
        "eval_csv": root / f"{baseline}_eval.csv",
        "summary_json": root / f"{baseline}_eval_summary.json",
        "details_json": root / f"{baseline}_eval_details.json",
        "ap_summary_json": root / f"{baseline}_ap_summary.json",
        "matched_pairs_csv": root / "matched_prediction_gt_pairs.csv",
        "per_class_metrics_csv": root / "per_class_metrics.csv",
        "confusion_matrix_csv": root / "class_confusion_matrix.csv",
        "visualization_root": visualization_root,
        "world_vis_dir": visualization_root / "world_gt_pred_compare",
        "summary_vis_dir": visualization_root / "summary",
    }


def configure_legacy_visualization(paths):
    """旧绘图函数通过模块全局变量读输出目录，统一在这里设置。"""
    legacy.BASELINE_ROOT = paths["root"]
    legacy.VIS_ROOT = paths["visualization_root"]
    legacy.WORLD_VIS_DIR = paths["world_vis_dir"]
    legacy.SUMMARY_VIS_DIR = paths["summary_vis_dir"]


def is_engineering_validation(prediction_types):
    return any(
        "label_oracle_engineering_validation" in prediction_type
        for prediction_type in prediction_types
    )


def build_scope_note(pred_class_set, gt_class_set, prediction_types):
    if is_engineering_validation(prediction_types):
        return (
            "当前结果包含 label_oracle_engineering_validation，属于工程链路验证，"
            "不是真实模型检测结果，不能作为真实检测性能结论。"
        )
    if pred_class_set == ["Car"]:
        return (
            "当前结果为 Car-only detection evaluation。classification_accuracy 仅表示"
            "已匹配目标中 Car 类是否一致，不应解释为完整多类别分类准确率。"
        )
    if len(gt_class_set) <= 1:
        gt_class_text = ", ".join(gt_class_set) if gt_class_set else "无有效类别"
        return (
            "当前 cooperative label_world 验证样本仅包含 "
            f"{gt_class_text}。classification_accuracy 不能解释为完整"
            " Car/Pedestrian/Cyclist 三类分类准确率。"
        )
    return "当前结果包含多类别预测，分类指标基于空间匹配成功的目标计算。"


def nullable_float(value):
    return None if pd.isna(value) else float(value)


def evaluate_baseline(baseline):
    if baseline not in BASELINES:
        raise ValueError(f"不支持的 baseline: {baseline}")

    config = BASELINES[baseline]
    paths = get_paths(baseline)
    configure_legacy_visualization(paths)
    paths["world_vis_dir"].mkdir(parents=True, exist_ok=True)
    paths["summary_vis_dir"].mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Evaluate and visualize {config['display_name']} baseline")
    print(f"Input: {paths['predictions_world']}")
    print(f"Output root: {paths['root']}")
    print("=" * 80)

    predictions_world = legacy.load_json(paths["predictions_world"])
    if not isinstance(predictions_world, list):
        raise ValueError("predictions_world.json 顶层必须是 list")
    sample_mapping = legacy.build_sample_mapping()

    eval_rows = []
    detailed_records = []
    ap_samples = []
    all_matched_pair_rows = []
    prediction_types = set()
    skipped_samples = []

    for idx, pred_record in enumerate(predictions_world):
        sample_id = str(pred_record.get("sample_id", ""))
        vehicle_id = str(pred_record.get("vehicle_id", sample_id))
        prediction_type = str(pred_record.get("prediction_type", "unknown"))
        prediction_types.add(prediction_type)

        if not sample_id:
            skipped_samples.append({"sample_id": None, "reason": "预测记录缺少 sample_id"})
            print(f"[WARN] [{idx + 1}] 预测记录缺少 sample_id，跳过")
            continue
        if vehicle_id not in sample_mapping:
            skipped_samples.append({
                "sample_id": sample_id,
                "vehicle_id": vehicle_id,
                "reason": "找不到 cooperative label_world 映射",
            })
            print(f"[WARN] {sample_id}: 找不到 cooperative label_world 映射: {vehicle_id}")
            continue

        label_world_path = sample_mapping[vehicle_id]["cooperative_label_path"]
        gt_objects = legacy.parse_gt_objects(label_world_path)
        pred_objects = pred_record.get("pred_objects", [])
        if not isinstance(pred_objects, list):
            raise TypeError(f"{sample_id}: pred_objects 必须是 list")

        ap_samples.append({
            "sample_id": sample_id,
            "gt_objects": gt_objects,
            "pred_objects": pred_objects,
        })
        matches, missed_gt, false_pred = legacy.match_predictions_to_gt(
            preds=pred_objects,
            gts=gt_objects,
            distance_threshold=5.0,
            class_aware=False,
        )

        num_gt = len(gt_objects)
        num_pred = len(pred_objects)
        num_match = len(matches)
        false_negative = len(missed_gt)
        false_positive = len(false_pred)
        count_error = abs(num_pred - num_gt)
        mean_loc_error_bev = (
            float(sum(match["distance_bev"] for match in matches) / num_match)
            if num_match > 0 else None
        )
        mean_loc_error_3d = (
            float(sum(match["distance_3d"] for match in matches) / num_match)
            if num_match > 0 else None
        )
        recall = num_match / num_gt if num_gt > 0 else None
        precision = num_match / num_pred if num_pred > 0 else None
        f1 = legacy.f1_score(precision, recall)
        matched_cls_correct = sum(match.get("class_correct", False) for match in matches)
        matched_cls_wrong = num_match - matched_cls_correct
        classification_accuracy = (
            matched_cls_correct / num_match if num_match > 0 else None
        )

        gt_counts = legacy.compute_class_counts(gt_objects)
        pred_counts = legacy.compute_class_counts(pred_objects)
        matched_pair_rows = legacy.build_match_pair_rows(
            sample_id=sample_id,
            pred_objects=pred_objects,
            gt_objects=gt_objects,
            matches=matches,
        )
        all_matched_pair_rows.extend(matched_pair_rows)

        row = {
            "sample_id": sample_id,
            "vehicle_id": vehicle_id,
            "infrastructure_id": pred_record.get("infrastructure_id"),
            "prediction_type": prediction_type,
            "num_gt": num_gt,
            "num_pred": num_pred,
            "num_match": num_match,
            "false_negative": false_negative,
            "false_positive": false_positive,
            "count_error": count_error,
            "mean_loc_error_bev": mean_loc_error_bev,
            "mean_loc_error_3d": mean_loc_error_3d,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "matched_cls_correct": matched_cls_correct,
            "matched_cls_wrong": matched_cls_wrong,
            "classification_accuracy": classification_accuracy,
            "gt_car": gt_counts["Car"],
            "gt_pedestrian": gt_counts["Pedestrian"],
            "gt_cyclist": gt_counts["Cyclist"],
            "pred_car": pred_counts["Car"],
            "pred_pedestrian": pred_counts["Pedestrian"],
            "pred_cyclist": pred_counts["Cyclist"],
        }
        eval_rows.append(row)
        detailed_records.append({
            "baseline": baseline,
            "sample_id": sample_id,
            "vehicle_id": vehicle_id,
            "infrastructure_id": pred_record.get("infrastructure_id"),
            "prediction_type": prediction_type,
            "label_world_path": label_world_path,
            "matches": matches,
            "matched_pairs": matched_pair_rows,
            "missed_gt_indices": missed_gt,
            "false_pred_indices": false_pred,
            "metrics": row,
        })

        legacy.plot_world_compare(
            sample_id=sample_id,
            gt_objects=gt_objects,
            pred_objects=pred_objects,
            matches=matches,
            missed_gt=missed_gt,
            false_pred=false_pred,
            save_path=paths["world_vis_dir"] / f"sample_{idx:03d}_{sample_id}_world_compare.png",
            classification_accuracy=classification_accuracy,
            precision=precision,
            recall=recall,
            baseline_label=config["display_name"],
        )
        print(
            f"[{idx + 1}/{len(predictions_world)}] {sample_id}: "
            f"GT={num_gt}, Pred={num_pred}, Match={num_match}, "
            f"FN={false_negative}, FP={false_positive}, "
            f"Precision={precision}, Recall={recall}, ClsAcc={classification_accuracy}"
        )

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(paths["eval_csv"], index=False, encoding="utf-8-sig")
    legacy.save_json(detailed_records, paths["details_json"])

    matched_pairs_df = pd.DataFrame(all_matched_pair_rows)
    matched_pairs_df.to_csv(paths["matched_pairs_csv"], index=False, encoding="utf-8-sig")
    per_class_df = pd.DataFrame(
        legacy.compute_per_class_metrics(ap_samples, all_matched_pair_rows)
    )
    per_class_df.to_csv(paths["per_class_metrics_csv"], index=False, encoding="utf-8-sig")
    confusion_df = legacy.build_confusion_matrix(all_matched_pair_rows)
    confusion_df.to_csv(paths["confusion_matrix_csv"], encoding="utf-8-sig")

    if not eval_df.empty:
        legacy.plot_summary(eval_df, baseline_label=config["display_name"])
        legacy.plot_confusion_matrix(confusion_df, paths["summary_vis_dir"] / "class_confusion_matrix.png")
        legacy.plot_per_class_metrics(per_class_df, paths["summary_vis_dir"] / "per_class_precision_recall_f1.png")
        legacy.plot_class_distribution(per_class_df, paths["summary_vis_dir"] / "gt_pred_class_distribution.png")
        legacy.plot_score_distribution(all_matched_pair_rows, ap_samples, paths["summary_vis_dir"] / "score_distribution_by_match.png")

    ap_summary = legacy.evaluate_average_precision(ap_samples)
    legacy.save_json(ap_summary, paths["ap_summary_json"])

    pred_class_set = sorted({
        legacy.normalize_class_name(pred.get("class", "Unknown"))
        for sample in ap_samples for pred in sample["pred_objects"]
    })
    gt_class_set = sorted({
        legacy.normalize_class_name(gt.get("class", "Unknown"))
        for sample in ap_samples for gt in sample["gt_objects"]
    })
    prediction_types = sorted(prediction_types)
    total_match = len(all_matched_pair_rows)
    matched_cls_correct = sum(row["class_correct"] for row in all_matched_pair_rows)
    matched_cls_wrong = total_match - matched_cls_correct

    per_class_metrics = {
        row["class"]: {
            "gt_count": int(row["gt_count"]),
            "pred_count": int(row["pred_count"]),
            "TP": int(row["TP"]),
            "FP": int(row["FP"]),
            "FN": int(row["FN"]),
            "precision": nullable_float(row["precision"]),
            "recall": nullable_float(row["recall"]),
            "f1": nullable_float(row["f1"]),
        }
        for _, row in per_class_df.iterrows()
    }
    summary = {
        "baseline": baseline,
        "prediction_type": (
            prediction_types[0] if len(prediction_types) == 1 else prediction_types
        ),
        "prediction_types": prediction_types,
        "num_samples": int(len(eval_df)),
        "num_skipped_samples": len(skipped_samples),
        "skipped_samples": skipped_samples,
        "input_predictions_world": str(paths["predictions_world"]),
        "output_csv": str(paths["eval_csv"]),
        "details_json": str(paths["details_json"]),
        "matched_pairs_csv": str(paths["matched_pairs_csv"]),
        "per_class_metrics_path": str(paths["per_class_metrics_csv"]),
        "confusion_matrix_path": str(paths["confusion_matrix_csv"]),
        "ap_summary_json": str(paths["ap_summary_json"]),
        "visualization_world_dir": str(paths["world_vis_dir"]),
        "visualization_summary_dir": str(paths["summary_vis_dir"]),
        "class_confusion_matrix_png": str(paths["summary_vis_dir"] / "class_confusion_matrix.png"),
        "per_class_metrics_png": str(paths["summary_vis_dir"] / "per_class_precision_recall_f1.png"),
        "matching_method": "class-agnostic greedy center matching in BEV, then class correctness analysis",
        "distance_threshold_meter": 5.0,
        "gt_classes_present": gt_class_set,
        "pred_classes_present": pred_class_set,
        "matched_cls_correct": int(matched_cls_correct),
        "matched_cls_wrong": int(matched_cls_wrong),
        "classification_accuracy": (
            matched_cls_correct / total_match if total_match > 0 else None
        ),
        "per_class_metrics": per_class_metrics,
        "evaluation_scope_note": build_scope_note(
            pred_class_set, gt_class_set, prediction_types
        ),
        "engineering_validation_warning": is_engineering_validation(prediction_types),
        "map_bev_iou_0.50": ap_summary["map"]["bev"]["iou_0.50"],
        "map_3d_iou_0.50": ap_summary["map"]["3d"]["iou_0.50"],
        "total_gt": int(eval_df["num_gt"].sum()) if not eval_df.empty else 0,
        "total_pred": int(eval_df["num_pred"].sum()) if not eval_df.empty else 0,
        "total_match": int(eval_df["num_match"].sum()) if not eval_df.empty else 0,
        "total_false_negative": (
            int(eval_df["false_negative"].sum()) if not eval_df.empty else 0
        ),
        "total_false_positive": (
            int(eval_df["false_positive"].sum()) if not eval_df.empty else 0
        ),
    }
    if not eval_df.empty:
        summary["mean_metrics"] = {
            key: float(value) for key, value in eval_df.mean(numeric_only=True).to_dict().items()
        }
        summary["mean_precision"] = nullable_float(eval_df["precision"].mean())
        summary["mean_recall"] = nullable_float(eval_df["recall"].mean())
        summary["mean_f1"] = nullable_float(eval_df["f1"].mean())

    legacy.save_json(summary, paths["summary_json"])
    print("=" * 80)
    print("评价与可视化完成")
    print(f"Eval CSV: {paths['eval_csv']}")
    print(f"Eval summary: {paths['summary_json']}")
    print(f"Details: {paths['details_json']}")
    print(json.dumps({
        "baseline": baseline,
        "num_samples": summary["num_samples"],
        "mean_precision": summary.get("mean_precision"),
        "mean_recall": summary.get("mean_recall"),
        "mean_f1": summary.get("mean_f1"),
        "classification_accuracy": summary["classification_accuracy"],
    }, indent=2, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser(
        description="统一评价 Vehicle/Infrastructure LiDAR-only world 坐标预测。"
    )
    parser.add_argument("--baseline", required=True, choices=sorted(BASELINES))
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_baseline(args.baseline)


if __name__ == "__main__":
    main()
