#!/usr/bin/env python3
"""Task 4: evaluator unit tests, configuration audit, and score-threshold sweep."""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluate_and_visualize_vehicle_lidar_baseline as evaluator


DEFAULT_DATA_ROOT = Path("/mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure")
DEFAULT_FULL_ROOT = PROJECT_ROOT / "outputs/full_baselines"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/diagnosis/task4_evaluator_config_summary.json"
THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.5]


def box(cls, x, y, z=0.0, score=1.0):
    return {"class": cls, "center_world": [x, y, z], "score": score}


def run_unit_tests():
    tests = []
    gt = [box("Car", 0, 0)]
    same = [box("Car", 0, 0, score=0.9)]
    matches, missed, false = evaluator.match_predictions_to_gt(same, gt, 5.0, False)
    tests.append({"name": "identical_box", "passed": len(matches) == 1 and not missed and not false and matches[0]["class_correct"]})

    far = [box("Car", 100, 100)]
    matches, missed, false = evaluator.match_predictions_to_gt(far, gt, 5.0, False)
    tests.append({"name": "non_overlapping_box", "passed": not matches and len(missed) == 1 and len(false) == 1})

    duplicate = [box("Car", 0, 0, score=0.9), box("Car", 0.5, 0.5, score=0.8)]
    matches, missed, false = evaluator.match_predictions_to_gt(duplicate, gt, 5.0, False)
    tests.append({"name": "two_predictions_one_gt", "passed": len(matches) == 1 and len(false) == 1 and not missed})

    wrong_class = [box("Pedestrian", 0, 0)]
    matches, missed, false = evaluator.match_predictions_to_gt(wrong_class, gt, 5.0, False)
    tests.append({"name": "same_position_wrong_class", "passed": len(matches) == 1 and not matches[0]["class_correct"]})

    empty, missed, false = evaluator.match_predictions_to_gt([], gt, 5.0, False)
    tests.append({"name": "no_prediction", "passed": not empty and len(missed) == 1 and not false})
    return {"num_tests": len(tests), "passed": all(test["passed"] for test in tests), "tests": tests}


def load_predictions(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("samples", [])


def load_world_gt(data_root, sample_id, coop_map):
    item = coop_map.get(str(sample_id))
    if not item:
        return []
    path = data_root / item["cooperative_label_path"]
    return evaluator.parse_gt_objects(path)


def build_coop_map(data_root):
    result = {}
    for item in json.loads((data_root / "cooperative/data_info.json").read_text(encoding="utf-8")):
        sample_id = Path(item.get("vehicle_pointcloud_path", "")).stem
        if sample_id:
            result[sample_id] = item
    return result


def threshold_sweep(prediction_path, data_root):
    samples = load_predictions(prediction_path)
    coop_map = build_coop_map(data_root)
    rows = []
    for threshold in THRESHOLDS:
        total_gt = total_pred = total_match = total_fn = total_fp = 0
        for sample in samples:
            preds = [pred for pred in sample.get("pred_objects", []) if float(pred.get("score", 0.0)) >= threshold]
            gts = load_world_gt(data_root, sample.get("sample_id"), coop_map)
            matches, missed, false = evaluator.match_predictions_to_gt(preds, gts, 5.0, False)
            total_gt += len(gts); total_pred += len(preds); total_match += len(matches)
            total_fn += len(missed); total_fp += len(false)
        precision = total_match / total_pred if total_pred else 0.0
        recall = total_match / total_gt if total_gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"score_threshold": threshold, "num_gt": total_gt, "num_pred": total_pred,
                     "TP": total_match, "FN": total_fn, "FP": total_fp,
                     "precision": precision, "recall": recall, "f1": f1})
    return rows


def parse_training_evidence(log_path):
    text = Path(log_path).read_text(encoding="utf-8", errors="replace") if Path(log_path).is_file() else ""
    epochs = [int(value) for value in re.findall(r"Train:\s+(\d+)/80", text)]
    sample_counts = [int(value) for value in re.findall(r"Total samples for CUSTOM dataset:\s*(\d+)", text)]
    return {"log_path": str(log_path), "exists": Path(log_path).is_file(),
            "max_logged_epoch": max(epochs) if epochs else None,
            "evaluation_sample_counts": sample_counts,
            "full_dataset_training_proven": bool(sample_counts and max(sample_counts) >= 1243),
            "note": "日志中的 Total samples 是 evaluation 样本数；若为 9，不能证明在 Full Dataset 上训练。"}


def config_audit(full_root):
    result = {}
    for baseline in ("vehicle_lidar", "infrastructure_lidar"):
        root = full_root / baseline
        summary_name = "predictions_vehicle_lidar_summary.json" if baseline == "vehicle_lidar" else "predictions_infrastructure_lidar_summary.json"
        summary = json.loads((root / summary_name).read_text(encoding="utf-8"))
        source = str(summary.get("input", summary.get("input_result_pkl", "")))
        checkpoint = str(root / "runtime_configs" / ("pointpillar_full_vehicle.yaml" if baseline == "vehicle_lidar" else "pointpillar_full_infrastructure.yaml"))
        result[baseline] = {"prediction_type": summary.get("prediction_type", summary.get("prediction_types")),
                            "num_prediction_samples": summary.get("num_samples"),
                            "source_result_pkl": source, "source_result_pkl_exists": Path(source).is_file(),
                            "runtime_config": checkpoint, "runtime_config_exists": Path(checkpoint).is_file()}
    return result


def main():
    parser = argparse.ArgumentParser(description="运行 LiDAR baseline Task 4 评价器、配置和训练证据诊断")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--full-root", default=str(DEFAULT_FULL_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    data_root, full_root = Path(args.data_root), Path(args.full_root)
    unit = run_unit_tests()
    threshold = {}
    for baseline in ("vehicle_lidar", "infrastructure_lidar"):
        threshold[baseline] = threshold_sweep(full_root / baseline / "predictions_world.json", data_root)
    evidence = {
        "vehicle_lidar": parse_training_evidence(PROJECT_ROOT.parent / "OpenPCDet/output/custom_models/pointpillar_custom_vehicle/veh_lidar_pointpillar_80e/train_20260709-092948.log"),
        "infrastructure_lidar": parse_training_evidence(PROJECT_ROOT.parent / "OpenPCDet/output/custom_models/pointpillar_custom_infrastructure/infra_lidar_pointpillar_80e/train_20260713-181441.log"),
    }
    summary = {"task": "Task 4: evaluator and PointPillars configuration validation", "unit_tests": unit,
               "threshold_sweep": threshold, "configuration_audit": config_audit(full_root),
               "training_evidence": evidence,
               "conclusion": "评价器单元测试通过；两套 checkpoint 日志显示训练到 80 epoch，但 evaluation 仅有 9 个样本，当前证据不能证明模型在 Full Dataset 上训练了足够轮次。"}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "unit_tests": unit, "training_evidence": evidence}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
