import argparse
import importlib
import json
import math
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from validate_predictions_format import validate_file


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INDEX_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "preprocessed"
    / "infrastructure_lidar_baseline_index.json"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "baselines"
    / "infrastructure_lidar"
    / "predictions_infrastructure_lidar.json"
)
PREDICTION_TYPE = "infrastructure_lidar_label_oracle_engineering_validation"
OBJECT_SOURCE = "infrastructure_lidar_label_oracle_engineering_validation"
OPENPCDET_PREDICTION_TYPE = "openpcdet_infrastructure_lidar_pointpillars"
OPENPCDET_OBJECT_SOURCE = "openpcdet_pointpillars"


class NumpyCompatUnpickler(pickle.Unpickler):
    """兼容不同 NumPy 版本生成的 OpenPCDet result.pkl。"""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core"):]
        try:
            importlib.import_module(module)
        except ModuleNotFoundError:
            pass
        return super().find_class(module, name)


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


def load_pickle(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 OpenPCDet result.pkl: {path}")
    with open(path, "rb") as f:
        return NumpyCompatUnpickler(f).load()


def as_finite_float(value, field_name):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 不是数值: {value!r}") from exc

    if not math.isfinite(result):
        raise ValueError(f"{field_name} 不是有限数值: {value!r}")
    return result


def parse_label_object(label, label_path, object_index):
    """把 infrastructure-side/label/virtuallidar 的一个标注转为统一预测对象。"""
    if not isinstance(label, dict):
        raise ValueError(f"第 {object_index} 个标注不是 JSON object")

    class_name = str(label.get("type", "")).strip()
    if not class_name:
        raise ValueError(f"第 {object_index} 个标注缺少非空 type")

    location = label.get("3d_location")
    dimensions = label.get("3d_dimensions")
    if not isinstance(location, dict):
        raise ValueError(f"第 {object_index} 个标注缺少 3d_location")
    if not isinstance(dimensions, dict):
        raise ValueError(f"第 {object_index} 个标注缺少 3d_dimensions")

    center_lidar = [
        as_finite_float(location.get("x"), "3d_location.x"),
        as_finite_float(location.get("y"), "3d_location.y"),
        as_finite_float(location.get("z"), "3d_location.z"),
    ]
    # DAIR-V2X 标注沿用 l/w/h 命名；统一预测格式采用 dx/dy/dz。
    dx = as_finite_float(dimensions.get("l"), "3d_dimensions.l")
    dy = as_finite_float(dimensions.get("w"), "3d_dimensions.w")
    dz = as_finite_float(dimensions.get("h"), "3d_dimensions.h")
    heading = as_finite_float(label.get("rotation"), "rotation")

    if dx <= 0.0 or dy <= 0.0 or dz <= 0.0:
        raise ValueError(
            f"第 {object_index} 个标注的尺寸必须为正数: dx={dx}, dy={dy}, dz={dz}"
        )

    return {
        "class": class_name,
        "score": 1.0,
        "center_lidar": center_lidar,
        "box_lidar": {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "heading": heading,
        },
        "source": OBJECT_SOURCE,
        "source_label_path": str(label_path),
    }


def load_label_as_predictions(label_path):
    """返回预测目标和解析警告。标签缺失、为空或不可读时返回空列表。"""
    if not label_path:
        return [], ["索引中缺少 infrastructure_label_path"]

    label_path = Path(label_path)
    if not label_path.exists():
        return [], [f"标注文件不存在: {label_path}"]

    try:
        labels = load_json(label_path)
    except (OSError, json.JSONDecodeError) as exc:
        return [], [f"标注文件读取失败: {label_path}: {exc}"]

    if labels is None:
        return [], []
    if not isinstance(labels, list):
        return [], [f"标注文件顶层应为 list: {label_path}"]

    objects = []
    warnings = []
    for object_index, label in enumerate(labels):
        try:
            objects.append(parse_label_object(label, label_path, object_index))
        except ValueError as exc:
            warnings.append(f"{label_path}: 跳过无效标注: {exc}")

    return objects, warnings


def build_prediction_record(index_record):
    sample_id = str(index_record.get("sample_id", "")).strip()
    infrastructure_id = str(index_record.get("infrastructure_id", "")).strip()
    if not sample_id:
        raise ValueError("索引记录缺少 sample_id")
    if not infrastructure_id:
        raise ValueError(f"sample_id={sample_id} 的索引记录缺少 infrastructure_id")

    label_path = index_record.get("infrastructure_label_path")
    pred_objects, warnings = load_label_as_predictions(label_path)
    record = {
        "sample_id": sample_id,
        "sensor": "infrastructure_lidar",
        "source_id": infrastructure_id,
        "infrastructure_id": infrastructure_id,
        "coordinate_system": "infrastructure_lidar",
        "prediction_type": PREDICTION_TYPE,
        "pred_objects": pred_objects,
    }
    if label_path:
        record["source_file"] = str(label_path)
    if warnings:
        record["export_warnings"] = warnings

    return record, warnings


def build_index_by_infrastructure_id(index_records):
    records_by_id = {}
    for index_number, record in enumerate(index_records):
        if not isinstance(record, dict):
            raise ValueError(f"索引记录 index={index_number} 必须是 object/dict")
        infrastructure_id = str(record.get("infrastructure_id", "")).strip()
        if not infrastructure_id:
            raise ValueError(f"索引记录 index={index_number} 缺少 infrastructure_id")
        if infrastructure_id in records_by_id:
            raise ValueError(f"索引中 infrastructure_id 重复: {infrastructure_id}")
        records_by_id[infrastructure_id] = record
    return records_by_id


def as_2d_boxes(value):
    boxes = np.asarray(value, dtype=float)
    if boxes.size == 0:
        return np.zeros((0, 7), dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ValueError(f"OpenPCDet boxes_lidar 形状异常: {boxes.shape}")
    return boxes


def build_openpcdet_prediction_records(input_path, index_records, score_threshold):
    det_annos = load_pickle(input_path)
    if not isinstance(det_annos, list):
        raise ValueError("OpenPCDet result.pkl 顶层应为 det_annos list")

    index_by_infrastructure_id = build_index_by_infrastructure_id(index_records)
    records = []
    seen_ids = set()

    for anno_index, anno in enumerate(det_annos):
        if not isinstance(anno, dict):
            raise ValueError(f"result.pkl 第 {anno_index} 条预测不是 dict")
        infrastructure_id = str(anno.get("frame_id", "")).strip()
        if not infrastructure_id:
            raise ValueError(f"result.pkl 第 {anno_index} 条预测缺少 frame_id")
        if infrastructure_id in seen_ids:
            raise ValueError(f"result.pkl 中 infrastructure_id 重复: {infrastructure_id}")
        seen_ids.add(infrastructure_id)

        index_record = index_by_infrastructure_id.get(infrastructure_id)
        if index_record is None:
            raise KeyError(
                "result.pkl 的 frame_id 无法匹配 infrastructure_lidar_baseline_index.json: "
                f"{infrastructure_id}"
            )

        names = np.asarray(anno.get("name", []))
        scores = np.asarray(anno.get("score", []), dtype=float)
        boxes = as_2d_boxes(anno.get("boxes_lidar", []))
        objects = []
        for box_index, box in enumerate(boxes):
            score = float(scores[box_index]) if box_index < len(scores) else 1.0
            if not math.isfinite(score) or score < score_threshold:
                continue
            class_name = str(names[box_index]) if box_index < len(names) else "Unknown"
            x, y, z, dx, dy, dz, heading = [float(value) for value in box[:7]]
            values = np.array([x, y, z, dx, dy, dz, heading], dtype=float)
            if not np.isfinite(values).all() or dx <= 0.0 or dy <= 0.0 or dz <= 0.0:
                continue
            objects.append(
                {
                    "class": class_name,
                    "score": score,
                    "center_lidar": [x, y, z],
                    "box_lidar": {
                        "dx": dx,
                        "dy": dy,
                        "dz": dz,
                        "heading": heading,
                    },
                    "source": OPENPCDET_OBJECT_SOURCE,
                }
            )

        sample_id = str(index_record.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"infrastructure_id={infrastructure_id} 对应索引缺少 sample_id")
        records.append(
            {
                "sample_id": sample_id,
                "sensor": "infrastructure_lidar",
                "source_id": infrastructure_id,
                "infrastructure_id": infrastructure_id,
                "coordinate_system": "infrastructure_lidar",
                "prediction_type": OPENPCDET_PREDICTION_TYPE,
                "source_file": str(input_path),
                "pred_objects": objects,
            }
        )

    return records


def build_summary(records, index_path, output_path, warnings, input_path=None, score_threshold=None):
    class_distribution = Counter(
        obj["class"] for record in records for obj in record["pred_objects"]
    )
    empty_samples = [
        record["sample_id"] for record in records if not record["pred_objects"]
    ]
    prediction_types = sorted({record["prediction_type"] for record in records})
    summary = {
        "input_index_path": str(index_path),
        "output_path": str(output_path),
        "num_samples": len(records),
        "total_pred_objects": sum(len(record["pred_objects"]) for record in records),
        "prediction_type": prediction_types[0] if len(prediction_types) == 1 else prediction_types,
        "class_distribution": dict(sorted(class_distribution.items())),
        "empty_samples": empty_samples,
        "num_empty_samples": len(empty_samples),
        "num_export_warnings": len(warnings),
        "export_warnings": warnings,
    }
    if input_path is None:
        summary["note"] = (
            "当前输出基于 infrastructure-side LiDAR 标注生成，仅用于工程链路验证，"
            "不代表真实 Infrastructure PointPillars 模型预测结果。"
        )
    else:
        summary["input_result_pkl"] = str(input_path)
        summary["score_threshold"] = score_threshold
        summary["note"] = "当前输出来自 OpenPCDet PointPillars 的真实模型推理结果。"
    return summary


def run_format_validation(output_path):
    validation = validate_file(output_path)
    for warning in validation.warnings:
        print(f"[VALIDATION WARN] {warning}")
    for error in validation.errors:
        print(f"[VALIDATION ERROR] {error}")

    print(
        "格式校验结果: "
        f"errors={len(validation.errors)}, warnings={len(validation.warnings)}"
    )
    if not validation.ok:
        raise RuntimeError(f"预测格式校验失败: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "从基础设施端虚拟 LiDAR 标注导出工程验证版 Infrastructure LiDAR-only 预测。"
        )
    )
    parser.add_argument(
        "--index",
        default=str(DEFAULT_INDEX_PATH),
        help="infrastructure_lidar_baseline_index.json 路径。",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="predictions_infrastructure_lidar.json 输出路径。",
    )
    parser.add_argument(
        "--input-result-pkl",
        help=(
            "真实 OpenPCDet tools/test.py --save_to_file 生成的 result.pkl。"
            "提供后导出真实模型预测，并使用 openpcdet_infrastructure_lidar_pointpillars 标识。"
        ),
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.1,
        help="仅在 --input-result-pkl 模式下生效；默认 0.1。",
    )
    parser.add_argument(
        "--label-oracle",
        action="store_true",
        help="显式导出标注工程验证版；不能用于真实模型评价。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    index_path = Path(args.index)
    output_path = Path(args.output)
    index_records = load_json(index_path)
    if not isinstance(index_records, list):
        raise ValueError(f"索引文件顶层必须是 list: {index_path}")

    if args.input_result_pkl and args.label_oracle:
        raise ValueError("--input-result-pkl 与 --label-oracle 不能同时使用")
    if not args.input_result_pkl and not args.label_oracle:
        raise ValueError(
            "请明确选择导出模式：真实模型使用 --input-result-pkl RESULT.PKL；"
            "工程验证使用 --label-oracle。"
        )

    warnings = []
    if args.input_result_pkl:
        records = build_openpcdet_prediction_records(
            input_path=Path(args.input_result_pkl),
            index_records=index_records,
            score_threshold=args.score_threshold,
        )
        for index_number, record in enumerate(records):
            print(
                f"[{index_number + 1}/{len(records)}] "
                f"sample_id={record['sample_id']}, "
                f"infrastructure_id={record['infrastructure_id']}, "
                f"pred_objects={len(record['pred_objects'])}"
            )
    else:
        records = []
        for index_number, index_record in enumerate(index_records):
            if not isinstance(index_record, dict):
                raise ValueError(f"索引记录 index={index_number} 必须是 object/dict")
            record, record_warnings = build_prediction_record(index_record)
            records.append(record)
            warnings.extend(
                f"sample_id={record['sample_id']}: {warning}" for warning in record_warnings
            )
            print(
                f"[{index_number + 1}/{len(index_records)}] "
                f"sample_id={record['sample_id']}, "
                f"infrastructure_id={record['infrastructure_id']}, "
                f"pred_objects={len(record['pred_objects'])}"
            )

    save_json(records, output_path)
    summary = build_summary(
        records=records,
        index_path=index_path,
        output_path=output_path,
        warnings=warnings,
        input_path=args.input_result_pkl,
        score_threshold=args.score_threshold if args.input_result_pkl else None,
    )
    summary_path = output_path.with_name("predictions_infrastructure_lidar_summary.json")
    save_json(summary, summary_path)

    run_format_validation(output_path)

    print("=" * 80)
    print("Infrastructure LiDAR-only predictions 导出完成")
    print(f"Output: {output_path}")
    print(f"Summary: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
