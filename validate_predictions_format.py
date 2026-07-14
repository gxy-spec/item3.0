import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SEARCH_ROOT = PROJECT_ROOT / "outputs" / "baselines"

REQUIRED_RECORD_FIELDS = {
    "sample_id": str,
    "coordinate_system": str,
    "prediction_type": str,
    "pred_objects": list,
}

RECOMMENDED_RECORD_FIELDS = ["sensor", "source_id"]
CENTER_FIELD_CANDIDATES = [
    "center_lidar",
    "center_world",
    "center_sensor",
    "center_camera",
]
BOX_FIELD_CANDIDATES = [
    "box_lidar",
    "box_world",
    "box_sensor",
    "box_camera",
]


class ValidationResult:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, message):
        self.errors.append(message)

    def warning(self, message):
        self.warnings.append(message)

    @property
    def ok(self):
        return len(self.errors) == 0


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_center(record_path, record_idx, obj_idx, obj, result):
    present = [field for field in CENTER_FIELD_CANDIDATES if field in obj]
    if not present:
        result.error(
            f"{record_path}: record[{record_idx}].pred_objects[{obj_idx}] "
            f"缺少中心点字段，候选字段为 {CENTER_FIELD_CANDIDATES}"
        )
        return

    field = present[0]
    value = obj[field]
    if not isinstance(value, list) or len(value) != 3 or not all(is_number(v) for v in value):
        result.error(
            f"{record_path}: record[{record_idx}].pred_objects[{obj_idx}].{field} "
            "必须是长度为 3 的数字列表"
        )


def validate_box(record_path, record_idx, obj_idx, obj, result):
    present = [field for field in BOX_FIELD_CANDIDATES if field in obj]
    if not present:
        result.error(
            f"{record_path}: record[{record_idx}].pred_objects[{obj_idx}] "
            f"缺少 box 字段，候选字段为 {BOX_FIELD_CANDIDATES}"
        )
        return

    field = present[0]
    value = obj[field]
    if not isinstance(value, dict):
        result.error(
            f"{record_path}: record[{record_idx}].pred_objects[{obj_idx}].{field} "
            "必须是 object/dict"
        )


def validate_prediction_object(record_path, record_idx, obj_idx, obj, result):
    if not isinstance(obj, dict):
        result.error(
            f"{record_path}: record[{record_idx}].pred_objects[{obj_idx}] 必须是 object/dict"
        )
        return

    if "class" not in obj:
        result.error(
            f"{record_path}: record[{record_idx}].pred_objects[{obj_idx}] 缺少必填字段 class"
        )
    elif not isinstance(obj["class"], str) or not obj["class"]:
        result.error(
            f"{record_path}: record[{record_idx}].pred_objects[{obj_idx}].class 必须是非空字符串"
        )

    if "score" not in obj:
        result.error(
            f"{record_path}: record[{record_idx}].pred_objects[{obj_idx}] 缺少必填字段 score"
        )
    elif not is_number(obj["score"]):
        result.error(
            f"{record_path}: record[{record_idx}].pred_objects[{obj_idx}].score 必须是数字"
        )

    validate_center(record_path, record_idx, obj_idx, obj, result)
    validate_box(record_path, record_idx, obj_idx, obj, result)


def validate_record(record_path, record_idx, record, result):
    if not isinstance(record, dict):
        result.error(f"{record_path}: record[{record_idx}] 必须是 object/dict")
        return

    for field, field_type in REQUIRED_RECORD_FIELDS.items():
        if field not in record:
            result.error(f"{record_path}: record[{record_idx}] 缺少必填字段 {field}")
            continue

        if not isinstance(record[field], field_type):
            type_name = field_type.__name__
            result.error(
                f"{record_path}: record[{record_idx}].{field} 类型错误，应为 {type_name}"
            )

        if field != "pred_objects" and isinstance(record.get(field), str) and not record[field]:
            result.error(f"{record_path}: record[{record_idx}].{field} 不能为空字符串")

    for field in RECOMMENDED_RECORD_FIELDS:
        if field not in record:
            result.warning(
                f"{record_path}: record[{record_idx}] 缺少推荐字段 {field}，"
                "不影响格式通过，但建议后续 baseline 补充"
            )
        elif not isinstance(record[field], str) or not record[field]:
            result.warning(
                f"{record_path}: record[{record_idx}].{field} 建议使用非空字符串"
            )

    pred_objects = record.get("pred_objects")
    if isinstance(pred_objects, list):
        for obj_idx, obj in enumerate(pred_objects):
            validate_prediction_object(record_path, record_idx, obj_idx, obj, result)


def validate_file(path):
    result = ValidationResult()
    path = Path(path)

    try:
        data = load_json(path)
    except Exception as exc:
        result.error(f"{path}: JSON 读取失败: {exc}")
        return result

    if not isinstance(data, list):
        result.error(f"{path}: 顶层结构必须是 list")
        return result

    for record_idx, record in enumerate(data):
        validate_record(path, record_idx, record, result)

    return result


def default_prediction_paths():
    if not DEFAULT_SEARCH_ROOT.exists():
        return []
    paths = []
    for path in sorted(DEFAULT_SEARCH_ROOT.glob("**/predictions_*.json")):
        if path.name.endswith("_summary.json"):
            continue
        paths.append(path)
    return paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="校验 baseline predictions_*.json 是否符合统一预测格式。"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help=(
            "要校验的 predictions_*.json 路径。"
            "如果不提供，则默认扫描 outputs/baselines/**/predictions_*.json。"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    paths = [Path(p) for p in args.paths] if args.paths else default_prediction_paths()

    if not paths:
        print("[WARN] 没有找到需要校验的 predictions_*.json 文件")
        return 0

    total_errors = 0
    total_warnings = 0

    for path in paths:
        result = validate_file(path)
        total_errors += len(result.errors)
        total_warnings += len(result.warnings)

        print("=" * 80)
        print(f"校验文件: {path}")
        print(f"errors: {len(result.errors)}, warnings: {len(result.warnings)}")

        for warning in result.warnings:
            print(f"[WARN] {warning}")

        for error in result.errors:
            print(f"[ERROR] {error}")

        if result.ok:
            print("[PASS] 格式检查通过")
        else:
            print("[FAIL] 格式检查失败")

    print("=" * 80)
    print(f"总计: errors={total_errors}, warnings={total_warnings}")

    return 1 if total_errors > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
