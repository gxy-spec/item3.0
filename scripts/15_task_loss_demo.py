import sys
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import PREPROCESS_DIR, LAMBDA_LOC, LAMBDA_COUNT


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def center(obj):
    return np.array(obj["center_world"], dtype=float)


def make_noisy_predictions(gt_objects, drop_prob=0.1, cls_error_prob=0.1, loc_noise_std=1.0, false_positive_prob=0.05):
    """
    构造模拟预测结果，用来测试 loss 代码。
    注意：这不是模型结果，只是为了检查任务损失计算流程。
    """
    preds = []

    for obj in gt_objects:
        if random.random() < drop_prob:
            continue

        pred = {
            "class": obj["class"],
            "center_world": (center(obj) + np.random.normal(0, loc_noise_std, size=3)).tolist(),
            "box_world": obj["box_world"]
        }

        if random.random() < cls_error_prob:
            pred["class"] = "wrong_class"

        preds.append(pred)

    if random.random() < false_positive_prob:
        preds.append({
            "class": "false_positive",
            "center_world": np.random.normal(0, 10, size=3).tolist(),
            "box_world": {
                "length": 4.0,
                "width": 1.8,
                "height": 1.5,
                "yaw": 0.0
            }
        })

    return preds


def greedy_match(gt_objects, pred_objects, max_match_dist=5.0):
    """
    简单最近邻匹配：
    每个 GT 最多匹配一个预测框，每个预测框也最多被匹配一次。
    这里先不做 IoU，是为了快速验证任务损失管线。
    """
    if len(gt_objects) == 0 or len(pred_objects) == 0:
        return []

    gt_centers = np.array([center(obj) for obj in gt_objects])
    pred_centers = np.array([center(obj) for obj in pred_objects])

    candidates = []

    for i in range(len(gt_objects)):
        for j in range(len(pred_objects)):
            dist = np.linalg.norm(gt_centers[i] - pred_centers[j])
            candidates.append((dist, i, j))

    candidates.sort(key=lambda x: x[0])

    matched_gt = set()
    matched_pred = set()
    matches = []

    for dist, i, j in candidates:
        if dist > max_match_dist:
            continue

        if i in matched_gt or j in matched_pred:
            continue

        matched_gt.add(i)
        matched_pred.add(j)
        matches.append((i, j, dist))

    return matches


def compute_task_loss(gt_objects, pred_objects, lambda_loc=1.0, lambda_count=1.0):
    K = len(gt_objects)
    K_hat = len(pred_objects)

    count_error = abs(K - K_hat)

    matches = greedy_match(gt_objects, pred_objects)

    if len(matches) == 0:
        cls_error = 1.0 if K > 0 else 0.0
        loc_error = 0.0 if K == 0 else 999.0
    else:
        cls_errors = []
        loc_errors = []

        for gt_idx, pred_idx, dist in matches:
            gt_obj = gt_objects[gt_idx]
            pred_obj = pred_objects[pred_idx]

            cls_errors.append(0.0 if gt_obj["class"] == pred_obj["class"] else 1.0)
            loc_errors.append(dist)

        cls_error = float(np.mean(cls_errors))
        loc_error = float(np.mean(loc_errors))

    task_loss = cls_error + lambda_loc * loc_error + lambda_count * count_error

    return {
        "num_gt": K,
        "num_pred": K_hat,
        "num_matched": len(matches),
        "cls_error": cls_error,
        "loc_error": loc_error,
        "count_error": count_error,
        "task_loss": task_loss
    }


def main():
    random.seed(0)
    np.random.seed(0)

    gt_path = PREPROCESS_DIR / "world_gt_objects.json"
    gt_records = load_json(gt_path)

    results = []

    num_samples = min(200, len(gt_records))

    for record in tqdm(gt_records[:num_samples], desc="Task loss demo"):
        gt_objects = record["objects"]

        # 情况1：完美预测，用来检查 loss 是否接近 0
        perfect_preds = [
            {
                "class": obj["class"],
                "center_world": obj["center_world"],
                "box_world": obj["box_world"]
            }
            for obj in gt_objects
        ]

        perfect_loss = compute_task_loss(
            gt_objects,
            perfect_preds,
            lambda_loc=LAMBDA_LOC,
            lambda_count=LAMBDA_COUNT
        )

        results.append({
            "sample_index": record["sample_index"],
            "mode": "perfect_prediction",
            **perfect_loss
        })

        # 情况2：模拟带噪声预测，用来检查 loss 是否能反映误差
        noisy_preds = make_noisy_predictions(gt_objects)

        noisy_loss = compute_task_loss(
            gt_objects,
            noisy_preds,
            lambda_loc=LAMBDA_LOC,
            lambda_count=LAMBDA_COUNT
        )

        results.append({
            "sample_index": record["sample_index"],
            "mode": "noisy_prediction",
            **noisy_loss
        })

    df = pd.DataFrame(results)

    save_path = PREPROCESS_DIR / "task_loss_demo_result.csv"
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print(f"任务损失 demo 结果保存完成：{save_path}")
    print("=" * 80)

    print("按预测模式统计：")
    print(df.groupby("mode")[["cls_error", "loc_error", "count_error", "task_loss"]].mean())


if __name__ == "__main__":
    main()