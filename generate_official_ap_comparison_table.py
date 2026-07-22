#!/usr/bin/env python3
"""Create a paper-style comparison table from official-aligned AP summaries."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "outputs" / "full_baselines"
BASELINES = [
    ("Vehicle Camera-only", "vehicle_camera", "ImVoxelNet", 1),
    ("Infrastructure Camera-only", "infrastructure_camera", "ImVoxelNet", 1),
    ("Vehicle LiDAR-only", "vehicle_lidar", "PointPillars", 20),
    ("Infrastructure LiDAR-only", "infrastructure_lidar", "PointPillars", 20),
]


def value(summary, view):
    ap = summary["metrics"][view]["car"]["iou_0.50"]["ap"]
    return None if ap is None else round(float(ap) * 100.0, 4)


def main():
    rows = []
    for display, key, model, epochs in BASELINES:
        path = ROOT / key / "official_local_aligned_ap_summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "Modality": "Image" if "Camera" in display else "Pointcloud",
            "Fusion": "Veh.-Only" if display.startswith("Vehicle") else "Inf.-Only",
            "Model": model,
            "Baseline": display,
            "Dataset": "DAIR-V2X-C Full common-val",
            "Training epochs": epochs,
            "AP_3D@0.50 Overall": value(summary, "3d"),
            "AP_BEV@0.50 Overall": value(summary, "bev"),
            "AP_3D@0.50 0-30m": "N/A",
            "AP_3D@0.50 30-50m": "N/A",
            "AP_3D@0.50 50-100m": "N/A",
            "AP_BEV@0.50 0-30m": "N/A",
            "AP_BEV@0.50 30-50m": "N/A",
            "AP_BEV@0.50 50-100m": "N/A",
            "AB(Byte)": "N/A",
            "num_samples": summary["num_evaluated_samples"],
            "evaluated_classes": ",".join(summary["evaluated_classes"]),
        })
    csv_path = ROOT / "official_ap_comparison.csv"
    json_path = ROOT / "official_ap_comparison.json"
    md_path = ROOT / "official_ap_comparison.md"
    fields = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({
        "table_scope": "Official-code Car AP on sensor-local Full common-val; values use the paper's 0-100 scale.",
        "distance_bins": "N/A: not computed because the official repository evaluator has no distance-bin API and sensor-local range must be defined before comparison",
        "interpretation_warning": "Camera rows are provisional 1-epoch checkpoints. LiDAR rows use 20-epoch checkpoints. Training schedules differ, so this table is a pipeline evaluation, not a final model ranking.",
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    headers = ["Modality", "Fusion", "Model", "Epoch", "Dataset", "AP3D@0.50 Overall", "APBEV@0.50 Overall", "0-30m", "30-50m", "50-100m"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join([
            row["Modality"], row["Fusion"], row["Model"], str(row["Training epochs"]),
            row["Dataset"], f"{row['AP_3D@0.50 Overall']:.2f}",
            f"{row['AP_BEV@0.50 Overall']:.2f}", "N/A", "N/A", "N/A",
        ]) + " |")
    lines += ["", "> Camera 为修复后 1 epoch 阶段性模型；LiDAR 为 20 epoch 模型。当前表格用于验证完整推理与官方 AP 链路，不能作为最终训练公平排名。"]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(csv_path)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
