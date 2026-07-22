#!/usr/bin/env python3
"""Check Full Dataset Camera training images/labels without modifying source data."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root.resolve()
    coop = json.loads((root / "cooperative/data_info.json").read_text(encoding="utf-8"))
    v_to_i = {
        Path(item["vehicle_pointcloud_path"]).stem: Path(item["infrastructure_pointcloud_path"]).stem
        for item in coop
    }
    result = {}
    for sensor, side in (("vehicle_camera", "vehicle-side"), ("infrastructure_camera", "infrastructure-side")):
        rows = {}
        for split in ("train", "val", "full_common_val"):
            vehicle_ids = (root / "vehicle-side/ImageSets" / f"{split}.txt").read_text(encoding="utf-8").split()
            ids = vehicle_ids if sensor == "vehicle_camera" else [v_to_i.get(x, x) for x in vehicle_ids]
            missing_images = [x for x in ids if not (root / side / "image" / f"{x}.jpg").is_file()]
            missing_labels = [x for x in ids if not (root / side / "label" / ("lidar" if sensor == "vehicle_camera" else "virtuallidar") / f"{x}.json").is_file()]
            missing_calibs = [x for x in ids if not (root / side / "calib/camera_intrinsic" / f"{x}.json").is_file()]
            rows[split] = {"num_requested": len(ids), "missing_images": missing_images, "missing_labels": missing_labels, "missing_calibs": missing_calibs, "ready": not (missing_images or missing_labels or missing_calibs)}
        result[sensor] = rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
