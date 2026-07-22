#!/usr/bin/env python3
"""Migrate pre-fix ImVoxelNet predictions from bottom-centre to box-centre z."""

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = load_json(args.input)
    corrected = 0
    for record in records:
        for obj in record.get("pred_objects", []):
            center = obj.get("center_lidar")
            box = obj.get("box_lidar", {})
            if not isinstance(center, list) or len(center) != 3 or "dz" not in box:
                raise ValueError(f"Invalid camera prediction object: {obj}")
            center[2] = float(center[2]) + float(box["dz"]) / 2.0
            obj["box_origin"] = "geometric_center"
            corrected += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "input": str(args.input),
        "output": str(args.output),
        "corrected_objects": corrected,
        "correction": "center_lidar.z += box_lidar.dz / 2",
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
