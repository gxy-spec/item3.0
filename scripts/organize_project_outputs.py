#!/usr/bin/env python3
"""Archive known non-production experiment artifacts without deleting data.

The script intentionally has a dry-run default.  It only moves artifacts that
were explicitly identified during the Camera empty-GT and box-origin audits.
"""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
FULL_ROOT = OUTPUT_ROOT / "full_baselines"
ARCHIVE_ROOT = OUTPUT_ROOT / "archive"


def archive_plan():
    return [
        # These Camera checkpoints were trained before training-info labels
        # were fixed, so their losses were zero and predictions were empty.
        (FULL_ROOT / "vehicle_camera" / "work_dir",
         ARCHIVE_ROOT / "invalid_empty_gt_camera" / "vehicle_camera" / "work_dir",
         "empty-GT training attempt"),
        (FULL_ROOT / "vehicle_camera" / "work_dir_restart",
         ARCHIVE_ROOT / "invalid_empty_gt_camera" / "vehicle_camera" / "work_dir_restart",
         "empty-GT training attempt"),
        (FULL_ROOT / "infrastructure_camera" / "work_dir_3e",
         ARCHIVE_ROOT / "invalid_empty_gt_camera" / "infrastructure_camera" / "work_dir_3e",
         "empty-GT training attempt"),
        # Smoke runs verify pipeline health but are not formal checkpoints.
        (FULL_ROOT / "vehicle_camera" / "smoke",
         ARCHIVE_ROOT / "camera_smoke_tests" / "vehicle_camera" / "smoke",
         "short smoke test"),
        (FULL_ROOT / "vehicle_camera" / "smoke_8",
         ARCHIVE_ROOT / "camera_smoke_tests" / "vehicle_camera" / "smoke_8",
         "short smoke test"),
        (FULL_ROOT / "vehicle_camera" / "smoke_8_3e",
         ARCHIVE_ROOT / "camera_smoke_tests" / "vehicle_camera" / "smoke_8_3e",
         "short smoke test"),
        (FULL_ROOT / "vehicle_camera" / "smoke_8_fixed",
         ARCHIVE_ROOT / "camera_smoke_tests" / "vehicle_camera" / "smoke_8_fixed",
         "short smoke test"),
        (FULL_ROOT / "vehicle_camera" / "smoke_preflight",
         ARCHIVE_ROOT / "camera_smoke_tests" / "vehicle_camera" / "smoke_preflight",
         "preflight smoke test"),
        (FULL_ROOT / "infrastructure_camera" / "smoke",
         ARCHIVE_ROOT / "camera_smoke_tests" / "infrastructure_camera" / "smoke",
         "short smoke test"),
        (FULL_ROOT / "infrastructure_camera" / "smoke_8_fixed",
         ARCHIVE_ROOT / "camera_smoke_tests" / "infrastructure_camera" / "smoke_8_fixed",
         "short smoke test"),
        # Preserve, but remove from the active baseline namespace.
        (FULL_ROOT / "camera_invalid_empty_gt_archive",
         ARCHIVE_ROOT / "provenance" / "camera_invalid_empty_gt_predictions",
         "pre-fix zero-prediction provenance"),
        (FULL_ROOT / "camera_bottom_center_archive",
         ARCHIVE_ROOT / "provenance" / "camera_bottom_center_predictions",
         "pre-fix box-origin provenance"),
    ]


def path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Actually move planned artifacts.")
    args = parser.parse_args()

    records = []
    for source, destination, reason in archive_plan():
        record = {
            "source": str(source.relative_to(PROJECT_ROOT)),
            "destination": str(destination.relative_to(PROJECT_ROOT)),
            "reason": reason,
            "exists": source.exists(),
            "bytes": path_size(source) if source.exists() else 0,
            "action": "planned" if source.exists() else "skipped_missing",
        }
        if args.apply and source.exists():
            if destination.exists():
                raise FileExistsError(f"Archive destination already exists: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            record["action"] = "moved"
        records.append(record)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "apply" if args.apply else "dry_run",
        "policy": "No deletion. Only known invalid or smoke Camera artifacts are archived.",
        "active_outputs_kept": [
            "outputs/full_baselines/<baseline>/predictions_*.json",
            "outputs/full_baselines/<baseline>/predictions_world.json",
            "outputs/full_baselines/<baseline>/eval*.json/csv",
            "outputs/full_baselines/<baseline>/official_local_aligned_ap_*",
            "outputs/full_baselines/<baseline>/train_1e_fixed/epoch_1.pth",
            "outputs/full_baselines/<lidar baseline>/runtime_configs",
            "outputs/baselines, outputs/diagnosis, outputs/preprocessed, outputs/common_multimodal_dataset",
        ],
        "records": records,
    }
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    (ARCHIVE_ROOT / "organization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = sum(record["bytes"] for record in records if record["exists"])
    print(json.dumps({
        "mode": manifest["mode"],
        "candidate_bytes": total,
        "candidate_gib": round(total / 1024 ** 3, 2),
        "manifest": str(ARCHIVE_ROOT / "organization_manifest.json"),
        "records": records,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
