#!/usr/bin/env python3
"""Safely expose split DAIR-V2X-C Full archives through the standard layout.

The Full Dataset packages image and velodyne directories beside the main
``cooperative-vehicle-infrastructure`` directory. This script creates directory
symlinks instead of moving or copying data, so the original archives remain
untouched and no nested duplicate directories are created.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path("/mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "full_baselines" / "dataset_preparation"

LAYOUTS = (
    ("vehicle_image", "cooperative-vehicle-infrastructure-vehicle-side-image", "vehicle-side/image"),
    ("vehicle_velodyne", "cooperative-vehicle-infrastructure-vehicle-side-velodyne", "vehicle-side/velodyne"),
    ("infrastructure_image", "cooperative-vehicle-infrastructure-infrastructure-side-image", "infrastructure-side/image"),
    ("infrastructure_velodyne", "cooperative-vehicle-infrastructure-infrastructure-side-velodyne", "infrastructure-side/velodyne"),
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def describe_target(path):
    if path.is_symlink():
        return {"state": "symlink", "link_target": os.readlink(path), "resolved_target": str(path.resolve())}
    if path.exists():
        return {"state": "directory" if path.is_dir() else "file", "resolved_target": str(path.resolve())}
    return {"state": "missing"}


def build_manifest(dataset_root, output_dir):
    standard_root = dataset_root / "cooperative-vehicle-infrastructure"
    if not standard_root.is_dir():
        raise FileNotFoundError(f"缺少标准数据根目录: {standard_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    file_manifest_path = output_dir / "full_dataset_layout_file_manifest.jsonl"
    summary_entries = []
    with file_manifest_path.open("w", encoding="utf-8") as manifest_file:
        for name, source_name, target_rel in LAYOUTS:
            source = dataset_root / source_name
            target = standard_root / target_rel
            if not source.is_dir():
                raise FileNotFoundError(f"缺少并列解压目录: {source}")
            files = sorted(
                (entry for entry in os.scandir(source) if entry.is_file()),
                key=lambda entry: entry.name,
            )
            empty_count = 0
            for entry in files:
                stat = entry.stat()
                empty = stat.st_size == 0
                empty_count += int(empty)
                manifest_file.write(json.dumps({
                    "group": name,
                    "action": "create_directory_symlink",
                    "source": entry.path,
                    "target": str(target / entry.name),
                    "size_bytes": stat.st_size,
                    "is_empty": empty,
                }, ensure_ascii=False) + "\n")
            summary_entries.append({
                "group": name,
                "action": "create_directory_symlink",
                "source_directory": str(source),
                "target_directory": str(target),
                "file_count": len(files),
                "empty_file_count": empty_count,
                "sample_files": [entry.name for entry in files[:5]],
                "target_before": describe_target(target),
            })

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "standard_root": str(standard_root),
        "operation": "directory_symlink_only",
        "raw_archives_modified": False,
        "file_manifest_jsonl": str(file_manifest_path),
        "entries": summary_entries,
    }
    manifest_path = output_dir / "full_dataset_layout_manifest.json"
    backup_path = output_dir / "full_dataset_layout_backup_snapshot.json"
    write_json(manifest_path, manifest)
    write_json(backup_path, {
        "created_at_utc": manifest["created_at_utc"],
        "purpose": "Snapshot of standard destination paths before layout application.",
        "dataset_root": str(dataset_root),
        "targets": [
            {"target_directory": entry["target_directory"], "before": entry["target_before"]}
            for entry in summary_entries
        ],
        "raw_archives_modified": False,
    })
    return manifest, manifest_path, backup_path


def apply_layout(manifest):
    created, already_correct = [], []
    for entry in manifest["entries"]:
        source = Path(entry["source_directory"]).resolve()
        target = Path(entry["target_directory"])
        if target.is_symlink():
            if target.resolve() == source:
                already_correct.append(str(target))
                continue
            raise FileExistsError(
                f"拒绝覆盖指向其他位置的软链接: {target} -> {target.resolve()}"
            )
        if target.exists():
            raise FileExistsError(f"拒绝覆盖已有路径: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(os.path.relpath(source, start=target.parent), target_is_directory=True)
        created.append(str(target))
    return created, already_correct


def main():
    parser = argparse.ArgumentParser(description="安全整理 DAIR-V2X-C Full Dataset 的 image/velodyne 目录")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--apply", action="store_true", help="确认后创建标准目录软链接；默认仅输出清单和备份快照")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    manifest, manifest_path, backup_path = build_manifest(dataset_root, output_dir)
    print(f"[PLAN] manifest: {manifest_path}")
    print(f"[PLAN] file list: {manifest['file_manifest_jsonl']}")
    print(f"[PLAN] backup snapshot: {backup_path}")
    for entry in manifest["entries"]:
        print(f"[PLAN] {entry['group']}: {entry['file_count']} files, {entry['source_directory']} -> {entry['target_directory']}")

    if not args.apply:
        print("[DRY RUN] 未修改数据目录。确认清单后使用 --apply 创建软链接。")
        return

    created, already_correct = apply_layout(manifest)
    result_path = output_dir / "full_dataset_layout_apply_result.json"
    write_json(result_path, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "directory_symlink_only",
        "created": created,
        "already_correct": already_correct,
        "raw_archives_modified": False,
    })
    print(f"[DONE] created={len(created)}, already_correct={len(already_correct)}")
    print(f"[DONE] apply result: {result_path}")


if __name__ == "__main__":
    main()
