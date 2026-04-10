#!/usr/bin/env python3
"""Delete clearly generated images while preserving crawled/retrieved images.

This script is intentionally conservative:
- Deletes files explicitly marked as generated in JSONL metadata.
- Deletes known generated-only cache directories.
- Preserves files marked as retrieved/crawled.
- Reports ambiguous legacy outputs that cannot be safely classified.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path("/mnt/hdd/xuran/multi_image_safety")
RAW = ROOT / "data" / "raw"


def load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def unlink_if_exists(path: Path, deleted: list[Path]) -> None:
    if path.exists():
        path.unlink()
        deleted.append(path)


def rmtree_if_exists(path: Path, deleted_dirs: list[Path]) -> None:
    if path.exists():
        shutil.rmtree(path)
        deleted_dirs.append(path)


def delete_generated_from_jsonl(
    jsonl_paths: list[Path],
    path_keys: tuple[str, ...] = ("image1_path", "image2_path"),
) -> tuple[list[Path], dict[str, int]]:
    deleted: list[Path] = []
    counts: dict[str, int] = defaultdict(int)
    for jsonl_path in jsonl_paths:
        for row in load_jsonl(jsonl_path):
            for idx, key in enumerate(path_keys, start=1):
                img_path = row.get(key)
                acquisition = str(row.get(f"image{idx}_acquisition", "")).strip().lower()
                if img_path and acquisition == "generated":
                    unlink_if_exists(Path(img_path), deleted)
                    counts[str(jsonl_path)] += 1
    return deleted, counts


def main() -> int:
    deleted_files: list[Path] = []
    deleted_dirs: list[Path] = []
    notes: list[str] = []

    # Path 2: only delete paths explicitly marked as generated.
    path2_jsonls = sorted((RAW / "path2").glob("samples_with_images*.jsonl"))
    path2_deleted, path2_counts = delete_generated_from_jsonl(path2_jsonls)
    deleted_files.extend(path2_deleted)
    if not path2_deleted:
        notes.append("Path 2: no images were explicitly marked as generated in current JSONL outputs.")

    # Path 3: current crawled/extracted image pools are retrieved. Remove generated cache dir if present.
    path3_generated_dir = RAW / "path3" / "generated_images"
    if path3_generated_dir.exists():
        rmtree_if_exists(path3_generated_dir, deleted_dirs)
    else:
        notes.append("Path 3: no generated_images directory found.")

    path3_jsonls = sorted((RAW / "path3").glob("method_a_with_images*.jsonl"))
    path3_deleted, path3_counts = delete_generated_from_jsonl(path3_jsonls)
    deleted_files.extend(path3_deleted)

    # Path 4: only delete paths explicitly marked as generated.
    path4_jsonls = sorted((RAW / "path4").glob("samples_with_images*.jsonl"))
    path4_deleted, path4_counts = delete_generated_from_jsonl(path4_jsonls)
    deleted_files.extend(path4_deleted)
    if not path4_deleted:
        notes.append(
            "Path 4: existing JSONL outputs do not record acquisition mode, so legacy worker images were preserved."
        )

    # Path 5: generated_images is generation-only; remove it entirely.
    path5_generated_dir = RAW / "path5" / "generated_images"
    if path5_generated_dir.exists():
        rmtree_if_exists(path5_generated_dir, deleted_dirs)
    else:
        notes.append("Path 5: no generated_images directory found.")

    # If crawled_image_info still contains generated rows in the future, drop them while preserving crawled rows.
    crawl_info = RAW / "path5" / "crawled_image_info.jsonl"
    if crawl_info.exists():
        rows = load_jsonl(crawl_info)
        kept_rows: list[dict] = []
        removed_generated_rows = 0
        for row in rows:
            dataset = str(row.get("dataset", "")).strip().lower()
            row_path = str(row.get("path", ""))
            if dataset == "t2i_generated" or "/generated_images/" in row_path:
                removed_generated_rows += 1
                continue
            kept_rows.append(row)
        if removed_generated_rows:
            with crawl_info.open("w", encoding="utf-8") as f:
                for row in kept_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            notes.append(f"Path 5: removed {removed_generated_rows} generated rows from crawled_image_info.jsonl.")

    print("Deleted generated files:", len(deleted_files))
    for path in deleted_files[:20]:
        print("FILE", path)
    if len(deleted_files) > 20:
        print(f"... and {len(deleted_files) - 20} more files")

    print("Deleted generated directories:", len(deleted_dirs))
    for path in deleted_dirs:
        print("DIR", path)

    if path2_counts:
        print("Path 2 generated deletions by JSONL:", dict(path2_counts))
    if path3_counts:
        print("Path 3 generated deletions by JSONL:", dict(path3_counts))
    if path4_counts:
        print("Path 4 generated deletions by JSONL:", dict(path4_counts))

    if notes:
        print("Notes:")
        for note in notes:
            print("-", note)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
