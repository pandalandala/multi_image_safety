#!/usr/bin/env python3
"""Rewrite absolute workspace paths in project text files to relative ones.

Examples:
  /mnt/hdd/xuran/multi_image_safety/... -> multi_image_safety/...
  /mnt/hdd/xuran/cache/...              -> cache/...
  /mnt/hdd/xuran/anaconda3/...          -> anaconda3/...

By default this script performs a dry run and prints the files that would
change. Use --apply to write modifications in place.
"""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

DEFAULT_SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "data",
    "logs",
    "legacy_logs",
}

TEXT_SUFFIXES = {
    ".py",
    ".sh",
    ".yaml",
    ".yml",
    ".json",
    ".jsonl",
    ".md",
    ".txt",
    ".cfg",
    ".ini",
    ".toml",
    ".env",
}


def should_skip(path: Path, skip_dirs: set[str]) -> bool:
    return any(part in skip_dirs for part in path.parts)


def is_text_candidate(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in {
        "Dockerfile",
        "Makefile",
        ".env",
    }


def replace_workspace_prefix(text: str, workspace_root: Path) -> str:
    normalized_root = workspace_root.as_posix().rstrip("/")
    return text.replace(f"{normalized_root}/", "")


def process_file(path: Path, workspace_root: Path, apply: bool) -> tuple[bool, int]:
    try:
        original = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False, 0

    updated = replace_workspace_prefix(original, workspace_root)
    if updated == original:
        return False, 0

    changes = original.count(f"{workspace_root.as_posix().rstrip('/')}/")
    if apply:
        path.write_text(updated, encoding="utf-8")
    return True, changes


def iter_candidate_files(root: Path, skip_dirs: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if should_skip(rel, skip_dirs):
            continue
        if not is_text_candidate(path):
            continue
        files.append(path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite absolute /mnt/hdd/xuran/... paths to relative workspace paths."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root to scan. Defaults to the directory containing this script.",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=WORKSPACE_ROOT,
        help="Workspace prefix to strip. Defaults to the parent of project root.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes in place. Without this flag the script only reports changes.",
    )
    parser.add_argument(
        "--include-logs",
        action="store_true",
        help="Also rewrite files under logs/ and legacy_logs/.",
    )
    parser.add_argument(
        "--include-data",
        action="store_true",
        help="Also rewrite text files under data/.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    workspace_root = args.workspace_root.resolve()
    skip_dirs = set(DEFAULT_SKIP_DIRS)
    if args.include_logs:
        skip_dirs.discard("logs")
        skip_dirs.discard("legacy_logs")
    if args.include_data:
        skip_dirs.discard("data")

    files = iter_candidate_files(root, skip_dirs)
    changed_files = 0
    changed_refs = 0

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] root={root}")
    print(f"[{mode}] workspace_root={workspace_root}")

    for path in files:
        changed, count = process_file(path, workspace_root, apply=args.apply)
        if not changed:
            continue
        rel = path.relative_to(root)
        changed_files += 1
        changed_refs += count
        action = "updated" if args.apply else "would update"
        print(f"{action}: {rel} ({count} replacements)")

    print(
        f"[{mode}] changed_files={changed_files}, replaced_references={changed_refs}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
