"""Persistent cross-path caches for reusable retrieval and pairing results."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from src.common.utils import DATA_DIR, ensure_dir

logger = logging.getLogger(__name__)

SHARED_REUSE_DIR = DATA_DIR / "shared_reuse"
RETRIEVAL_CACHE_DIR = SHARED_REUSE_DIR / "retrieval"
PAIR_CACHE_DIR = SHARED_REUSE_DIR / "path5_pairs"
SHARED_IMAGE_DIR = SHARED_REUSE_DIR / "images"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _stable_hash(parts: Iterable[str]) -> str:
    data = "||".join(str(part) for part in parts)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _safe_load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.warning("Ignoring unreadable shared cache file: %s", path)
        return {}


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _retrieval_cache_path(
    query: str,
    *,
    purpose: str,
    min_width: int,
    min_height: int,
) -> Path:
    normalized = _normalize_text(query)
    key = _stable_hash([purpose, normalized, str(min_width), str(min_height)])
    return RETRIEVAL_CACHE_DIR / f"{key}.json"


def load_retrieval_cache(
    query: str,
    *,
    purpose: str,
    min_width: int,
    min_height: int,
    allowed_source_types: tuple[str, ...] | None = None,
) -> list[dict]:
    """Return reusable retrieval records with existing files."""
    cache_path = _retrieval_cache_path(
        query,
        purpose=purpose,
        min_width=min_width,
        min_height=min_height,
    )
    data = _safe_load_json(cache_path)
    items = data.get("items", [])
    valid = []
    for item in items:
        path = Path(str(item.get("path", "")).strip())
        if not path.exists():
            continue
        source_type = str(item.get("source_type", "")).strip().lower()
        if allowed_source_types and source_type not in allowed_source_types:
            continue
        valid.append(dict(item))
    return valid


def save_retrieval_cache(
    query: str,
    entries: list[dict],
    *,
    purpose: str,
    min_width: int,
    min_height: int,
    path_name: str,
) -> None:
    """Persist reusable retrieval records for future path runs."""
    if not entries:
        return
    cache_path = _retrieval_cache_path(
        query,
        purpose=purpose,
        min_width=min_width,
        min_height=min_height,
    )
    existing = _safe_load_json(cache_path)
    combined: dict[tuple[str, str], dict] = {}
    for item in existing.get("items", []):
        item_path = str(item.get("path", "")).strip()
        source_type = str(item.get("source_type", "")).strip().lower()
        if item_path:
            combined[(source_type, item_path)] = item
    for item in entries:
        item_path = str(item.get("path", "")).strip()
        source_type = str(item.get("source_type", "")).strip().lower()
        if not item_path or not Path(item_path).exists():
            continue
        enriched = dict(item)
        enriched.setdefault("source_type", source_type or "unknown")
        enriched.setdefault("created_by_path", path_name)
        enriched["updated_at"] = _now_iso()
        combined[(enriched["source_type"], item_path)] = enriched

    payload = {
        "query": _normalize_text(query),
        "purpose": purpose,
        "min_width": int(min_width),
        "min_height": int(min_height),
        "last_updated": _now_iso(),
        "items": list(combined.values()),
    }
    _atomic_write_json(cache_path, payload)


def materialize_shared_image(source_path: str | Path, *, source_type: str) -> str:
    """Copy an image into the shared reusable-image pool and return its path."""
    source = Path(source_path)
    if not source.exists():
        return str(source)
    stat = source.stat()
    suffix = source.suffix.lower() or ".png"
    token = _stable_hash(
        [str(source.resolve()), str(stat.st_size), str(stat.st_mtime_ns), source_type]
    )
    dest_dir = ensure_dir(SHARED_IMAGE_DIR / source_type)
    dest = dest_dir / f"{token}{suffix}"
    if not dest.exists():
        shutil.copy2(source, dest)
    return str(dest)


def reuse_cached_image(entry: dict, output_path: Path) -> bool:
    """Copy a cached reusable image entry to the requested output path."""
    source = Path(str(entry.get("path", "")).strip())
    if not source.exists():
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output_path)
    return True


def _pair_key(info1: dict, info2: dict) -> str:
    parts = []
    for info in (info1, info2):
        parts.append(
            "||".join(
                [
                    _normalize_text(str(info.get("path", ""))),
                    _normalize_text(str(info.get("class_label", info.get("class", "")))),
                    _normalize_text(str(info.get("description", info.get("caption", "")))),
                ]
            )
        )
    return _stable_hash(sorted(parts))


def load_cached_path5_pair_result(info1: dict, info2: dict) -> dict | None:
    path = PAIR_CACHE_DIR / f"{_pair_key(info1, info2)}.json"
    data = _safe_load_json(path)
    result = data.get("result")
    if isinstance(result, dict):
        return result
    return None


def save_cached_path5_pair_result(info1: dict, info2: dict, result: dict) -> None:
    if not result:
        return
    path = PAIR_CACHE_DIR / f"{_pair_key(info1, info2)}.json"
    payload = {
        "saved_at": _now_iso(),
        "info1": {
            "path": info1.get("path", ""),
            "class_label": info1.get("class_label", info1.get("class", "")),
            "description": info1.get("description", info1.get("caption", "")),
        },
        "info2": {
            "path": info2.get("path", ""),
            "class_label": info2.get("class_label", info2.get("class", "")),
            "description": info2.get("description", info2.get("caption", "")),
        },
        "result": result,
    }
    _atomic_write_json(path, payload)
