"""Search locally downloaded image datasets by text query.

Supported datasets (enable each in config/pipeline.yaml after downloading):
  - MSCOCO 2017: caption-based search (~118K train images, rich 5-caption annotations)
  - Open Images V7: class-label search (download a subset of classes)
  - ImageNet (ILSVRC 2012): synset-description search (~1.2M images)
  - CC12M+ImageNet21K recap: metadata.jsonl + image_path search across both recap sources

Each dataset index is lazy-loaded once and kept in memory for fast repeated queries.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import shutil
import sqlite3
import threading
import time
import math
from pathlib import Path
from typing import Optional

from PIL import Image

from src.common.retrieval_queries import (
    build_compact_retrieval_query,
    build_retrieval_query_variants,
    simple_subject_bonus,
)
from src.common.utils import load_config

logger = logging.getLogger(__name__)

# ── Lazy-loaded singleton indices ──────────────────────────────────────────

_coco_index: Optional[dict] = None
_open_images_index: Optional[dict] = None
_imagenet_index: Optional[dict] = None
_recap_index_paths: Optional[dict] = None
_local_query_cache: dict[tuple[str, int, int, int, str], list[dict]] = {}
_coco_index_lock = threading.Lock()
_open_images_index_lock = threading.Lock()
_imagenet_index_lock = threading.Lock()
_recap_index_paths_lock = threading.Lock()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"


def _log_progress(
    prefix: str,
    processed: int,
    start_time: float,
    *,
    total: int | None = None,
    every: int = 50000,
    force: bool = False,
) -> None:
    if not force and (processed <= 0 or processed % every != 0):
        return
    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0.0
    if total:
        pct = processed / total * 100
        logger.info(
            "%s: %d/%d (%.1f%%) elapsed=%s rate=%.0f items/s",
            prefix,
            processed,
            total,
            pct,
            _format_elapsed(elapsed),
            rate,
        )
    else:
        logger.info(
            "%s: %d processed elapsed=%s rate=%.0f items/s",
            prefix,
            processed,
            _format_elapsed(elapsed),
            rate,
        )


def _get_local_datasets_config() -> dict:
    try:
        config = load_config()
        return config.get("local_datasets", {})
    except Exception:
        return {}


def _tokenize(text: str) -> list[str]:
    return [tok for tok in re.findall(r"\w+", str(text or "").lower()) if tok]


def _get_local_search_target(dataset_name: str, requested: int) -> int:
    env_specific = os.environ.get(f"MIS_LOCAL_SEARCH_TARGET_{dataset_name.upper()}", "").strip()
    env_global = os.environ.get("MIS_LOCAL_SEARCH_TARGET_CANDIDATES", "").strip()
    for raw in (env_specific, env_global):
        if raw:
            try:
                return max(1, min(requested, int(raw)))
            except ValueError:
                logger.warning("Invalid local search target %r for %s", raw, dataset_name)
    return max(1, requested)


def _get_recap_limit(num_results: int) -> int:
    env_value = os.environ.get("MIS_RECAP_QUERY_LIMIT", "").strip()
    if env_value:
        try:
            return max(num_results, int(env_value))
        except ValueError:
            logger.warning("Invalid MIS_RECAP_QUERY_LIMIT=%r; falling back to default", env_value)
    return max(20, num_results * 6)


def _get_local_source_mix_enabled() -> bool:
    return _env_flag("MIS_LOCAL_SOURCE_MIX", default=False)


def _get_local_source_soft_cap(num_results: int, active_sources: int) -> int:
    env_value = os.environ.get("MIS_LOCAL_SEARCH_SOFT_CAP", "").strip()
    if env_value:
        try:
            return max(1, int(env_value))
        except ValueError:
            logger.warning("Invalid MIS_LOCAL_SEARCH_SOFT_CAP=%r; falling back to default", env_value)
    if active_sources <= 0:
        return max(1, num_results)
    return max(1, math.ceil(num_results / active_sources))


def _merge_source_diverse_results(
    results_by_source: dict[str, list[dict]],
    order: list[str],
    num_results: int,
) -> list[dict]:
    """Mix candidates from all local datasets instead of letting one source dominate."""
    queues = {
        source: list(results_by_source.get(source, []))
        for source in order
        if results_by_source.get(source)
    }
    if not queues:
        return []

    selected: list[dict] = []
    seen_paths: set[str] = set()
    active_sources = [source for source in order if source in queues]
    soft_cap = _get_local_source_soft_cap(num_results, len(active_sources))
    taken_by_source = {source: 0 for source in active_sources}

    def _take_from_source(source: str) -> bool:
        queue = queues.get(source, [])
        while queue:
            item = queue.pop(0)
            path = item.get("path", "")
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            selected.append(item)
            taken_by_source[source] = taken_by_source.get(source, 0) + 1
            return True
        return False

    while len(selected) < num_results:
        progressed = False
        for source in active_sources:
            if taken_by_source.get(source, 0) >= soft_cap:
                continue
            if _take_from_source(source):
                progressed = True
                if len(selected) >= num_results:
                    break
        if not progressed:
            break

    while len(selected) < num_results:
        progressed = False
        for source in active_sources:
            if _take_from_source(source):
                progressed = True
                if len(selected) >= num_results:
                    break
        if not progressed:
            break

    if selected:
        counts: dict[str, int] = {}
        for item in selected:
            provider = str(item.get("provider", item.get("dataset", "unknown")))
            counts[provider] = counts.get(provider, 0) + 1
        logger.info(
            "Local search mixed select: total=%d source_breakdown=%s",
            len(selected),
            ", ".join(f"{source}:{count}" for source, count in sorted(counts.items())),
        )
    return selected


def _check_image_size(path: Path, min_width: int, min_height: int) -> tuple[bool, int, int]:
    """Return (passes, width, height)."""
    try:
        img = Image.open(path)
        w, h = img.size
        return (w >= min_width and h >= min_height), w, h
    except Exception:
        return False, 0, 0


def _match_score(query_words: set[str], text: str, query_raw: str = "") -> float:
    """Score how well *query_words* matches *text*. Higher is better."""
    if not query_words:
        return 0.0
    text_lower = text.lower()
    text_words = set(re.findall(r"\w+", text_lower))
    overlap = len(query_words & text_words)
    score = overlap / len(query_words)
    # Bonus for exact substring match
    if query_raw and query_raw in text_lower:
        score += 0.5
    return score


def _build_query_variants(query: str) -> list[tuple[str, set[str]]]:
    variants = []
    for variant in build_retrieval_query_variants(query):
        qwords = set(re.findall(r"\w+", variant.lower()))
        if qwords:
            variants.append((variant.lower(), qwords))
    return variants


def _best_variant_match_score(query: str, text: str) -> tuple[float, str]:
    """Return best adaptive query-variant score and the matching variant."""
    best_score = 0.0
    best_variant = query.lower()
    for variant_raw, variant_words in _build_query_variants(query):
        score = _match_score(variant_words, text, variant_raw)
        if score > best_score:
            best_score = score
            best_variant = variant_raw
    return best_score, best_variant


def _infer_class_label(*texts: str, fallback: str = "") -> str:
    """Infer a simple 1-2 word class label from free text."""
    for text in texts:
        text = str(text or "").strip()
        if not text:
            continue
        label = build_compact_retrieval_query(text, max_words=2)
        if label:
            return label
    fallback = str(fallback or "").strip()
    if fallback:
        label = build_compact_retrieval_query(fallback, max_words=2)
        if label:
            return label
    return ""


def _fts_escape(token: str) -> str:
    return token.replace('"', ' ')


def _build_fts_query(query: str) -> str:
    clauses: list[str] = []
    for variant_raw, variant_words in _build_query_variants(query):
        tokens = [tok for tok in re.findall(r"\w+", variant_raw.lower()) if tok]
        if not tokens:
            continue
        clause = " AND ".join(f'"{_fts_escape(tok)}"' for tok in tokens)
        if clause:
            clauses.append(f"({clause})")
    return " OR ".join(clauses[:8])


# ── MSCOCO ─────────────────────────────────────────────────────────────────

def _load_coco_index() -> dict:
    global _coco_index
    if _coco_index is not None:
        return _coco_index
    with _coco_index_lock:
        if _coco_index is not None:
            return _coco_index

        cfg = _get_local_datasets_config().get("mscoco", {})
        if not cfg.get("enabled", False):
            _coco_index = {}
            return _coco_index

        root = Path(_get_local_datasets_config().get("root", ""))
        images_dir = root / cfg.get("images_dir", "coco/train2017")
        ann_file = root / cfg.get("annotations", "coco/annotations/captions_train2017.json")

        if not ann_file.exists():
            logger.warning("MSCOCO annotations not found: %s", ann_file)
            _coco_index = {}
            return _coco_index

        stage_start = time.time()
        logger.info("Loading MSCOCO caption index from %s ...", ann_file)
        with open(ann_file, "r") as f:
            data = json.load(f)
        logger.info(
            "MSCOCO JSON parsed: images=%d annotations=%d elapsed=%s",
            len(data.get("images", [])),
            len(data.get("annotations", [])),
            _format_elapsed(time.time() - stage_start),
        )

        index: dict[int, dict] = {}
        token_index: dict[str, set[int]] = {}
        image_start = time.time()
        images = data.get("images", [])
        for i, img in enumerate(images, start=1):
            index[img["id"]] = {
                "path": str(images_dir / img["file_name"]),
                "width": img.get("width", 0),
                "height": img.get("height", 0),
                "captions": [],
            }
            _log_progress("MSCOCO image map", i, image_start, total=len(images), every=25000)
        _log_progress("MSCOCO image map", len(images), image_start, total=len(images), force=True)

        annotation_start = time.time()
        annotations = data.get("annotations", [])
        for i, ann in enumerate(annotations, start=1):
            img_id = ann["image_id"]
            if img_id in index:
                caption = ann["caption"].lower()
                index[img_id]["captions"].append(caption)
                for token in set(_tokenize(caption)):
                    token_index.setdefault(token, set()).add(img_id)
            _log_progress("MSCOCO caption attach", i, annotation_start, total=len(annotations), every=100000)
        _log_progress(
            "MSCOCO caption attach",
            len(annotations),
            annotation_start,
            total=len(annotations),
            force=True,
        )

        _coco_index = {"images": index, "token_index": token_index}
        logger.info(
            "MSCOCO index loaded: %d images, %d captions total_elapsed=%s",
            len(index),
            sum(len(v["captions"]) for v in index.values()),
            _format_elapsed(time.time() - stage_start),
        )
        return _coco_index


def _search_coco(query: str, num_results: int, min_w: int, min_h: int) -> list[dict]:
    index = _load_coco_index()
    if not index:
        return []

    query_lower = query.lower()
    if not _build_query_variants(query_lower):
        return []

    scored: list[tuple[float, dict]] = []
    candidate_ids: set[int] = set()
    for _, query_words in _build_query_variants(query_lower):
        for token in query_words:
            candidate_ids.update(index.get("token_index", {}).get(token, set()))
    candidates = [index["images"][img_id] for img_id in candidate_ids] if candidate_ids else list(index["images"].values())

    for meta in candidates:
        if meta["width"] < min_w or meta["height"] < min_h:
            continue
        best = 0.0
        best_cap = ""
        for cap in meta["captions"]:
            s, _ = _best_variant_match_score(query_lower, cap)
            s += simple_subject_bonus(cap)
            if s > best:
                best = s
                best_cap = cap
        if best >= 0.3:
            scored.append((best, {
                "path": meta["path"],
                "caption": best_cap,
                "width": meta["width"],
                "height": meta["height"],
                "provider": "mscoco",
                "dataset": "mscoco",
                "class": _infer_class_label(best_cap, fallback=query_lower),
                "class_label": _infer_class_label(best_cap, fallback=query_lower),
                "score": best,
            }))

    scored.sort(key=lambda x: -x[0])
    return [item for _, item in scored[:num_results]]


# ── Open Images ────────────────────────────────────────────────────────────

def _load_open_images_index() -> dict:
    global _open_images_index
    if _open_images_index is not None:
        return _open_images_index
    with _open_images_index_lock:
        if _open_images_index is not None:
            return _open_images_index

        cfg = _get_local_datasets_config().get("open_images", {})
        if not cfg.get("enabled", False):
            _open_images_index = {}
            return _open_images_index

        root = Path(_get_local_datasets_config().get("root", ""))
        images_dir = root / cfg.get("images_dir", "open_images/train")
        labels_csv = root / cfg.get("labels_csv", "open_images/oidv7-class-descriptions.csv")
        ann_csv = root / cfg.get("annotations_csv",
                                  "open_images/oidv7-train-annotations-human-imagelabels.csv")

        if not labels_csv.exists():
            logger.warning("Open Images class descriptions not found: %s", labels_csv)
            _open_images_index = {}
            return _open_images_index
        if not ann_csv.exists():
            logger.warning("Open Images annotations not found: %s", ann_csv)
            _open_images_index = {}
            return _open_images_index

        if not images_dir.exists():
            logger.warning("Open Images image directory not found: %s", images_dir)
            _open_images_index = {}
            return _open_images_index

        # 1) class label_name -> display name
        stage_start = time.time()
        logger.info("Loading Open Images class descriptions ...")
        class_names: dict[str, str] = {}
        labels_start = time.time()
        with open(labels_csv, "r", newline="") as f:
            for i, row in enumerate(csv.reader(f), start=1):
                if len(row) >= 2:
                    class_names[row[0]] = row[1].strip().lower()
                _log_progress("Open Images class descriptions", i, labels_start, every=5000)
        _log_progress(
            "Open Images class descriptions",
            len(class_names),
            labels_start,
            force=True,
        )

        # 2) image_id -> list[display_name]
        logger.info("Loading Open Images annotations (this may take a moment) ...")
        image_labels: dict[str, list[str]] = {}
        ann_start = time.time()
        with open(ann_csv, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            processed = 0
            for processed, row in enumerate(reader, start=1):
                if len(row) < 3:
                    continue
                image_id, _, label_name = row[0], row[1], row[2]
                confidence = float(row[3]) if len(row) > 3 else 1.0
                if confidence < 1:
                    continue  # only keep high-confidence labels
                desc = class_names.get(label_name, "")
                if desc:
                    image_labels.setdefault(image_id, []).append(desc)
                _log_progress("Open Images annotations", processed, ann_start, every=500000)
        _log_progress("Open Images annotations", processed, ann_start, force=True)

        logger.info("Scanning Open Images image files under %s ...", images_dir)
        file_map: dict[str, str] = {}
        scan_start = time.time()
        processed_files = 0
        for processed_files, img_path in enumerate(images_dir.rglob("*"), start=1):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            file_map.setdefault(img_path.stem, str(img_path))
            _log_progress("Open Images file scan", processed_files, scan_start, every=100000)
        _log_progress("Open Images file scan", processed_files, scan_start, force=True)

        index: dict[str, dict] = {}
        token_index: dict[str, set[str]] = {}
        build_start = time.time()
        image_items = list(image_labels.items())
        for i, (image_id, labels) in enumerate(image_items, start=1):
            img_path = file_map.get(image_id)
            if not img_path:
                continue
            index[image_id] = {"path": str(img_path), "labels": labels}
            for label in labels:
                for token in set(_tokenize(label)):
                    token_index.setdefault(token, set()).add(image_id)
            _log_progress(
                "Open Images index build",
                i,
                build_start,
                total=len(image_items),
                every=100000,
            )
        _log_progress(
            "Open Images index build",
            len(image_items),
            build_start,
            total=len(image_items),
            force=True,
        )

        _open_images_index = {"images": index, "token_index": token_index}
        logger.info(
            "Open Images index loaded: %d images total_elapsed=%s",
            len(index),
            _format_elapsed(time.time() - stage_start),
        )
        return _open_images_index


def _search_open_images(query: str, num_results: int, min_w: int, min_h: int) -> list[dict]:
    index = _load_open_images_index()
    if not index:
        return []

    query_lower = query.lower()
    if not _build_query_variants(query_lower):
        return []

    scored: list[tuple[float, str, str]] = []
    candidate_ids: set[str] = set()
    for _, query_words in _build_query_variants(query_lower):
        for token in query_words:
            candidate_ids.update(index.get("token_index", {}).get(token, set()))
    candidate_items = (
        [(image_id, index["images"][image_id]) for image_id in candidate_ids if image_id in index["images"]]
        if candidate_ids
        else list(index["images"].items())
    )

    for image_id, meta in candidate_items:
        best = 0.0
        best_label = ""
        for label in meta["labels"]:
            s, _ = _best_variant_match_score(query_lower, label)
            s += simple_subject_bonus(label)
            if s > best:
                best = s
                best_label = label
        if best >= 0.5:
            scored.append((best, meta["path"], best_label))

    scored.sort(key=lambda x: -x[0])

    results: list[dict] = []
    for score, path_str, label in scored:
        if len(results) >= num_results:
            break
        p = Path(path_str)
        if not p.exists():
            continue
        ok, w, h = _check_image_size(p, min_w, min_h)
        if not ok:
            continue
        results.append({
            "path": path_str,
            "caption": label,
            "width": w,
            "height": h,
            "provider": "open_images",
            "dataset": "open_images",
            "class": _infer_class_label(label, fallback=query_lower),
            "class_label": _infer_class_label(label, fallback=query_lower),
            "score": score,
        })
    return results


# ── ImageNet ───────────────────────────────────────────────────────────────

def _load_imagenet_index() -> dict:
    global _imagenet_index
    if _imagenet_index is not None:
        return _imagenet_index
    with _imagenet_index_lock:
        if _imagenet_index is not None:
            return _imagenet_index

        cfg = _get_local_datasets_config().get("imagenet", {})
        if not cfg.get("enabled", False):
            _imagenet_index = {}
            return _imagenet_index

        root = Path(_get_local_datasets_config().get("root", ""))
        images_dir = root / cfg.get("images_dir", "imagenet/train")

        # Find synset mapping file (multiple common formats)
        synsets_file = None
        for candidate in [
            root / cfg.get("synsets_file", "imagenet/LOC_synset_mapping.txt"),
            root / "imagenet" / "LOC_synset_mapping.txt",
            root / "imagenet" / "synset_words.txt",
        ]:
            if candidate.exists():
                synsets_file = candidate
                break

        if synsets_file is None or not images_dir.exists():
            logger.warning("ImageNet data not found (images_dir=%s)", images_dir)
            _imagenet_index = {}
            return _imagenet_index

        stage_start = time.time()
        logger.info("Loading ImageNet synset mapping from %s ...", synsets_file)
        index: dict[str, dict] = {}
        with open(synsets_file, "r") as f:
            processed = 0
            for processed, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                # Handles both "nXXXX description" and "nXXXX\tdescription"
                parts = line.split(None, 1) if "\t" not in line else line.split("\t", 1)
                if len(parts) < 2:
                    continue
                synset_id = parts[0].strip()
                description = parts[1].strip().lower()
                synset_dir = images_dir / synset_id
                if synset_dir.is_dir():
                    index[synset_id] = {
                        "dir": str(synset_dir),
                        "description": description,
                    }
                _log_progress("ImageNet synset scan", processed, stage_start, every=500)

        _imagenet_index = index
        _log_progress("ImageNet synset scan", processed, stage_start, force=True)
        logger.info(
            "ImageNet index loaded: %d synsets total_elapsed=%s",
            len(index),
            _format_elapsed(time.time() - stage_start),
        )
        return _imagenet_index


def _search_imagenet(query: str, num_results: int, min_w: int, min_h: int) -> list[dict]:
    index = _load_imagenet_index()
    if not index:
        return []

    query_lower = query.lower()
    if not _build_query_variants(query_lower):
        return []

    scored: list[tuple[float, str, dict]] = []
    for synset_id, meta in index.items():
        desc = meta["description"]
        s, _ = _best_variant_match_score(query_lower, desc)
        # Also check individual comma-separated synonyms
        for synonym in desc.split(","):
            syn = synonym.strip()
            if query_lower in syn:
                s = max(s, 0.8)
                break
        s += simple_subject_bonus(desc.split(",")[0].strip())
        if s >= 0.4:
            scored.append((s, synset_id, meta))

    scored.sort(key=lambda x: -x[0])

    results: list[dict] = []
    for score, synset_id, meta in scored:
        if len(results) >= num_results:
            break
        synset_dir = Path(meta["dir"])
        # Pick up to 3 images per synset
        images = sorted(synset_dir.glob("*.JPEG"))[:3]
        if not images:
            images = sorted(synset_dir.glob("*.jpeg"))[:3]
        if not images:
            images = sorted(synset_dir.glob("*.jpg"))[:3]
        for img_path in images:
            if len(results) >= num_results:
                break
            ok, w, h = _check_image_size(img_path, min_w, min_h)
            if not ok:
                continue
            results.append({
                "path": str(img_path),
                "caption": meta["description"].split(",")[0].strip(),
                "width": w,
                "height": h,
                "provider": "imagenet",
                "dataset": "imagenet",
                "class": _infer_class_label(meta["description"], fallback=query_lower),
                "class_label": _infer_class_label(meta["description"], fallback=query_lower),
                "score": score,
            })
    return results


# ── CC12M + ImageNet21K Recap ───────────────────────────────────────────────

def _get_recap_paths() -> dict:
    global _recap_index_paths
    if _recap_index_paths is not None:
        return _recap_index_paths
    with _recap_index_paths_lock:
        if _recap_index_paths is not None:
            return _recap_index_paths

        cfg = _get_local_datasets_config().get("cc12m_imagenet_recap", {})
        if not cfg.get("enabled", False):
            _recap_index_paths = {}
            return _recap_index_paths

        root = Path(_get_local_datasets_config().get("root", ""))
        metadata_path = root / cfg.get(
            "metadata",
            "CC12M_and_Imagenet21K_Recap/metadata.jsonl",
        )
        index_db = root / cfg.get(
            "index_db",
            "CC12M_and_Imagenet21K_Recap/metadata_search.sqlite3",
        )

        if not metadata_path.exists():
            logger.warning("Recap metadata not found: %s", metadata_path)
            _recap_index_paths = {}
            return _recap_index_paths

        _recap_index_paths = {
            "metadata": metadata_path,
            "index_db": index_db,
        }
        return _recap_index_paths


def _build_recap_index(metadata_path: Path, index_db: Path) -> None:
    stage_start = time.time()
    logger.info("Building recap SQLite index from %s ...", metadata_path)
    index_db.parent.mkdir(parents=True, exist_ok=True)
    tmp_db = index_db.with_suffix(index_db.suffix + ".tmp")
    tmp_db.unlink(missing_ok=True)

    con = sqlite3.connect(tmp_db)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA temp_store=MEMORY")
        con.execute("PRAGMA cache_size=-200000")
        con.execute(
            """
            CREATE VIRTUAL TABLE recap_fts USING fts5(
                source UNINDEXED,
                image_id UNINDEXED,
                class_text,
                category_text,
                recaption,
                recaption_short,
                image_path UNINDEXED,
                tokenize='unicode61'
            )
            """
        )

        batch = []
        inserted = 0
        with open(metadata_path, "r") as f:
            processed = 0
            for processed, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                batch.append((
                    str(row.get("source", "") or ""),
                    str(row.get("id", "") or ""),
                    str(row.get("class", "") or ""),
                    str(row.get("category", "") or ""),
                    str(row.get("recaption", "") or ""),
                    str(row.get("recaption_short", "") or ""),
                    str(row.get("image_path", "") or ""),
                ))
                if len(batch) >= 5000:
                    con.executemany(
                        """
                        INSERT INTO recap_fts (
                            source, image_id, class_text, category_text,
                            recaption, recaption_short, image_path
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        batch,
                    )
                    inserted += len(batch)
                    batch.clear()
                    _log_progress("Recap index build", inserted, stage_start, every=200000)

        if batch:
            con.executemany(
                """
                INSERT INTO recap_fts (
                    source, image_id, class_text, category_text,
                    recaption, recaption_short, image_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                batch,
            )
            inserted += len(batch)
        con.commit()
        _log_progress("Recap index build", inserted, stage_start, force=True)
        logger.info(
            "Recap index build complete: %d rows indexed total_elapsed=%s",
            inserted,
            _format_elapsed(time.time() - stage_start),
        )
    finally:
        con.close()

    tmp_db.replace(index_db)


def _recap_index_lock_path(index_db: Path) -> Path:
    return index_db.with_suffix(index_db.suffix + ".lock")


def _recap_index_is_valid(index_db: Path) -> bool:
    if not index_db.exists() or index_db.stat().st_size <= 0:
        return False
    try:
        con = sqlite3.connect(index_db)
        try:
            row = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='recap_fts'"
            ).fetchone()
            if not row:
                return False
            con.execute("SELECT rowid FROM recap_fts LIMIT 1").fetchone()
            return True
        finally:
            con.close()
    except sqlite3.Error:
        return False


def _wait_for_recap_index(index_db: Path, timeout_s: float = 7200.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _recap_index_is_valid(index_db):
            return True
        time.sleep(1.0)
    return _recap_index_is_valid(index_db)


def _rebuild_recap_index_if_needed(metadata_path: Path, index_db: Path) -> None:
    lock_path = _recap_index_lock_path(index_db)
    tmp_db = index_db.with_suffix(index_db.suffix + ".tmp")
    timeout_s = float(os.environ.get("MIS_RECAP_INDEX_BUILD_TIMEOUT_S", "7200"))

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            logger.info("Waiting for recap index build lock: %s", lock_path)
            if _wait_for_recap_index(index_db, timeout_s=timeout_s):
                return
            stale_for_s = time.time() - lock_path.stat().st_mtime if lock_path.exists() else 0.0
            if stale_for_s > timeout_s:
                logger.warning(
                    "Recap index lock looks stale (%.1fs); forcing rebuild: %s",
                    stale_for_s,
                    lock_path,
                )
                lock_path.unlink(missing_ok=True)
                continue
            raise RuntimeError(
                f"Timed out waiting for recap index to become valid: {index_db}"
            )

    try:
        if _recap_index_is_valid(index_db) and index_db.stat().st_mtime >= metadata_path.stat().st_mtime:
            return
        if index_db.exists():
            logger.warning("Existing recap index is missing/invalid; rebuilding: %s", index_db)
            index_db.unlink(missing_ok=True)
        tmp_db.unlink(missing_ok=True)
        _build_recap_index(metadata_path, index_db)
    finally:
        lock_path.unlink(missing_ok=True)


def _ensure_recap_index() -> dict:
    paths = _get_recap_paths()
    if not paths:
        return {}

    metadata_path = paths["metadata"]
    index_db = paths["index_db"]
    needs_rebuild = (
        (not index_db.exists())
        or (index_db.stat().st_mtime < metadata_path.stat().st_mtime)
        or (not _recap_index_is_valid(index_db))
    )
    if needs_rebuild:
        _rebuild_recap_index_if_needed(metadata_path, index_db)
    return paths


def _search_recap_source(
    query: str,
    num_results: int,
    min_w: int,
    min_h: int,
    source_filter: str,
) -> list[dict]:
    paths = _ensure_recap_index()
    if not paths:
        return []

    fts_query = _build_fts_query(query)
    if not fts_query:
        return []

    limit = _get_recap_limit(num_results)
    rows = None
    for attempt in range(2):
        con = sqlite3.connect(paths["index_db"])
        try:
            rows = con.execute(
                """
                SELECT
                    source, image_id, class_text, category_text,
                    recaption, recaption_short, image_path,
                    bm25(recap_fts) AS rank
                FROM recap_fts
                WHERE recap_fts MATCH ? AND source = ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, source_filter, limit),
            ).fetchall()
            break
        except sqlite3.OperationalError as exc:
            if ("no such table: recap_fts" not in str(exc).lower()) or attempt > 0:
                raise
            logger.warning(
                "Recap index query hit missing FTS table; forcing index rebuild and retrying once: %s",
                paths["index_db"],
            )
            _rebuild_recap_index_if_needed(paths["metadata"], paths["index_db"])
        finally:
            con.close()
    if rows is None:
        return []

    query_lower = query.lower()
    results: list[tuple[float, dict]] = []
    provider = "recap_cc12m" if source_filter == "cc12m" else "recap_imagenet"
    for source, image_id, class_text, category_text, recaption, recaption_short, image_path, rank in rows:
        p = Path(image_path)
        if not p.exists():
            continue
        ok, w, h = _check_image_size(p, min_w, min_h)
        if not ok:
            continue
        search_text = " ".join(
            part for part in [category_text, class_text, recaption_short, recaption] if part
        ).lower()
        score, _ = _best_variant_match_score(query_lower, search_text)
        score += simple_subject_bonus((category_text or recaption_short or class_text or "").lower())
        if query_lower and query_lower in search_text:
            score += 0.2
        caption = recaption_short or category_text or class_text or ""
        class_label = _infer_class_label(category_text, class_text, recaption_short, fallback=query_lower)
        results.append((
            score,
            {
                "path": str(p),
                "caption": caption,
                "width": w,
                "height": h,
                "provider": provider,
                "score": score,
                "dataset": provider,
                "source": source,
                "class": class_label,
                "class_label": class_label,
                "class_text": class_text,
                "category_text": category_text,
            },
        ))

    results.sort(key=lambda item: -item[0])
    return [item for _, item in results[:num_results]]


def _search_recap_imagenet(query: str, num_results: int, min_w: int, min_h: int) -> list[dict]:
    return _search_recap_source(query, num_results, min_w, min_h, "imagenet")


def _search_recap_cc12m(query: str, num_results: int, min_w: int, min_h: int) -> list[dict]:
    return _search_recap_source(query, num_results, min_w, min_h, "cc12m")


def _get_local_search_order() -> list[str]:
    """Return the ordered local-dataset search priority."""
    cfg = _get_local_datasets_config()
    raw = os.environ.get("MIS_LOCAL_DATASET_ORDER", "")
    if raw.strip():
        order = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        order = list(cfg.get("search_order", ["mscoco", "open_images", "imagenet", "recap_imagenet", "recap_cc12m"]))

    valid = {"mscoco", "open_images", "imagenet", "recap_imagenet", "recap_cc12m"}
    normalized = [name for name in order if name in valid]
    for fallback in ("mscoco", "open_images", "imagenet", "recap_imagenet", "recap_cc12m"):
        if fallback not in normalized:
            normalized.append(fallback)
    return normalized


def _search_open_images_sequential(
    query: str,
    remaining: int,
    min_width: int,
    min_height: int,
) -> list[dict]:
    """Search Open Images with support for nested train/<label>/image files."""
    return _search_open_images(query, remaining, min_width, min_height)


def _search_imagenet_sequential(
    query: str,
    remaining: int,
    min_width: int,
    min_height: int,
) -> list[dict]:
    """Search ImageNet after higher-priority local datasets are exhausted."""
    return _search_imagenet(query, remaining, min_width, min_height)


def _search_coco_sequential(
    query: str,
    remaining: int,
    min_width: int,
    min_height: int,
) -> list[dict]:
    """Search MSCOCO first for short phrase / caption-style matches."""
    return _search_coco(query, remaining, min_width, min_height)


def _search_recap_imagenet_sequential(
    query: str,
    remaining: int,
    min_width: int,
    min_height: int,
) -> list[dict]:
    """Search the recap dataset's ImageNet slice."""
    return _search_recap_imagenet(query, remaining, min_width, min_height)


def _search_recap_cc12m_sequential(
    query: str,
    remaining: int,
    min_width: int,
    min_height: int,
) -> list[dict]:
    """Search the recap dataset's CC12M slice."""
    return _search_recap_cc12m(query, remaining, min_width, min_height)


# ── Public API ─────────────────────────────────────────────────────────────

def search_local(
    query: str,
    num_results: int = 10,
    min_width: int = 256,
    min_height: int = 256,
) -> list[dict]:
    """Search all enabled local datasets for images matching *query*.

    Returns list of dicts, each with: path, caption, width, height, provider, score.
    Sorted by match score (highest first), deduplicated by path.
    """
    cfg = _get_local_datasets_config()
    if not cfg.get("enabled", False):
        return []

    order = _get_local_search_order()
    normalized_query = " ".join(str(query).strip().lower().split())
    cache_policy = "|".join([
        ",".join(order),
        os.environ.get("MIS_LOCAL_SEARCH_MODE", "fast").strip().lower(),
        f"mix={int(_get_local_source_mix_enabled())}",
        os.environ.get("MIS_LOCAL_SEARCH_EARLY_STOP_AT", "4").strip(),
        os.environ.get("MIS_LOCAL_SEARCH_TARGET_CANDIDATES", "").strip(),
        os.environ.get("MIS_RECAP_QUERY_LIMIT", "").strip(),
    ])
    cache_key = (normalized_query, int(num_results), int(min_width), int(min_height), cache_policy)
    cached = _local_query_cache.get(cache_key)
    if cached is not None:
        return [dict(item) for item in cached[:num_results]]

    searchers = {
        "mscoco": _search_coco_sequential,
        "open_images": _search_open_images_sequential,
        "imagenet": _search_imagenet_sequential,
        "recap_imagenet": _search_recap_imagenet_sequential,
        "recap_cc12m": _search_recap_cc12m_sequential,
    }

    fast_mode = os.environ.get("MIS_LOCAL_SEARCH_MODE", "fast").strip().lower()
    mix_sources = _get_local_source_mix_enabled()
    if mix_sources:
        results_by_source: dict[str, list[dict]] = {}
        for dataset_name in order:
            target = max(num_results, _get_local_search_target(dataset_name, num_results))
            dataset_start = time.time()
            logger.info(
                "Local search: query=%r dataset=%s target=%d mode=mixed current=0/%d",
                query,
                dataset_name,
                target,
                num_results,
            )
            dataset_results = searchers[dataset_name](query, target, min_width, min_height)
            logger.info(
                "Local search done: query=%r dataset=%s hits=%d elapsed=%s",
                query,
                dataset_name,
                len(dataset_results),
                _format_elapsed(time.time() - dataset_start),
            )
            results_by_source[dataset_name] = dataset_results
        mixed_results = _merge_source_diverse_results(results_by_source, order, num_results)
        _local_query_cache[cache_key] = [dict(item) for item in mixed_results]
        return mixed_results

    ordered_results: list[dict] = []
    seen: set[str] = set()
    for dataset_name in order:
        remaining = num_results - len(ordered_results)
        if remaining <= 0:
            break
        target = _get_local_search_target(dataset_name, remaining)
        dataset_start = time.time()
        logger.info(
            "Local search: query=%r dataset=%s target=%d current=%d/%d",
            query,
            dataset_name,
            target,
            len(ordered_results),
            num_results,
        )
        dataset_results = searchers[dataset_name](query, target, min_width, min_height)
        logger.info(
            "Local search done: query=%r dataset=%s hits=%d elapsed=%s",
            query,
            dataset_name,
            len(dataset_results),
            _format_elapsed(time.time() - dataset_start),
        )
        for result in dataset_results:
            path = result["path"]
            if path in seen:
                continue
            seen.add(path)
            ordered_results.append(result)
            if len(ordered_results) >= num_results:
                break
        if fast_mode == "fast" and ordered_results:
            early_stop_target = min(num_results, max(1, int(os.environ.get("MIS_LOCAL_SEARCH_EARLY_STOP_AT", "4"))))
            if len(ordered_results) >= early_stop_target:
                break
    _local_query_cache[cache_key] = [dict(item) for item in ordered_results]
    return ordered_results


def copy_local_image(src_path: str | Path, dst_path: str | Path) -> bool:
    """Copy a local dataset image to the output location. Returns True on success."""
    src, dst = Path(src_path), Path(dst_path)
    if not src.exists():
        return False
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    except Exception:
        dst.unlink(missing_ok=True)
        return False
