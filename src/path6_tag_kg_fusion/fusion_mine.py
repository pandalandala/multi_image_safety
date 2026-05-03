"""Fuse TAG chains with Path 1 KG concept pairs for enhanced coverage.

Prioritizes pairs appearing in longer chains (more covert) and combines
TAG endpoint pairs with KG-mined pairs, deduplicating by concept pair key.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from src.common.utils import load_jsonl

logger = logging.getLogger(__name__)


def fuse_with_path1(
    tag_pairs: list[dict],
    path1_data_dir: Path | None = None,
) -> list[dict]:
    """
    Combine TAG pairs with Path 1 KG pairs, deduplicating by (concept1, concept2, category).
    TAG pairs take priority since they have richer chain metadata.
    """
    seen = set()
    fused = []

    # TAG pairs first (higher priority — have chain metadata)
    for pair in tag_pairs:
        key = (pair["concept1"].lower(), pair["concept2"].lower(), pair["category"])
        rev_key = (pair["concept2"].lower(), pair["concept1"].lower(), pair["category"])
        if key not in seen and rev_key not in seen:
            seen.add(key)
            pair["fusion_source"] = "tag"
            fused.append(pair)

    # Path 1 pairs as supplement
    if path1_data_dir is None:
        from src.common.utils import DATA_DIR
        path1_data_dir = DATA_DIR / "raw" / "path1"

    fusion_target = int(os.environ.get("MIS_PATH6_FUSION_TARGET_PAIRS", "5000"))
    path1_sources = [
        ("kg", path1_data_dir / "filtered_pairs.jsonl"),
        ("kg_supplement", path1_data_dir / "all_mined_pairs.jsonl"),
        ("kg_supplement", path1_data_dir / "llm_pairs.jsonl"),
        ("kg_supplement", path1_data_dir / "numberbatch_pairs.jsonl"),
    ]
    total_added = 0
    for fusion_source, path1_file in path1_sources:
        if not path1_file.exists():
            continue
        path1_pairs = load_jsonl(path1_file)
        added = 0
        for pair in path1_pairs:
            key = (pair.get("concept1", "").lower(), pair.get("concept2", "").lower(), pair.get("category", ""))
            rev_key = (pair.get("concept2", "").lower(), pair.get("concept1", "").lower(), pair.get("category", ""))
            if key not in seen and rev_key not in seen:
                seen.add(key)
                pair["fusion_source"] = fusion_source
                pair["hop_count"] = pair.get("hop_count", 1)
                pair["covertness_score"] = pair.get("covertness_score", 2)
                fused.append(pair)
                added += 1
                total_added += 1
                if len(fused) >= fusion_target:
                    break
        logger.info("Added %d Path 1 pairs from %s", added, path1_file.name)
        if len(fused) >= fusion_target:
            logger.info("Fusion target %d reached; stopping Path 1 supplementation", fusion_target)
            break
    if total_added == 0:
        logger.info("No Path 1 data found under %s — using TAG pairs only", path1_data_dir)

    # Sort by covertness (high to low), then by hop count (long chains first)
    fused.sort(key=lambda p: (-p.get("covertness_score", 1), -p.get("hop_count", 1)))
    logger.info("Total fused pairs: %d", len(fused))
    return fused
