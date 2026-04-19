"""CLIP-based reranking helpers for local and external image retrieval."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from src.common.clip_utils import (
    cosine_similarity,
    download_image_url,
    encode_image,
    encode_text,
    passes_download_filter,
    retrieve_from_laion,
)
from src.common.retrieval_queries import build_retrieval_query_variants

logger = logging.getLogger(__name__)

_TEXT_EMBED_CACHE: dict[str, object] = {}


def get_retrieval_clip_candidate_limit() -> int:
    """Return how many retrieval candidates to score with CLIP."""
    return max(1, int(os.environ.get("MIS_RETRIEVAL_CLIP_TOP_K", "5")))


def get_retrieval_clip_keep_count(default: int = 5) -> int:
    """Return how many top CLIP-ranked candidates to keep per query."""
    return max(1, int(os.environ.get("MIS_RETRIEVAL_CLIP_KEEP", str(default))))


def get_retrieval_clip_min_score(source: str, default: float | None = None) -> float:
    """Return the minimum CLIP score required to accept a retrieval candidate.

    Source-specific env vars take precedence:
      - MIS_RETRIEVAL_CLIP_MIN_SCORE_LOCAL
      - MIS_RETRIEVAL_CLIP_MIN_SCORE_WEB
    Fallbacks:
      - MIS_RETRIEVAL_CLIP_MIN_SCORE
      - provided default
      - built-in defaults (local=0.18, web=0.20)
    """
    source = str(source).strip().lower()
    source_default = 0.18 if source == "local" else 0.20
    if default is None:
        default = source_default

    specific_key = f"MIS_RETRIEVAL_CLIP_MIN_SCORE_{source.upper()}" if source else ""
    if specific_key and specific_key in os.environ:
        return float(os.environ[specific_key])
    if "MIS_RETRIEVAL_CLIP_MIN_SCORE" in os.environ:
        return float(os.environ["MIS_RETRIEVAL_CLIP_MIN_SCORE"])
    return float(default)


def should_accept_retrieval_candidate(score: float, source: str) -> bool:
    """Return whether a CLIP-ranked candidate is good enough to keep."""
    return float(score) >= get_retrieval_clip_min_score(source)


def score_query_image_alignment(query_text: str, image_path: str | Path) -> float:
    """Return CLIP cosine similarity between a text query and an image."""
    query_text = str(query_text).strip()
    if not query_text:
        return float("-inf")
    if query_text not in _TEXT_EMBED_CACHE:
        _TEXT_EMBED_CACHE[query_text] = encode_text(query_text)
    text_emb = _TEXT_EMBED_CACHE[query_text]
    image_emb = encode_image(image_path)
    return cosine_similarity(image_emb, text_emb)


def rank_local_results_by_clip(
    query_text: str,
    results: list[dict],
    *,
    max_candidates: int | None = None,
    min_width: int = 256,
    min_height: int = 256,
) -> list[dict]:
    """Rank local dataset results by CLIP text-image alignment."""
    limit = max_candidates or get_retrieval_clip_candidate_limit()
    scored: list[dict] = []
    seen: set[str] = set()

    for result in results:
        path = str(result.get("path", ""))
        if not path or path in seen:
            continue
        seen.add(path)
        image_path = Path(path)
        if not image_path.exists():
            continue
        if not passes_download_filter(image_path, min_width=min_width, min_height=min_height):
            continue
        try:
            clip_score = score_query_image_alignment(query_text, image_path)
        except Exception as exc:
            logger.debug("Failed to CLIP-score local image %s for %r: %s", image_path, query_text, exc)
            continue
        enriched = dict(result)
        enriched["clip_score"] = clip_score
        scored.append(enriched)
        if len(scored) >= limit:
            break

    scored.sort(key=lambda item: float(item.get("clip_score", float("-inf"))), reverse=True)
    return scored


def download_web_results_ranked(
    query_text: str,
    results: list[dict],
    temp_dir: str | Path,
    *,
    max_candidates: int | None = None,
    min_width: int = 256,
    min_height: int = 256,
    caption_validator=None,
) -> list[dict]:
    """Download and CLIP-rerank external retrieval results."""
    limit = max_candidates or get_retrieval_clip_candidate_limit()
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    scored: list[dict] = []
    attempted = 0
    for idx, result in enumerate(results):
        if attempted >= limit:
            break
        url = str(result.get("url", "")).strip()
        if not url:
            continue
        caption = str(result.get("caption", "")).strip()
        if caption_validator is not None and caption and not caption_validator(caption):
            continue

        attempted += 1
        tmp_path = temp_dir / f"candidate_{idx:03d}.png"
        tmp_path.unlink(missing_ok=True)
        try:
            download_image_url(url, tmp_path)
            if not passes_download_filter(tmp_path, min_width=min_width, min_height=min_height):
                tmp_path.unlink(missing_ok=True)
                continue
            clip_score = score_query_image_alignment(query_text, tmp_path)
        except Exception as exc:
            logger.debug("Failed to CLIP-score downloaded image for %r from %s: %s", query_text, url, exc)
            tmp_path.unlink(missing_ok=True)
            continue

        enriched = dict(result)
        enriched["clip_score"] = clip_score
        enriched["tmp_path"] = str(tmp_path)
        scored.append(enriched)

    scored.sort(key=lambda item: float(item.get("clip_score", float("-inf"))), reverse=True)
    return scored


def cleanup_ranked_tmp_paths(results: list[dict], *, keep: int = 0) -> None:
    """Delete temporary downloaded files created during web reranking."""
    for item in results[keep:]:
        tmp_path = str(item.get("tmp_path", "")).strip()
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def retrieve_web_results_with_variants(
    query_text: str,
    *,
    num_results: int = 10,
    aesthetic_score_min: float = 4.5,
    min_width: int = 256,
    min_height: int = 256,
) -> list[dict]:
    """Query external backends with adaptive query variants and dedupe URLs.

    Retrieval queries may be slightly different from the final CLIP scoring
    query. The original *query_text* should still be used when reranking the
    downloaded images.
    """
    merged: list[dict] = []
    seen_urls: set[str] = set()

    for variant in build_retrieval_query_variants(query_text):
        try:
            results = retrieve_from_laion(
                variant,
                num_results=num_results,
                aesthetic_score_min=aesthetic_score_min,
                min_width=min_width,
                min_height=min_height,
            )
        except Exception as exc:
            logger.debug("Variant retrieval failed for %r via %r: %s", query_text, variant, exc)
            continue

        for result in results:
            url = str(result.get("url", "")).strip()
            key = url or f"{result.get('provider', '')}:{result.get('caption', '')}"
            if not key or key in seen_urls:
                continue
            seen_urls.add(key)
            enriched = dict(result)
            enriched["retrieval_query"] = variant
            merged.append(enriched)
            if len(merged) >= max(num_results, get_retrieval_clip_candidate_limit()):
                return merged
    return merged
