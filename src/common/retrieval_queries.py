"""Adaptive query planning helpers for local and web image retrieval."""

from __future__ import annotations

import os
import re

COMMON_RETRIEVAL_STOPWORDS = {
    "a", "an", "the", "of", "for", "to", "and", "or", "with", "without",
    "in", "on", "at", "by", "from", "into", "onto", "near", "beside",
    "behind", "inside", "outside", "under", "over", "against",
}

ABSTRACT_QUERY_SUFFIXES = (
    "ism",
    "ist",
    "ity",
    "tion",
    "sion",
    "ment",
    "ness",
    "ship",
    "ology",
    "graphy",
    "tize",
    "lize",
)

KNOWN_POOR_RETRIEVAL_TERMS = {
    "petn",
    "subcreation",
    "politize",
    "cybervandalism",
    "neuromechanism",
    "bioproduction",
}

GENERIC_QUERY_WORDS = {
    "a",
    "an",
    "the",
    "photo",
    "picture",
    "image",
    "images",
    "object",
    "objects",
    "scene",
    "scenes",
    "simple",
    "plain",
    "clean",
    "clear",
    "closeup",
    "close-up",
    "isolated",
    "background",
    "of",
}

TAIL_SPLIT_WORDS = {
    "with",
    "in",
    "on",
    "at",
    "near",
    "beside",
    "by",
    "under",
    "over",
    "inside",
    "outside",
    "against",
}

CLUTTER_HINTS = {
    "crowd",
    "busy",
    "many",
    "multiple",
    "group",
    "collection",
    "street",
    "market",
    "room",
    "people",
    "background",
    "surrounded",
}

SUBJECT_HINTS = {
    "close-up",
    "closeup",
    "single",
    "plain background",
    "simple background",
    "white background",
    "isolated",
}


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def _normalize_phrase(text: str) -> str:
    text = _normalize_spaces(text).lower()
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    return text


def _strip_generic_query_words(text: str) -> str:
    words = [w for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", _normalize_phrase(text)) if w not in GENERIC_QUERY_WORDS]
    return " ".join(words)


def _extract_subject_head(text: str) -> str:
    words = _normalize_phrase(text).split()
    if not words:
        return ""
    for idx, word in enumerate(words):
        if idx > 0 and word in TAIL_SPLIT_WORDS:
            return " ".join(words[:idx])
    return " ".join(words)


def get_retrieval_max_query_words(default: int = 2) -> int:
    """Return the maximum number of words to use for retrieval keywords."""
    return max(1, int(os.environ.get("MIS_RETRIEVAL_MAX_QUERY_WORDS", str(default))))


def build_compact_retrieval_query(query_text: str, *, max_words: int | None = None) -> str:
    """Build a short common-keyword retrieval query from a longer phrase."""
    text = _normalize_phrase(query_text)
    if not text:
        return ""
    if max_words is None:
        max_words = get_retrieval_max_query_words()

    stripped = _strip_generic_query_words(text)
    subject = _extract_subject_head(stripped or text)
    tokens = [tok for tok in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", subject) if tok not in COMMON_RETRIEVAL_STOPWORDS]
    if not tokens:
        tokens = [tok for tok in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", stripped or text) if tok not in COMMON_RETRIEVAL_STOPWORDS]
    if not tokens:
        return _normalize_spaces(query_text)
    return " ".join(tokens[:max_words])


def is_retrieval_friendly_query(query_text: str, *, max_words: int | None = None) -> bool:
    """Return whether a query looks simple and visually grounded enough for retrieval."""
    compact = build_compact_retrieval_query(query_text, max_words=max_words)
    if not compact:
        return False

    tokens = compact.split()
    if not tokens:
        return False

    limit = get_retrieval_max_query_words() if max_words is None else max(1, int(max_words))
    if len(tokens) > limit:
        return False

    for token in tokens:
        if token in KNOWN_POOR_RETRIEVAL_TERMS:
            return False
        if not re.fullmatch(r"[a-z]+(?:-[a-z]+)?", token):
            return False
        if len(token) < 3 or len(token) > 14:
            return False
        if token.endswith(ABSTRACT_QUERY_SUFFIXES):
            return False
    return True


def get_retrieval_query_mode() -> str:
    """Return retrieval query planning mode."""
    return str(os.environ.get("MIS_RETRIEVAL_QUERY_MODE", "adaptive")).strip().lower()


def get_retrieval_query_variant_limit(default: int = 4) -> int:
    """Return maximum number of query variants to try."""
    return max(1, int(os.environ.get("MIS_RETRIEVAL_QUERY_VARIANTS", str(default))))


def build_retrieval_query_variants(query_text: str, *, max_variants: int | None = None) -> list[str]:
    """Build a small set of retrieval-friendly query variants.

    The original query always stays first. In adaptive mode, we also try:
      - a de-noised phrase without generic photo words
      - a subject-head phrase before location/context tails
      - a photo-oriented variant encouraging simple, subject-centric shots
    """
    query_text = _normalize_spaces(query_text)
    if not query_text:
        return []

    mode = get_retrieval_query_mode()
    if mode == "literal":
        return [build_compact_retrieval_query(query_text)]

    if max_variants is None:
        max_variants = get_retrieval_query_variant_limit()

    compact = build_compact_retrieval_query(query_text)
    stripped = build_compact_retrieval_query(_strip_generic_query_words(query_text) or query_text)
    subject = build_compact_retrieval_query(_extract_subject_head(_strip_generic_query_words(query_text) or query_text))
    candidates = [
        compact,
        stripped,
        subject,
        " ".join(subject.split()[:1]) if subject else "",
        " ".join(compact.split()[:1]) if compact else "",
    ]

    variants: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = _normalize_spaces(candidate)
        if not candidate:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        variants.append(candidate)
        if len(variants) >= max_variants:
            break
    return variants


def simple_subject_bonus(candidate_text: str) -> float:
    """Heuristic bonus for captions/labels suggesting a simple prominent subject."""
    text = _normalize_phrase(candidate_text)
    if not text:
        return 0.0

    bonus = 0.0
    word_count = len(text.split())
    if word_count <= 4:
        bonus += 0.08
    elif word_count <= 8:
        bonus += 0.03
    else:
        bonus -= 0.02

    if any(hint in text for hint in SUBJECT_HINTS):
        bonus += 0.08
    if any(hint in text for hint in CLUTTER_HINTS):
        bonus -= 0.08
    return bonus
