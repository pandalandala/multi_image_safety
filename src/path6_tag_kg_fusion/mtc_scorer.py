"""Multimodal Toxicity Covertness (MTC) scoring for TAG+KG samples.

Extends the base covertness scoring with TAG-specific factors:
  - Hop count: longer chains = more covert
  - Cross-modal emergence: harm only appears when combining visual + textual
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def compute_mtc_score(sample: dict) -> int:
    """
    Compute MTC score (1-5) for a TAG+KG sample.

    Factors:
      1. Hop count (from TAG chain length)
      2. CLIP combined similarity (higher = less covert, but we already filtered)
      3. Individual concept safety (lower individual sim = more covert)
    """
    hop_count = sample.get("hop_count", 1)
    combined_sim = sample.get("clip_combined_sim", 0.0)
    endpoint_sims = sample.get("clip_endpoint_sims", [0.1, 0.1])

    # Base score from hop count
    if hop_count >= 4:
        base = 5
    elif hop_count >= 3:
        base = 4
    elif hop_count >= 2:
        base = 3
    else:
        base = 2

    # Adjust based on how covert the endpoints are
    avg_endpoint = sum(endpoint_sims) / max(len(endpoint_sims), 1)
    if avg_endpoint < 0.10:
        base = min(5, base + 1)  # Very safe-looking endpoints
    elif avg_endpoint > 0.20:
        base = max(1, base - 1)  # Endpoints are closer to harm

    return max(1, min(5, base))


def batch_score_mtc(samples: list[dict]) -> list[dict]:
    """Score a batch of samples with MTC."""
    for sample in samples:
        sample["mtc_score"] = compute_mtc_score(sample)
    avg = sum(s["mtc_score"] for s in samples) / max(len(samples), 1)
    logger.info("MTC scoring: %d samples, avg score %.2f", len(samples), avg)
    return samples
