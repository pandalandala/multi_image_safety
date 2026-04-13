"""Filter concept pairs using CLIP embeddings and LLM validation.

Ensures concept pairs satisfy the compositional harm criteria:
  - Each concept individually has low harm similarity (< theta_safe)
  - Combined concept embedding approaches harm vector (> theta_harm)
"""

from __future__ import annotations

import logging

import numpy as np

from src.common.clip_utils import encode_text, cosine_similarity
from src.path5_embedding_pair.harm_vector import build_harm_vectors

logger = logging.getLogger(__name__)


def filter_pairs_clip(
    pairs: list[dict],
    theta_safe: float = 0.40,
    theta_harm: float = 0.35,
    min_pass_rate: float = 0.10,
    fallback_top_k: int = 3000,
) -> list[dict]:
    """
    Filter concept pairs using CLIP text embeddings.

    For each pair:
      - encode_text(concept1) -> emb1
      - encode_text(concept2) -> emb2
      - harm_vector for the pair's category
      - cos(emb1, harm) < theta_safe
      - cos(emb2, harm) < theta_safe
      - cos(normalize(emb1 + emb2), harm) > theta_harm

    If hard filter pass rate < min_pass_rate, falls back to top-K ranking
    by covertness score.
    """
    harm_vectors = build_harm_vectors()

    all_scored = []
    for pair in pairs:
        cat = pair["category"]
        if cat not in harm_vectors:
            continue

        harm_vec = harm_vectors[cat]

        emb1 = encode_text(f"a photo of {pair['concept1']}")
        emb2 = encode_text(f"a photo of {pair['concept2']}")

        sim1 = float(cosine_similarity(emb1, harm_vec))
        sim2 = float(cosine_similarity(emb2, harm_vec))

        combined = emb1 + emb2
        combined = combined / (np.linalg.norm(combined) + 1e-8)
        combined_sim = float(cosine_similarity(combined, harm_vec))

        pair["clip_individual_sims"] = [sim1, sim2]
        pair["clip_combined_sim"] = combined_sim
        pair["_covertness"] = combined_sim - max(sim1, sim2)
        all_scored.append(pair)

    # Diagnostic logging
    if all_scored:
        indiv = [s for p in all_scored for s in p["clip_individual_sims"]]
        combi = [p["clip_combined_sim"] for p in all_scored]
        logger.info(
            "CLIP score distribution (individual): n=%d, min=%.3f, max=%.3f, "
            "mean=%.3f, p25=%.3f, p50=%.3f, p75=%.3f",
            len(indiv), np.min(indiv), np.max(indiv), np.mean(indiv),
            np.percentile(indiv, 25), np.percentile(indiv, 50), np.percentile(indiv, 75),
        )
        logger.info(
            "CLIP score distribution (combined): n=%d, min=%.3f, max=%.3f, "
            "mean=%.3f, p25=%.3f, p50=%.3f, p75=%.3f",
            len(combi), np.min(combi), np.max(combi), np.mean(combi),
            np.percentile(combi, 25), np.percentile(combi, 50), np.percentile(combi, 75),
        )

    # Hard filter
    passed = [
        p for p in all_scored
        if p["clip_individual_sims"][0] < theta_safe
        and p["clip_individual_sims"][1] < theta_safe
        and p["clip_combined_sim"] > theta_harm
    ]

    pass_rate = len(passed) / max(len(all_scored), 1)
    logger.info(
        "CLIP hard filter: %d / %d pairs passed (%.1f%%, theta_safe=%.2f, theta_harm=%.2f)",
        len(passed), len(all_scored), pass_rate * 100, theta_safe, theta_harm,
    )

    # Fallback: if hard filter too strict, rank by covertness and take top-K
    if pass_rate < min_pass_rate and len(all_scored) > 0:
        logger.warning(
            "Hard filter pass rate %.1f%% < %.0f%% threshold. "
            "Falling back to top-%d ranking by covertness score.",
            pass_rate * 100, min_pass_rate * 100, fallback_top_k,
        )
        candidates = [p for p in all_scored if p["clip_combined_sim"] > theta_harm]
        if not candidates:
            candidates = all_scored
        candidates.sort(key=lambda p: -p["_covertness"])
        passed = candidates[:fallback_top_k]
        logger.info("Fallback yielded %d pairs", len(passed))

    # Cleanup internal key
    for p in passed:
        p.pop("_covertness", None)

    return passed


def rank_pairs_by_covertness(pairs: list[dict]) -> list[dict]:
    """Rank pairs by how covert the harm is (lower individual sim, higher combined)."""
    for pair in pairs:
        sims = pair.get("clip_individual_sims", [0.0, 0.0])
        combined = pair.get("clip_combined_sim", 0.0)
        avg_individual = sum(sims) / max(len(sims), 1)
        pair["covertness_rank"] = combined - avg_individual
    pairs.sort(key=lambda p: -p.get("covertness_rank", 0))
    return pairs
