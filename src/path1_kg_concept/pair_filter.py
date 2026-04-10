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
    theta_safe: float = 0.25,
    theta_harm: float = 0.35,
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
    """
    harm_vectors = build_harm_vectors()

    passed = []
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

        if sim1 < theta_safe and sim2 < theta_safe and combined_sim > theta_harm:
            pair["clip_individual_sims"] = [sim1, sim2]
            pair["clip_combined_sim"] = combined_sim
            passed.append(pair)

    logger.info("CLIP filter: %d / %d pairs passed", len(passed), len(pairs))
    return passed


def rank_pairs_by_covertness(pairs: list[dict]) -> list[dict]:
    """Rank pairs by how covert the harm is (lower individual sim, higher combined)."""
    for pair in pairs:
        sims = pair.get("clip_individual_sims", [0.0, 0.0])
        combined = pair.get("clip_combined_sim", 0.0)
        # Covertness = high combined harm with low individual harm
        avg_individual = sum(sims) / max(len(sims), 1)
        pair["covertness_rank"] = combined - avg_individual
    pairs.sort(key=lambda p: -p.get("covertness_rank", 0))
    return pairs
