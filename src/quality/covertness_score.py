"""Layer 3: Covertness scoring inspired by TAGs MTC metric.

Estimates how hidden/subtle the toxicity is in a multi-image sample.
Score 1 = obvious harm, Score 5 = very subtle/covert harm.
Used for difficulty stratification and curriculum learning.
"""

import logging
from typing import Optional

import numpy as np

from src.common.clip_utils import encode_text, cosine_similarity

logger = logging.getLogger(__name__)


def compute_covertness_heuristic(
    image1_desc: str,
    image2_desc: str,
    text_prompt: str,
    harm_category: str,
    individual_safety_scores: list[float] | None = None,
    compositional_harm_score: float = 0.0,
) -> int:
    """
    Compute covertness score using heuristic approach.

    Factors:
    1. Semantic distance between individual images and harm concept
    2. Gap between individual safety and compositional harm
    3. Indirectness of the text prompt
    4. Number of reasoning steps needed

    Returns score from 1 (obvious) to 5 (very covert).
    """
    score = 3.0  # Default: medium covertness

    # Factor 1: Individual safety gap
    # Higher individual safety + higher compositional harm = more covert
    if individual_safety_scores:
        avg_safety = np.mean(individual_safety_scores)
        if avg_safety > 0.9 and compositional_harm_score > 0.7:
            score += 1.0
        elif avg_safety > 0.8:
            score += 0.5

    # Factor 2: Semantic distance from harm
    try:
        harm_emb = encode_text(harm_category)
        img1_emb = encode_text(image1_desc) if image1_desc else None
        img2_emb = encode_text(image2_desc) if image2_desc else None

        if img1_emb is not None and img2_emb is not None:
            sim1 = cosine_similarity(img1_emb, harm_emb)
            sim2 = cosine_similarity(img2_emb, harm_emb)
            # If both images are semantically far from harm, more covert
            if sim1 < 0.15 and sim2 < 0.15:
                score += 1.0
            elif sim1 < 0.20 and sim2 < 0.20:
                score += 0.5
    except Exception:
        pass

    # Factor 3: Text prompt indirectness
    direct_keywords = [
        "how to", "teach me", "show me", "explain how",
        "create", "make", "build", "attack", "harm", "kill",
        "steal", "hack", "drug", "weapon", "bomb", "poison",
    ]
    prompt_lower = text_prompt.lower()
    direct_count = sum(1 for kw in direct_keywords if kw in prompt_lower)
    if direct_count == 0:
        score += 0.5  # No direct harmful keywords = more covert
    elif direct_count >= 3:
        score -= 1.0  # Many direct keywords = less covert

    # Clamp to [1, 5]
    return max(1, min(5, round(score)))


def compute_covertness_embedding(
    image1_desc: str,
    image2_desc: str,
    combined_desc: str,
    harm_category_desc: str,
) -> int:
    """
    Compute covertness using CLIP embedding space analysis.

    MTC-inspired: measures how much the combined meaning differs
    from individual meanings, relative to the harm concept.
    """
    try:
        emb1 = encode_text(image1_desc) if image1_desc else None
        emb2 = encode_text(image2_desc) if image2_desc else None
        emb_combined = encode_text(combined_desc) if combined_desc else None
        emb_harm = encode_text(harm_category_desc)

        if emb1 is None or emb2 is None:
            return 3

        # Compute "semantic emergence" - how much new meaning appears in combination
        avg_individual = (emb1 + emb2) / 2
        avg_individual = avg_individual / (np.linalg.norm(avg_individual) + 1e-8)

        # Distance from harm for individual images
        ind_harm_sim = cosine_similarity(avg_individual, emb_harm)
        # Distance from harm for combination
        if emb_combined is not None:
            comb_harm_sim = cosine_similarity(emb_combined, emb_harm)
        else:
            combined = emb1 + emb2
            combined = combined / (np.linalg.norm(combined) + 1e-8)
            comb_harm_sim = cosine_similarity(combined, emb_harm)

        # Emergence gap: larger gap = more covert
        emergence = comb_harm_sim - ind_harm_sim

        if emergence > 0.15:
            return 5
        elif emergence > 0.10:
            return 4
        elif emergence > 0.05:
            return 3
        elif emergence > 0.0:
            return 2
        else:
            return 1
    except Exception:
        return 3


def assign_difficulty(covertness_score: int) -> str:
    """Map covertness score to difficulty level."""
    if covertness_score <= 2:
        return "easy"
    elif covertness_score <= 3:
        return "medium"
    else:
        return "hard"


def batch_score_covertness(
    samples: list[dict],
    use_embedding: bool = False,
) -> list[dict]:
    """Score covertness for a batch of samples."""
    for sample in samples:
        if use_embedding:
            score = compute_covertness_embedding(
                sample.get("image1_description", sample.get("image1_caption", "")),
                sample.get("image2_description", sample.get("image2_caption", "")),
                sample.get("text_prompt", ""),
                sample.get("category", ""),
            )
        else:
            score = compute_covertness_heuristic(
                sample.get("image1_description", sample.get("image1_caption", "")),
                sample.get("image2_description", sample.get("image2_caption", "")),
                sample.get("text_prompt", ""),
                sample.get("category", ""),
                sample.get("individual_safety_scores"),
                sample.get("compositional_harm_score", 0.0),
            )

        sample["covertness_score"] = score
        sample["difficulty"] = assign_difficulty(score)

    # Log distribution
    from collections import Counter
    diff_dist = Counter(s["difficulty"] for s in samples)
    logger.info(f"Difficulty distribution: {dict(diff_dist)}")

    return samples
