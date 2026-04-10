"""Generate benign images for each concept in a pair.

Reuses the shared image generation pipeline (SD3.5 Large Turbo).
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.common.image_generation import generate_image

logger = logging.getLogger(__name__)


def generate_concept_images(
    pairs: list[dict],
    output_dir: Path,
    start_id: int = 0,
) -> list[dict]:
    """
    For each concept pair, generate a benign image for each concept.
    Returns pairs enriched with image paths.
    """
    results = []
    for i, pair in enumerate(pairs):
        sample_id = start_id + i
        img_dir = output_dir / "images" / str(sample_id)

        img1_path = img_dir / "image1.png"
        img2_path = img_dir / "image2.png"

        # Generate benign prompts that produce safe images
        prompt1 = f"A clear, high-quality photograph of {pair['concept1']}, well-lit, neutral background"
        prompt2 = f"A clear, high-quality photograph of {pair['concept2']}, well-lit, neutral background"

        success1 = img1_path.exists() or generate_image(prompt1, img1_path)
        success2 = img2_path.exists() or generate_image(prompt2, img2_path)

        if success1 and success2:
            pair["image1_path"] = str(img1_path)
            pair["image2_path"] = str(img2_path)
            pair["image1_description"] = prompt1
            pair["image2_description"] = prompt2
            pair["sample_id"] = sample_id
            results.append(pair)
            if (i + 1) % 50 == 0:
                logger.info("Generated images for %d / %d pairs", i + 1, len(pairs))
        else:
            logger.debug(
                "Failed to generate images for pair %d: %s + %s",
                sample_id, pair["concept1"], pair["concept2"],
            )

    logger.info("Image generation: %d / %d pairs succeeded", len(results), len(pairs))
    return results
