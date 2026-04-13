"""Acquire images for TAG+KG fusion pairs.

Strategy: retrieve real images first via configured backends (Wikimedia,
Openverse, Pexels, Pixabay).  Falls back to T2I generation only when
retrieval fails for a concept.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from src.common.clip_utils import (
    retrieve_from_laion,
    download_image_url,
    passes_download_filter,
)
from src.common.image_generation import generate_image
from src.common.utils import is_english

logger = logging.getLogger(__name__)


def _retrieve_concept_image(
    concept: str,
    output_path: Path,
    num_results: int = 10,
) -> bool:
    """Try to retrieve a real image for *concept* from configured backends.

    Downloads candidates one-by-one until a quality-passing image is saved
    to *output_path*.  Returns True on success, False if all candidates fail.
    """
    query = f"a photo of {concept}"
    results = retrieve_from_laion(
        query,
        num_results=num_results,
        aesthetic_score_min=4.5,
        min_width=512,
        min_height=512,
    )
    for result in results:
        url = result.get("url", "")
        if not url:
            continue
        caption = result.get("caption", "")
        if caption and not is_english(caption):
            continue
        try:
            download_image_url(url, output_path)
            img = Image.open(output_path)
            img.verify()
            if passes_download_filter(output_path, min_width=512, min_height=512):
                return True
            output_path.unlink(missing_ok=True)
        except Exception:
            output_path.unlink(missing_ok=True)
    return False


def generate_fusion_images(
    pairs: list[dict],
    output_dir: Path,
    start_id: int = 0,
) -> list[dict]:
    """Acquire images for fusion pairs.

    Acquisition order per concept:
      1. Retrieve a real photo from external backends.
      2. Fall back to T2I generation if retrieval fails.

    Returns pairs enriched with image paths.
    """
    results = []
    retrieve_ok = 0
    generate_ok = 0

    for i, pair in enumerate(pairs):
        sample_id = start_id + i
        img_dir = output_dir / "images" / str(sample_id)

        img1_path = img_dir / "image1.png"
        img2_path = img_dir / "image2.png"

        # ── Concept 1 ──────────────────────────────────────────────
        if img1_path.exists():
            success1 = True
        elif _retrieve_concept_image(pair["concept1"], img1_path):
            success1 = True
            retrieve_ok += 1
        else:
            prompt1 = f"A clear, high-quality photograph of {pair['concept1']}, well-lit, neutral background"
            success1 = generate_image(prompt1, img1_path)
            if success1:
                generate_ok += 1

        # ── Concept 2 ──────────────────────────────────────────────
        if img2_path.exists():
            success2 = True
        elif _retrieve_concept_image(pair["concept2"], img2_path):
            success2 = True
            retrieve_ok += 1
        else:
            prompt2 = f"A clear, high-quality photograph of {pair['concept2']}, well-lit, neutral background"
            success2 = generate_image(prompt2, img2_path)
            if success2:
                generate_ok += 1

        if success1 and success2:
            desc1 = pair["concept1"]
            desc2 = pair["concept2"]
            pair["image1_path"] = str(img1_path)
            pair["image2_path"] = str(img2_path)
            pair["image1_description"] = f"a photo of {desc1}"
            pair["image2_description"] = f"a photo of {desc2}"
            pair["sample_id"] = sample_id
            results.append(pair)
            if (i + 1) % 50 == 0:
                logger.info(
                    "Acquired images for %d / %d fusion pairs (retrieved=%d, generated=%d)",
                    i + 1, len(pairs), retrieve_ok, generate_ok,
                )
        else:
            logger.debug(
                "Failed to acquire images for pair %d: %s + %s",
                sample_id, pair["concept1"], pair["concept2"],
            )

    logger.info(
        "Fusion image acquisition: %d / %d pairs succeeded (retrieved=%d, generated=%d)",
        len(results), len(pairs), retrieve_ok, generate_ok,
    )
    return results
