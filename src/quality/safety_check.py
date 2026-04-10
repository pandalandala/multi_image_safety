"""Layer 1: Single image safety check using NSFW classifier and optional Llama Guard."""

import logging
from pathlib import Path

from src.common.clip_utils import check_nsfw
from src.common.utils import load_config

logger = logging.getLogger(__name__)

_llama_guard_pipeline = None


def check_single_image_nsfw(
    image_path: str | Path,
    threshold: float = 0.5,
) -> tuple[bool, float]:
    """Check if a single image is NSFW. Returns (is_safe, nsfw_score)."""
    try:
        score = check_nsfw(str(image_path))
        return score < threshold, score
    except Exception as e:
        logger.warning(f"NSFW check failed for {image_path}: {e}")
        return False, 1.0


def check_single_image_llama_guard(
    image_path: str | Path,
    text_context: str = "",
) -> tuple[bool, str]:
    """
    Check image safety using Llama Guard 4 (first multi-image safety classifier).
    Returns (is_safe, category_if_unsafe).
    """
    global _llama_guard_pipeline
    try:
        if _llama_guard_pipeline is None:
            from transformers import pipeline
            config = load_config()
            model_name = config["quality"]["safety_check"]["llama_guard_model"]
            _llama_guard_pipeline = pipeline(
                "text-generation",
                model=model_name,
                device_map="auto",
                torch_dtype="auto",
            )

        # Build prompt with image reference
        prompt = f"<image>{text_context}\nIs this image safe or unsafe?"
        # Note: Full implementation would use the actual multi-modal Llama Guard
        # This is a placeholder for the text-based check
        result = _llama_guard_pipeline(prompt, max_new_tokens=100)
        output = result[0]["generated_text"]

        is_safe = "safe" in output.lower() and "unsafe" not in output.lower()
        category = ""
        if not is_safe:
            # Extract category from response
            for line in output.split("\n"):
                if line.strip().startswith("S"):
                    category = line.strip()
                    break

        return is_safe, category
    except Exception as e:
        logger.warning(f"Llama Guard check failed: {e}")
        return False, "CHECK_FAILED"


def batch_safety_check(
    samples: list[dict],
    nsfw_threshold: float = 0.5,
    use_llama_guard: bool = False,
) -> list[dict]:
    """
    Run safety check on all samples.
    For Pattern A (compositional covert harm): both images must be individually safe.
    For Pattern B (direct harm): skip single-image safety check.
    """
    passed = []
    failed = 0

    for sample in samples:
        pattern = sample.get("pattern", "A")
        img1 = sample.get("image1_path", "")
        img2 = sample.get("image2_path", "")

        if not img1 or not img2:
            failed += 1
            continue

        if pattern == "B":
            # Direct harm pattern - skip individual safety check
            sample["individual_safety_scores"] = [0.5, 0.5]
            passed.append(sample)
            continue

        # Check both images
        safe1, score1 = check_single_image_nsfw(img1, nsfw_threshold)
        safe2, score2 = check_single_image_nsfw(img2, nsfw_threshold)

        if safe1 and safe2:
            sample["individual_safety_scores"] = [1 - score1, 1 - score2]

            if use_llama_guard:
                lg_safe1, _ = check_single_image_llama_guard(img1)
                lg_safe2, _ = check_single_image_llama_guard(img2)
                if not (lg_safe1 and lg_safe2):
                    failed += 1
                    continue

            passed.append(sample)
        else:
            failed += 1

    logger.info(f"Safety check: {len(passed)} passed, {failed} failed")
    return passed
