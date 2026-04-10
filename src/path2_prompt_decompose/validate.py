"""Step 4: Validate decomposed samples - single image safety + compositional harm."""

import logging
from pathlib import Path

from src.common.clip_utils import check_nsfw
from src.common.utils import load_jsonl, save_jsonl, DATA_DIR

logger = logging.getLogger(__name__)


def validate_single_image_safety(
    image_path: str,
    nsfw_threshold: float = 0.5,
) -> tuple[bool, float]:
    """Check that a single image is safe (low NSFW score)."""
    try:
        score = check_nsfw(image_path)
        return score < nsfw_threshold, score
    except Exception as e:
        logger.warning(f"NSFW check failed for {image_path}: {e}")
        return False, 1.0


def validate_sample(
    sample: dict,
    nsfw_threshold: float = 0.5,
) -> dict | None:
    """
    Validate a single sample:
    1. Each image individually passes safety check
    2. Record safety scores in metadata

    Full compositional harm verification is done in quality/composition_verify.py
    """
    img1 = sample.get("image1_path", "")
    img2 = sample.get("image2_path", "")

    if not img1 or not img2:
        return None

    safe1, score1 = validate_single_image_safety(img1, nsfw_threshold)
    safe2, score2 = validate_single_image_safety(img2, nsfw_threshold)

    # For Pattern A (compositional covert harm), both images must be individually safe
    pattern = sample.get("pattern", "A")
    if pattern == "A" and not (safe1 and safe2):
        logger.debug(
            f"Sample {sample.get('sample_id', '?')} failed: "
            f"img1 NSFW={score1:.3f}, img2 NSFW={score2:.3f}"
        )
        return None

    # For Pattern B (direct harm), at least one image can be harmful - that's expected
    # But we still record scores
    sample["individual_safety_scores"] = [1 - score1, 1 - score2]
    return sample


def run(
    input_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    nsfw_threshold: float = 0.5,
):
    """Validate all samples from image acquisition step."""
    if input_file is None:
        input_file = DATA_DIR / "raw" / "path2" / "samples_with_images.jsonl"
    if output_dir is None:
        output_dir = DATA_DIR / "raw" / "path2"

    samples = load_jsonl(input_file)
    logger.info(f"Validating {len(samples)} samples")

    valid_samples = []
    for sample in samples:
        result = validate_sample(sample, nsfw_threshold)
        if result:
            valid_samples.append(result)

    save_jsonl(valid_samples, Path(output_dir) / "validated_samples.jsonl")
    logger.info(
        f"Validation: {len(valid_samples)}/{len(samples)} passed "
        f"({len(valid_samples)/len(samples)*100:.1f}%)"
    )
    return valid_samples


if __name__ == "__main__":
    from src.common.utils import setup_logging
    setup_logging()
    run()
