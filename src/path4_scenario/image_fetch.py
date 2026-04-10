"""Step 3: Fetch images for intent-injected scenarios.

Uses mixed approach: external retrieval for realistic scenes and the shared
configurable T2I backend for specific objects/compositions.
"""

import logging
from pathlib import Path

from src.common.clip_utils import (
    retrieve_from_laion,
    download_image_url,
    passes_download_filter,
)
from src.common.image_generation import should_force_regenerate_images
from src.common.utils import load_jsonl, save_jsonl, DATA_DIR

logger = logging.getLogger(__name__)


def fetch_image_for_description(
    description: str,
    output_path: Path,
    prefer_retrieval: bool = False,
) -> bool:
    """Fetch a single image, trying generation first (LAION is offline)."""
    if prefer_retrieval:
        if _try_retrieval(description, output_path):
            return True
        return _try_generation(description, output_path)
    else:
        if _try_generation(description, output_path):
            return True
        return _try_retrieval(description, output_path)


def _try_retrieval(description: str, output_path: Path) -> bool:
    """Try to retrieve an image from LAION-5B."""
    from PIL import Image

    results = retrieve_from_laion(description, num_results=10)
    for result in results:
        url = result.get("url", "")
        if not url:
            continue
        try:
            download_image_url(url, output_path)
            img = Image.open(output_path)
            img.verify()
            if not passes_download_filter(output_path, min_width=512, min_height=512):
                if output_path.exists():
                    output_path.unlink()
                continue
            return True
        except Exception:
            if output_path.exists():
                output_path.unlink()
            continue
    return False


def _try_generation(description: str, output_path: Path) -> bool:
    """Try to generate an image using T2I."""
    try:
        from src.common.image_generation import generate_image
        return generate_image(description, output_path)
    except Exception as e:
        logger.debug(f"T2I generation failed: {e}")
        return False


def fetch_images_for_samples(
    samples: list[dict],
    output_dir: Path,
    start_id: int = 0,
    prefer_retrieval: bool = True,
) -> list[dict]:
    """Fetch images for all intent-injected samples."""
    results = []
    force_regenerate = should_force_regenerate_images()

    for i, sample in enumerate(samples):
        sample_id = start_id + i
        img_dir = output_dir / "images" / str(sample_id)
        img1_path = img_dir / "image1.png"
        img2_path = img_dir / "image2.png"

        desc1 = sample.get("image1_description", "")
        desc2 = sample.get("image2_description", "")

        if not desc1 or not desc2:
            continue

        acquisition1 = str(sample.get("image1_acquisition", "")).strip().lower()
        acquisition2 = str(sample.get("image2_acquisition", "")).strip().lower()

        if img1_path.exists() and (not force_regenerate or acquisition1 in {"retrieved", "crawled"}):
            success1 = True
            acquisition1 = acquisition1 or "retrieved"
        else:
            if img1_path.exists() and force_regenerate:
                img1_path.unlink(missing_ok=True)
            from src.path2_prompt_decompose.acquire_images import acquire_single_image
            success1, acquisition1 = acquire_single_image(
                desc1,
                img1_path,
                prefer_generation=not prefer_retrieval,
            )

        if img2_path.exists() and (not force_regenerate or acquisition2 in {"retrieved", "crawled"}):
            success2 = True
            acquisition2 = acquisition2 or "retrieved"
        else:
            if img2_path.exists() and force_regenerate:
                img2_path.unlink(missing_ok=True)
            from src.path2_prompt_decompose.acquire_images import acquire_single_image
            success2, acquisition2 = acquire_single_image(
                desc2,
                img2_path,
                prefer_generation=not prefer_retrieval,
            )

        if success1 and success2:
            sample["image1_path"] = str(img1_path)
            sample["image2_path"] = str(img2_path)
            sample["image1_acquisition"] = acquisition1
            sample["image2_acquisition"] = acquisition2
            sample["sample_id"] = sample_id
            results.append(sample)

        if (i + 1) % 50 == 0:
            logger.info(f"Fetched images for {i+1}/{len(samples)}, success: {len(results)}")
            save_jsonl(results, output_dir / "samples_with_images.jsonl")

    return results


def run(
    input_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    start_id: int = 6000,  # After path2 (0-2999) and path3 (3000-5999)
):
    """Main entry point."""
    if input_file is None:
        input_file = DATA_DIR / "raw" / "path4" / "intent_injected_samples.jsonl"
    if output_dir is None:
        output_dir = DATA_DIR / "raw" / "path4"
    output_dir = Path(output_dir)

    samples = load_jsonl(input_file)
    logger.info(f"Fetching images for {len(samples)} samples")

    results = fetch_images_for_samples(samples, output_dir, start_id=start_id)
    save_jsonl(results, output_dir / "samples_with_images.jsonl")
    logger.info(f"Successfully fetched images for {len(results)}/{len(samples)} samples")
    return results


if __name__ == "__main__":
    from src.common.utils import setup_logging
    setup_logging()
    run()
