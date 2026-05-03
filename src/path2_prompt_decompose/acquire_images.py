"""Step 3: Acquire images for decomposed prompts via T2I generation or retrieval.

Dual-track approach:
  Track A: retrieval from configured external image backends
  Track B: T2I generation using the shared configurable backend
"""

import logging
from pathlib import Path
from typing import Optional

from src.common.utils import (
    get_min_image_size,
    load_jsonl,
    save_jsonl,
    DATA_DIR,
)
from src.common.local_datasets import search_local, copy_local_image
from src.common.image_generation import (
    generate_image,
    should_force_regenerate_images,
)
from src.common.retrieval_rerank import (
    cleanup_ranked_tmp_paths,
    download_web_results_ranked,
    get_retrieval_clip_candidate_limit,
    rank_local_results_by_clip,
    retrieve_web_results_with_variants,
    should_accept_retrieval_candidate,
)

logger = logging.getLogger(__name__)


def _has_valid_description(description: str | None) -> bool:
    return bool(str(description).strip()) if description is not None else False


def _existing_acquisition_mode(sample: dict, image_index: int) -> str:
    """Return the stored acquisition mode for an image if present."""
    return str(sample.get(f"image{image_index}_acquisition", "")).strip().lower()


def _should_preserve_existing_image(
    sample: dict,
    image_path: Path,
    image_index: int,
    force_regenerate: bool,
) -> bool:
    """Keep retrieved/crawled images on forced reruns; only overwrite generated ones."""
    if not image_path.exists():
        return False
    if not force_regenerate:
        return True

    acquisition = _existing_acquisition_mode(sample, image_index)
    return acquisition in {"retrieved", "crawled", "local"}


def _delete_existing_generated_image(
    sample: dict,
    image_path: Path,
    image_index: int,
    force_regenerate: bool,
) -> None:
    """Delete existing generated images before reacquiring them."""
    if not image_path.exists() or not force_regenerate:
        return
    if _should_preserve_existing_image(sample, image_path, image_index, force_regenerate):
        return
    image_path.unlink(missing_ok=True)


def retrieve_image(
    description: str,
    output_path: str | Path,
    num_results: int = 10,
) -> tuple[bool, str]:
    """Search local datasets first, then web retrieval, and report the source."""
    output_path = Path(output_path)
    min_w, min_h = get_min_image_size()

    local_results = search_local(
        description,
        num_results=max(num_results, get_retrieval_clip_candidate_limit()),
        min_width=min_w,
        min_height=min_h,
    )
    ranked_local = rank_local_results_by_clip(
        description,
        local_results,
        min_width=min_w,
        min_height=min_h,
    )
    if ranked_local:
        best_local = ranked_local[0]
        best_local_score = float(best_local.get("clip_score", float("-inf")))
        if should_accept_retrieval_candidate(best_local_score, "local"):
            if copy_local_image(best_local["path"], output_path):
                return True, "local"
        else:
            logger.debug(
                "Best local image for %r scored %.3f, below threshold; continuing to web retrieval",
                description,
                best_local_score,
            )

    results = retrieve_web_results_with_variants(
        description,
        num_results=max(num_results, get_retrieval_clip_candidate_limit()),
        min_width=min_w,
        min_height=min_h,
    )
    if not results:
        return False, ""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranked_web = download_web_results_ranked(
        description,
        results,
        output_path.parent / ".retrieval_tmp",
        min_width=min_w,
        min_height=min_h,
    )
    if ranked_web:
        best_web = ranked_web[0]
        best_web_score = float(best_web.get("clip_score", float("-inf")))
        if should_accept_retrieval_candidate(best_web_score, "web"):
            best_tmp = Path(best_web["tmp_path"])
            output_path.unlink(missing_ok=True)
            best_tmp.replace(output_path)
            cleanup_ranked_tmp_paths(ranked_web, keep=1)
            return True, "retrieved"
        logger.debug(
            "Best web image for %r scored %.3f, below threshold; falling back to generation",
            description,
            best_web_score,
        )
        cleanup_ranked_tmp_paths(ranked_web, keep=0)
    return False, ""


def acquire_single_image(
    description: str,
    output_path: str | Path,
    *,
    prefer_generation: bool = True,
) -> tuple[bool, str]:
    """Acquire one image via retrieval/generation, with configurable preference order."""
    if not _has_valid_description(description):
        logger.warning("Skipping image acquisition for empty description at %s", output_path)
        return False, ""

    if prefer_generation:
        if generate_image(description, output_path):
            return True, "generated"
        retrieved, source = retrieve_image(description, output_path)
        if retrieved:
            return True, source
    else:
        retrieved, source = retrieve_image(description, output_path)
        if retrieved:
            return True, source
        if generate_image(description, output_path):
            return True, "generated"
    return False, ""


def acquire_images_for_sample(
    sample: dict,
    sample_id: int,
    output_dir: Path,
    prefer_generation: bool = True,
    force_regenerate: bool = False,
) -> Optional[dict]:
    """Acquire both images for a decomposed sample."""
    img_dir = output_dir / "images" / str(sample_id)
    img1_path = img_dir / "image1.png"
    img2_path = img_dir / "image2.png"

    desc1 = sample["image1_description"]
    desc2 = sample["image2_description"]

    _delete_existing_generated_image(sample, img1_path, 1, force_regenerate)
    _delete_existing_generated_image(sample, img2_path, 2, force_regenerate)

    if _should_preserve_existing_image(sample, img1_path, 1, force_regenerate):
        success1, acquisition1 = True, _existing_acquisition_mode(sample, 1) or "retrieved"
    else:
        success1, acquisition1 = acquire_single_image(
            desc1,
            img1_path,
            prefer_generation=prefer_generation,
        )

    if _should_preserve_existing_image(sample, img2_path, 2, force_regenerate):
        success2, acquisition2 = True, _existing_acquisition_mode(sample, 2) or "retrieved"
    else:
        success2, acquisition2 = acquire_single_image(
            desc2,
            img2_path,
            prefer_generation=prefer_generation,
        )

    if success1 and success2:
        sample["image1_path"] = str(img1_path)
        sample["image2_path"] = str(img2_path)
        sample["image1_acquisition"] = acquisition1
        sample["image2_acquisition"] = acquisition2
        sample["sample_id"] = sample_id
        return sample

    logger.info(
        "Failed to acquire images for sample %s (image1=%s, image2=%s)",
        sample_id,
        success1,
        success2,
    )
    return None


def run(
    input_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    prefer_generation: bool = True,
    start_id: int = 0,
):
    """Main entry point: acquire images for all decomposed prompts."""
    if input_file is None:
        input_file = DATA_DIR / "raw" / "path2" / "decomposed_prompts.jsonl"
    if output_dir is None:
        output_dir = DATA_DIR / "raw" / "path2"
    output_dir = Path(output_dir)

    samples = load_jsonl(input_file)
    logger.info(f"Acquiring images for {len(samples)} decomposed prompts")

    results = []
    force_regenerate = should_force_regenerate_images()
    for i, sample in enumerate(samples):
        sample_id = start_id + i
        result = acquire_images_for_sample(
            sample,
            sample_id,
            output_dir,
            prefer_generation,
            force_regenerate=force_regenerate,
        )
        if result:
            results.append(result)

        if (i + 1) % 50 == 0:
            logger.info(f"Processed {i + 1}/{len(samples)}, success: {len(results)}")
            # Save intermediate results
            save_jsonl(results, output_dir / "samples_with_images.jsonl")

    save_jsonl(results, output_dir / "samples_with_images.jsonl")
    logger.info(f"Acquired images for {len(results)}/{len(samples)} samples")
    return results


if __name__ == "__main__":
    from src.common.utils import setup_logging
    setup_logging()
    run()
