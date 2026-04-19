"""Acquire images for each concept in a pair.

Strategy (per concept):
  1. Search local datasets (MSCOCO, Open Images, ImageNet)
  2. Retrieve from web backends (Wikimedia, Openverse, Pexels, Pixabay)
  3. Fall back to T2I generation only when all retrieval fails
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.common.image_generation import generate_image, is_image_generation_allowed
from src.common.local_datasets import search_local, copy_local_image
from src.common.retrieval_queries import build_compact_retrieval_query
from src.common.retrieval_rerank import (
    cleanup_ranked_tmp_paths,
    download_web_results_ranked,
    get_retrieval_clip_candidate_limit,
    rank_local_results_by_clip,
    retrieve_web_results_with_variants,
    should_accept_retrieval_candidate,
)
from src.common.shared_reuse import (
    load_retrieval_cache,
    materialize_shared_image,
    reuse_cached_image,
    save_retrieval_cache,
)
from src.common.utils import get_min_image_size, is_english

logger = logging.getLogger(__name__)


def _get_retrieval_worker_count() -> int:
    """Return per-process retrieval concurrency for local/web image fetching."""
    env_value = os.environ.get("MIS_RETRIEVAL_WORKERS_PER_GPU", "").strip()
    if env_value:
        try:
            return max(1, int(env_value))
        except ValueError:
            logger.warning("Invalid MIS_RETRIEVAL_WORKERS_PER_GPU=%r; falling back to default", env_value)
    cpu_count = os.cpu_count() or 8
    return max(2, min(8, cpu_count // 4 if cpu_count >= 8 else 2))


def _retrieve_concept_image(
    concept: str,
    output_path: Path,
    num_results: int = 10,
) -> str:
    """Try to retrieve a real image for *concept*.

    Returns the source tag ("local", "web") on success, or "" on failure.
    The image is saved to *output_path*.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    query = build_compact_retrieval_query(concept)
    if not query:
        query = str(concept).strip()
    min_w, min_h = get_min_image_size()

    cached_local = load_retrieval_cache(
        query,
        purpose="single_image",
        min_width=min_w,
        min_height=min_h,
        allowed_source_types=("local",),
    )
    for entry in cached_local:
        if reuse_cached_image(entry, output_path):
            logger.debug("Reused cached local retrieval for %r", concept)
            return "local"

    # ── 1. Local datasets ──────────────────────────────────────────
    local_results = search_local(concept, num_results=max(num_results, get_retrieval_clip_candidate_limit()), min_width=min_w, min_height=min_h)
    ranked_local = rank_local_results_by_clip(query, local_results, min_width=min_w, min_height=min_h)
    if ranked_local:
        best_local = ranked_local[0]
        best_local_score = float(best_local.get("clip_score", float("-inf")))
        if should_accept_retrieval_candidate(best_local_score, "local"):
            if copy_local_image(best_local["path"], output_path):
                logger.debug("Selected local image for %r with CLIP score %.3f", concept, best_local_score)
                save_retrieval_cache(
                    query,
                    [{
                        "path": best_local["path"],
                        "source_type": "local",
                        "provider": best_local.get("provider", "local"),
                        "caption": best_local.get("caption", ""),
                        "class_label": best_local.get("class_label", best_local.get("class", "")),
                        "clip_score": best_local_score,
                    }],
                    purpose="single_image",
                    min_width=min_w,
                    min_height=min_h,
                    path_name="path1",
                )
                return "local"
        else:
            logger.debug("Best local image for %r scored %.3f, below threshold; continuing to web retrieval", concept, best_local_score)

    cached_web = load_retrieval_cache(
        query,
        purpose="single_image",
        min_width=min_w,
        min_height=min_h,
        allowed_source_types=("web",),
    )
    for entry in cached_web:
        if reuse_cached_image(entry, output_path):
            logger.debug("Reused cached web retrieval for %r", concept)
            return "web"

    # ── 2. Web retrieval backends ──────────────────────────────────
    web_results = retrieve_web_results_with_variants(
        query,
        num_results=max(num_results, get_retrieval_clip_candidate_limit()),
        aesthetic_score_min=4.5,
        min_width=min_w,
        min_height=min_h,
    )
    ranked_web = download_web_results_ranked(
        query,
        web_results,
        output_path.parent / f".retrieval_tmp_{output_path.stem}",
        min_width=min_w,
        min_height=min_h,
        caption_validator=is_english,
    )
    if ranked_web:
        best_web = ranked_web[0]
        best_web_score = float(best_web.get("clip_score", float("-inf")))
        if should_accept_retrieval_candidate(best_web_score, "web"):
            best_tmp = Path(best_web["tmp_path"])
            output_path.unlink(missing_ok=True)
            best_tmp.replace(output_path)
            cleanup_ranked_tmp_paths(ranked_web, keep=1)
            logger.debug("Selected web image for %r with CLIP score %.3f", concept, best_web_score)
            shared_path = materialize_shared_image(output_path, source_type="web")
            save_retrieval_cache(
                query,
                [{
                    "path": shared_path,
                    "source_type": "web",
                    "provider": best_web.get("provider", "web"),
                    "caption": best_web.get("caption", ""),
                    "class_label": best_web.get("class_label", best_web.get("class", "")),
                    "clip_score": best_web_score,
                    "url": best_web.get("url", ""),
                }],
                purpose="single_image",
                min_width=min_w,
                min_height=min_h,
                path_name="path1",
            )
            return "web"
        logger.debug("Best web image for %r scored %.3f, below threshold; falling back to generation", concept, best_web_score)
        cleanup_ranked_tmp_paths(ranked_web, keep=0)

    cached_gen = load_retrieval_cache(
        query,
        purpose="single_image",
        min_width=min_w,
        min_height=min_h,
        allowed_source_types=("gen",),
    )
    for entry in cached_gen:
        if reuse_cached_image(entry, output_path):
            logger.debug("Reused cached generated image for %r", concept)
            return "gen"

    return ""


def generate_concept_images(
    pairs: list[dict],
    output_dir: Path,
    start_id: int = 0,
) -> list[dict]:
    """
    For each concept pair, acquire an image for each concept.

    Acquisition order per concept:
      1. Local dataset search (MSCOCO / Open Images / ImageNet)
      2. Web retrieval (Wikimedia / Openverse / Pexels / Pixabay)
      3. T2I generation (SD3.5 Large Turbo) — last resort

    Returns pairs enriched with image paths.
    """
    results = []
    local_ok = 0
    web_ok = 0
    gen_ok = 0
    retrieval_workers = _get_retrieval_worker_count()

    task_specs = []
    pair_states: list[dict] = []
    for i, pair in enumerate(pairs):
        sample_id = start_id + i
        img_dir = output_dir / "images" / str(sample_id)

        img1_path = img_dir / "image1.png"
        img2_path = img_dir / "image2.png"
        pair_states.append({
            "pair": pair,
            "sample_id": sample_id,
            "img1_path": img1_path,
            "img2_path": img2_path,
            "src1": "existing" if img1_path.exists() else "",
            "src2": "existing" if img2_path.exists() else "",
        })
        if not img1_path.exists():
            task_specs.append((i, 1, pair["concept1"], img1_path))
        if not img2_path.exists():
            task_specs.append((i, 2, pair["concept2"], img2_path))

    if task_specs:
        logger.info(
            "Retrieval phase: fetching %d concept images with %d worker threads",
            len(task_specs),
            retrieval_workers,
        )
        with ThreadPoolExecutor(max_workers=retrieval_workers) as executor:
            futures = {
                executor.submit(_retrieve_concept_image, concept, img_path): (pair_idx, slot)
                for pair_idx, slot, concept, img_path in task_specs
            }
            for future in as_completed(futures):
                pair_idx, slot = futures[future]
                src = future.result()
                pair_states[pair_idx][f"src{slot}"] = src

    for state in pair_states:
        pair = state["pair"]
        sample_id = state["sample_id"]
        img1_path = state["img1_path"]
        img2_path = state["img2_path"]

        # ── Concept 1 ──────────────────────────────────────────────
        if img1_path.exists():
            success1 = True
        else:
            src1 = state["src1"]
            if src1 == "local":
                success1 = True
                local_ok += 1
            elif src1 == "web":
                success1 = True
                web_ok += 1
            else:
                if is_image_generation_allowed("path1"):
                    prompt1 = f"A clear, high-quality photograph of {pair['concept1']}, well-lit, neutral background"
                    success1 = generate_image(prompt1, img1_path)
                    if success1:
                        gen_ok += 1
                        query1 = build_compact_retrieval_query(pair["concept1"]) or str(pair["concept1"]).strip()
                        min_w, min_h = get_min_image_size()
                        save_retrieval_cache(
                            query1,
                            [{
                                "path": materialize_shared_image(img1_path, source_type="gen"),
                                "source_type": "gen",
                                "provider": "t2i_generated",
                                "caption": pair["concept1"],
                                "class_label": build_compact_retrieval_query(pair["concept1"], max_words=2) or str(pair["concept1"]).strip(),
                            }],
                            purpose="single_image",
                            min_width=min_w,
                            min_height=min_h,
                            path_name="path1",
                        )
                else:
                    success1 = False

        # ── Concept 2 ──────────────────────────────────────────────
        if img2_path.exists():
            success2 = True
        else:
            src2 = state["src2"]
            if src2 == "local":
                success2 = True
                local_ok += 1
            elif src2 == "web":
                success2 = True
                web_ok += 1
            else:
                if is_image_generation_allowed("path1"):
                    prompt2 = f"A clear, high-quality photograph of {pair['concept2']}, well-lit, neutral background"
                    success2 = generate_image(prompt2, img2_path)
                    if success2:
                        gen_ok += 1
                        query2 = build_compact_retrieval_query(pair["concept2"]) or str(pair["concept2"]).strip()
                        min_w, min_h = get_min_image_size()
                        save_retrieval_cache(
                            query2,
                            [{
                                "path": materialize_shared_image(img2_path, source_type="gen"),
                                "source_type": "gen",
                                "provider": "t2i_generated",
                                "caption": pair["concept2"],
                                "class_label": build_compact_retrieval_query(pair["concept2"], max_words=2) or str(pair["concept2"]).strip(),
                            }],
                            purpose="single_image",
                            min_width=min_w,
                            min_height=min_h,
                            path_name="path1",
                        )
                else:
                    success2 = False

        if success1 and success2:
            pair["image1_path"] = str(img1_path)
            pair["image2_path"] = str(img2_path)
            pair["image1_description"] = f"a photo of {pair['concept1']}"
            pair["image2_description"] = f"a photo of {pair['concept2']}"
            pair["sample_id"] = sample_id
            results.append(pair)
            if len(results) % 50 == 0:
                logger.info(
                    "Acquired images for %d / %d pairs (local=%d, web=%d, gen=%d)",
                    len(results), len(pairs), local_ok, web_ok, gen_ok,
                )
        else:
            logger.debug(
                "Failed to acquire images for pair %d: %s + %s",
                sample_id, pair["concept1"], pair["concept2"],
            )

    logger.info(
        "Image acquisition: %d / %d pairs succeeded (local=%d, web=%d, gen=%d)",
        len(results), len(pairs), local_ok, web_ok, gen_ok,
    )
    return results
