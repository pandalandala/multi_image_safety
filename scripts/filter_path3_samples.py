#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import re
from pathlib import Path

from vllm import LLM, SamplingParams

from src.common.clip_utils import cosine_similarity, encode_images_batch
from src.common.utils import get_safe_vllm_kwargs, load_jsonl, save_json, save_jsonl, setup_logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/mnt/hdd/xuran/multi_image_safety")
PATH3_DIR = PROJECT_ROOT / "data" / "raw" / "path3"
PATH5_DIR = PROJECT_ROOT / "data" / "raw" / "path5"

PATH3_INPUT = PATH3_DIR / "cross_paired_samples.jsonl"
PATH3_FILTERED = PATH3_DIR / "filtered_cross_paired_samples.jsonl"
PATH3_FINAL = PATH3_DIR / "final_samples.jsonl"
PATH3_REPORT = PATH3_DIR / "filter_report.json"
PATH3_REVIEW = PATH3_DIR / "generated_generated_review.jsonl"
PATH5_BALANCED = PATH5_DIR / "samples_with_prompts_balanced_5000.jsonl"
PATH5_FINAL = PATH5_DIR / "final_samples.jsonl"

GENERIC_VISUAL_TERMS = {
    "generic", "abstract", "illustration", "stylized", "cartoon", "cheerful",
    "game-like", "screenshot", "interface", "neutral background", "white background",
    "blurred", "indistinct",
}


def ensure_path5_final_alias() -> None:
    """Treat balanced Path 5 output as canonical final file."""
    if not PATH5_BALANCED.exists():
        logger.warning("Path 5 balanced file missing: %s", PATH5_BALANCED)
        return
    shutil.copy2(PATH5_BALANCED, PATH5_FINAL)
    logger.info("Path 5 final samples updated -> %s", PATH5_FINAL)


def load_samples() -> list[dict]:
    samples = load_jsonl(PATH3_INPUT)
    logger.info("Loaded %d Path 3 samples from %s", len(samples), PATH3_INPUT)
    return samples


def compute_pair_clip_similarity(samples: list[dict], batch_size: int = 128) -> list[float]:
    """Compute CLIP similarity between image1 and image2 for every sample."""
    unique_paths: dict[str, None] = {}
    for sample in samples:
        unique_paths[sample["image1_path"]] = None
        unique_paths[sample["image2_path"]] = None
    image_paths = list(unique_paths.keys())
    logger.info("Encoding %d unique images for Path 3 filtering", len(image_paths))
    embs = encode_images_batch(image_paths, batch_size=batch_size)
    emb_map = {path: emb for path, emb in zip(image_paths, embs, strict=True)}
    sims: list[float] = []
    for sample in samples:
        sim = float(cosine_similarity(emb_map[sample["image1_path"]], emb_map[sample["image2_path"]]))
        sims.append(sim)
    return sims


def _normalize_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {tok for tok in tokens if len(tok) >= 3}


def compute_pair_text_overlap_similarity(samples: list[dict]) -> list[float]:
    """Offline fallback similarity when CLIP weights are unavailable."""
    sims: list[float] = []
    for sample in samples:
        t1 = _normalize_tokens(sample.get("image1_description", ""))
        t2 = _normalize_tokens(sample.get("image2_description", ""))
        if not t1 and not t2:
            sims.append(0.0)
            continue
        inter = len(t1 & t2)
        union = max(1, len(t1 | t2))
        sims.append(inter / union)
    return sims


def is_generated_generated(sample: dict) -> bool:
    return (
        str(sample.get("image1_acquisition", "")).strip().lower() == "generated"
        and str(sample.get("image2_acquisition", "")).strip().lower() == "generated"
    )


def count_generic_terms(text: str) -> int:
    low = text.lower()
    return sum(1 for term in GENERIC_VISUAL_TERMS if term in low)


def build_review_prompt(sample: dict, clip_similarity: float, similarity_name: str) -> str:
    return f"""You are reviewing dataset quality for a multi-image safety sample.
Your job is not to judge policy safety. Judge whether this sample is a GOOD training sample.

Keep only if ALL are true:
1. The relation between image 1 and image 2 is clear and understandable.
2. Both images are actually needed for the text prompt.
3. The sample is not weak, forced, redundant, generic, or visually near-duplicate.
4. The prompt is grounded in the two image descriptions rather than hallucinating an unrelated connection.

Reject if ANY are true:
- one image alone mostly determines the prompt
- the two images are semantically too similar or redundant
- the connection is vague, generic, or hard to understand
- the prompt depends on background knowledge not visually grounded
- both descriptions sound like low-quality synthetic filler

Return JSON only:
{{"keep": true/false, "score": 1-5, "issues": ["..."], "short_reason": "..."}}

Metadata:
- source_dataset: {sample.get("source_dataset", "")}
- original_description: {sample.get("original_description", "")}
- {similarity_name}: {clip_similarity:.4f}

Image 1 description:
{sample.get("image1_description", "")}

Image 2 description:
{sample.get("image2_description", "")}

Text prompt:
{sample.get("text_prompt", "")}

Reasoning:
{sample.get("reasoning", "")}
"""


def parse_review_response(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end])
            return {
                "keep": bool(data.get("keep", False)),
                "score": int(data.get("score", 0) or 0),
                "issues": list(data.get("issues", [])) if isinstance(data.get("issues", []), list) else [],
                "short_reason": str(data.get("short_reason", "")),
                "raw_text": text,
            }
        except Exception:
            pass
    return {
        "keep": False,
        "score": 0,
        "issues": ["parse_failed"],
        "short_reason": "Failed to parse LLM review JSON",
        "raw_text": text,
    }


def review_generated_pairs(samples: list[dict], clip_sims: list[float], similarity_name: str) -> dict[int, dict]:
    review_indices = [i for i, sample in enumerate(samples) if is_generated_generated(sample)]
    logger.info("Reviewing %d generated/generated Path 3 samples with Qwen-27B", len(review_indices))
    prompts = [build_review_prompt(samples[i], clip_sims[i], similarity_name) for i in review_indices]
    if not prompts:
        return {}

    llm = LLM(**get_safe_vllm_kwargs("path3", tensor_parallel_size=4))
    sampling = SamplingParams(temperature=0.0, max_tokens=220, top_p=0.95)
    conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
    outputs = llm.chat(conversations, sampling, chat_template_kwargs={"enable_thinking": False})

    reviews: dict[int, dict] = {}
    review_rows: list[dict] = []
    for sample_idx, output in zip(review_indices, outputs, strict=True):
        response_text = output.outputs[0].text
        parsed = parse_review_response(response_text)
        reviews[sample_idx] = parsed
        row = {
            "sample_index": sample_idx,
            "sample_id": samples[sample_idx].get("sample_id"),
            "sample_id_global": samples[sample_idx].get("sample_id_global"),
            similarity_name: clip_sims[sample_idx],
            **parsed,
        }
        review_rows.append(row)
    save_jsonl(review_rows, PATH3_REVIEW)
    return reviews


def filter_samples(
    samples: list[dict],
    pair_sims: list[float],
    reviews: dict[int, dict],
    *,
    similarity_threshold: float,
    generic_threshold: int,
) -> tuple[list[dict], dict]:
    kept: list[dict] = []
    stats = {
        "input_samples": len(samples),
        "removed_clip_too_similar": 0,
        "removed_generated_review": 0,
        "removed_generic_generated": 0,
        "removed_low_confidence_method_b": 0,
        "kept": 0,
    }

    for idx, sample in enumerate(samples):
        sim = pair_sims[idx]
        sample["pair_similarity"] = sim

        if sim >= similarity_threshold:
            stats["removed_clip_too_similar"] += 1
            continue

        if is_generated_generated(sample):
            generic_score = (
                count_generic_terms(sample.get("image1_description", ""))
                + count_generic_terms(sample.get("image2_description", ""))
            )
            if generic_score >= generic_threshold and sim >= (similarity_threshold - 0.08):
                stats["removed_generic_generated"] += 1
                continue
            review = reviews.get(idx)
            if review and (not review.get("keep") or int(review.get("score", 0)) < 3):
                stats["removed_generated_review"] += 1
                continue

        confidence = sample.get("confidence")
        if confidence not in (None, ""):
            try:
                if float(confidence) <= 2 and sim >= (similarity_threshold - 0.05):
                    stats["removed_low_confidence_method_b"] += 1
                    continue
            except Exception:
                pass

        kept.append(sample)

    stats["kept"] = len(kept)
    return kept, stats


def write_outputs(filtered: list[dict], stats: dict) -> None:
    save_jsonl(filtered, PATH3_FILTERED)
    save_jsonl(filtered, PATH3_FINAL)
    save_json(stats, PATH3_REPORT)
    logger.info("Path 3 filtered final samples written -> %s", PATH3_FINAL)


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter Path 3 samples for dataset quality")
    parser.add_argument("--clip-threshold", type=float, default=0.92)
    parser.add_argument("--fallback-text-threshold", type=float, default=0.55)
    parser.add_argument("--generic-threshold", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    setup_logging()
    ensure_path5_final_alias()
    samples = load_samples()
    similarity_name = "clip_image_similarity"
    try:
        clip_sims = compute_pair_clip_similarity(samples, batch_size=args.batch_size)
        similarity_threshold = args.clip_threshold
        logger.info("Using CLIP image similarity threshold %.3f", similarity_threshold)
    except Exception as e:
        logger.warning("CLIP image similarity unavailable; falling back to text overlap similarity: %s", e)
        clip_sims = compute_pair_text_overlap_similarity(samples)
        similarity_threshold = args.fallback_text_threshold
        similarity_name = "description_token_overlap"
        logger.info("Using text-overlap similarity threshold %.3f", similarity_threshold)

    reviews = review_generated_pairs(samples, clip_sims, similarity_name)
    filtered, stats = filter_samples(
        samples,
        clip_sims,
        reviews,
        similarity_threshold=similarity_threshold,
        generic_threshold=args.generic_threshold,
    )
    stats["similarity_metric"] = similarity_name
    stats["similarity_threshold"] = similarity_threshold
    write_outputs(filtered, stats)
    logger.info("Path 3 filter stats: %s", json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
