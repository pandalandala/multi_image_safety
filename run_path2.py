#!/usr/bin/env python3
"""Path 2: Prompt Decomposition pipeline."""

import argparse
import logging
import os

from src.common.utils import (
    DATA_DIR,
    apply_path2_runtime_defaults,
    clear_step_state,
    env_flag_is_true,
    finish_step,
    is_step_complete,
    jsonl_record_count,
    start_step,
    setup_logging,
)

logger = logging.getLogger(__name__)


def run(use_api: bool = False) -> None:
    """Run Path 2 end-to-end, skipping completed steps by default."""
    defaults = apply_path2_runtime_defaults()
    output_dir = DATA_DIR / "raw" / "path2"
    step1_output = output_dir / "collected_prompts.jsonl"
    step2_output = output_dir / "decomposed_prompts.jsonl"
    step3_output = output_dir / "samples_with_images.jsonl"
    step4_output = output_dir / "validated_samples.jsonl"
    force_rerun = env_flag_is_true("MIS_FORCE_RERUN_COMPLETED_STEPS")

    def step_complete(step_name: str, output_file, *, min_records: int = 1) -> bool:
        return (not force_rerun) and is_step_complete(
            output_dir,
            step_name,
            expected_outputs=[output_file],
            validator=lambda: jsonl_record_count(output_file) >= min_records,
        )

    logger.info("=" * 60)
    logger.info("PATH 2: Prompt Decomposition")
    logger.info("=" * 60)
    logger.info(
        "Path 2 runtime defaults: CUDA_VISIBLE_DEVICES=%s, max_model_len=%s, "
        "gpu_memory_utilization=%s, batch_size=%s",
        os.environ["CUDA_VISIBLE_DEVICES"],
        os.environ["MIS_PATH2_MAX_MODEL_LEN"],
        os.environ["MIS_PATH2_GPU_MEMORY_UTILIZATION"],
        os.environ["MIS_PATH2_VLLM_BATCH_SIZE"],
    )
    if defaults["HF_HOME"] == os.environ.get("HF_HOME"):
        logger.info("Using Hugging Face cache at %s", os.environ["HF_HOME"])

    if step_complete("step1_collect_prompts", step1_output):
        logger.info("[Step 1/4] Skipping prompt collection; completion marker found")
    else:
        clear_step_state(output_dir, "step1_collect_prompts", stale_paths=[step1_output])
        start_step(output_dir, "step1_collect_prompts")
        logger.info("[Step 1/4] Collecting malicious prompts...")
        from src.path2_prompt_decompose.collect_prompts import run as collect

        prompts = collect()
        if jsonl_record_count(step1_output) < 1:
            raise RuntimeError("Path 2 Step 1 did not produce collected prompts")
        finish_step(
            output_dir,
            "step1_collect_prompts",
            expected_outputs=[step1_output],
            metadata={"records": len(prompts)},
        )

    if step_complete("step2_decompose_prompts", step2_output):
        logger.info("[Step 2/4] Skipping prompt decomposition; completion marker found")
    else:
        clear_step_state(output_dir, "step2_decompose_prompts", stale_paths=[step2_output])
        start_step(output_dir, "step2_decompose_prompts")
        logger.info("[Step 2/4] Decomposing prompts...")
        from src.path2_prompt_decompose.decompose import run as decompose

        decomposed = decompose(use_api=use_api)
        if jsonl_record_count(step2_output) < 1:
            raise RuntimeError("Path 2 Step 2 did not produce decomposed prompts")
        finish_step(
            output_dir,
            "step2_decompose_prompts",
            expected_outputs=[step2_output],
            metadata={"records": len(decomposed)},
        )

    if step_complete("step3_acquire_images", step3_output):
        logger.info("[Step 3/4] Skipping image acquisition; completion marker found")
    else:
        stale_outputs = [step3_output, *sorted(output_dir.glob("samples_with_images_gpu*.jsonl")), *sorted(output_dir.glob("_step3_chunk_*.jsonl"))]
        clear_step_state(output_dir, "step3_acquire_images", stale_paths=stale_outputs)
        start_step(output_dir, "step3_acquire_images")
        logger.info("[Step 3/4] Acquiring images...")
        from src.path2_prompt_decompose.acquire_images import run as acquire

        acquired = acquire(prefer_generation=True)
        if jsonl_record_count(step3_output) < 1:
            raise RuntimeError("Path 2 Step 3 did not produce image-backed samples")
        finish_step(
            output_dir,
            "step3_acquire_images",
            expected_outputs=[step3_output],
            metadata={"records": len(acquired)},
        )

    if step_complete("step4_validate_samples", step4_output):
        logger.info("[Step 4/4] Skipping validation; completion marker found")
    else:
        clear_step_state(output_dir, "step4_validate_samples", stale_paths=[step4_output])
        start_step(output_dir, "step4_validate_samples")
        logger.info("[Step 4/4] Validating samples...")
        from src.path2_prompt_decompose.validate import run as validate

        validated = validate()
        if jsonl_record_count(step4_output) < 1:
            raise RuntimeError("Path 2 Step 4 did not produce validated samples")
        finish_step(
            output_dir,
            "step4_validate_samples",
            expected_outputs=[step4_output],
            metadata={"records": len(validated)},
        )

    logger.info("PATH 2 complete!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Path 2: Prompt Decomposition")
    parser.add_argument("--use-api", action="store_true", help="Use API for LLM calls")
    parser.add_argument("--clean", action="store_true", help="Clear all step states before running")
    args = parser.parse_args()

    setup_logging()
    if args.clean:
        from src.common.utils import clear_all_step_states
        clear_all_step_states(DATA_DIR / "raw" / "path2")
    run(use_api=args.use_api)


if __name__ == "__main__":
    main()
