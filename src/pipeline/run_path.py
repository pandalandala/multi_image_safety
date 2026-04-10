"""Unified entry point for running individual data construction paths."""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from src.common.utils import (
    apply_path2_runtime_defaults,
    setup_logging,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path("/mnt/hdd/xuran/multi_image_safety")


def run_script(script_name: str) -> None:
    """Run the current top-level pipeline script for a path."""
    script_path = PROJECT_ROOT / script_name
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")


def run_path2(use_api: bool = False):
    """Run Path 2: Prompt Decomposition pipeline."""
    defaults = apply_path2_runtime_defaults()
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
    result = [sys.executable, str(PROJECT_ROOT / "run_path2.py")]
    if use_api:
        result.append("--use-api")
    completed = subprocess.run(result, cwd=str(PROJECT_ROOT))
    if completed.returncode != 0:
        raise RuntimeError(f"run_path2.py failed with exit code {completed.returncode}")


def run_path3():
    """Run Path 3: Dataset Expansion pipeline."""
    logger.info("=" * 60)
    logger.info("PATH 3: Dataset Expansion")
    logger.info("=" * 60)
    run_script("run_path3_expand.py")

    logger.info("PATH 3 complete!")


def run_path4(use_api: bool = False):
    """Run Path 4: Scenario Construction pipeline."""
    logger.info("=" * 60)
    logger.info("PATH 4: Scenario Construction")
    logger.info("=" * 60)
    run_script("run_path4_scenario.py")

    logger.info("PATH 4 complete!")


def run_path5(use_api: bool = False):
    """Run Path 5: image pair mining + prompt generation pipeline."""
    logger.info("=" * 60)
    logger.info("PATH 5: Image Pair Mining")
    logger.info("=" * 60)
    run_script("run_path5_embedding.py")

    logger.info("PATH 5 complete!")


PATH_RUNNERS = {
    2: run_path2,
    3: run_path3,
    4: run_path4,
    5: run_path5,
}


def main():
    parser = argparse.ArgumentParser(description="Run a specific data construction path")
    parser.add_argument("path", type=int, choices=[2, 3, 4, 5], help="Path number to run")
    parser.add_argument("--use-api", action="store_true", help="Use API for LLM calls")
    args = parser.parse_args()

    setup_logging()

    runner = PATH_RUNNERS[args.path]
    if args.path in [2, 4, 5]:
        runner(use_api=args.use_api)
    else:
        runner()


if __name__ == "__main__":
    main()
