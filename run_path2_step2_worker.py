#!/usr/bin/env python3
"""Clean Step 2 worker process for local/API prompt decomposition."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.common.utils import apply_gpu_runtime_profile, log_gpu_runtime_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--use-api", action="store_true")
    parser.add_argument("--max-prompts", type=int, default=4500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = apply_gpu_runtime_profile(
        path_name="path2",
        task_type="llm",
        preferred_gpu_count=4,
        requested_tensor_parallel_size=4,
        requested_batch_size=int(os.environ.get("MIS_PATH2_VLLM_BATCH_SIZE", "64")),
    )

    os.environ.setdefault("HF_HOME", "/mnt2/xuran_hdd/cache")
    os.environ.setdefault("MIS_HF_DOWNLOAD_VERBOSE", os.environ.get("MIS_HF_DOWNLOAD_VERBOSE", "0"))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    token_path = Path(os.environ["HF_HOME"]) / "token"
    if token_path.exists():
        os.environ.setdefault("HF_TOKEN", token_path.read_text(encoding="utf-8").strip())

    from src.common.utils import setup_logging
    from src.path2_prompt_decompose.decompose import run

    setup_logging()
    import logging
    log_gpu_runtime_profile(logging.getLogger(__name__), profile, "Path 2 Step 2 worker")

    results = run(
        input_file=args.input_file,
        output_dir=args.output_dir,
        use_api=args.use_api,
        max_prompts=args.max_prompts,
    )
    print(f"Decomposed {len(results)} prompts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
