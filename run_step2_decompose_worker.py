#!/usr/bin/env python3
"""Clean Step 2 worker process for local/API prompt decomposition."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.common.utils import get_visible_gpu_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--use-api", action="store_true")
    parser.add_argument("--max-prompts", type=int, default=4500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    os.environ.setdefault("HF_HOME", "/mnt2/xuran_hdd/cache")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", get_visible_gpu_csv())
    os.environ.setdefault("MIS_HF_DOWNLOAD_VERBOSE", os.environ.get("MIS_HF_DOWNLOAD_VERBOSE", "0"))
    os.environ.setdefault("MIS_PATH2_GPU_MEMORY_UTILIZATION", "0.68")
    os.environ.setdefault("MIS_PATH2_MAX_MODEL_LEN", "4096")
    os.environ.setdefault("MIS_PATH2_VLLM_BATCH_SIZE", "64")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    token_path = Path(os.environ["HF_HOME"]) / "token"
    if token_path.exists():
        os.environ.setdefault("HF_TOKEN", token_path.read_text(encoding="utf-8").strip())

    from src.common.utils import setup_logging
    from src.path2_prompt_decompose.decompose import run

    setup_logging()

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
