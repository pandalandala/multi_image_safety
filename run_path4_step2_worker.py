#!/usr/bin/env python3
"""File-backed worker for Path 4 Step 2 intent injection."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-combinations", type=int, default=4000)
    parser.add_argument("--max-per-scene", type=int, default=2)
    parser.add_argument("--use-api", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    os.environ.setdefault("HF_HOME", "/mnt2/xuran_hdd/cache")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    token_path = Path(os.environ["HF_HOME"]) / "token"
    if token_path.exists():
        os.environ.setdefault("HF_TOKEN", token_path.read_text(encoding="utf-8").strip())

    from src.common.utils import setup_logging
    from src.path4_scenario.intent_inject import run

    setup_logging()
    results = run(
        input_file=args.input_file,
        output_dir=args.output_dir,
        use_api=args.use_api,
        max_combinations=args.max_combinations,
        max_per_scene=args.max_per_scene,
    )
    print(f"Intent injection complete: {len(results)} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
