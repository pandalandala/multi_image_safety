#!/usr/bin/env python3
"""
Path 2 Step 2 launcher.

Launches the actual vLLM decomposition work in a second, cleaner Python
process so the worker can start before any project import accidentally touches
CUDA state. If the preferred multiprocessing mode fails, retry once with a
fallback mode.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.common.utils import (
    apply_gpu_runtime_profile,
    clear_step_state,
    finish_step,
    is_step_complete,
    jsonl_record_count,
    log_gpu_runtime_profile,
    start_step,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--use-api", action="store_true")
    parser.add_argument("--max-prompts", type=int, default=4500)
    return parser.parse_args()


def _worker_cmd(args: argparse.Namespace) -> list[str]:
    worker = Path(__file__).with_name("run_path2_step2_worker.py")
    cmd = [
        sys.executable,
        str(worker),
        "--input-file",
        args.input_file,
        "--output-dir",
        args.output_dir,
        "--max-prompts",
        str(args.max_prompts),
    ]
    if args.use_api:
        cmd.append("--use-api")
    return cmd


def _run_once(args: argparse.Namespace, mp_method: str) -> int:
    env = os.environ.copy()
    env.setdefault("HF_HOME", "/mnt2/xuran_hdd/cache")
    env.setdefault("MIS_HF_DOWNLOAD_VERBOSE", os.environ.get("MIS_HF_DOWNLOAD_VERBOSE", "0"))
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    env["VLLM_WORKER_MULTIPROC_METHOD"] = mp_method

    token_path = Path(env["HF_HOME"]) / "token"
    if token_path.exists():
        env.setdefault("HF_TOKEN", token_path.read_text(encoding="utf-8").strip())

    return subprocess.run(_worker_cmd(args), env=env, cwd=str(Path(__file__).parent)).returncode


def main() -> int:
    args = parse_args()
    from src.common.utils import setup_logging
    setup_logging()
    import logging

    profile = apply_gpu_runtime_profile(
        path_name="path2",
        task_type="llm",
        preferred_gpu_count=4,
        requested_tensor_parallel_size=4,
        requested_batch_size=int(os.environ.get("MIS_PATH2_VLLM_BATCH_SIZE", "64")),
    )
    log_gpu_runtime_profile(logging.getLogger(__name__), profile, "Path 2 Step 2 launcher")
    output_dir = Path(args.output_dir)
    output_file = output_dir / "decomposed_prompts.jsonl"

    if is_step_complete(
        output_dir,
        "step2_decompose_prompts",
        expected_outputs=[output_file],
        validator=lambda: jsonl_record_count(output_file) >= 1,
    ):
        print(f"Step 2 already complete; marker found for {output_file}")
        return 0

    clear_step_state(output_dir, "step2_decompose_prompts", stale_paths=[output_file])
    start_step(output_dir, "step2_decompose_prompts")

    # Prefer fork on Linux when we can keep the worker process pristine.
    attempts = ["fork", "spawn"]
    tried: list[tuple[str, int]] = []

    for mp_method in attempts:
        rc = _run_once(args, mp_method)
        tried.append((mp_method, rc))
        if rc == 0:
            if jsonl_record_count(output_file) < 1:
                print("Step 2 finished without producing decomposed prompts", file=sys.stderr)
                return 1
            finish_step(
                output_dir,
                "step2_decompose_prompts",
                expected_outputs=[output_file],
                metadata={"records": jsonl_record_count(output_file)},
            )
            if len(tried) > 1:
                print(f"Step 2 succeeded after retrying with VLLM_WORKER_MULTIPROC_METHOD={mp_method}")
            return 0
        print(
            f"Step 2 attempt failed with VLLM_WORKER_MULTIPROC_METHOD={mp_method} "
            f"(exit code {rc})",
            file=sys.stderr,
        )

    print(f"Step 2 failed after attempts: {tried}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
