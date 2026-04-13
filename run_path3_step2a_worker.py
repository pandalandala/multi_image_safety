#!/usr/bin/env python3
"""File-backed worker for Path 3 Method A vLLM decomposition."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
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

    from src.common.schema import Pattern, SourcePath
    from src.common.utils import load_jsonl, save_jsonl, setup_logging
    from src.path3_dataset_expand.element_decompose import prepare_batch_decompose_prompts
    from vllm import LLM, SamplingParams

    setup_logging()

    images_with_desc = load_jsonl(args.input_file)
    prompts = prepare_batch_decompose_prompts(images_with_desc)
    conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]

    print(f"Loading LLM for {len(prompts)} decompose prompts...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_custom_all_reduce=True,
    )
    sampling_params = SamplingParams(temperature=0.7, max_tokens=2048, top_p=0.9)
    outputs = llm.chat(conversations, sampling_params, chat_template_kwargs={"enable_thinking": False})

    results = []
    for item, output in zip(images_with_desc, outputs):
        text = output.outputs[0].text
        try:
            start = text.rfind("{")
            end = text.rfind("}") + 1
            data = json.loads(text[start:end]) if start >= 0 and end > 0 else {}
        except Exception:
            continue
        if "image1_description" not in data or "image2_description" not in data:
            continue
        results.append(
            {
                "original_description": item.get("description", ""),
                "original_category": item.get("category", ""),
                "category": item.get("category", "CRIME"),
                "image1_description": data["image1_description"],
                "image2_description": data["image2_description"],
                "text_prompt": data.get("text_prompt", ""),
                "reasoning": data.get("decomposition_reasoning", ""),
                "pattern": Pattern.A.value,
                "source_path": SourcePath.PATH3.value,
                "source_dataset": item.get("dataset", ""),
            }
        )

    save_jsonl(results, args.output_file)
    print(f"Method A: {len(results)}/{len(images_with_desc)} samples saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
