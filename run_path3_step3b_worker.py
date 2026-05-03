#!/usr/bin/env python3
"""File-backed worker for Path 3 Method B vLLM cross-pairing."""

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
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    token_path = Path(os.environ["HF_HOME"]) / "token"
    if token_path.exists():
        os.environ.setdefault("HF_TOKEN", token_path.read_text(encoding="utf-8").strip())

    from src.common.schema import Pattern, SourcePath
    from src.common.utils import get_hf_home, load_jsonl, save_jsonl, setup_logging
    from src.path3_dataset_expand.cross_pair import parse_cross_pair_response
    from vllm import LLM, SamplingParams

    os.environ.setdefault("HF_HOME", get_hf_home())

    setup_logging()

    prompt_data = load_jsonl(args.input_file)
    conversations = [[{"role": "user", "content": item["prompt"]}] for item in prompt_data]

    print(f"Loading LLM for {len(conversations)} cross-pair prompts...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_custom_all_reduce=True,
    )
    sampling_params = SamplingParams(temperature=0.7, max_tokens=1024, top_p=0.9)
    outputs = llm.chat(conversations, sampling_params, chat_template_kwargs={"enable_thinking": False})

    results = []
    rejected = 0
    for item, output in zip(prompt_data, outputs):
        text = output.outputs[0].text
        parsed = parse_cross_pair_response(text)
        if not parsed:
            rejected += 1
            continue
        results.append(
            {
                "image1_path": item["info1"].get("path", ""),
                "image2_path": item["info2"].get("path", ""),
                "image1_description": item["info1"].get("description", ""),
                "image2_description": item["info2"].get("description", ""),
                "image1_source": item["info1"].get("dataset", ""),
                "image2_source": item["info2"].get("dataset", ""),
                "category": parsed.get("category", "CRIME"),
                "text_prompt": parsed["text_prompt"],
                "reasoning": parsed.get("reasoning", ""),
                "confidence": parsed.get("confidence", 3),
                "pairing_mode": item.get("pairing_mode", ""),
                "category_hint": item.get("category_hint", ""),
                "pattern": Pattern.B.value,
                "source_path": SourcePath.PATH3.value,
            }
        )

    save_jsonl(results, args.output_file)
    print(f"Method B: {len(results)} accepted, {rejected} rejected out of {len(prompt_data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
