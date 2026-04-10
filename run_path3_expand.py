#!/usr/bin/env python3
"""
Path 3: Dataset Expansion via image datasets
  Step 1: Load HF safety datasets (+ optional external retrieval / T2I fallback)
  Step 2: Method A — batch decompose captioned images into benign pairs
          (text-only output at this stage; images can be added later)
  Step 3: Method B — LLM cross-image pairing over existing image pools
          (outputs already include real image paths)

Data sources:
  Tier 1: HF safety datasets with captions (SPA-VL, VLGuard, Hateful Memes) -> Method A
  Tier 2: HF safety datasets (UnsafeBench, BeaverTails-V, MIS) -> Method B image pool
  Tier 3: External retrieval + configurable T2I fallback (if too few images from Tier 1+2)
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

os.environ["HF_HOME"] = "/mnt2/xuran_hdd/cache"
sys.path.insert(0, str(Path(__file__).parent))

from src.common.utils import (
    DATA_DIR,
    clear_step_state,
    env_flag_is_true,
    finish_step,
    get_effective_tensor_parallel_size,
    get_visible_gpu_csv,
    is_step_complete,
    jsonl_record_count,
    load_config,
    load_jsonl,
    save_jsonl,
    start_step,
    setup_logging,
)
from src.path3_dataset_expand.status import write_path3_status

setup_logging()
logger = logging.getLogger(__name__)

PROJ = Path(__file__).parent
PYTHON = sys.executable
OUTPUT_DIR = DATA_DIR / "raw" / "path3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
force_rerun = env_flag_is_true("MIS_FORCE_RERUN_COMPLETED_STEPS")
ALL_GPU_IDS = get_visible_gpu_csv()


def step_complete(step_name: str, output_file: Path, *, min_records: int = 1) -> bool:
    """Check whether a Path 3 step has an explicit completion marker."""
    return (not force_rerun) and is_step_complete(
        OUTPUT_DIR,
        step_name,
        expected_outputs=[output_file],
        validator=lambda: jsonl_record_count(output_file) >= min_records,
    )


def _load_existing_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file when it already exists."""
    return load_jsonl(path) if path.exists() else []


def _merge_records(
    existing: list[dict],
    new: list[dict],
    key_fn,
    *,
    require_existing_file: bool = False,
) -> list[dict]:
    """Merge record lists by key, preferring newer non-empty fields."""
    merged: dict[object, dict] = {}
    order: list[object] = []
    for records in (existing, new):
        for item in records:
            key = key_fn(item)
            if not key:
                continue
            if require_existing_file:
                path = item.get("path", "")
                if path and not Path(path).exists():
                    continue
            if key not in merged:
                merged[key] = dict(item)
                order.append(key)
                continue
            combined = dict(merged[key])
            for field, value in item.items():
                if value not in ("", None, [], {}):
                    combined[field] = value
            merged[key] = combined
    return [merged[key] for key in order]


def _image_info_key(item: dict) -> str:
    """Stable key for image metadata records."""
    path = item.get("path", "")
    if path:
        return path
    return "::".join([
        item.get("dataset", ""),
        item.get("category", ""),
        item.get("query", ""),
        item.get("description", ""),
    ])


def _method_a_input_key(item: dict) -> tuple[str, str, str]:
    """Key pending Method A work by source + category + source description."""
    return (
        item.get("dataset", ""),
        item.get("category", ""),
        item.get("description", ""),
    )


def _method_a_result_key(item: dict) -> tuple[str, str, str]:
    """Key Method A outputs compatibly with Method A inputs."""
    return (
        item.get("source_dataset", ""),
        item.get("original_category", ""),
        item.get("original_description", ""),
    )


def _pair_key(path1: str, path2: str) -> tuple[str, str]:
    """Order-invariant key for an image pair."""
    return tuple(sorted((path1 or "", path2 or "")))


def _method_b_prompt_key(item: dict) -> tuple[str, str]:
    """Key pending Method B work by its image pair."""
    return _pair_key(
        item.get("info1", {}).get("path", ""),
        item.get("info2", {}).get("path", ""),
    )


def _method_b_result_key(item: dict) -> tuple[str, str]:
    """Key Method B outputs by their image pair."""
    return _pair_key(
        item.get("image1_path", ""),
        item.get("image2_path", ""),
    )


def _run_method_a(images_with_desc: list[dict]) -> list[dict]:
    """Run Method A incrementally, reusing existing outputs when possible."""
    logger.info("=" * 60)
    logger.info("PATH 3 STEP 2: Method A — vLLM element decompose (subprocess)")
    logger.info("=" * 60)

    method_a_output = OUTPUT_DIR / "method_a_decomposed.jsonl"
    existing_results = _load_existing_jsonl(method_a_output)
    if step_complete("step2_method_a_decompose", method_a_output):
        logger.info("Method A: skipping because completion marker exists")
        return existing_results
    if not images_with_desc:
        logger.info("Method A: skipped (no images with descriptions)")
        return existing_results

    existing_keys = {_method_a_result_key(item) for item in existing_results}
    pending_inputs = [
        item for item in images_with_desc
        if _method_a_input_key(item) not in existing_keys
    ]
    if not pending_inputs:
        if existing_results:
            finish_step(
                OUTPUT_DIR,
                "step2_method_a_decompose",
                expected_outputs=[method_a_output],
                metadata={"records": len(existing_results), "new_records": 0},
            )
        logger.info(f"Method A: reusing {len(existing_results)} existing samples")
        return existing_results

    config = load_config()
    local_cfg = config["llm"]["local"]
    llm_tensor_parallel_size = get_effective_tensor_parallel_size(
        local_cfg.get("tensor_parallel_size")
    )

    input_file = OUTPUT_DIR / "_method_a_input.jsonl"
    temp_output = OUTPUT_DIR / "_method_a_new.jsonl"
    clear_step_state(OUTPUT_DIR, "step2_method_a_decompose", stale_paths=[input_file, temp_output])
    start_step(OUTPUT_DIR, "step2_method_a_decompose")
    save_jsonl(pending_inputs, input_file)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ALL_GPU_IDS
    env["HF_HOME"] = "/mnt2/xuran_hdd/cache"
    env["PYTHONPATH"] = str(PROJ)

    code = f"""
import json, sys
sys.path.insert(0, '{PROJ}')
from src.common.utils import setup_logging, load_jsonl, save_jsonl
from src.path3_dataset_expand.element_decompose import prepare_batch_decompose_prompts
from src.common.schema import Pattern, SourcePath
from vllm import LLM, SamplingParams
setup_logging()

images_with_desc = load_jsonl('{input_file}')
prompts = prepare_batch_decompose_prompts(images_with_desc)
conversations = [[{{"role": "user", "content": p}}] for p in prompts]

print(f'Loading LLM for {{len(prompts)}} decompose prompts...')
llm = LLM(
    model='{local_cfg["model_path"]}',
    tensor_parallel_size={llm_tensor_parallel_size},
    trust_remote_code=True,
    max_model_len=8192,
    enforce_eager=True,
    gpu_memory_utilization={local_cfg.get("gpu_memory_utilization", 0.9)},
)
sampling_params = SamplingParams(temperature=0.7, max_tokens=2048, top_p=0.9)
outputs = llm.chat(conversations, sampling_params, chat_template_kwargs={{"enable_thinking": False}})

results = []
for item, output in zip(images_with_desc, outputs):
    text = output.outputs[0].text
    try:
        start, end = text.rfind("{{"), text.rfind("}}") + 1
        data = json.loads(text[start:end]) if start >= 0 and end > 0 else {{}}
        if "image1_description" in data and "image2_description" in data:
            results.append({{
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
            }})
    except Exception:
        pass

save_jsonl(results, '{temp_output}')
print(f'Method A: {{len(results)}}/{{len(images_with_desc)}} samples saved')
"""
    result = subprocess.run([PYTHON, "-c", code], env=env, cwd=str(PROJ))
    input_file.unlink(missing_ok=True)
    if result.returncode != 0:
        logger.warning("Method A subprocess failed — continuing with existing Method A results")
        temp_output.unlink(missing_ok=True)
        return existing_results

    new_results = _load_existing_jsonl(temp_output)
    temp_output.unlink(missing_ok=True)
    merged = _merge_records(existing_results, new_results, _method_a_result_key)
    save_jsonl(merged, method_a_output)
    finish_step(
        OUTPUT_DIR,
        "step2_method_a_decompose",
        expected_outputs=[method_a_output],
        metadata={"records": len(merged), "new_records": len(new_results)},
    )
    logger.info(
        "Method A: reused %d, added %d, total %d",
        len(existing_results),
        len(new_results),
        len(merged),
    )
    return merged


def _run_method_b(all_image_infos: list[dict]) -> list[dict]:
    """Run Method B incrementally, reusing existing outputs when possible."""
    logger.info("=" * 60)
    logger.info("PATH 3 STEP 3: Method B — LLM cross-image pairing (subprocess)")
    logger.info("=" * 60)

    method_b_output = OUTPUT_DIR / "method_b_cross_paired.jsonl"
    existing_results = _load_existing_jsonl(method_b_output)
    if step_complete("step3_method_b_cross_pair", method_b_output):
        logger.info("Method B: skipping because completion marker exists")
        return existing_results
    if len(all_image_infos) < 20:
        logger.info("Method B: skipped (fewer than 20 images)")
        return existing_results

    from src.path3_dataset_expand.cross_pair import run as run_cross_pair

    existing_keys = {_method_b_result_key(item) for item in existing_results}
    prompt_data = run_cross_pair(all_image_infos, OUTPUT_DIR)
    pending_prompt_data = [
        item for item in prompt_data
        if _method_b_prompt_key(item) not in existing_keys
    ]
    if not pending_prompt_data:
        if existing_results:
            finish_step(
                OUTPUT_DIR,
                "step3_method_b_cross_pair",
                expected_outputs=[method_b_output],
                metadata={"records": len(existing_results), "new_records": 0},
            )
        logger.info(
            "Method B: reusing %d existing samples (no new candidate pairs)",
            len(existing_results),
        )
        return existing_results

    config = load_config()
    local_cfg = config["llm"]["local"]
    llm_tensor_parallel_size = get_effective_tensor_parallel_size(
        local_cfg.get("tensor_parallel_size")
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ALL_GPU_IDS
    env["HF_HOME"] = "/mnt2/xuran_hdd/cache"
    env["PYTHONPATH"] = str(PROJ)

    method_b_input = OUTPUT_DIR / "_method_b_input.jsonl"
    temp_output = OUTPUT_DIR / "_method_b_new.jsonl"
    clear_step_state(
        OUTPUT_DIR,
        "step3_method_b_cross_pair",
        stale_paths=[method_b_input, temp_output, OUTPUT_DIR / "_method_b_pairs.jsonl"],
    )
    start_step(OUTPUT_DIR, "step3_method_b_cross_pair")
    save_jsonl(pending_prompt_data, method_b_input)

    code = f"""
import sys
sys.path.insert(0, '{PROJ}')
from src.common.utils import setup_logging, load_jsonl, save_jsonl
from src.path3_dataset_expand.cross_pair import parse_cross_pair_response
from src.common.schema import Pattern, SourcePath
from vllm import LLM, SamplingParams
setup_logging()

prompt_data = load_jsonl('{method_b_input}')
conversations = [[{{"role": "user", "content": item["prompt"]}}] for item in prompt_data]

print(f'Loading LLM for {{len(conversations)}} cross-pair prompts...')
llm = LLM(
    model='{local_cfg["model_path"]}',
    tensor_parallel_size={llm_tensor_parallel_size},
    trust_remote_code=True,
    max_model_len=8192,
    enforce_eager=True,
    gpu_memory_utilization={local_cfg.get("gpu_memory_utilization", 0.9)},
)
sampling_params = SamplingParams(temperature=0.7, max_tokens=1024, top_p=0.9)
outputs = llm.chat(conversations, sampling_params, chat_template_kwargs={{"enable_thinking": False}})

results = []
rejected = 0
for item, output in zip(prompt_data, outputs):
    text = output.outputs[0].text
    parsed = parse_cross_pair_response(text)
    if parsed:
        results.append({{
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
        }})
    else:
        rejected += 1

save_jsonl(results, '{temp_output}')
print(f'Method B: {{len(results)}} accepted, {{rejected}} rejected out of {{len(prompt_data)}}')
"""
    result = subprocess.run([PYTHON, "-c", code], env=env, cwd=str(PROJ))
    method_b_input.unlink(missing_ok=True)
    if result.returncode != 0:
        logger.warning("Method B subprocess failed — continuing with existing Method B results")
        temp_output.unlink(missing_ok=True)
        return existing_results

    new_results = _load_existing_jsonl(temp_output)
    temp_output.unlink(missing_ok=True)
    merged = _merge_records(existing_results, new_results, _method_b_result_key)
    save_jsonl(merged, method_b_output)
    (OUTPUT_DIR / "_method_b_pairs.jsonl").unlink(missing_ok=True)
    finish_step(
        OUTPUT_DIR,
        "step3_method_b_cross_pair",
        expected_outputs=[method_b_output],
        metadata={"records": len(merged), "new_records": len(new_results)},
    )
    logger.info(
        "Method B: reused %d, added %d, total %d",
        len(existing_results),
        len(new_results),
        len(merged),
    )
    return merged


def main() -> None:
    """Run the active Path 3 expansion flow."""
    logger.info("=" * 60)
    logger.info("PATH 3 STEP 1: Collecting images from HF datasets + external retrieval")
    logger.info("=" * 60)

    all_infos_file = OUTPUT_DIR / "all_image_infos.jsonl"
    if step_complete("step1_collect_image_pool", all_infos_file):
        logger.info("Skipping Path 3 Step 1; completion marker found: %s", all_infos_file)
        all_image_infos = _load_existing_jsonl(all_infos_file)
    else:
        clear_step_state(OUTPUT_DIR, "step1_collect_image_pool")
        start_step(OUTPUT_DIR, "step1_collect_image_pool")
        from src.path3_dataset_expand.expand import collect_all_data

        images_with_desc, _, _, all_image_infos = collect_all_data(
            OUTPUT_DIR,
            max_images_per_dataset=2000,
            max_laion_per_category=200,
        )

        existing_all_image_infos = _load_existing_jsonl(all_infos_file)
        all_image_infos = _merge_records(
            existing_all_image_infos,
            all_image_infos,
            _image_info_key,
            require_existing_file=True,
        )
        save_jsonl(all_image_infos, all_infos_file)
        finish_step(
            OUTPUT_DIR,
            "step1_collect_image_pool",
            expected_outputs=[all_infos_file],
            metadata={"records": len(all_image_infos)},
        )

    images_with_desc = [info for info in all_image_infos if info.get("description", "").strip()]
    logger.info("Total images: %d", len(all_image_infos))
    logger.info("Images with descriptions (for Method A): %d", len(images_with_desc))

    if len(images_with_desc) < 100 and not step_complete("step1_5_t2i_fallback", all_infos_file):
        logger.info("=" * 60)
        logger.info("PATH 3 STEP 1.5: T2I image generation fallback (subprocess)")
        logger.info("=" * 60)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ALL_GPU_IDS
        env["HF_HOME"] = "/mnt2/xuran_hdd/cache"
        env["PYTHONPATH"] = str(PROJ)

        temp_info_file = OUTPUT_DIR / "_t2i_generated_infos.jsonl"
        clear_step_state(OUTPUT_DIR, "step1_5_t2i_fallback", stale_paths=[temp_info_file])
        start_step(OUTPUT_DIR, "step1_5_t2i_fallback")
        code = f"""
import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = '{ALL_GPU_IDS}'
os.environ['HF_HOME'] = '/mnt2/xuran_hdd/cache'
sys.path.insert(0, '{PROJ}')
from src.common.utils import setup_logging, save_jsonl
from src.path3_dataset_expand.expand import generate_images_from_queries
setup_logging()

paths, sources, infos = generate_images_from_queries('{OUTPUT_DIR}', max_per_category=20)
save_jsonl(infos, '{temp_info_file}')
print(f'T2I fallback: {{len(infos)}} images generated')
"""
        rc = subprocess.run([PYTHON, "-c", code], env=env, cwd=str(PROJ)).returncode

        if rc == 0 and temp_info_file.exists():
            t2i_infos = _load_existing_jsonl(temp_info_file)
            all_image_infos = _merge_records(
                all_image_infos,
                t2i_infos,
                _image_info_key,
                require_existing_file=True,
            )
            images_with_desc = [info for info in all_image_infos if info.get("description", "").strip()]
            save_jsonl(all_image_infos, all_infos_file)
            logger.info(
                "After T2I fallback: %d total, %d with descriptions",
                len(all_image_infos),
                len(images_with_desc),
            )
            temp_info_file.unlink(missing_ok=True)
            finish_step(
                OUTPUT_DIR,
                "step1_5_t2i_fallback",
                expected_outputs=[all_infos_file],
                metadata={"records": len(all_image_infos)},
            )
        else:
            logger.warning("T2I fallback subprocess failed")

    method_a_results = _run_method_a(images_with_desc)
    method_b_results = _run_method_b(all_image_infos)

    all_results = method_a_results + method_b_results
    merged_output = OUTPUT_DIR / "cross_paired_samples.jsonl"
    clear_step_state(OUTPUT_DIR, "step4_merge_output", stale_paths=[merged_output])
    start_step(OUTPUT_DIR, "step4_merge_output")
    save_jsonl(all_results, merged_output)
    finish_step(
        OUTPUT_DIR,
        "step4_merge_output",
        expected_outputs=[merged_output],
        metadata={"records": len(all_results)},
    )

    method_a_with_images = len(_load_existing_jsonl(OUTPUT_DIR / "method_a_with_images.jsonl"))
    status_path = write_path3_status(
        OUTPUT_DIR,
        method_a_text_only=len(method_a_results),
        method_b_with_images=len(method_b_results),
        merged_total=len(all_results),
        method_a_with_images=method_a_with_images,
    )

    print("\n" + "=" * 80)
    print(f"✓ PATH 3 EXPAND COMPLETE: {len(all_results)} samples")
    print(f"  Method A text-only:      {len(method_a_results)}")
    print(f"  Method B with images:    {len(method_b_results)}")
    print(f"  Method A with images:    {method_a_with_images}")
    print(f"  Mixed merged output:     {merged_output}")
    print(f"  Status note:             {status_path}")
    print("  Next step if needed:     python run_path3_acquire_images.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
