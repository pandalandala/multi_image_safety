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

os.environ.setdefault("HF_HOME", "/mnt2/xuran_hdd/cache")
sys.path.insert(0, str(Path(__file__).parent))

from src.common.utils import (
    DATA_DIR,
    apply_gpu_runtime_profile,
    clear_step_state,
    env_flag_is_true,
    fail_step,
    finish_step,
    get_effective_tensor_parallel_size,
    get_hf_home,
    get_visible_gpu_csv,
    is_step_complete,
    jsonl_record_count,
    load_config,
    load_jsonl,
    log_gpu_runtime_profile,
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

if "--clean" in sys.argv:
    from src.common.utils import clear_all_step_states
    clear_all_step_states(OUTPUT_DIR)
    sys.argv.remove("--clean")

force_rerun = env_flag_is_true("MIS_FORCE_RERUN_COMPLETED_STEPS")
LLM_PROFILE = apply_gpu_runtime_profile(
    path_name="path3",
    task_type="llm",
    preferred_gpu_count=4,
    requested_tensor_parallel_size=4,
)
log_gpu_runtime_profile(logger, LLM_PROFILE, "Path 3 LLM")
ALL_GPU_IDS = LLM_PROFILE["selected_gpu_csv"]
HF_HOME = get_hf_home()
LLM_MAX_MODEL_LEN = LLM_PROFILE["max_model_len"]
LLM_GPU_MEMORY_UTILIZATION = LLM_PROFILE["gpu_memory_utilization"]
LLM_TENSOR_PARALLEL_SIZE = get_effective_tensor_parallel_size(LLM_PROFILE["tensor_parallel_size"])
if LLM_TENSOR_PARALLEL_SIZE == 3:
    logger.warning(
        "Path 3 selected 3 visible GPUs for Qwen/Qwen3.5-27B; forcing tensor_parallel_size=2 "
        "because 3-way TP is incompatible with this model."
    )
    LLM_TENSOR_PARALLEL_SIZE = 2


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


def _build_worker_env(gpu_ids: str | None = None) -> dict[str, str]:
    """Build a stable subprocess runtime environment for Path 3 workers."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids or ALL_GPU_IDS
    env["HF_HOME"] = HF_HOME
    env["PYTHONPATH"] = str(PROJ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return env


def _run_worker_script(script_name: str, args: list[str], gpu_ids: str | None = None) -> int:
    """Run a file-backed worker subprocess for Path 3."""
    worker = PROJ / script_name
    result = subprocess.run(
        [PYTHON, str(worker), *args],
        env=_build_worker_env(gpu_ids),
        cwd=str(PROJ),
    )
    return result.returncode


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

    input_file = OUTPUT_DIR / "_method_a_input.jsonl"
    temp_output = OUTPUT_DIR / "_method_a_new.jsonl"
    clear_step_state(OUTPUT_DIR, "step2_method_a_decompose", stale_paths=[input_file, temp_output])
    start_step(
        OUTPUT_DIR,
        "step2_method_a_decompose",
        cleanup_paths=[input_file, temp_output, method_a_output],
    )
    save_jsonl(pending_inputs, input_file)

    result = _run_worker_script(
        "run_path3_step2a_worker.py",
        [
            "--input-file",
            str(input_file),
            "--output-file",
            str(temp_output),
            "--model-path",
            str(local_cfg["model_path"]),
            "--tensor-parallel-size",
            str(LLM_TENSOR_PARALLEL_SIZE),
            "--max-model-len",
            str(LLM_MAX_MODEL_LEN),
            "--gpu-memory-utilization",
            str(LLM_GPU_MEMORY_UTILIZATION),
        ],
    )
    input_file.unlink(missing_ok=True)
    if result != 0:
        logger.warning("Method A subprocess failed — continuing with existing Method A results")
        fail_step(
            OUTPUT_DIR,
            "step2_method_a_decompose",
            error=f"Method A worker failed with exit code {result}",
        )
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

    method_b_input = OUTPUT_DIR / "_method_b_input.jsonl"
    temp_output = OUTPUT_DIR / "_method_b_new.jsonl"
    clear_step_state(
        OUTPUT_DIR,
        "step3_method_b_cross_pair",
        stale_paths=[method_b_input, temp_output, OUTPUT_DIR / "_method_b_pairs.jsonl"],
    )
    start_step(
        OUTPUT_DIR,
        "step3_method_b_cross_pair",
        cleanup_paths=[method_b_input, temp_output, OUTPUT_DIR / "_method_b_pairs.jsonl", method_b_output],
    )
    save_jsonl(pending_prompt_data, method_b_input)

    result = _run_worker_script(
        "run_path3_step3b_worker.py",
        [
            "--input-file",
            str(method_b_input),
            "--output-file",
            str(temp_output),
            "--model-path",
            str(local_cfg["model_path"]),
            "--tensor-parallel-size",
            str(LLM_TENSOR_PARALLEL_SIZE),
            "--max-model-len",
            str(LLM_MAX_MODEL_LEN),
            "--gpu-memory-utilization",
            str(LLM_GPU_MEMORY_UTILIZATION),
        ],
    )
    method_b_input.unlink(missing_ok=True)
    if result != 0:
        logger.warning("Method B subprocess failed — continuing with existing Method B results")
        fail_step(
            OUTPUT_DIR,
            "step3_method_b_cross_pair",
            error=f"Method B worker failed with exit code {result}",
        )
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
        clear_step_state(OUTPUT_DIR, "step1_collect_image_pool", stale_paths=[all_infos_file])
        start_step(OUTPUT_DIR, "step1_collect_image_pool", cleanup_paths=[all_infos_file])
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
        env["HF_HOME"] = HF_HOME
        env["PYTHONPATH"] = str(PROJ)

        temp_info_file = OUTPUT_DIR / "_t2i_generated_infos.jsonl"
        clear_step_state(OUTPUT_DIR, "step1_5_t2i_fallback", stale_paths=[temp_info_file])
        start_step(
            OUTPUT_DIR,
            "step1_5_t2i_fallback",
            cleanup_paths=[temp_info_file],
        )
        code = f"""
import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = '{ALL_GPU_IDS}'
os.environ['HF_HOME'] = '{HF_HOME}'
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
            fail_step(
                OUTPUT_DIR,
                "step1_5_t2i_fallback",
                error=f"T2I fallback subprocess failed with exit code {rc}",
            )

    method_a_results = _run_method_a(images_with_desc)
    method_b_results = _run_method_b(all_image_infos)

    all_results = method_a_results + method_b_results
    merged_output = OUTPUT_DIR / "cross_paired_samples.jsonl"
    clear_step_state(OUTPUT_DIR, "step4_merge_output", stale_paths=[merged_output])
    start_step(OUTPUT_DIR, "step4_merge_output", cleanup_paths=[merged_output])
    save_jsonl(all_results, merged_output)
    finish_step(
        OUTPUT_DIR,
        "step4_merge_output",
        expected_outputs=[merged_output],
        metadata={"records": len(all_results)},
    )

    # ── Step 4: Method A image acquisition (T2I, parallel per GPU) ──────────────
    logger.info("=" * 60)
    logger.info("PATH 3 STEP 4: Method A image acquisition (T2I parallel subprocesses)")
    logger.info("=" * 60)
    method_a_with_images_file = OUTPUT_DIR / "method_a_with_images.jsonl"
    rc_step4 = _run_worker_script("run_path3_step4_images.py", [])
    if rc_step4 != 0:
        logger.warning("Step 4 image acquisition failed (exit code %d); continuing", rc_step4)

    method_a_with_images = len(_load_existing_jsonl(method_a_with_images_file))
    # Re-read merged output (step 4 may have updated it)
    final_results = _load_existing_jsonl(merged_output)
    status_path = write_path3_status(
        OUTPUT_DIR,
        method_a_text_only=len(method_a_results),
        method_b_with_images=len(method_b_results),
        merged_total=len(final_results),
        method_a_with_images=method_a_with_images,
    )

    print("\n" + "=" * 80)
    if rc_step4 == 0:
        print(f"✓ PATH 3 COMPLETE: {len(final_results)} samples")
    else:
        print(f"! PATH 3 PARTIAL: Step 4 image acquisition failed (exit code {rc_step4})")
    print(f"  Method A text-only:      {len(method_a_results)}")
    print(f"  Method A with images:    {method_a_with_images}")
    print(f"  Method B with images:    {len(method_b_results)}")
    print(f"  Mixed merged output:     {merged_output}")
    print(f"  Status note:             {status_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
