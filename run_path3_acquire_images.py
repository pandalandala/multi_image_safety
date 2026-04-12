#!/usr/bin/env python3
"""
Path 3 Step 4: Acquire images for Method A's text-only decomposed descriptions.
Uses 4 parallel T2I subprocesses (one per GPU), same pattern as Path 2 Step 3.

Only processes method_a_decomposed.jsonl (Method B already has images).
After completion, merges Method A (with images) + Method B into cross_paired_samples.jsonl.

Runtime estimate: ~1-2 hours (2865 samples × 2 images, 4 GPUs)
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

os.environ["HF_HOME"] = "/mnt2/xuran_hdd/cache"
sys.path.insert(0, str(Path(__file__).parent))

from src.common.utils import (
    apply_gpu_runtime_profile,
    clear_step_state,
    env_flag_is_true,
    finish_step,
    get_visible_gpu_ids,
    is_step_complete,
    jsonl_record_count,
    setup_logging,
    load_jsonl,
    log_gpu_runtime_profile,
    save_jsonl,
    DATA_DIR,
    start_step,
)
from src.path3_dataset_expand.status import write_path3_status

setup_logging()
logger = logging.getLogger(__name__)
IMAGE_PROFILE = apply_gpu_runtime_profile(
    path_name="path3",
    task_type="image",
    preferred_gpu_count=4,
)
log_gpu_runtime_profile(logger, IMAGE_PROFILE, "Path 3 image acquire")

PROJ = Path(__file__).parent
PYTHON = sys.executable
OUTPUT_DIR = DATA_DIR / "raw" / "path3"
INPUT_FILE = OUTPUT_DIR / "method_a_decomposed.jsonl"
METHOD_A_WITH_IMAGES = OUTPUT_DIR / "method_a_with_images.jsonl"
METHOD_B_FILE = OUTPUT_DIR / "method_b_cross_paired.jsonl"
MERGED_OUTPUT = OUTPUT_DIR / "cross_paired_samples.jsonl"
force_rerun = env_flag_is_true("MIS_FORCE_RERUN_COMPLETED_STEPS")


def step_complete(step_name: str, output_file: Path, *, min_records: int = 1) -> bool:
    """Check whether a Path 3 image-acquisition step has a completion marker."""
    return (not force_rerun) and is_step_complete(
        OUTPUT_DIR,
        step_name,
        expected_outputs=[output_file],
        validator=lambda: jsonl_record_count(output_file) >= min_records,
    )
def main() -> None:
    """Acquire images for Method A and refresh the merged Path 3 output."""
    if step_complete("step5_method_a_acquire_images", METHOD_A_WITH_IMAGES):
        method_a_results = load_jsonl(METHOD_A_WITH_IMAGES)
        method_b_results = load_jsonl(METHOD_B_FILE) if METHOD_B_FILE.exists() else []
        all_results = load_jsonl(MERGED_OUTPUT) if MERGED_OUTPUT.exists() else (method_a_results + method_b_results)
        logger.info("Skipping Path 3 image acquisition; completion marker found")
        print("\n" + "=" * 80)
        print("✓ PATH 3 IMAGE ACQUISITION ALREADY COMPLETE")
        print(f"  Method A with images: {len(method_a_results)}")
        print(f"  Merged output: {MERGED_OUTPUT}")
        print("=" * 80)
        return

    if not INPUT_FILE.exists():
        logger.error(f"Input not found: {INPUT_FILE}")
        sys.exit(1)

    samples = load_jsonl(INPUT_FILE)
    logger.info(f"Loaded {len(samples)} Method A samples (text-only, need images)")
    clear_step_state(
        OUTPUT_DIR,
        "step5_method_a_acquire_images",
        stale_paths=[
            METHOD_A_WITH_IMAGES,
            MERGED_OUTPUT,
            *sorted(OUTPUT_DIR.glob("method_a_images_gpu*.jsonl")),
            *sorted(OUTPUT_DIR.glob("_p3img_chunk_*.jsonl")),
        ],
    )
    start_step(OUTPUT_DIR, "step5_method_a_acquire_images")

    for i, sample in enumerate(samples):
        sample["sample_id_global"] = 3000 + i  # Path 3 ID range: 3000+

    free_gpus = get_visible_gpu_ids(max_gpus=None)
    logger.info(f"Using visible GPUs for Method A image acquisition: {free_gpus}")
    if not free_gpus:
        logger.error("No GPUs configured for Method A image acquisition!")
        sys.exit(1)

    num_workers = len(free_gpus)
    force_regenerate = os.environ.get("MIS_FORCE_REGENERATE_IMAGES", "").strip() == "1"
    n = len(samples)
    chunk_size = (n + num_workers - 1) // num_workers
    logger.info(f"Splitting {n} samples across {num_workers} GPUs (~{chunk_size} each)")

    chunk_files = []
    chunk_sizes = []
    for i in range(num_workers):
        chunk = samples[i * chunk_size : (i + 1) * chunk_size]
        cf = OUTPUT_DIR / f"_p3img_chunk_{i}.jsonl"
        save_jsonl(chunk, cf)
        chunk_files.append(cf)
        chunk_sizes.append(len(chunk))

    procs = []
    for i, gpu_id in enumerate(free_gpus[:num_workers]):
        chunk_out = OUTPUT_DIR / f"method_a_images_gpu{gpu_id}.jsonl"
        code = f"""
import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu_id}'
os.environ['HF_HOME'] = '/mnt2/xuran_hdd/cache'
sys.path.insert(0, '{PROJ}')
import logging
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [GPU{gpu_id}] %(levelname)s %(message)s')

from src.common.utils import load_jsonl, save_jsonl
from src.path2_prompt_decompose.acquire_images import acquire_images_for_sample
from pathlib import Path

samples = load_jsonl('{chunk_files[i]}')
output_dir = Path('{OUTPUT_DIR}')
results = []

for j, sample in enumerate(samples):
    sample_id = sample.get('sample_id_global', j)
    result = acquire_images_for_sample(
        sample,
        sample_id,
        output_dir,
        prefer_generation=True,
        force_regenerate={force_regenerate},
    )
    if result:
        results.append(result)
    if (j + 1) % 50 == 0:
        save_jsonl(results, '{chunk_out}')
        print(f'GPU {gpu_id}: {{j+1}}/{{len(samples)}} done, {{len(results)}} succeeded')

save_jsonl(results, '{chunk_out}')
print(f'GPU {gpu_id}: DONE — {{len(results)}}/{{len(samples)}} succeeded')
"""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["HF_HOME"] = "/mnt2/xuran_hdd/cache"
        env["PYTHONPATH"] = str(PROJ)
        proc = subprocess.Popen([PYTHON, "-c", code], env=env, cwd=str(PROJ))
        procs.append((proc, chunk_out, gpu_id))
        logger.info(f"Started GPU {gpu_id} PID={proc.pid} ({chunk_sizes[i]} samples)")

    worker_failures = []
    for proc, _, gpu_id in procs:
        proc.wait()
        if proc.returncode != 0:
            logger.warning(f"GPU {gpu_id} subprocess exited with code {proc.returncode}")
            worker_failures.append((gpu_id, proc.returncode))

    method_a_results = []
    for _, chunk_out, _ in procs:
        if chunk_out.exists():
            method_a_results.extend(load_jsonl(chunk_out))

    method_a_results.sort(key=lambda x: x.get("sample_id_global", 0))
    save_jsonl(method_a_results, METHOD_A_WITH_IMAGES)
    if worker_failures:
        logger.error("Path 3 image acquisition incomplete because workers failed: %s", worker_failures)
        sys.exit(1)
    logger.info(f"Method A with images: {len(method_a_results)}/{n}")

    for cf in chunk_files:
        cf.unlink(missing_ok=True)

    method_b_results = load_jsonl(METHOD_B_FILE) if METHOD_B_FILE.exists() else []
    all_results = method_a_results + method_b_results
    save_jsonl(all_results, MERGED_OUTPUT)
    finish_step(
        OUTPUT_DIR,
        "step5_method_a_acquire_images",
        expected_outputs=[METHOD_A_WITH_IMAGES, MERGED_OUTPUT],
        metadata={"records": len(method_a_results), "merged_records": len(all_results)},
    )

    status_path = write_path3_status(
        OUTPUT_DIR,
        method_a_text_only=len(samples),
        method_b_with_images=len(method_b_results),
        merged_total=len(all_results),
        method_a_with_images=len(method_a_results),
    )

    print("\n" + "=" * 80)
    print("✓ PATH 3 IMAGE ACQUISITION COMPLETE")
    print(f"  Method A: {len(method_a_results)}/{n} acquired images")
    print(f"  Method B: {len(method_b_results)} (already had images)")
    print(f"  Merged:   {len(all_results)} → {MERGED_OUTPUT}")
    print(f"  Status:   {status_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
