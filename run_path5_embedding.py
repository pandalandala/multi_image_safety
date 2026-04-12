#!/usr/bin/env python3
"""
Path 5: Image pair mining + prompt generation
  Step 1: Acquire benign-adjacent images per harm category (external retrieval
          or configurable T2I fallback)
  Step 2: LLM cross-image pairing — find harmful combinations + generate text prompts
          (replaces broken CLIP harm-vector pairing; runs in isolated subprocess)

Runtime estimate: ~1-2 hours
GPUs: use all 4 visible GPUs for T2I fallback; vLLM runs tp=4 across the same 4 GPUs
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
import shutil

os.environ.setdefault("HF_HOME", "/mnt2/xuran_hdd/cache")
sys.path.insert(0, str(Path(__file__).parent))

from src.common.utils import (
    apply_gpu_runtime_profile,
    clear_step_state,
    setup_logging,
    load_config,
    load_jsonl,
    save_jsonl,
    DATA_DIR,
    env_flag_is_true,
    finish_step,
    get_effective_tensor_parallel_size,
    is_step_complete,
    jsonl_record_count,
    log_gpu_runtime_profile,
    start_step,
)
from src.common.image_generation import should_force_regenerate_images

setup_logging()
logger = logging.getLogger(__name__)

PROJ = Path(__file__).parent
PYTHON = sys.executable
OUTPUT_DIR = DATA_DIR / "raw" / "path5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if "--clean" in sys.argv:
    from src.common.utils import clear_all_step_states
    clear_all_step_states(OUTPUT_DIR)
    sys.argv.remove("--clean")

MAX_PATH5_CANDIDATE_PAIRS = 10000
force_rerun = env_flag_is_true("MIS_FORCE_RERUN_COMPLETED_STEPS")


def step_complete(step_name: str, output_file: Path, *, min_records: int = 1, validator=None) -> bool:
    """Check whether a Path 5 step has an explicit completion marker."""
    return (not force_rerun) and is_step_complete(
        OUTPUT_DIR,
        step_name,
        expected_outputs=[output_file],
        validator=validator or (lambda: jsonl_record_count(output_file) >= min_records),
    )


def _normalize_crawled_infos(items: list[dict]) -> list[dict]:
    """Ensure each crawled image record has a usable text description."""
    normalized = []
    for item in items:
        info = dict(item)
        description = (
            info.get("description", "").strip()
            or info.get("caption", "").strip()
            or info.get("query", "").strip()
        )
        info["description"] = description
        normalized.append(info)
    return normalized

# ── Step 1: Acquire images per harm category ─────────────────────────────────
logger.info("=" * 60)
logger.info("PATH 5 STEP 1: Acquire images per harm category")
logger.info("=" * 60)

config = load_config()
laion_enabled = config.get("laion", {}).get("enabled", False)
refresh_crawl = os.environ.get("PATH5_REFRESH_CRAWL", "").strip() == "1"
force_regenerate_generated = should_force_regenerate_images()

crawled_file = OUTPUT_DIR / "crawled_image_info.jsonl"
crawled = []

existing_count = 0
if step_complete(
    "step1_acquire_images",
    crawled_file,
    validator=lambda: sum(
        1 for item in load_jsonl(crawled_file)
        if Path(item.get("path", "")).exists()
    ) >= 10,
):
    cached = _normalize_crawled_infos(load_jsonl(crawled_file))
    crawled = [c for c in cached if Path(c.get("path", "")).exists()]
    logger.info("Skipping Step 1; completion marker found")
elif crawled_file.exists():
    cached = _normalize_crawled_infos(load_jsonl(crawled_file))
    existing_count = sum(1 for c in cached if Path(c.get("path", "")).exists())
    only_generated_cache = bool(cached) and all(
        c.get("dataset", "") == "t2i_generated" for c in cached
    )
    should_reuse_cache = existing_count > 0 and not refresh_crawl and not force_rerun
    if force_regenerate_generated and only_generated_cache:
        for item in cached:
            path = Path(item.get("path", ""))
            if path.exists():
                path.unlink(missing_ok=True)
        generated_dir = OUTPUT_DIR / "generated_images"
        if generated_dir.exists():
            shutil.rmtree(generated_dir, ignore_errors=True)
        crawled_file.unlink(missing_ok=True)
        should_reuse_cache = False
    if should_reuse_cache:
        logger.info(f"Reusing {existing_count} existing crawled images")
        crawled = [c for c in cached if Path(c.get("path", "")).exists()]
        save_jsonl(crawled, crawled_file)

if not crawled and laion_enabled:
    logger.info("External retrieval enabled — crawling configured image backends")
    from src.path5_embedding_pair.crawl_laion import run as run_crawl
    crawled = _normalize_crawled_infos(
        run_crawl(output_dir=OUTPUT_DIR, max_per_category=500)
    )
    if crawled:
        save_jsonl(crawled, crawled_file)
    logger.info(f"Crawled {len(crawled)} images total")

if not crawled:
    if not laion_enabled:
        image_profile = apply_gpu_runtime_profile(
            path_name="path5",
            task_type="image",
            preferred_gpu_count=4,
        )
        log_gpu_runtime_profile(logger, image_profile, "Path 5 Step 1")
        logger.info("Refreshing T2I fallback pool to match the current query list")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = image_profile["selected_gpu_csv"]
        env["HF_HOME"] = "/mnt2/xuran_hdd/cache"
        env["PYTHONPATH"] = str(PROJ)

        sdxl_code = f"""
import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = '{image_profile["selected_gpu_csv"]}'
os.environ['HF_HOME'] = '/mnt2/xuran_hdd/cache'
sys.path.insert(0, '{PROJ}')
from src.common.utils import setup_logging, save_jsonl
from src.path3_dataset_expand.expand import generate_images_from_queries
setup_logging()

paths, sources, infos = generate_images_from_queries('{OUTPUT_DIR}', max_per_category=40)
save_jsonl(infos, '{crawled_file}')
print(f'T2I generated {{len(infos)}} images')
"""
        rc = subprocess.run([PYTHON, "-c", sdxl_code], env=env, cwd=str(PROJ)).returncode
        if rc == 0 and crawled_file.exists():
            crawled = _normalize_crawled_infos(load_jsonl(crawled_file))
            save_jsonl(crawled, crawled_file)
        else:
            logger.error("T2I generation subprocess failed")

logger.info(f"Step 1 complete: {len(crawled)} images")
if len(crawled) >= 10 and not step_complete("step1_acquire_images", crawled_file):
    finish_step(
        OUTPUT_DIR,
        "step1_acquire_images",
        expected_outputs=[crawled_file],
        metadata={"records": len(crawled)},
    )

if len(crawled) < 10:
    print("\n" + "=" * 80)
    print(f"✗ PATH 5 ABORTED: only {len(crawled)} images (need at least 10)")
    print("=" * 80)
    sys.exit(1)

# ── Step 2: LLM cross-image pairing (subprocess) ────────────────────────────
logger.info("=" * 60)
logger.info("PATH 5 STEP 2: LLM cross-image pairing (subprocess)")
logger.info("=" * 60)

output_file = OUTPUT_DIR / "samples_with_prompts.jsonl"
if step_complete("step2_cross_pair_prompts", output_file):
    results = load_jsonl(output_file)
    logger.info("Skipping Step 2; completion marker found")
    candidate_pairs = []
    print("\n" + "=" * 80)
    print(f"✓ PATH 5 COMPLETE: {len(results)} samples with text prompts")
    print(f"  Images: {len(crawled)}")
    print("  Candidate pairs: skipped (reused existing output)")
    print(f"  Final samples: {len(results)}")
    print(f"  Output: {output_file}")
    print("=" * 80)
    sys.exit(0)
clear_step_state(
    OUTPUT_DIR,
    "step2_cross_pair_prompts",
    stale_paths=[output_file, OUTPUT_DIR / "_cross_pair_input.jsonl", OUTPUT_DIR / "_method_b_pairs.jsonl"],
)
start_step(OUTPUT_DIR, "step2_cross_pair_prompts")

# Generate candidate pairs (no GPU needed)
from src.path3_dataset_expand.cross_pair import (
    generate_candidate_pairs,
    prepare_cross_pair_prompts,
)

candidate_pairs = generate_candidate_pairs(crawled, max_total=MAX_PATH5_CANDIDATE_PAIRS)
prompts = prepare_cross_pair_prompts(candidate_pairs)
logger.info(f"Generated {len(candidate_pairs)} candidate pairs, {len(prompts)} prompts")

# Save input for subprocess
method_b_input = OUTPUT_DIR / "_cross_pair_input.jsonl"
prompt_data = []
prompt_idx = 0
for pair in candidate_pairs:
    desc1 = pair["info1"].get("description", "")
    desc2 = pair["info2"].get("description", "")
    if not desc1 or not desc2:
        continue
    from src.common.utils import is_english
    if not is_english(desc1) or not is_english(desc2):
        continue
    prompt_data.append({
        "prompt": prompts[prompt_idx] if prompt_idx < len(prompts) else "",
        "info1": pair["info1"],
        "info2": pair["info2"],
        "pairing_mode": pair["pairing_mode"],
        "category_hint": pair.get("category_hint", ""),
    })
    prompt_idx += 1

save_jsonl(prompt_data, method_b_input)

results = []

if prompt_data:
    llm_profile = apply_gpu_runtime_profile(
        path_name="path5",
        task_type="llm",
        preferred_gpu_count=4,
        requested_tensor_parallel_size=4,
    )
    log_gpu_runtime_profile(logger, llm_profile, "Path 5 Step 2")
    local_cfg = config["llm"]["local"]
    llm_tensor_parallel_size = llm_profile["tensor_parallel_size"]
    worker_script = PROJ / "run_path5_step2_worker.py"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = llm_profile["selected_gpu_csv"]
    env["HF_HOME"] = "/mnt2/xuran_hdd/cache"
    env["PYTHONPATH"] = str(PROJ)

    rc = subprocess.run(
        [
            PYTHON,
            str(worker_script),
            "--input-file",
            str(method_b_input),
            "--output-file",
            str(output_file),
            "--model-path",
            str(local_cfg["model_path"]),
            "--tensor-parallel-size",
            str(llm_tensor_parallel_size),
            "--max-model-len",
            str(llm_profile["max_model_len"]),
            "--gpu-memory-utilization",
            str(llm_profile["gpu_memory_utilization"]),
        ],
        env=env,
        cwd=str(PROJ),
    ).returncode
    if rc != 0:
        logger.error("LLM cross-pairing subprocess failed")
        raise SystemExit(rc)
    else:
        results = load_jsonl(output_file) if output_file.exists() else []
        logger.info(f"LLM cross-pairing: {len(results)} samples generated")
        finish_step(
            OUTPUT_DIR,
            "step2_cross_pair_prompts",
            expected_outputs=[output_file],
            metadata={"records": len(results), "candidate_pairs": len(candidate_pairs)},
        )
        # Cleanup temp
        method_b_input.unlink(missing_ok=True)
        (OUTPUT_DIR / "_method_b_pairs.jsonl").unlink(missing_ok=True)
else:
    logger.warning("No valid prompt data — skipping LLM step")

print("\n" + "=" * 80)
print(f"✓ PATH 5 COMPLETE: {len(results)} samples with text prompts")
print(f"  Images: {len(crawled)}")
print(f"  Candidate pairs: {len(candidate_pairs)}")
print(f"  Final samples: {len(results)}")
print(f"  Output: {output_file}")
print("=" * 80)
