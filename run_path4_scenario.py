#!/usr/bin/env python3
"""
Path 4: Scenario-based image pair generation
  Step 1: LLM generates everyday scene-activity pairs (40 categories × 25 activities)
  Step 2: LLM injects dangerous intent into scenes (17 safety modes × 12 harm categories)
  Step 3: Fetch images via LAION retrieval + T2I (4 parallel subprocesses, 1 per GPU)

Runtime estimate: ~2-3 hours
GPUs: 0,1,2,3 — each step runs in an isolated subprocess so GPU memory is
fully released before the next step begins.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt2/xuran_hdd/cache")
sys.path.insert(0, str(Path(__file__).parent))

from src.common.utils import (
    clear_step_state,
    setup_logging,
    load_config,
    load_jsonl,
    save_jsonl,
    DATA_DIR,
    env_flag_is_true,
    finish_step,
    get_visible_gpu_csv,
    get_visible_gpu_ids,
    is_step_complete,
    jsonl_record_count,
    start_step,
)

setup_logging()
logger = logging.getLogger(__name__)

PROJ = Path(__file__).parent
PYTHON = sys.executable
OUTPUT_DIR = DATA_DIR / "raw" / "path4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if "--clean" in sys.argv:
    from src.common.utils import clear_all_step_states
    clear_all_step_states(OUTPUT_DIR)
    sys.argv.remove("--clean")

FINAL_OUTPUT = OUTPUT_DIR / "samples_with_images.jsonl"
SCENES_OUTPUT = OUTPUT_DIR / "generated_scenes.jsonl"
INTENT_OUTPUT = OUTPUT_DIR / "intent_injected_samples.jsonl"
force_rerun = env_flag_is_true("MIS_FORCE_RERUN_COMPLETED_STEPS")
FREE_GPUS = get_visible_gpu_ids()
ALL_GPU_IDS = get_visible_gpu_csv()


def step_complete(step_name: str, output_file: Path, *, min_records: int = 1) -> bool:
    """Check whether a Path 4 step has an explicit completion marker."""
    return (not force_rerun) and is_step_complete(
        OUTPUT_DIR,
        step_name,
        expected_outputs=[output_file],
        validator=lambda: jsonl_record_count(output_file) >= min_records,
    )


def run_subprocess(code: str, gpu_ids: str | None = None) -> int:
    """Run Python code in an isolated subprocess. Returns exit code."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids or ALL_GPU_IDS
    env["HF_HOME"] = "/mnt2/xuran_hdd/cache"
    env["PYTHONPATH"] = str(PROJ)
    result = subprocess.run([PYTHON, "-c", code], env=env, cwd=str(PROJ))
    return result.returncode


# ── Step 1: Scene generation ─────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("PATH 4 STEP 1: LLM scene generation")
logger.info("=" * 60)

if step_complete("step1_generate_scenes", SCENES_OUTPUT):
    logger.info("Skipping Step 1; completion marker found")
else:
    clear_step_state(OUTPUT_DIR, "step1_generate_scenes", stale_paths=[SCENES_OUTPUT])
    start_step(OUTPUT_DIR, "step1_generate_scenes")
    rc = run_subprocess(f"""
import sys; sys.path.insert(0, '{PROJ}')
from src.common.utils import setup_logging; setup_logging()
from src.path4_scenario.scene_gen import run
scenes = run(output_dir='{OUTPUT_DIR}', use_api=False)
print(f'Generated {{len(scenes)}} scene-activity pairs')
""")
    if rc != 0:
        raise RuntimeError("Step 1 (scene_gen) failed")
    finish_step(
        OUTPUT_DIR,
        "step1_generate_scenes",
        expected_outputs=[SCENES_OUTPUT],
        metadata={"records": jsonl_record_count(SCENES_OUTPUT)},
    )
logger.info(f"Scene generation complete → {SCENES_OUTPUT}")

# ── Step 2: Intent injection ──────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("PATH 4 STEP 2: LLM intent injection")
logger.info("=" * 60)

if step_complete("step2_inject_intent", INTENT_OUTPUT):
    logger.info("Skipping Step 2; completion marker found")
else:
    clear_step_state(OUTPUT_DIR, "step2_inject_intent", stale_paths=[INTENT_OUTPUT])
    start_step(OUTPUT_DIR, "step2_inject_intent")
    rc = run_subprocess(f"""
import sys; sys.path.insert(0, '{PROJ}')
from src.common.utils import setup_logging; setup_logging()
from src.path4_scenario.intent_inject import run
results = run(
    input_file='{SCENES_OUTPUT}',
    output_dir='{OUTPUT_DIR}',
    use_api=False,
    max_combinations=3000,
)
print(f'Intent injection complete: {{len(results)}} samples')
""")
    if rc != 0:
        raise RuntimeError("Step 2 (intent_inject) failed")
    finish_step(
        OUTPUT_DIR,
        "step2_inject_intent",
        expected_outputs=[INTENT_OUTPUT],
        metadata={"records": jsonl_record_count(INTENT_OUTPUT)},
    )
logger.info(f"Intent injection complete → {INTENT_OUTPUT}")

# ── Step 3: Image fetch (4 parallel subprocesses, 1 per GPU) ─────────────────
logger.info("=" * 60)
logger.info("PATH 4 STEP 3: Image fetch (4-GPU parallel subprocesses)")
logger.info("=" * 60)

NUM_WORKERS = len(FREE_GPUS)

if force_rerun or os.environ.get("MIS_FORCE_REGENERATE_IMAGES", "").strip() == "1":
    clear_step_state(
        OUTPUT_DIR,
        "step3_fetch_images",
        stale_paths=[
            FINAL_OUTPUT,
            *[OUTPUT_DIR / f"samples_with_images_gpu{gpu_id}.jsonl" for gpu_id in FREE_GPUS],
            *sorted(OUTPUT_DIR.glob("_chunk_*.jsonl")),
        ],
    )

if step_complete("step3_fetch_images", FINAL_OUTPUT) and os.environ.get("MIS_FORCE_REGENERATE_IMAGES", "").strip() != "1":
    all_results = load_jsonl(FINAL_OUTPUT)
    logger.info(f"Skipping Step 3; completion marker found: {len(all_results)} samples")
else:
    clear_step_state(
        OUTPUT_DIR,
        "step3_fetch_images",
        stale_paths=[
            FINAL_OUTPUT,
            *[OUTPUT_DIR / f"samples_with_images_gpu{gpu_id}.jsonl" for gpu_id in FREE_GPUS],
            *sorted(OUTPUT_DIR.glob("_chunk_*.jsonl")),
        ],
    )
    start_step(OUTPUT_DIR, "step3_fetch_images")
    samples = load_jsonl(INTENT_OUTPUT)
    n = len(samples)
    chunk_size = (n + NUM_WORKERS - 1) // NUM_WORKERS
    logger.info(f"Splitting {n} samples into {NUM_WORKERS} chunks of ~{chunk_size}")

    chunk_files = []
    for i in range(NUM_WORKERS):
        chunk = samples[i * chunk_size: (i + 1) * chunk_size]
        chunk_file = OUTPUT_DIR / f"_chunk_{i}.jsonl"
        save_jsonl(chunk, chunk_file)
        chunk_files.append(chunk_file)

    procs = []
    for i, gpu_id in enumerate(FREE_GPUS):
        worker_dir = OUTPUT_DIR / f"worker_{gpu_id}"
        chunk_out = OUTPUT_DIR / f"samples_with_images_gpu{gpu_id}.jsonl"
        code = f"""
import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu_id}'
os.environ['HF_HOME'] = '/mnt2/xuran_hdd/cache'
sys.path.insert(0, '{PROJ}')
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [GPU{gpu_id}] %(levelname)s %(message)s')

from src.common.utils import load_jsonl, save_jsonl
from src.path4_scenario.image_fetch import fetch_images_for_samples
from pathlib import Path

samples = load_jsonl('{chunk_files[i]}')
worker_dir = Path('{worker_dir}')
worker_dir.mkdir(parents=True, exist_ok=True)

results = fetch_images_for_samples(
    samples,
    worker_dir,
    start_id={6000 + i * chunk_size},
    prefer_retrieval=False,
)
save_jsonl(results, '{chunk_out}')
print(f'GPU {gpu_id}: {{len(results)}}/{{len(samples)}} images fetched')
"""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["HF_HOME"] = "/mnt2/xuran_hdd/cache"
        env["PYTHONPATH"] = str(PROJ)
        p = subprocess.Popen([PYTHON, "-c", code], env=env, cwd=str(PROJ))
        procs.append((p, chunk_out, gpu_id))
        logger.info(f"Started GPU {gpu_id} fetch subprocess PID={p.pid} (~{chunk_size} samples)")

    worker_failures = []
    for p, _, gpu_id in procs:
        p.wait()
        if p.returncode != 0:
            logger.warning(f"GPU {gpu_id} fetch subprocess exited with code {p.returncode}")
            worker_failures.append((gpu_id, p.returncode))

    all_results = []
    for _, chunk_out, _ in procs:
        if chunk_out.exists():
            chunk = load_jsonl(chunk_out)
            all_results.extend(chunk)

    save_jsonl(all_results, FINAL_OUTPUT)
    if worker_failures:
        raise RuntimeError(f"Path 4 Step 3 incomplete because workers failed: {worker_failures}")
    finish_step(
        OUTPUT_DIR,
        "step3_fetch_images",
        expected_outputs=[FINAL_OUTPUT],
        metadata={"records": len(all_results), "input_records": n},
    )

    for cf in chunk_files:
        cf.unlink(missing_ok=True)

print("\n" + "=" * 80)
print(f"✓ PATH 4 COMPLETE: {len(all_results)} samples with images")
print(f"Output: {FINAL_OUTPUT}")
print("=" * 80)
