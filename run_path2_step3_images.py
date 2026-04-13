#!/usr/bin/env python3
"""
Path 2 Step 3: Acquire images for decomposed prompts
Strategy: 4 parallel subprocesses (one per GPU) for configurable T2I + retrieval

Runtime estimate: ~2 hours (4× RTX A6000, 3989 samples × 2 images)
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

os.environ["HF_HOME"] = "/mnt2/xuran_hdd/cache"
token_path = Path(os.environ["HF_HOME"]) / "token"
if token_path.exists():
    os.environ.setdefault("HF_TOKEN", token_path.read_text(encoding="utf-8").strip())
sys.path.insert(0, str(Path(__file__).parent))

from src.common.utils import (
    DATA_DIR,
    apply_gpu_runtime_profile,
    get_visible_gpu_ids,
    load_jsonl,
    log_gpu_runtime_profile,
    save_jsonl,
    setup_logging,
)
from src.common.image_generation import ensure_t2i_model_cached
from src.common.utils import clear_step_state, finish_step, is_step_complete, jsonl_record_count, start_step

setup_logging()
logger = logging.getLogger(__name__)
IMAGE_PROFILE = apply_gpu_runtime_profile(
    path_name="path2",
    task_type="image",
    preferred_gpu_count=4,
)
log_gpu_runtime_profile(logger, IMAGE_PROFILE, "Path 2 Step 3")

PROJ = Path(__file__).parent
PYTHON = sys.executable
INPUT_FILE  = DATA_DIR / "raw" / "path2" / "decomposed_prompts.jsonl"
OUTPUT_DIR  = DATA_DIR / "raw" / "path2"
FINAL_OUTPUT = OUTPUT_DIR / "samples_with_images.jsonl"

if is_step_complete(
    OUTPUT_DIR,
    "step3_acquire_images",
    expected_outputs=[FINAL_OUTPUT],
    validator=lambda: jsonl_record_count(FINAL_OUTPUT) >= 1,
):
    print("\n" + "=" * 80)
    print(f"Step 3 already complete: {FINAL_OUTPUT}")
    print("=" * 80)
    sys.exit(0)

clear_step_state(
    OUTPUT_DIR,
    "step3_acquire_images",
    stale_paths=[
        FINAL_OUTPUT,
        *sorted(OUTPUT_DIR.glob("samples_with_images_gpu*.jsonl")),
        *sorted(OUTPUT_DIR.glob("_step3_chunk_*.jsonl")),
    ],
)
start_step(OUTPUT_DIR, "step3_acquire_images")

# ── Use all visible GPUs ──────────────────────────────────────────────────────
free_gpus = get_visible_gpu_ids(max_gpus=None)
logger.info(f"Using visible GPUs for image acquisition: {free_gpus}")
if not free_gpus:
    logger.error("No GPUs configured for image acquisition!")
    sys.exit(1)

logger.info("Pre-caching active T2I model before spawning GPU workers")
ensure_t2i_model_cached()

num_workers = len(free_gpus)
force_regenerate = os.environ.get("MIS_FORCE_REGENERATE_IMAGES", "").strip() == "1"

if force_regenerate:
    FINAL_OUTPUT.unlink(missing_ok=True)
    for gpu_id in free_gpus[:num_workers]:
        (OUTPUT_DIR / f"samples_with_images_gpu{gpu_id}.jsonl").unlink(missing_ok=True)

# ── Load + split samples ──────────────────────────────────────────────────────
samples = load_jsonl(INPUT_FILE)
for i, s in enumerate(samples):
    s["sample_id_global"] = i   # stable ID before splitting

n = len(samples)
chunk_size = (n + num_workers - 1) // num_workers
logger.info(f"Splitting {n} samples across {num_workers} GPUs (~{chunk_size} each)")

chunk_files = []
chunk_sizes = []
for i in range(num_workers):
    chunk = samples[i * chunk_size: (i + 1) * chunk_size]
    cf = OUTPUT_DIR / f"_step3_chunk_{i}.jsonl"
    save_jsonl(chunk, cf)
    chunk_files.append(cf)
    chunk_sizes.append(len(chunk))

# ── Launch one subprocess per GPU ─────────────────────────────────────────────
procs = []
for i, gpu_id in enumerate(free_gpus[:num_workers]):
    chunk_out = OUTPUT_DIR / f"samples_with_images_gpu{gpu_id}.jsonl"
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
    p = subprocess.Popen([PYTHON, "-c", code], env=env, cwd=str(PROJ))
    procs.append((p, chunk_out, gpu_id))
    logger.info(f"Started GPU {gpu_id} PID={p.pid} ({chunk_sizes[i]} samples)")

# Wait for all
worker_failures = []
for p, _, gpu_id in procs:
    p.wait()
    if p.returncode != 0:
        logger.warning(f"GPU {gpu_id} subprocess exited with code {p.returncode}")
        worker_failures.append((gpu_id, p.returncode))

# Merge + sort by sample_id_global
all_results = []
for _, chunk_out, _ in procs:
    if chunk_out.exists():
        chunk = load_jsonl(chunk_out)
        all_results.extend(chunk)

all_results.sort(key=lambda x: x.get("sample_id_global", 0))
save_jsonl(all_results, FINAL_OUTPUT)
if worker_failures:
    print(f"Step 3 incomplete because workers failed: {worker_failures}", file=sys.stderr)
    raise SystemExit(1)
finish_step(
    OUTPUT_DIR,
    "step3_acquire_images",
    expected_outputs=[FINAL_OUTPUT],
    metadata={"records": len(all_results), "input_records": n},
)

# Cleanup temp files
for cf in chunk_files:
    cf.unlink(missing_ok=True)

print("\n" + "=" * 80)
print(f"✓ STEP 3 COMPLETE: {len(all_results)}/{n} samples acquired images")
print(f"Output: {FINAL_OUTPUT}")
print("=" * 80)
