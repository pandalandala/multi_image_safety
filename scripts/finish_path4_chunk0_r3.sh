#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/mnt/hdd/xuran/multi_image_safety"
OUTPUT_DIR="$PROJECT_ROOT/data/raw/path4"

source /mnt/hdd/xuran/anaconda3/bin/activate mis_safety
source "$PROJECT_ROOT/scripts/_load_local_env.sh"

export HF_HOME="${HF_HOME:-/mnt2/xuran_hdd/cache}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

rm -rf \
  "$OUTPUT_DIR/worker_0_r4" \
  "$OUTPUT_DIR/worker_0_r5" \
  "$OUTPUT_DIR/worker_0_r6" \
  "$OUTPUT_DIR/_chunk0_r3_sub_0.jsonl" \
  "$OUTPUT_DIR/_chunk0_r3_sub_1.jsonl" \
  "$OUTPUT_DIR/_chunk0_r3_sub_2.jsonl" \
  "$OUTPUT_DIR/_chunk0_r3_out_gpu4.jsonl" \
  "$OUTPUT_DIR/_chunk0_r3_out_gpu5.jsonl" \
  "$OUTPUT_DIR/_chunk0_r3_out_gpu6.jsonl" \
  "$OUTPUT_DIR/samples_with_images_gpu0.jsonl"

cd "$PROJECT_ROOT"

python -u - <<'PY'
import os
import sys
import subprocess
from pathlib import Path

from src.common.utils import load_jsonl, save_jsonl, finish_step

proj = Path("/mnt/hdd/xuran/multi_image_safety")
out = proj / "data" / "raw" / "path4"
chunk0 = load_jsonl(out / "_chunk_0.jsonl")
gpus = [4, 5, 6]
sub_size = (len(chunk0) + len(gpus) - 1) // len(gpus)

sub_files = []
sub_outs = []
sub_procs = []

for idx, gpu in enumerate(gpus):
    sub = chunk0[idx * sub_size : (idx + 1) * sub_size]
    sub_file = out / f"_chunk0_r3_sub_{idx}.jsonl"
    sub_out = out / f"_chunk0_r3_out_gpu{gpu}.jsonl"
    worker_dir = out / f"worker_0_r{gpu}"
    log_file = proj / "logs" / f"path4_chunk0_r3_gpu{gpu}.log"

    save_jsonl(sub, sub_file)
    sub_files.append(sub_file)
    sub_outs.append(sub_out)

    code = f"""
from pathlib import Path
from src.common.utils import load_jsonl, save_jsonl
from src.path4_scenario.image_fetch import fetch_images_for_samples

samples = load_jsonl(r'{sub_file}')
worker_dir = Path(r'{worker_dir}')
worker_dir.mkdir(parents=True, exist_ok=True)

results = fetch_images_for_samples(
    samples,
    worker_dir,
    start_id={6000 + idx * sub_size},
    prefer_retrieval=False,
)
save_jsonl(results, r'{sub_out}')
print('gpu {gpu} done', len(results), '/', len(samples), flush=True)
"""

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["MIS_GPU_CANDIDATES"] = str(gpu)
    env["HF_HOME"] = os.environ.get("HF_HOME", "/mnt2/xuran_hdd/cache")
    env["PYTHONPATH"] = str(proj)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    log_fp = open(log_file, "w")
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", code],
        cwd=str(proj),
        env=env,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
    )
    sub_procs.append((proc, gpu, log_file, log_fp))
    print(f"started gpu {gpu} pid={proc.pid} samples={len(sub)} log={log_file}", flush=True)

failures = []
for proc, gpu, log_file, log_fp in sub_procs:
    rc = proc.wait()
    log_fp.close()
    print(f"gpu {gpu} exited rc={rc}", flush=True)
    if rc != 0:
        failures.append((gpu, rc, str(log_file)))

if failures:
    raise RuntimeError(f"worker failures: {failures}")

chunk0_results = []
for sub_out in sub_outs:
    rows = load_jsonl(sub_out)
    print(f"{sub_out.name}: {len(rows)}", flush=True)
    chunk0_results.extend(rows)

chunk0_results.sort(key=lambda x: int(x.get("sample_id", -1)))
samples_gpu0 = out / "samples_with_images_gpu0.jsonl"
save_jsonl(chunk0_results, samples_gpu0)
print(f"chunk0 merged: {len(chunk0_results)} -> {samples_gpu0}", flush=True)

final_parts = [
    out / "samples_with_images_gpu0.jsonl",
    out / "samples_with_images_gpu1.jsonl",
    out / "samples_with_images_gpu2.jsonl",
    out / "samples_with_images_gpu3.jsonl",
]
all_results = []
for part in final_parts:
    rows = load_jsonl(part)
    print(f"{part.name}: {len(rows)}", flush=True)
    all_results.extend(rows)

all_results.sort(key=lambda x: int(x.get("sample_id", -1)))
final_output = out / "samples_with_images.jsonl"
save_jsonl(all_results, final_output)

input_records = len(load_jsonl(out / "intent_injected_samples.jsonl"))
finish_step(
    out,
    "step3_fetch_images",
    expected_outputs=[final_output],
    metadata={"records": len(all_results), "input_records": input_records},
)

for cf in list(out.glob("_chunk_*.jsonl")) + sub_files + sub_outs:
    cf.unlink(missing_ok=True)

print(f"finalized path4: {len(all_results)}/{input_records}", flush=True)
print(final_output, flush=True)
PY
