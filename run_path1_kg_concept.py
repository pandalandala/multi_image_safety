#!/usr/bin/env python3
"""
Path 1: KG Concept Pair Mining
  Step 1: Mine concept pairs from Numberbatch + LLM generation
  Step 2: CLIP-based pair filtering
  Step 3: Generate benign images for each concept (SD3.5 Large Turbo, 4 visible GPUs)
  Step 4: LLM generates connecting text prompts (vLLM on 4 visible GPUs)

Runtime estimate: ~3-4 hours
"""

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt2/xuran_hdd/cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
sys.path.insert(0, str(Path(__file__).parent))

from src.common.utils import (
    clear_step_state,
    get_effective_tensor_parallel_size,
    setup_logging,
    load_config,
    load_jsonl,
    save_jsonl,
    DATA_DIR,
    env_flag_is_true,
    finish_step,
    is_step_complete,
    jsonl_record_count,
    start_step,
)

setup_logging()
logger = logging.getLogger(__name__)

PROJ = Path(__file__).parent
PYTHON = sys.executable
OUTPUT_DIR = DATA_DIR / "raw" / "path1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_gpu_ids(default: str = "0,1,2,3") -> list[int]:
    """Parse the visible GPU list from the environment."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", default)
    gpu_ids = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not gpu_ids:
        raise RuntimeError("No GPUs configured for Path 1")
    return gpu_ids


def get_llm_runtime_limits() -> tuple[dict, int, float, int]:
    """Clamp Path 1 vLLM settings to stable defaults unless overridden."""
    config = load_config()
    local_cfg = config["llm"]["local"]
    max_model_len = min(
        int(local_cfg.get("max_model_len", 8192)),
        int(os.environ.get("MIS_PATH1_MAX_MODEL_LEN", "4096")),
    )
    gpu_memory_utilization = min(
        float(local_cfg.get("gpu_memory_utilization", 0.9)),
        float(os.environ.get("MIS_PATH1_GPU_MEMORY_UTILIZATION", "0.68")),
    )
    tensor_parallel_size = get_effective_tensor_parallel_size(
        local_cfg.get("tensor_parallel_size")
    )
    return local_cfg, max_model_len, gpu_memory_utilization, tensor_parallel_size


def run_subprocess(code: str, gpu_ids: str | None = None) -> int:
    """Run Python code in an isolated subprocess."""
    env = os.environ.copy()
    if gpu_ids is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    env["HF_HOME"] = os.environ["HF_HOME"]
    env["PYTHONPATH"] = str(PROJ)
    result = subprocess.run([PYTHON, "-c", code], env=env, cwd=str(PROJ))
    return result.returncode


visible_gpus = get_gpu_ids()
all_gpu_ids = ",".join(str(gpu_id) for gpu_id in visible_gpus)
single_gpu_id = str(visible_gpus[0])
local_cfg, llm_max_model_len, llm_gpu_memory_utilization, llm_tensor_parallel_size = get_llm_runtime_limits()
force_rerun = env_flag_is_true("MIS_FORCE_RERUN_COMPLETED_STEPS")


def step_complete(step_name: str, output_file: Path, *, min_records: int = 1) -> bool:
    """Check whether a Path 1 step has an explicit completion marker."""
    return (not force_rerun) and is_step_complete(
        OUTPUT_DIR,
        step_name,
        expected_outputs=[output_file],
        validator=lambda: jsonl_record_count(output_file) >= min_records,
    )


# ── Step 1: Mine concept pairs ──────────────────────────────────────────────
logger.info("=" * 60)
logger.info("PATH 1 STEP 1a: Numberbatch concept pair mining")
logger.info("=" * 60)

nb_pairs_file = OUTPUT_DIR / "numberbatch_pairs.jsonl"
llm_pairs_file = OUTPUT_DIR / "llm_pairs.jsonl"
all_mined_file = OUTPUT_DIR / "all_mined_pairs.jsonl"

# Step 1a: Numberbatch mining (CPU only, no GPU needed)
if step_complete("step1a_numberbatch_pairs", nb_pairs_file):
    logger.info("Skipping Step 1a; completion marker found")
else:
    clear_step_state(OUTPUT_DIR, "step1a_numberbatch_pairs", stale_paths=[nb_pairs_file])
    start_step(OUTPUT_DIR, "step1a_numberbatch_pairs")
    rc = run_subprocess(f"""
import sys; sys.path.insert(0, '{PROJ}')
from src.common.utils import setup_logging, save_jsonl, get_category_harm_descriptions
from src.path1_kg_concept.concept_mine import load_numberbatch, mine_numberbatch_pairs
setup_logging()

descriptions = get_category_harm_descriptions()
embeddings = load_numberbatch()
pairs = mine_numberbatch_pairs(embeddings, descriptions, max_pairs_per_category=200)
save_jsonl(pairs, '{nb_pairs_file}')
print(f'Numberbatch mining: {{len(pairs)}} pairs found')
""", gpu_ids="")
    if rc != 0:
        logger.warning("Numberbatch mining failed — continuing with LLM pairs only")
    elif jsonl_record_count(nb_pairs_file) >= 1:
        finish_step(
            OUTPUT_DIR,
            "step1a_numberbatch_pairs",
            expected_outputs=[nb_pairs_file],
            metadata={"records": jsonl_record_count(nb_pairs_file)},
        )

# Step 1b: LLM concept pair generation
logger.info("=" * 60)
logger.info("PATH 1 STEP 1b: LLM concept pair generation (vLLM)")
logger.info("=" * 60)

if step_complete("step1b_llm_pairs", llm_pairs_file):
    logger.info("Skipping Step 1b; completion marker found")
else:
    clear_step_state(OUTPUT_DIR, "step1b_llm_pairs", stale_paths=[llm_pairs_file])
    start_step(OUTPUT_DIR, "step1b_llm_pairs")
    rc = run_subprocess(f"""
import sys, json, re
sys.path.insert(0, '{PROJ}')
from src.common.utils import setup_logging, save_jsonl, get_category_harm_descriptions
from src.path1_kg_concept.concept_mine import mine_llm_pairs, parse_llm_pairs
from vllm import LLM, SamplingParams
setup_logging()

descriptions = get_category_harm_descriptions()
prompt_items = mine_llm_pairs(descriptions, pairs_per_category=60)
conversations = [[{{"role": "user", "content": item["prompt"]}}] for item in prompt_items]

print(f'Loading LLM for {{len(conversations)}} concept mining prompts...')
llm = LLM(
    model='{local_cfg["model_path"]}',
    tensor_parallel_size={llm_tensor_parallel_size},
    trust_remote_code=True,
    max_model_len={llm_max_model_len},
    enforce_eager=True,
    gpu_memory_utilization={llm_gpu_memory_utilization},
    disable_custom_all_reduce=True,
)
sampling_params = SamplingParams(temperature=0.8, max_tokens=4096, top_p=0.95)
outputs = llm.chat(conversations, sampling_params, chat_template_kwargs={{"enable_thinking": False}})

all_pairs = []
for item, output in zip(prompt_items, outputs):
    text = output.outputs[0].text
    pairs = parse_llm_pairs(text, item["category"])
    all_pairs.extend(pairs)
    print(f'  {{item["category"]}}: {{len(pairs)}} pairs parsed')

save_jsonl(all_pairs, '{llm_pairs_file}')
print(f'LLM mining: {{len(all_pairs)}} total pairs')
""", gpu_ids=all_gpu_ids)
    if rc != 0:
        logger.error("LLM concept mining failed")
        if not nb_pairs_file.exists():
            logger.error("No pairs available — aborting Path 1")
            sys.exit(1)
    elif jsonl_record_count(llm_pairs_file) >= 1:
        finish_step(
            OUTPUT_DIR,
            "step1b_llm_pairs",
            expected_outputs=[llm_pairs_file],
            metadata={"records": jsonl_record_count(llm_pairs_file)},
        )

# Merge all mined pairs
nb_pairs = load_jsonl(nb_pairs_file) if nb_pairs_file.exists() else []
llm_pairs = load_jsonl(llm_pairs_file) if llm_pairs_file.exists() else []
all_mined = nb_pairs + llm_pairs
save_jsonl(all_mined, all_mined_file)
logger.info(f"Total mined pairs: {len(all_mined)} (Numberbatch: {len(nb_pairs)}, LLM: {len(llm_pairs)})")

if len(all_mined) < 10:
    logger.error("Too few pairs mined — aborting Path 1")
    sys.exit(1)

# ── Step 2: CLIP-based filtering ────────────────────────────────────────────
logger.info("=" * 60)
logger.info("PATH 1 STEP 2: CLIP pair filtering")
logger.info("=" * 60)

filtered_file = OUTPUT_DIR / "filtered_pairs.jsonl"
if step_complete("step2_filter_pairs", filtered_file):
    logger.info("Skipping Step 2; completion marker found")
else:
    clear_step_state(OUTPUT_DIR, "step2_filter_pairs", stale_paths=[filtered_file])
    start_step(OUTPUT_DIR, "step2_filter_pairs")
    rc = run_subprocess(f"""
import sys; sys.path.insert(0, '{PROJ}')
from src.common.utils import setup_logging, load_jsonl, save_jsonl, load_config
from src.path1_kg_concept.pair_filter import filter_pairs_clip, rank_pairs_by_covertness
setup_logging()

config = load_config()
clip_cfg = config.get("clip", {{}})
pairs = load_jsonl('{all_mined_file}')
filtered = filter_pairs_clip(
    pairs,
    theta_safe=clip_cfg.get("theta_safe", 0.25),
    theta_harm=clip_cfg.get("theta_harm", 0.35),
)
ranked = rank_pairs_by_covertness(filtered)
save_jsonl(ranked, '{filtered_file}')
print(f'CLIP filter: {{len(ranked)}} / {{len(pairs)}} pairs passed')
""", gpu_ids=single_gpu_id)
    if rc != 0:
        logger.warning("CLIP filtering failed — using all mined pairs")
        filtered_file = all_mined_file
    elif jsonl_record_count(filtered_file) >= 1:
        finish_step(
            OUTPUT_DIR,
            "step2_filter_pairs",
            expected_outputs=[filtered_file],
            metadata={"records": jsonl_record_count(filtered_file)},
        )

filtered_pairs = load_jsonl(filtered_file)
logger.info(f"Filtered pairs: {len(filtered_pairs)}")

# ── Step 3: Generate images ─────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("PATH 1 STEP 3: Image generation (4-GPU parallel)")
logger.info("=" * 60)

images_file = OUTPUT_DIR / "pairs_with_images.jsonl"

if step_complete("step3_generate_images", images_file) and os.environ.get("MIS_FORCE_REGENERATE_IMAGES", "").strip() != "1":
    all_results = load_jsonl(images_file)
    logger.info(f"Skipping Step 3; completion marker found: {len(all_results)} pairs")
else:
    clear_step_state(
        OUTPUT_DIR,
        "step3_generate_images",
        stale_paths=[
            images_file,
            *sorted(OUTPUT_DIR.glob("pairs_with_images_gpu*.jsonl")),
            *sorted(OUTPUT_DIR.glob("_img_chunk_*.jsonl")),
        ],
    )
    start_step(OUTPUT_DIR, "step3_generate_images")
    free_gpus = visible_gpus
    num_workers = len(free_gpus)
    n = len(filtered_pairs)
    chunk_size = (n + num_workers - 1) // num_workers

    chunk_files = []
    for i in range(num_workers):
        chunk = filtered_pairs[i * chunk_size: (i + 1) * chunk_size]
        cf = OUTPUT_DIR / f"_img_chunk_{i}.jsonl"
        save_jsonl(chunk, cf)
        chunk_files.append(cf)

    procs = []
    for i, gpu_id in enumerate(free_gpus):
        chunk_out = OUTPUT_DIR / f"pairs_with_images_gpu{gpu_id}.jsonl"
        code = f"""
import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu_id}'
os.environ['HF_HOME'] = '/mnt2/xuran_hdd/cache'
sys.path.insert(0, '{PROJ}')
import logging
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [GPU{gpu_id}] %(levelname)s %(message)s')

from src.common.utils import load_jsonl, save_jsonl
from src.path1_kg_concept.image_gen import generate_concept_images
from pathlib import Path

pairs = load_jsonl('{chunk_files[i]}')
results = generate_concept_images(
    pairs,
    Path('{OUTPUT_DIR}'),
    start_id={10000 + i * chunk_size},
)
save_jsonl(results, '{chunk_out}')
print(f'GPU {gpu_id}: {{len(results)}}/{{len(pairs)}} images generated')
"""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["HF_HOME"] = os.environ["HF_HOME"]
        env["PYTHONPATH"] = str(PROJ)
        p = subprocess.Popen([PYTHON, "-c", code], env=env, cwd=str(PROJ))
        procs.append((p, chunk_out, gpu_id))
        logger.info(f"Started GPU {gpu_id} PID={p.pid} (~{chunk_size} pairs)")

    worker_failures = []
    for p, _, gpu_id in procs:
        p.wait()
        if p.returncode != 0:
            logger.warning(f"GPU {gpu_id} image gen subprocess exited with code {p.returncode}")
            worker_failures.append((gpu_id, p.returncode))

    all_results = []
    for _, chunk_out, _ in procs:
        if chunk_out.exists():
            all_results.extend(load_jsonl(chunk_out))

    save_jsonl(all_results, images_file)
    if worker_failures:
        logger.error("Path 1 Step 3 incomplete because workers failed: %s", worker_failures)
        sys.exit(1)
    finish_step(
        OUTPUT_DIR,
        "step3_generate_images",
        expected_outputs=[images_file],
        metadata={"records": len(all_results), "input_records": len(filtered_pairs)},
    )

    # Cleanup temp files
    for cf in chunk_files:
        cf.unlink(missing_ok=True)

logger.info(f"Pairs with images: {len(all_results)}")

# ── Step 4: Generate connecting text prompts ─────────────────────────────────
logger.info("=" * 60)
logger.info("PATH 1 STEP 4: LLM connecting prompt generation (vLLM)")
logger.info("=" * 60)

final_output = OUTPUT_DIR / "validated_samples.jsonl"

if step_complete("step4_generate_prompts", final_output):
    logger.info("Skipping Step 4; completion marker found")
    final_results = load_jsonl(final_output)
else:
    clear_step_state(OUTPUT_DIR, "step4_generate_prompts", stale_paths=[final_output])
    start_step(OUTPUT_DIR, "step4_generate_prompts")
    rc = run_subprocess(f"""
import sys, json, re
sys.path.insert(0, '{PROJ}')
from src.common.utils import setup_logging, load_jsonl, save_jsonl
from src.path1_kg_concept.prompt_create import prepare_prompt_gen_prompts, parse_prompt_gen_response
from src.common.schema import Pattern, SourcePath
from vllm import LLM, SamplingParams
setup_logging()

pairs = load_jsonl('{images_file}')
prompt_data = prepare_prompt_gen_prompts(pairs)
conversations = [[{{"role": "user", "content": item["prompt"]}}] for item in prompt_data]

print(f'Loading LLM for {{len(conversations)}} prompt generation requests...')
llm = LLM(
    model='{local_cfg["model_path"]}',
    tensor_parallel_size={llm_tensor_parallel_size},
    trust_remote_code=True,
    max_model_len={llm_max_model_len},
    enforce_eager=True,
    gpu_memory_utilization={llm_gpu_memory_utilization},
    disable_custom_all_reduce=True,
)
sampling_params = SamplingParams(temperature=0.7, max_tokens=1024, top_p=0.9)
outputs = llm.chat(conversations, sampling_params, chat_template_kwargs={{"enable_thinking": False}})

results = []
for item, output in zip(prompt_data, outputs):
    text = output.outputs[0].text
    parsed = parse_prompt_gen_response(text, item["pair"])
    if parsed:
        pair = item["pair"]
        results.append({{
            "image1_path": pair.get("image1_path", ""),
            "image2_path": pair.get("image2_path", ""),
            "image1_description": pair.get("image1_description", ""),
            "image2_description": pair.get("image2_description", ""),
            "concept1": pair.get("concept1", ""),
            "concept2": pair.get("concept2", ""),
            "category": pair.get("category", "CRIME"),
            "sub_category": parsed.get("sub_category", ""),
            "text_prompt": parsed["text_prompt"],
            "safety_response": parsed.get("safety_response", ""),
            "reasoning": pair.get("reasoning", ""),
            "pattern": Pattern.A.value,
            "source_path": SourcePath.PATH1.value,
            "clip_combined_sim": pair.get("clip_combined_sim", 0.0),
            "covertness_rank": pair.get("covertness_rank", 0.0),
        }})

save_jsonl(results, '{final_output}')
print(f'Path 1: {{len(results)}}/{{len(prompt_data)}} samples with text prompts')
""", gpu_ids=all_gpu_ids)

    if rc != 0:
        logger.error("Prompt generation failed")
        sys.exit(1)

    final_results = load_jsonl(final_output)
    finish_step(
        OUTPUT_DIR,
        "step4_generate_prompts",
        expected_outputs=[final_output],
        metadata={"records": len(final_results)},
    )
print("\n" + "=" * 80)
print(f"PATH 1 COMPLETE: {len(final_results)} samples")
print(f"  Numberbatch pairs: {len(nb_pairs)}")
print(f"  LLM pairs: {len(llm_pairs)}")
print(f"  After CLIP filter: {len(filtered_pairs)}")
print(f"  With images: {len(all_results)}")
print(f"  Final samples: {len(final_results)}")
print(f"  Output: {final_output}")
print("=" * 80)
