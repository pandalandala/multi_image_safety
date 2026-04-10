#!/usr/bin/env python3
"""
Path 6: TAG+KG Fusion for Covert Toxicity
  Step 1: LLM generates concept chains (toxicity association graphs)
  Step 2: CLIP scoring and filtering of chains
  Step 3: Fuse with Path 1 KG pairs (if available)
  Step 4: Generate images for fusion pairs (SD3.5 Large Turbo, 4 visible GPUs)
  Step 5: LLM generates connecting text prompts (vLLM on 4 visible GPUs)

Runtime estimate: ~3-4 hours
"""

import logging
import os
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
OUTPUT_DIR = DATA_DIR / "raw" / "path6"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if "--clean" in sys.argv:
    from src.common.utils import clear_all_step_states
    clear_all_step_states(OUTPUT_DIR)
    sys.argv.remove("--clean")


def get_gpu_ids(default: str = "0,1,2,3") -> list[int]:
    """Parse the visible GPU list from the environment."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", default)
    gpu_ids = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not gpu_ids:
        raise RuntimeError("No GPUs configured for Path 6")
    return gpu_ids


def get_llm_runtime_limits() -> tuple[dict, int, float, int]:
    """Clamp Path 6 vLLM settings to stable defaults unless overridden."""
    config = load_config()
    local_cfg = config["llm"]["local"]
    max_model_len = min(
        int(local_cfg.get("max_model_len", 8192)),
        int(os.environ.get("MIS_PATH6_MAX_MODEL_LEN", "4096")),
    )
    gpu_memory_utilization = min(
        float(local_cfg.get("gpu_memory_utilization", 0.9)),
        float(os.environ.get("MIS_PATH6_GPU_MEMORY_UTILIZATION", "0.68")),
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
    """Check whether a Path 6 step has an explicit completion marker."""
    return (not force_rerun) and is_step_complete(
        OUTPUT_DIR,
        step_name,
        expected_outputs=[output_file],
        validator=lambda: jsonl_record_count(output_file) >= min_records,
    )


# ── Step 1: LLM generates concept chains ─────────────────────────────────────
logger.info("=" * 60)
logger.info("PATH 6 STEP 1: LLM TAG chain generation (vLLM)")
logger.info("=" * 60)

chains_file = OUTPUT_DIR / "raw_chains.jsonl"

if step_complete("step1_generate_chains", chains_file):
    logger.info("Skipping Step 1; completion marker found")
else:
    clear_step_state(OUTPUT_DIR, "step1_generate_chains", stale_paths=[chains_file])
    start_step(OUTPUT_DIR, "step1_generate_chains")
    rc = run_subprocess(f"""
import sys
sys.path.insert(0, '{PROJ}')
from src.common.utils import setup_logging, save_jsonl, get_category_harm_descriptions
from src.path6_tag_kg_fusion.tag_builder import prepare_chain_gen_prompts, parse_chain_response
from vllm import LLM, SamplingParams
setup_logging()

descriptions = get_category_harm_descriptions()
prompt_items = prepare_chain_gen_prompts(descriptions, chains_per_category=40)
conversations = [[{{"role": "user", "content": item["prompt"]}}] for item in prompt_items]

print(f'Loading LLM for {{len(conversations)}} TAG chain prompts...')
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

all_chains = []
for item, output in zip(prompt_items, outputs):
    text = output.outputs[0].text
    chains = parse_chain_response(text, item["category"])
    all_chains.extend(chains)
    print(f'  {{item["category"]}}: {{len(chains)}} chains parsed')

save_jsonl(all_chains, '{chains_file}')
print(f'Total chains generated: {{len(all_chains)}}')
""", gpu_ids=all_gpu_ids)
    if rc != 0:
        logger.error("TAG chain generation failed")
        sys.exit(1)
    finish_step(
        OUTPUT_DIR,
        "step1_generate_chains",
        expected_outputs=[chains_file],
        metadata={"records": jsonl_record_count(chains_file)},
    )

raw_chains = load_jsonl(chains_file)
logger.info(f"Raw chains: {len(raw_chains)}")

# ── Step 2: CLIP scoring and filtering ───────────────────────────────────────
logger.info("=" * 60)
logger.info("PATH 6 STEP 2: CLIP chain scoring")
logger.info("=" * 60)

scored_file = OUTPUT_DIR / "scored_chains.jsonl"

if step_complete("step2_score_chains", scored_file):
    logger.info("Skipping Step 2; completion marker found")
    scored_chains = load_jsonl(scored_file)
else:
    clear_step_state(OUTPUT_DIR, "step2_score_chains", stale_paths=[scored_file])
    start_step(OUTPUT_DIR, "step2_score_chains")
    rc = run_subprocess(f"""
import sys; sys.path.insert(0, '{PROJ}')
from src.common.utils import setup_logging, load_jsonl, save_jsonl, load_config
from src.path6_tag_kg_fusion.tag_builder import score_chains_clip, extract_endpoint_pairs
setup_logging()

config = load_config()
clip_cfg = config.get("clip", {{}})
chains = load_jsonl('{chains_file}')
scored = score_chains_clip(
    chains,
    theta_safe=clip_cfg.get("theta_safe", 0.40),
    theta_harm=min(clip_cfg.get("theta_harm", 0.35), 0.30),  # Slightly relaxed for chains
)
save_jsonl(scored, '{scored_file}')
print(f'Scored chains: {{len(scored)}} / {{len(chains)}} passed')
""", gpu_ids=single_gpu_id)
    if rc != 0:
        logger.warning("CLIP scoring failed — using unscored chains with endpoint extraction")
        # Fallback: just extract endpoint pairs without CLIP filtering
        from src.path6_tag_kg_fusion.tag_builder import extract_endpoint_pairs
        pairs = extract_endpoint_pairs(raw_chains)
        save_jsonl(pairs, OUTPUT_DIR / "fusion_pairs.jsonl")
        scored_chains = raw_chains
    else:
        scored_chains = load_jsonl(scored_file)
        finish_step(
            OUTPUT_DIR,
            "step2_score_chains",
            expected_outputs=[scored_file],
            metadata={"records": len(scored_chains)},
        )

# ── Step 3: Extract endpoint pairs and fuse with Path 1 ─────────────────────
logger.info("=" * 60)
logger.info("PATH 6 STEP 3: Fusion with Path 1 KG pairs")
logger.info("=" * 60)

fusion_file = OUTPUT_DIR / "fusion_pairs.jsonl"

if step_complete("step3_fuse_pairs", fusion_file):
    logger.info("Skipping Step 3; completion marker found")
    fused_pairs = load_jsonl(fusion_file)
else:
    clear_step_state(OUTPUT_DIR, "step3_fuse_pairs", stale_paths=[fusion_file])
    start_step(OUTPUT_DIR, "step3_fuse_pairs")
    from src.path6_tag_kg_fusion.tag_builder import extract_endpoint_pairs
    from src.path6_tag_kg_fusion.fusion_mine import fuse_with_path1

    tag_pairs = extract_endpoint_pairs(scored_chains)
    fused_pairs = fuse_with_path1(tag_pairs)
    save_jsonl(fused_pairs, fusion_file)
    finish_step(
        OUTPUT_DIR,
        "step3_fuse_pairs",
        expected_outputs=[fusion_file],
        metadata={"records": len(fused_pairs)},
    )

logger.info(f"Fusion pairs: {len(fused_pairs)}")

# ── Step 4: Generate images ─────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("PATH 6 STEP 4: Image generation (4-GPU parallel)")
logger.info("=" * 60)

images_file = OUTPUT_DIR / "pairs_with_images.jsonl"

if step_complete("step4_generate_images", images_file) and os.environ.get("MIS_FORCE_REGENERATE_IMAGES", "").strip() != "1":
    all_results = load_jsonl(images_file)
    logger.info(f"Skipping Step 4; completion marker found: {len(all_results)} pairs")
else:
    clear_step_state(
        OUTPUT_DIR,
        "step4_generate_images",
        stale_paths=[
            images_file,
            *sorted(OUTPUT_DIR.glob("pairs_with_images_gpu*.jsonl")),
            *sorted(OUTPUT_DIR.glob("_img_chunk_*.jsonl")),
        ],
    )
    start_step(OUTPUT_DIR, "step4_generate_images")
    free_gpus = visible_gpus
    num_workers = len(free_gpus)
    n = len(fused_pairs)
    chunk_size = (n + num_workers - 1) // num_workers

    chunk_files = []
    for i in range(num_workers):
        chunk = fused_pairs[i * chunk_size: (i + 1) * chunk_size]
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
from src.path6_tag_kg_fusion.sample_gen import generate_fusion_images
from pathlib import Path

pairs = load_jsonl('{chunk_files[i]}')
results = generate_fusion_images(
    pairs,
    Path('{OUTPUT_DIR}'),
    start_id={20000 + i * chunk_size},
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
        logger.error("Path 6 Step 4 incomplete because workers failed: %s", worker_failures)
        sys.exit(1)
    finish_step(
        OUTPUT_DIR,
        "step4_generate_images",
        expected_outputs=[images_file],
        metadata={"records": len(all_results), "input_records": len(fused_pairs)},
    )
    for cf in chunk_files:
        cf.unlink(missing_ok=True)

logger.info(f"Pairs with images: {len(all_results)}")

# ── Step 5: Generate connecting text prompts + MTC scoring ───────────────────
logger.info("=" * 60)
logger.info("PATH 6 STEP 5: LLM prompt generation + MTC scoring (vLLM)")
logger.info("=" * 60)

final_output = OUTPUT_DIR / "validated_samples.jsonl"

if step_complete("step5_generate_prompts", final_output):
    logger.info("Skipping Step 5; completion marker found")
    final_results = load_jsonl(final_output)
else:
    clear_step_state(OUTPUT_DIR, "step5_generate_prompts", stale_paths=[final_output])
    start_step(OUTPUT_DIR, "step5_generate_prompts")
    rc = run_subprocess(f"""
import sys, json, re
sys.path.insert(0, '{PROJ}')
from src.common.utils import setup_logging, load_jsonl, save_jsonl
from src.path1_kg_concept.prompt_create import prepare_prompt_gen_prompts, parse_prompt_gen_response
from src.path6_tag_kg_fusion.mtc_scorer import batch_score_mtc
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
            "full_chain": pair.get("full_chain", []),
            "hop_count": pair.get("hop_count", 1),
            "clip_combined_sim": pair.get("clip_combined_sim", 0.0),
            "clip_endpoint_sims": pair.get("clip_endpoint_sims", []),
            "covertness_score": pair.get("covertness_score", 3),
            "fusion_source": pair.get("fusion_source", "tag"),
            "pattern": Pattern.A.value,
            "source_path": SourcePath.PATH6.value,
        }})

# Apply MTC scoring
results = batch_score_mtc(results)

save_jsonl(results, '{final_output}')
print(f'Path 6: {{len(results)}}/{{len(prompt_data)}} samples generated')
""", gpu_ids=all_gpu_ids)

    if rc != 0:
        logger.error("Prompt generation + MTC scoring failed")
        sys.exit(1)

    final_results = load_jsonl(final_output)
    finish_step(
        OUTPUT_DIR,
        "step5_generate_prompts",
        expected_outputs=[final_output],
        metadata={"records": len(final_results)},
    )

print("\n" + "=" * 80)
print(f"PATH 6 COMPLETE: {len(final_results)} samples")
print(f"  Raw chains: {len(raw_chains)}")
print(f"  Scored chains: {len(scored_chains)}")
print(f"  Fusion pairs: {len(fused_pairs)}")
print(f"  With images: {len(all_results)}")
print(f"  Final samples: {len(final_results)}")
print(f"  Output: {final_output}")
print("=" * 80)
