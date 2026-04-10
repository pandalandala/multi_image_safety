"""Common utility functions for the multi-image safety dataset pipeline."""

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Any

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/mnt/hdd/xuran/multi_image_safety")
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = PROJECT_ROOT / "prompts"


def _is_truthy_env(value: str | None) -> bool:
    """Interpret common truthy environment variable values."""
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on", "debug"}


def load_config(config_name: str = "pipeline.yaml") -> dict:
    """Load a YAML config file."""
    config_path = CONFIG_DIR / config_name
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_taxonomy() -> dict:
    """Load the harm taxonomy configuration."""
    return load_config("taxonomy.yaml")


def load_prompt_sources() -> dict:
    """Load the prompt sources configuration."""
    return load_config("prompt_sources.yaml")


def is_english(text: str, threshold: float = 0.6) -> bool:
    """
    Lightweight English detection. Returns True if text appears to be English.
    Uses ASCII letter ratio + CJK/Cyrillic/Arabic character detection.
    """
    if not text or len(text.strip()) < 3:
        return False
    text = text.strip()
    # Count ASCII letters vs total non-whitespace characters
    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return False
    ascii_letters = sum(1 for c in non_space if c.isascii())
    ratio = ascii_letters / len(non_space)
    # Check for CJK, Cyrillic, Arabic, Thai, Devanagari characters
    for c in text:
        cp = ord(c)
        if (0x4E00 <= cp <= 0x9FFF or   # CJK Unified
            0x3040 <= cp <= 0x30FF or    # Hiragana/Katakana
            0xAC00 <= cp <= 0xD7AF or    # Hangul
            0x0400 <= cp <= 0x04FF or    # Cyrillic
            0x0600 <= cp <= 0x06FF or    # Arabic
            0x0E00 <= cp <= 0x0E7F or    # Thai
            0x0900 <= cp <= 0x097F):     # Devanagari
            return False
    return ratio >= threshold


def load_prompt_template(template_name: str) -> str:
    """Load an LLM prompt template file."""
    template_path = PROMPTS_DIR / template_name
    with open(template_path) as f:
        return f.read()


def save_jsonl(data: list[dict], output_path: str | Path) -> None:
    """Save data as JSONL (one JSON object per line)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(data)} items to {output_path}")


def load_jsonl(input_path: str | Path) -> list[dict]:
    """Load data from a JSONL file."""
    data = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_json(data: Any, output_path: str | Path) -> None:
    """Save data as a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved to {output_path}")


def load_json(input_path: str | Path) -> Any:
    """Load data from a JSON file."""
    with open(input_path) as f:
        return json.load(f)


def sha256_file(file_path: str | Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def env_flag_is_true(name: str) -> bool:
    """Interpret common truthy environment-variable values."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def get_visible_gpu_ids(default: str = "0,1,2,3", max_gpus: int | None = 4) -> list[int]:
    """Return the GPU ids exposed through CUDA_VISIBLE_DEVICES."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", default)
    gpu_ids = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if max_gpus is not None:
        return gpu_ids[:max_gpus]
    return gpu_ids


def get_visible_gpu_csv(default: str = "0,1,2,3", max_gpus: int | None = 4) -> str:
    """Return visible GPU ids as a CSV string."""
    return ",".join(str(gpu_id) for gpu_id in get_visible_gpu_ids(default=default, max_gpus=max_gpus))


def get_effective_tensor_parallel_size(requested: int | None = None, default: str = "0,1,2,3") -> int:
    """Clamp tensor parallelism to the number of visible GPUs."""
    visible_gpu_count = max(1, len(get_visible_gpu_ids(default=default, max_gpus=None)))
    if requested is None:
        return visible_gpu_count
    return max(1, min(int(requested), visible_gpu_count))


def jsonl_file_has_records(path: str | Path) -> bool:
    """Return True when a JSONL file exists and has at least one non-empty line."""
    path = Path(path)
    if not path.exists() or not path.is_file():
        return False
    with open(path) as f:
        for line in f:
            if line.strip():
                return True
    return False


def jsonl_record_count(path: str | Path) -> int:
    """Count non-empty JSONL records."""
    path = Path(path)
    if not path.exists() or not path.is_file():
        return 0
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def get_step_state_dir(output_dir: str | Path) -> Path:
    """Return the per-path directory for step state markers."""
    return ensure_dir(Path(output_dir) / ".step_state")


def get_step_marker_path(output_dir: str | Path, step_name: str, status: str) -> Path:
    """Return the marker path for a step status."""
    return get_step_state_dir(output_dir) / f"{step_name}.{status}.json"


def step_completion_marker_exists(output_dir: str | Path, step_name: str) -> bool:
    """Return True when a step completion marker exists."""
    return get_step_marker_path(output_dir, step_name, "done").exists()


def is_step_complete(
    output_dir: str | Path,
    step_name: str,
    *,
    expected_outputs: list[str | Path] | tuple[str | Path, ...] = (),
    validator=None,
) -> bool:
    """Return True only when a step completion marker exists and outputs validate."""
    marker_path = get_step_marker_path(output_dir, step_name, "done")
    if not marker_path.exists():
        return False

    for output in expected_outputs:
        if not Path(output).exists():
            return False

    if validator is not None:
        try:
            return bool(validator())
        except Exception:
            logger.warning("Step validator failed for %s", step_name, exc_info=True)
            return False

    return True


def start_step(output_dir: str | Path, step_name: str, metadata: dict[str, Any] | None = None) -> Path:
    """Write a running marker for a step and clear any stale completion marker."""
    running_path = get_step_marker_path(output_dir, step_name, "running")
    done_path = get_step_marker_path(output_dir, step_name, "done")
    done_path.unlink(missing_ok=True)
    payload = {
        "step": step_name,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }
    running_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return running_path


def finish_step(
    output_dir: str | Path,
    step_name: str,
    *,
    expected_outputs: list[str | Path] | tuple[str | Path, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write the completion marker for a step and clear the running marker."""
    running_path = get_step_marker_path(output_dir, step_name, "running")
    done_path = get_step_marker_path(output_dir, step_name, "done")
    output_summaries = []
    for output in expected_outputs:
        output_path = Path(output)
        summary: dict[str, Any] = {"path": str(output_path), "exists": output_path.exists()}
        if output_path.suffix == ".jsonl" and output_path.exists():
            summary["records"] = jsonl_record_count(output_path)
        output_summaries.append(summary)

    payload = {
        "step": step_name,
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "outputs": output_summaries,
        "metadata": metadata or {},
    }
    done_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    running_path.unlink(missing_ok=True)
    return done_path


def clear_step_state(
    output_dir: str | Path,
    step_name: str,
    *,
    stale_paths: list[str | Path] | tuple[str | Path, ...] = (),
) -> None:
    """Remove step markers and any stale step outputs before rerunning the step."""
    get_step_marker_path(output_dir, step_name, "running").unlink(missing_ok=True)
    get_step_marker_path(output_dir, step_name, "done").unlink(missing_ok=True)

    for stale_path in stale_paths:
        path = Path(stale_path)
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def clear_all_step_states(output_dir: str | Path) -> int:
    """Remove all step state markers so pipeline can re-run cleanly."""
    state_dir = get_step_state_dir(output_dir)
    if not state_dir.exists():
        return 0
    count = 0
    for marker in state_dir.glob("*.json"):
        marker.unlink(missing_ok=True)
        count += 1
    if count:
        logger.info("Cleared %d step state markers from %s", count, state_dir)
    return count


def get_safe_vllm_kwargs(
    path_name: str = "",
    *,
    model_path: str | None = None,
    tensor_parallel_size: int | None = None,
) -> dict:
    """Return conservative vLLM LLM() constructor kwargs.

    Reads from pipeline.yaml and clamps to safe defaults proven to work
    on GPUs with ~33-47 GB free VRAM.
    """
    config = load_config()
    local_cfg = config["llm"]["local"]
    env_prefix = f"MIS_{path_name.upper()}_" if path_name else "MIS_"

    max_model_len = min(
        int(local_cfg.get("max_model_len", 4096)),
        int(os.environ.get(f"{env_prefix}MAX_MODEL_LEN", "4096")),
    )
    gpu_mem_util = min(
        float(local_cfg.get("gpu_memory_utilization", 0.68)),
        float(os.environ.get(f"{env_prefix}GPU_MEMORY_UTILIZATION", "0.68")),
    )

    return {
        "model": model_path or local_cfg["model_path"],
        "tensor_parallel_size": tensor_parallel_size or int(local_cfg.get("tensor_parallel_size", 4)),
        "trust_remote_code": True,
        "max_model_len": max_model_len,
        "enforce_eager": True,
        "gpu_memory_utilization": gpu_mem_util,
        "disable_custom_all_reduce": True,
    }


def get_hf_home() -> str:
    """Get the HuggingFace cache directory."""
    config = load_config()
    hf_home = config.get("hf_home", "/mnt2/xuran_hdd/cache")
    os.environ["HF_HOME"] = hf_home
    return hf_home


def get_hf_token() -> str | None:
    """Return the Hugging Face token from env or the configured cache dir."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token

    hf_home = Path(os.environ.get("HF_HOME") or get_hf_home())
    token_path = hf_home / "token"
    if not token_path.exists():
        return None

    token = token_path.read_text(encoding="utf-8").strip()
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        return token
    return None


def apply_path2_runtime_defaults() -> dict[str, str]:
    """Apply stable runtime defaults for Path 2 unless already overridden."""
    defaults = {
        "HF_HOME": "/mnt2/xuran_hdd/cache",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "OMP_NUM_THREADS": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "MIS_PATH2_MAX_MODEL_LEN": "4096",
        "MIS_PATH2_GPU_MEMORY_UTILIZATION": "0.68",
        "MIS_PATH2_VLLM_BATCH_SIZE": "64",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    return defaults


def configure_hf_download_debug() -> bool:
    """Enable detailed Hugging Face Hub logs when requested via env."""
    if not _is_truthy_env(os.environ.get("MIS_HF_DOWNLOAD_VERBOSE")):
        return False

    logging.getLogger("huggingface_hub").setLevel(logging.DEBUG)
    logging.getLogger("huggingface_hub.file_download").setLevel(logging.DEBUG)
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.INFO)

    try:
        from huggingface_hub.utils import logging as hf_logging

        hf_logging.set_verbosity_debug()
    except Exception:
        logger.debug("Failed to enable huggingface_hub debug verbosity", exc_info=True)

    logger.info(
        "Enabled detailed Hugging Face download logging via MIS_HF_DOWNLOAD_VERBOSE=%s",
        os.environ.get("MIS_HF_DOWNLOAD_VERBOSE"),
    )
    return True


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    configure_hf_download_debug()


def get_all_categories() -> list[dict]:
    """Get a flat list of all harm categories from taxonomy."""
    taxonomy = load_taxonomy()
    categories = []
    for tier in taxonomy["tiers"].values():
        for cat in tier["categories"]:
            categories.append(cat)
    return categories


def get_category_harm_descriptions() -> dict[str, str]:
    """Get harm descriptions for each category (used for CLIP harm vectors)."""
    categories = get_all_categories()
    descriptions = {}
    for cat in categories:
        cat_id = cat["id"]
        desc = cat["description"]
        # Also include sub-categories for richer harm vectors
        subs = cat.get("sub_categories", [])
        full_desc = f"{desc}. Examples: {', '.join(subs[:5])}"
        descriptions[cat_id] = full_desc
    return descriptions
