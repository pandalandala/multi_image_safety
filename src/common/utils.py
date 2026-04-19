"""Common utility functions for the multi-image safety dataset pipeline."""

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
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


_min_image_size_cache: tuple[int, int] | None = None


def get_min_image_size() -> tuple[int, int]:
    """Return (min_width, min_height) from pipeline.yaml → laion section."""
    global _min_image_size_cache
    if _min_image_size_cache is not None:
        return _min_image_size_cache
    try:
        config = load_config()
        laion_cfg = config.get("laion", {})
        w = int(laion_cfg.get("min_width", 256))
        h = int(laion_cfg.get("min_height", 256))
    except Exception:
        w, h = 256, 256
    _min_image_size_cache = (w, h)
    return _min_image_size_cache


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


def _parse_gpu_id_csv(raw: str | None, default: str) -> list[int]:
    """Parse a CSV GPU list into integer GPU ids."""
    value = raw or default
    gpu_ids = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            gpu_ids.append(int(part))
        except ValueError:
            logger.warning("Ignoring invalid GPU id %r in %r", part, value)
    return gpu_ids


def get_gpu_candidate_ids(default: str = "0,1,2,3,4,5,6,7") -> list[int]:
    """Return candidate GPU ids that this run is allowed to use."""
    raw = (
        os.environ.get("MIS_GPU_CANDIDATES")
        or os.environ.get("CUDA_VISIBLE_DEVICES")
        or default
    )
    gpu_ids = _parse_gpu_id_csv(raw, default)
    if not gpu_ids:
        gpu_ids = _parse_gpu_id_csv(default, default)
    return gpu_ids


def should_use_all_visible_gpus() -> bool:
    """Return True when runs should try to consume all user-exposed GPUs."""
    raw = os.environ.get("MIS_USE_ALL_VISIBLE_GPUS", "1")
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def get_visible_gpu_ids(default: str = "0,1,2,3", max_gpus: int | None = 4) -> list[int]:
    """Return the GPU ids exposed through CUDA_VISIBLE_DEVICES."""
    gpu_ids = _parse_gpu_id_csv(os.environ.get("CUDA_VISIBLE_DEVICES"), default)
    if max_gpus is not None:
        return gpu_ids[:max_gpus]
    return gpu_ids


def get_visible_gpu_csv(default: str = "0,1,2,3", max_gpus: int | None = 4) -> str:
    """Return visible GPU ids as a CSV string."""
    return ",".join(str(gpu_id) for gpu_id in get_visible_gpu_ids(default=default, max_gpus=max_gpus))


def get_effective_tensor_parallel_size(requested: int | None = None, default: str = "0,1,2,3") -> int:
    """Clamp tensor parallelism to the number of visible GPUs."""
    visible_gpu_count = max(1, len(get_visible_gpu_ids(default=default, max_gpus=None)))
    if should_use_all_visible_gpus():
        if requested is None:
            return visible_gpu_count
        requested = int(requested)
        if requested >= 4:
            return visible_gpu_count
    if requested is None:
        return visible_gpu_count
    return max(1, min(int(requested), visible_gpu_count))


def query_gpu_inventory(default: str = "0,1,2,3,4,5,6,7") -> list[dict[str, Any]]:
    """Return per-GPU free-memory inventory for candidate GPUs."""
    candidate_ids = set(get_gpu_candidate_ids(default=default))
    if not candidate_ids:
        return []

    try:
        gpu_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if gpu_query.returncode != 0:
            raise RuntimeError(gpu_query.stderr.strip() or gpu_query.stdout.strip())
        rows = [line.strip() for line in gpu_query.stdout.splitlines() if line.strip()]
    except Exception as exc:
        logger.warning("Failed to query GPU inventory via nvidia-smi: %s", exc)
        try:
            import torch

            torch_inventory = []
            for gpu_id in sorted(candidate_ids):
                with torch.cuda.device(gpu_id):
                    free_bytes, total_bytes = torch.cuda.mem_get_info()
                torch_inventory.append(
                    {
                        "index": gpu_id,
                        "uuid": "",
                        "total_gb": float(total_bytes) / (1024.0 ** 3),
                        "free_gb": float(free_bytes) / (1024.0 ** 3),
                        "utilization": 0.0,
                        "process_count": 0,
                    }
                )
            if torch_inventory:
                return torch_inventory
        except Exception:
            logger.debug("Failed to query GPU inventory via torch.cuda", exc_info=True)
        return [
            {
                "index": gpu_id,
                "uuid": "",
                "total_gb": 48.0,
                "free_gb": 48.0,
                "utilization": 0.0,
                "process_count": 0,
            }
            for gpu_id in sorted(candidate_ids)
        ]

    process_counts: dict[str, int] = {}
    try:
        proc_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        proc_rows = [line.strip() for line in proc_query.stdout.splitlines() if line.strip()]
        for row in proc_rows:
            if "No running compute processes found" in row:
                break
            parts = [part.strip() for part in row.split(",")]
            if len(parts) < 3:
                continue
            gpu_uuid = parts[0]
            process_counts[gpu_uuid] = process_counts.get(gpu_uuid, 0) + 1
    except Exception:
        logger.debug("Failed to query GPU process inventory", exc_info=True)

    inventory = []
    for row in rows:
        parts = [part.strip() for part in row.split(",")]
        if len(parts) < 5:
            continue
        try:
            index = int(parts[0])
            if index not in candidate_ids:
                continue
            total_gb = float(parts[2]) / 1024.0
            free_gb = float(parts[3]) / 1024.0
            util_value = parts[4]
            utilization = float(util_value) if util_value.upper() != "N/A" else 0.0
        except ValueError:
            continue
        uuid = parts[1]
        inventory.append(
            {
                "index": index,
                "uuid": uuid,
                "total_gb": total_gb,
                "free_gb": free_gb,
                "utilization": utilization,
                "process_count": process_counts.get(uuid, 0),
            }
        )

    return sorted(inventory, key=lambda item: item["index"])


def _gpu_preference_key(gpu: dict[str, Any]) -> tuple[float, float, int, int]:
    """Sort GPUs by more free memory, less load, fewer processes, lower id."""
    return (-float(gpu["free_gb"]), float(gpu["utilization"]), int(gpu["process_count"]), int(gpu["index"]))


def format_gpu_inventory(inventory: list[dict[str, Any]]) -> str:
    """Render GPU inventory compactly for logs."""
    parts = []
    for gpu in inventory:
        parts.append(
            "GPU{index} free={free:.1f}/{total:.1f}GiB util={util:.0f}% procs={procs}".format(
                index=gpu["index"],
                free=gpu["free_gb"],
                total=gpu["total_gb"],
                util=gpu["utilization"],
                procs=gpu["process_count"],
            )
        )
    return "; ".join(parts)


def select_gpu_runtime_profile(
    *,
    path_name: str,
    task_type: str,
    preferred_gpu_count: int = 8,
    requested_tensor_parallel_size: int | None = None,
    requested_gpu_memory_utilization: float | None = None,
    requested_max_model_len: int | None = None,
    requested_batch_size: int | None = None,
    default_candidates: str = "0,1,2,3,4,5,6,7",
) -> dict[str, Any]:
    """Select the best GPU set for an LLM or image-generation task."""
    mode = os.environ.get("MIS_GPU_MODE", "adaptive").strip().lower() or "adaptive"
    inventory = query_gpu_inventory(default=default_candidates)
    if not inventory:
        raise RuntimeError("No candidate GPUs found for runtime selection")

    candidate_ids = [gpu["index"] for gpu in inventory]
    if should_use_all_visible_gpus() and preferred_gpu_count >= 4:
        preferred_gpu_count = len(candidate_ids)
    preferred_gpu_count = max(1, preferred_gpu_count)
    sorted_inventory = sorted(inventory, key=_gpu_preference_key)

    if task_type == "image":
        min_free_gb = float(
            os.environ.get(
                "MIS_GPU_IMAGE_MIN_FREE_GB",
                os.environ.get("MIS_GPU_MIN_FREE_GB", "8"),
            )
        )
        eligible = [gpu for gpu in sorted_inventory if gpu["free_gb"] >= min_free_gb]
        if mode == "strict4":
            target_count = preferred_gpu_count
        else:
            target_count = min(preferred_gpu_count, len(eligible))
        if target_count < 1:
            raise RuntimeError(
                "No GPUs satisfy image-generation free-memory threshold "
                f"({min_free_gb:.1f} GiB). Inventory: {format_gpu_inventory(inventory)}"
            )
        selected = eligible[:target_count]
        fallback_reason = ""
        if len(selected) < preferred_gpu_count:
            fallback_reason = (
                f"preferred {preferred_gpu_count} GPUs for image work, "
                f"selected {len(selected)} because only {len(eligible)} met the "
                f"{min_free_gb:.1f} GiB free-memory threshold"
            )
        return {
            "path_name": path_name,
            "task_type": task_type,
            "mode": mode,
            "selected_gpu_ids": [gpu["index"] for gpu in selected],
            "selected_gpu_csv": ",".join(str(gpu["index"]) for gpu in selected),
            "gpu_count": len(selected),
            "worker_count": len(selected),
            "tensor_parallel_size": 1,
            "fallback_mode": "none" if not fallback_reason else "reduced_workers",
            "fallback_reason": fallback_reason,
            "inventory": inventory,
        }

    path_prefix = f"MIS_{path_name.upper()}_"
    base_max_model_len = int(
        os.environ.get(f"{path_prefix}MAX_MODEL_LEN", str(requested_max_model_len or 4096))
    )
    base_gpu_util = float(
        os.environ.get(
            f"{path_prefix}GPU_MEMORY_UTILIZATION",
            str(requested_gpu_memory_utilization if requested_gpu_memory_utilization is not None else 0.68),
        )
    )
    base_batch_size = int(
        os.environ.get(f"{path_prefix}VLLM_BATCH_SIZE", str(requested_batch_size or 64))
    )
    requested_tp = requested_tensor_parallel_size or preferred_gpu_count
    if should_use_all_visible_gpus() and requested_tp >= 4:
        requested_tp = len(candidate_ids)
    profile_counts = [min(preferred_gpu_count, requested_tp, len(candidate_ids))]
    if mode in {"adaptive", "auto"}:
        for count in (3, 2, 1):
            if count <= len(candidate_ids) and count not in profile_counts:
                profile_counts.append(count)

    attempted: list[str] = []
    for count in profile_counts:
        if count >= 4:
            gpu_util = base_gpu_util
            max_model_len = base_max_model_len
            batch_size = base_batch_size
        elif count == 3:
            gpu_util = min(base_gpu_util, 0.62)
            max_model_len = min(base_max_model_len, 4096)
            batch_size = max(16, (base_batch_size * 3) // 4)
        elif count == 2:
            gpu_util = min(base_gpu_util, 0.55)
            max_model_len = min(base_max_model_len, 3072)
            batch_size = max(8, base_batch_size // 2)
        else:
            gpu_util = min(base_gpu_util, 0.40)
            max_model_len = min(base_max_model_len, 2048)
            batch_size = max(4, base_batch_size // 4)

        eligible = [
            gpu for gpu in sorted_inventory
            if gpu["free_gb"] >= (gpu["total_gb"] * gpu_util)
        ]
        attempted.append(
            f"{count}gpu@util={gpu_util:.2f}: eligible={len(eligible)}"
        )
        if len(eligible) < count:
            continue
        selected = eligible[:count]
        fallback_reason = ""
        fallback_mode = "none"
        if count != profile_counts[0]:
            fallback_mode = "adaptive_llm"
            fallback_reason = (
                f"preferred {profile_counts[0]}-GPU llm profile was unavailable; "
                f"using {count} GPUs with max_model_len={max_model_len}, "
                f"gpu_memory_utilization={gpu_util:.2f}, batch_size={batch_size}"
            )
        return {
            "path_name": path_name,
            "task_type": task_type,
            "mode": mode,
            "selected_gpu_ids": [gpu["index"] for gpu in selected],
            "selected_gpu_csv": ",".join(str(gpu["index"]) for gpu in selected),
            "gpu_count": len(selected),
            "worker_count": len(selected),
            "tensor_parallel_size": max(1, min(requested_tp, len(selected))),
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_util,
            "batch_size": batch_size,
            "fallback_mode": fallback_mode,
            "fallback_reason": fallback_reason,
            "inventory": inventory,
        }

    raise RuntimeError(
        f"No {task_type} GPU profile for {path_name} satisfied free-memory constraints. "
        f"Attempts: {', '.join(attempted)}. Inventory: {format_gpu_inventory(inventory)}"
    )


def apply_gpu_runtime_profile(
    *,
    path_name: str,
    task_type: str,
    preferred_gpu_count: int = 4,
    requested_tensor_parallel_size: int | None = None,
    requested_gpu_memory_utilization: float | None = None,
    requested_max_model_len: int | None = None,
    requested_batch_size: int | None = None,
    default_candidates: str = "0,1,2,3,4,5,6,7",
) -> dict[str, Any]:
    """Select GPUs for the current process and export the chosen runtime profile."""
    profile = select_gpu_runtime_profile(
        path_name=path_name,
        task_type=task_type,
        preferred_gpu_count=preferred_gpu_count,
        requested_tensor_parallel_size=requested_tensor_parallel_size,
        requested_gpu_memory_utilization=requested_gpu_memory_utilization,
        requested_max_model_len=requested_max_model_len,
        requested_batch_size=requested_batch_size,
        default_candidates=default_candidates,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = profile["selected_gpu_csv"]
    os.environ["MIS_SELECTED_GPUS"] = profile["selected_gpu_csv"]
    os.environ["MIS_GPU_COUNT"] = str(profile["gpu_count"])
    os.environ["MIS_GPU_FALLBACK_MODE"] = str(profile["fallback_mode"])
    os.environ["MIS_GPU_FALLBACK_REASON"] = str(profile["fallback_reason"])

    if task_type == "llm":
        path_prefix = f"MIS_{path_name.upper()}_"
        os.environ[f"{path_prefix}MAX_MODEL_LEN"] = str(profile["max_model_len"])
        os.environ[f"{path_prefix}GPU_MEMORY_UTILIZATION"] = f"{profile['gpu_memory_utilization']:.2f}"
        os.environ[f"{path_prefix}VLLM_BATCH_SIZE"] = str(profile["batch_size"])

    return profile


def log_gpu_runtime_profile(logger: logging.Logger, profile: dict[str, Any], label: str) -> None:
    """Log the selected GPU runtime profile and current inventory."""
    logger.info(
        "%s GPU runtime: selected_gpus=%s gpu_count=%s tensor_parallel_size=%s "
        "worker_count=%s mode=%s fallback_mode=%s fallback_reason=%s",
        label,
        profile.get("selected_gpu_csv", ""),
        profile.get("gpu_count", 0),
        profile.get("tensor_parallel_size", 1),
        profile.get("worker_count", 0),
        profile.get("mode", ""),
        profile.get("fallback_mode", ""),
        profile.get("fallback_reason", "") or "none",
    )
    if _is_truthy_env(os.environ.get("MIS_GPU_VERBOSE", "1")):
        logger.info("%s GPU inventory: %s", label, format_gpu_inventory(profile.get("inventory", [])))


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


def _normalize_step_paths(
    paths: list[str | Path] | tuple[str | Path, ...] = (),
) -> list[str]:
    normalized: list[str] = []
    for path in paths:
        raw = str(path).strip()
        if raw and raw not in normalized:
            normalized.append(raw)
    return normalized


def _read_step_marker_payload(marker_path: str | Path) -> dict[str, Any]:
    path = Path(marker_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        logger.warning("Failed to parse step marker: %s", path, exc_info=True)
        return {}


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


def start_step(
    output_dir: str | Path,
    step_name: str,
    metadata: dict[str, Any] | None = None,
    *,
    cleanup_paths: list[str | Path] | tuple[str | Path, ...] = (),
) -> Path:
    """Write a running marker for a step and clear any stale completion marker."""
    running_path = get_step_marker_path(output_dir, step_name, "running")
    done_path = get_step_marker_path(output_dir, step_name, "done")
    failed_path = get_step_marker_path(output_dir, step_name, "failed")
    done_path.unlink(missing_ok=True)
    failed_path.unlink(missing_ok=True)
    payload = {
        "step": step_name,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
        "cleanup_paths": _normalize_step_paths(cleanup_paths),
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
    failed_path = get_step_marker_path(output_dir, step_name, "failed")
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
    failed_path.unlink(missing_ok=True)
    return done_path


def fail_step(
    output_dir: str | Path,
    step_name: str,
    *,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
    cleanup_paths: list[str | Path] | tuple[str | Path, ...] = (),
    auto_cleanup: bool | None = None,
) -> Path:
    """Record a failed step and optionally remove that step's outputs."""
    running_path = get_step_marker_path(output_dir, step_name, "running")
    done_path = get_step_marker_path(output_dir, step_name, "done")
    failed_path = get_step_marker_path(output_dir, step_name, "failed")

    running_payload = _read_step_marker_payload(running_path)
    payload_cleanup_paths = running_payload.get("cleanup_paths", []) if running_payload else []
    combined_cleanup = _normalize_step_paths(tuple(payload_cleanup_paths) + tuple(cleanup_paths))

    if auto_cleanup is None:
        auto_cleanup = _is_truthy_env(os.environ.get("MIS_AUTO_CLEAN_FAILED_STEP_OUTPUTS", "1"))

    removed_paths: list[str] = []
    if auto_cleanup:
        for stale_path in combined_cleanup:
            path = Path(stale_path)
            if not path.exists():
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            removed_paths.append(str(path))

    payload = {
        "step": step_name,
        "status": "failed",
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "error": error or "",
        "cleanup_paths": combined_cleanup,
        "removed_paths": removed_paths,
        "metadata": metadata or {},
    }
    failed_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    running_path.unlink(missing_ok=True)
    done_path.unlink(missing_ok=True)
    return failed_path


def clear_step_state(
    output_dir: str | Path,
    step_name: str,
    *,
    stale_paths: list[str | Path] | tuple[str | Path, ...] = (),
) -> None:
    """Remove step markers and any stale step outputs before rerunning the step."""
    running_path = get_step_marker_path(output_dir, step_name, "running")
    done_path = get_step_marker_path(output_dir, step_name, "done")
    failed_path = get_step_marker_path(output_dir, step_name, "failed")
    running_payload = _read_step_marker_payload(running_path)
    failed_payload = _read_step_marker_payload(failed_path)
    registered_cleanup = []
    if running_payload:
        registered_cleanup.extend(running_payload.get("cleanup_paths", []))
    if failed_payload:
        registered_cleanup.extend(failed_payload.get("cleanup_paths", []))

    running_path.unlink(missing_ok=True)
    done_path.unlink(missing_ok=True)
    failed_path.unlink(missing_ok=True)

    for stale_path in _normalize_step_paths(tuple(registered_cleanup) + tuple(stale_paths)):
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
    effective_tp = tensor_parallel_size or int(local_cfg.get("tensor_parallel_size", 4))
    effective_tp = get_effective_tensor_parallel_size(effective_tp)

    return {
        "model": model_path or local_cfg["model_path"],
        "tensor_parallel_size": effective_tp,
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
    default_candidates = (
        os.environ.get("MIS_GPU_CANDIDATES")
        or os.environ.get("CUDA_VISIBLE_DEVICES")
        or "0,1,2,3,4,5,6,7"
    )
    defaults = {
        "HF_HOME": "/mnt2/xuran_hdd/cache",
        "MIS_GPU_CANDIDATES": default_candidates,
        "OMP_NUM_THREADS": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
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
