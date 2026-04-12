"""Shared text-to-image generation utilities with configurable backends."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

from src.common.clip_utils import (
    cosine_similarity,
    encode_image,
    encode_text,
    passes_download_filter,
)
from src.common.utils import get_hf_home, get_hf_token, load_config

logger = logging.getLogger(__name__)

_PIPELINES: dict[tuple[str, str, str], object] = {}
_TEXT_EMBED_CACHE: dict[str, object] = {}


def _normalize_description(description: str | None) -> str:
    """Return a stripped prompt string, or an empty string for invalid descriptions."""
    if description is None:
        return ""
    return str(description).strip()


def should_force_regenerate_images() -> bool:
    """Return True when callers should ignore cached/generated image files."""
    return os.environ.get("MIS_FORCE_REGENERATE_IMAGES", "").strip() == "1"


def _torch_dtype(dtype_name: str | None) -> torch.dtype:
    """Map config strings to torch dtypes."""
    name = (dtype_name or "float16").strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(name, torch.float16)


def _get_root_generation_config() -> dict:
    """Load the project-level image generation config."""
    return load_config().get("image_generation", {})


def _resolve_backend_config(backend: str | None = None) -> tuple[str, dict, dict]:
    """Resolve the active backend name, backend-specific config, and root config."""
    root_cfg = _get_root_generation_config()
    backend_name = backend or root_cfg.get("backend", "sdxl")
    backend_cfg = dict(root_cfg.get("backends", {}).get(backend_name, {}))

    if not backend_cfg and backend_name == "sdxl":
        # Backward-compatibility with the old flat SDXL config.
        backend_cfg = {
            "pipeline": "sdxl",
            "model": root_cfg.get("model", "stabilityai/stable-diffusion-xl-base-1.0"),
            "torch_dtype": root_cfg.get("torch_dtype", "float16"),
            "variant": root_cfg.get("variant", "fp16"),
            "use_safetensors": root_cfg.get("use_safetensors", True),
            "num_inference_steps": root_cfg.get("num_inference_steps", 30),
            "guidance_scale": root_cfg.get("guidance_scale", 7.5),
        }

    if not backend_cfg:
        raise ValueError(f"Unknown image_generation backend: {backend_name}")

    backend_cfg.setdefault("pipeline", "sdxl" if backend_name == "sdxl" else "sd3")
    backend_cfg.setdefault("model", root_cfg.get("model", "stabilityai/stable-diffusion-xl-base-1.0"))
    backend_cfg.setdefault("torch_dtype", root_cfg.get("torch_dtype", "float16"))
    backend_cfg.setdefault("num_inference_steps", root_cfg.get("num_inference_steps", 30))
    backend_cfg.setdefault("guidance_scale", root_cfg.get("guidance_scale", 7.5))
    return backend_name, backend_cfg, root_cfg


def _reraise_gated_repo_error(model_id: str, exc: Exception) -> None:
    """Rewrite gated-repo failures with a clearer local setup message."""
    from huggingface_hub.errors import GatedRepoError

    if not isinstance(exc, GatedRepoError):
        return

    hf_home = os.environ.get("HF_HOME") or get_hf_home()
    token = get_hf_token()
    if token:
        detail = (
            f"HF_TOKEN is set, but it does not currently have access to the gated repo "
            f"'{model_id}'. Request approval on https://huggingface.co/{model_id} and "
            f"refresh the token in {hf_home}/token or the HF_TOKEN environment variable."
        )
    else:
        detail = (
            f"No Hugging Face token is configured for gated repo '{model_id}'. "
            f"Run `huggingface-cli login`, or write a token with access to "
            f"{hf_home}/token, or export HF_TOKEN before starting the pipeline."
        )
    raise RuntimeError(detail) from exc


def _build_pipeline(backend_name: str, backend_cfg: dict, root_cfg: dict):
    """Load the requested pipeline lazily."""
    pipeline_type = backend_cfg.get("pipeline", "sdxl")
    model_id = backend_cfg["model"]
    device = root_cfg.get("device", "cuda")
    dtype = _torch_dtype(backend_cfg.get("torch_dtype"))
    hf_token = get_hf_token()
    cache_key = (backend_name, model_id, device)
    if cache_key in _PIPELINES:
        return _PIPELINES[cache_key]

    try:
        if pipeline_type == "sd3":
            from diffusers import StableDiffusion3Pipeline

            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                token=hf_token,
            )
        elif pipeline_type == "sdxl":
            from diffusers import StableDiffusionXLPipeline

            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                variant=backend_cfg.get("variant", "fp16"),
                use_safetensors=bool(backend_cfg.get("use_safetensors", True)),
                token=hf_token,
            )
        else:
            raise ValueError(f"Unsupported pipeline type: {pipeline_type}")
    except Exception as exc:
        _reraise_gated_repo_error(model_id, exc)
        raise

    pipe = pipe.to(device)
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    elif hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    logger.info("Loaded T2I pipeline backend=%s model=%s", backend_name, model_id)
    _PIPELINES[cache_key] = pipe
    return pipe


def ensure_t2i_model_cached(backend: str | None = None) -> str:
    """Pre-download the active T2I model into the shared HF cache."""
    backend_name, backend_cfg, _ = _resolve_backend_config(backend)
    model_id = backend_cfg["model"]
    hf_token = get_hf_token()

    try:
        from huggingface_hub import snapshot_download

        snapshot_path = snapshot_download(
            repo_id=model_id,
            token=hf_token,
            resume_download=True,
        )
        logger.info(
            "Pre-cached T2I model backend=%s model=%s snapshot=%s",
            backend_name,
            model_id,
            snapshot_path,
        )
        return snapshot_path
    except Exception as exc:
        logger.exception(
            "Failed to pre-cache T2I model backend=%s model=%s",
            backend_name,
            model_id,
        )
        _reraise_gated_repo_error(model_id, exc)
        raise


def get_t2i_pipeline(backend: str | None = None):
    """Return the active shared T2I pipeline."""
    backend_name, backend_cfg, root_cfg = _resolve_backend_config(backend)
    return _build_pipeline(backend_name, backend_cfg, root_cfg)


def _get_retry_config(root_cfg: dict, backend_name: str) -> tuple[int, float, bool]:
    """Return bounded retry parameters for the active backend."""
    retry_cfg = root_cfg.get("quality_retry", {})
    if not retry_cfg.get("enabled", True):
        return 1, 0.0, True
    attempts_by_backend = retry_cfg.get("max_attempts_by_backend", {})
    threshold_by_backend = retry_cfg.get("clip_threshold_by_backend", {})

    max_attempts = int(attempts_by_backend.get(backend_name, retry_cfg.get("max_attempts", 1)))
    max_attempts = max(1, max_attempts)
    clip_threshold = float(threshold_by_backend.get(backend_name, retry_cfg.get("clip_threshold", 0.0)))
    if not retry_cfg.get("retry_on_low_clip", True):
        clip_threshold = 0.0
    keep_best = bool(retry_cfg.get("keep_best_if_below_threshold", True))
    return max_attempts, clip_threshold, keep_best


def _score_prompt_image_alignment(prompt: str, image_path: Path) -> float:
    """Score prompt/image alignment with CLIP cosine similarity."""
    if prompt not in _TEXT_EMBED_CACHE:
        _TEXT_EMBED_CACHE[prompt] = encode_text(prompt)
    text_emb = _TEXT_EMBED_CACHE[prompt]
    image_emb = encode_image(image_path)
    return cosine_similarity(image_emb, text_emb)


def generate_image(
    description: str,
    output_path: str | Path,
    width: int | None = None,
    height: int | None = None,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    backend: str | None = None,
    save_format: str | None = None,
    save_quality: int = 90,
    output_resolution: int | None = None,
) -> bool:
    """
    Generate a single image using the configured backend.

    A bounded CLIP-based retry loop can regenerate low-alignment outputs without
    letting generation run indefinitely.

    Extra parameters for disk-efficient output:
      save_format:  "JPEG" or "PNG" (default: inferred from output_path extension)
      save_quality: JPEG quality 1-100 (default 90, ignored for PNG)
      output_resolution: if set, resize the image to this square size before saving
    """
    backend_name, backend_cfg, root_cfg = _resolve_backend_config(backend)
    prompt_text = _normalize_description(description)
    if not prompt_text:
        logger.warning("Skipping T2I generation for empty image description: %r", description)
        return False
    pipe = get_t2i_pipeline(backend_name)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    width = int(width or root_cfg.get("width", 1024))
    height = int(height or root_cfg.get("height", 1024))
    steps = int(num_inference_steps or backend_cfg.get("num_inference_steps", 30))
    scale = float(guidance_scale if guidance_scale is not None else backend_cfg.get("guidance_scale", 7.5))
    negative_prompt = root_cfg.get("negative_prompt", "")
    max_sequence_length = backend_cfg.get("max_sequence_length", root_cfg.get("max_sequence_length"))
    min_width = int(root_cfg.get("min_width", 512))
    min_height = int(root_cfg.get("min_height", 512))
    max_attempts, clip_threshold, keep_best = _get_retry_config(root_cfg, backend_name)

    best_tmp: Path | None = None
    best_score = float("-inf")

    try:
        for attempt in range(1, max_attempts + 1):
            tmp_path = out_path.with_name(f"{out_path.stem}.attempt{attempt}{out_path.suffix}")
            generator_device = "cpu" if str(root_cfg.get("device", "cuda")).startswith("cpu") else root_cfg.get("device", "cuda")
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
            generator = torch.Generator(device=generator_device).manual_seed(seed)

            kwargs = {
                "prompt": prompt_text,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "guidance_scale": scale,
                "generator": generator,
            }
            if max_sequence_length is not None and backend_cfg.get("pipeline") == "sd3":
                kwargs["max_sequence_length"] = int(max_sequence_length)

            image = pipe(**kwargs).images[0]
            if output_resolution and (image.width != output_resolution or image.height != output_resolution):
                image = image.resize((output_resolution, output_resolution), resample=3)  # LANCZOS
            fmt = save_format or ("JPEG" if tmp_path.suffix.lower() in (".jpg", ".jpeg") else None)
            if fmt and fmt.upper() == "JPEG":
                image = image.convert("RGB")
                image.save(tmp_path, format="JPEG", quality=save_quality)
            else:
                image.save(tmp_path)

            passed_filter = passes_download_filter(tmp_path, min_width=min_width, min_height=min_height)
            clip_score = _score_prompt_image_alignment(prompt_text, tmp_path) if passed_filter else float("-inf")
            accept = passed_filter and (clip_threshold <= 0.0 or clip_score >= clip_threshold)

            if passed_filter and clip_score > best_score:
                if best_tmp is not None and best_tmp.exists():
                    best_tmp.unlink(missing_ok=True)
                best_tmp = tmp_path
                best_score = clip_score
            else:
                tmp_path.unlink(missing_ok=True)

            if accept:
                if out_path.exists():
                    out_path.unlink()
                assert best_tmp is not None
                best_tmp.rename(out_path)
                logger.info(
                    "Generated image accepted backend=%s attempt=%d clip=%.3f path=%s",
                    backend_name,
                    attempt,
                    clip_score,
                    out_path,
                )
                return True

            logger.info(
                "Regenerating low-quality image backend=%s attempt=%d/%d clip=%.3f threshold=%.3f prompt=%r",
                backend_name,
                attempt,
                max_attempts,
                clip_score,
                clip_threshold,
                prompt_text[:80],
            )

        if keep_best and best_tmp is not None and best_tmp.exists():
            if out_path.exists():
                out_path.unlink()
            best_tmp.rename(out_path)
            logger.warning(
                "Kept best generated image below CLIP threshold backend=%s best_clip=%.3f path=%s",
                backend_name,
                best_score,
                out_path,
            )
            return True

        return False
    except Exception as e:
        logger.warning("T2I generation failed for %r: %s", prompt_text[:80], e)
        return False
    finally:
        for attempt in range(1, max_attempts + 1):
            tmp_path = out_path.with_name(f"{out_path.stem}.attempt{attempt}{out_path.suffix}")
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
