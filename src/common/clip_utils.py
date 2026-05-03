"""CLIP encoding, retrieval, and NSFW detection utilities."""

import html
import json
import logging
import os
import re
import threading
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy-loaded global models
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_nsfw_pipeline = None
_clip_model_lock = threading.Lock()

DEFAULT_CLIP_RETRIEVAL_URL = "https://knn.laion.ai/knn-service"
DEFAULT_OPENVERSE_API_URL = "https://api.openverse.org/v1/images/"
DEFAULT_NASA_IMAGES_API_URL = "https://images-api.nasa.gov"
DEFAULT_METMUSEUM_API_URL = "https://collectionapi.metmuseum.org/public/collection/v1"
WATERMARK_HINTS = (
    "watermark",
    "watermarked",
    "shutterstock",
    "getty",
    "istock",
    "alamy",
    "depositphotos",
    "dreamstime",
    "123rf",
    "adobe stock",
    "stock photo",
    "sample image",
    "preview only",
)
NON_PHOTO_HINTS = (
    "logo",
    "icon",
    "vector",
    "illustration",
    "clipart",
    "graphic",
    "emblem",
    "badge",
    "symbol",
    "diagram",
    "infographic",
    "poster",
    "sticker",
    "banner",
    "cartoon",
    "drawing",
    "render",
    "3d render",
    "screenshot",
    "interface",
    "blueprint",
    "map",
    "coat of arms",
    "seal",
    "flag",
)
DEFAULT_HTTP_HEADERS = {
    "User-Agent": os.environ.get(
        "MIS_HTTP_USER_AGENT",
        os.environ.get(
            "MIS_WIKIMEDIA_USER_AGENT",
            "multi-image-safety-research/0.1 (non-commercial research use; contact: unset)",
        ),
    ),
}

# ── Cross-process rate limiter for web API backends ───────────────────────
# Uses a shared lock file so parallel GPU workers don't flood the API.
import fcntl
import time as _time_mod

_RATE_LIMIT_DIR = Path("/tmp/mis_ratelimits")
_RATE_LIMIT_DIR.mkdir(exist_ok=True)
_BACKEND_SESSION_ID = os.environ.setdefault(
    "MIS_BACKEND_SESSION_ID",
    f"{int(_time_mod.time())}-{os.getpid()}",
)
_BACKEND_DISABLE_DIR = Path("/tmp/mis_backend_state") / _BACKEND_SESSION_ID
_BACKEND_DISABLE_DIR.mkdir(parents=True, exist_ok=True)
_BACKEND_SKIP_LOGGED: set[str] = set()

# Per-backend minimum intervals (seconds between consecutive requests)
_BACKEND_MIN_INTERVALS: dict[str, float] = {
    "wikimedia_commons": float(os.environ.get("MIS_WIKIMEDIA_MIN_INTERVAL", "3.0")),
    "openverse": float(os.environ.get("MIS_OPENVERSE_MIN_INTERVAL", "0.5")),
    "pexels": float(os.environ.get("MIS_PEXELS_MIN_INTERVAL", "0.2")),
    "pixabay": float(os.environ.get("MIS_PIXABAY_MIN_INTERVAL", "0.2")),
}


def _disabled_backend_marker(backend: str) -> Path:
    """Return the run-scoped marker file for a disabled backend."""
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", backend)
    return _BACKEND_DISABLE_DIR / f"{safe_name}.disabled"


def _disable_backend_for_run(backend: str, reason: str) -> None:
    """Disable a backend for the current run after a connection/API failure."""
    marker = _disabled_backend_marker(backend)
    try:
        marker.write_text(reason[:2000] + "\n", encoding="utf-8")
    except Exception:
        logger.debug("Failed to persist disabled-backend marker for %s", backend, exc_info=True)
    logger.warning("Disabling backend %s for this run after failure: %s", backend, reason)


def _is_backend_disabled_for_run(backend: str) -> bool:
    """Return True when a backend has been disabled for this run."""
    marker = _disabled_backend_marker(backend)
    if marker.exists():
        if backend not in _BACKEND_SKIP_LOGGED:
            try:
                reason = marker.read_text(encoding="utf-8").strip()
            except Exception:
                reason = ""
            logger.warning(
                "Skipping backend %s for this run because it previously failed%s",
                backend,
                f": {reason}" if reason else "",
            )
            _BACKEND_SKIP_LOGGED.add(backend)
        return True
    return False


def _backend_rate_limit(backend: str) -> None:
    """Block until enough time has passed since the last request to *backend*."""
    interval = _BACKEND_MIN_INTERVALS.get(backend, 0.0)
    if interval <= 0:
        return
    lock_path = _RATE_LIMIT_DIR / f"{backend}.lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            last_ts = float(f.read().strip())
        except (ValueError, OSError):
            last_ts = 0.0
        wait = interval - (_time_mod.time() - last_ts)
        if wait > 0:
            _time_mod.sleep(wait)
        f.seek(0)
        f.truncate()
        f.write(str(_time_mod.time()))
        fcntl.flock(f, fcntl.LOCK_UN)


class ImageSourceError(RuntimeError):
    """Raised when a configured image backend cannot be queried safely."""


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_clip_model(
    model_name: str = "ViT-L-14",
    pretrained: str = "openai",
):
    """Load CLIP model (lazy singleton)."""
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is not None:
        return _clip_model, _clip_preprocess, _clip_tokenizer
    with _clip_model_lock:
        if _clip_model is not None:
            return _clip_model, _clip_preprocess, _clip_tokenizer

        import open_clip

        device = _get_device()
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model.eval()

        _clip_model = model
        _clip_preprocess = preprocess
        _clip_tokenizer = tokenizer
        logger.info(f"Loaded CLIP model {model_name}/{pretrained} on {device}")
        return model, preprocess, tokenizer


def encode_image(image_path: str | Path, normalize: bool = True) -> np.ndarray:
    """Encode a single image using CLIP. Returns a 1-D embedding."""
    model, preprocess, _ = load_clip_model()
    device = _get_device()

    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.encode_image(image_tensor)
        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().flatten()


def encode_images_batch(
    image_paths: list[str | Path],
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    """Encode multiple images in batches. Returns (N, D) array."""
    model, preprocess, _ = load_clip_model()
    device = _get_device()

    all_features = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(preprocess(img))
            except Exception as e:
                logger.warning(f"Failed to load {p}: {e}")
                # Use a zero tensor as placeholder
                images.append(torch.zeros(3, 224, 224))

        batch_tensor = torch.stack(images).to(device)
        with torch.no_grad():
            features = model.encode_image(batch_tensor)
            if normalize:
                features = features / features.norm(dim=-1, keepdim=True)
        all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def encode_text(text: str, normalize: bool = True) -> np.ndarray:
    """Encode a single text using CLIP. Returns a 1-D embedding."""
    model, _, tokenizer = load_clip_model()
    device = _get_device()

    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().flatten()


def encode_texts_batch(
    texts: list[str],
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    """Encode multiple texts in batches. Returns (N, D) array."""
    model, _, tokenizer = load_clip_model()
    device = _get_device()

    all_features = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            features = model.encode_text(tokens)
            if normalize:
                features = features / features.norm(dim=-1, keepdim=True)
        all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix. a: (N,D), b: (M,D) -> (N,M)."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


def find_harm_pairs(
    image_embeddings: np.ndarray,
    image_paths: list[str],
    harm_vector: np.ndarray,
    theta_safe: float = 0.40,
    theta_harm: float = 0.35,
    max_pairs: int = 1000,
) -> list[tuple[int, int, float]]:
    """
    Find image pairs where each image is individually safe but their
    combination approaches a harm vector in CLIP embedding space.

    Args:
        image_embeddings: (N, D) array of CLIP image embeddings
        image_paths: List of image file paths
        harm_vector: (D,) CLIP text embedding of harm description
        theta_safe: Max individual image-harm similarity
        theta_harm: Min combined pair-harm similarity
        max_pairs: Maximum number of pairs to return

    Returns:
        List of (idx1, idx2, combined_score) tuples
    """
    N = len(image_embeddings)
    # Step 1: Filter individually safe images
    individual_sims = image_embeddings @ harm_vector
    safe_mask = individual_sims < theta_safe
    safe_indices = np.where(safe_mask)[0]

    logger.info(
        f"Found {len(safe_indices)}/{N} individually safe images "
        f"(threshold={theta_safe})"
    )

    if len(safe_indices) < 2:
        return []

    # Step 2: Find pairs whose combined embedding exceeds harm threshold
    safe_embeddings = image_embeddings[safe_indices]
    pairs = []

    # Vectorized pairwise combined score computation in blocks
    block_size = 500
    for i in range(0, len(safe_indices), block_size):
        block_i = safe_embeddings[i : i + block_size]
        for j in range(i, len(safe_indices), block_size):
            block_j = safe_embeddings[j : j + block_size]

            # combined[a,b] = block_i[a] + block_j[b], then normalize
            combined = block_i[:, None, :] + block_j[None, :, :]  # (B1, B2, D)
            norms = np.linalg.norm(combined, axis=2, keepdims=True) + 1e-8
            combined = combined / norms
            scores = combined @ harm_vector  # (B1, B2)

            # Mask out duplicate/self pairs when blocks overlap
            if i == j:
                mask = np.triu(np.ones(scores.shape, dtype=bool), k=1)
                scores = np.where(mask, scores, -1.0)

            hits = np.argwhere(scores > theta_harm)
            for ii, jj in hits:
                real_i = safe_indices[i + int(ii)]
                real_j = safe_indices[j + int(jj)]
                pairs.append((int(real_i), int(real_j), float(scores[ii, jj])))

            if len(pairs) >= max_pairs:
                pairs.sort(key=lambda x: x[2], reverse=True)
                return pairs[:max_pairs]

    pairs.sort(key=lambda x: x[2], reverse=True)
    logger.info(f"Found {len(pairs)} harm pairs (threshold={theta_harm})")
    return pairs[:max_pairs]


def check_nsfw(image_path: str | Path) -> float:
    """
    Check NSFW score of an image using a lightweight classifier.
    Returns probability of being NSFW (0-1).
    """
    global _nsfw_pipeline
    if _nsfw_pipeline is None:
        from transformers import pipeline

        _nsfw_pipeline = pipeline(
            "image-classification",
            model="Falconsai/nsfw_image_detection",
            device=0 if torch.cuda.is_available() else -1,
        )

    image = Image.open(image_path).convert("RGB")
    results = _nsfw_pipeline(image)
    for r in results:
        if r["label"] == "nsfw":
            return r["score"]
    return 0.0


def _is_laion_enabled() -> bool:
    """Check if external image retrieval is enabled in config."""
    try:
        from src.common.utils import load_config
        config = load_config()
        return config.get("laion", {}).get("enabled", False)
    except Exception:
        return False


def _get_retrieval_config() -> dict:
    """Load the retrieval backend configuration."""
    try:
        from src.common.utils import load_config
        config = load_config()
        return config.get("laion", {})
    except Exception:
        return {}


def _strict_backend_errors_enabled() -> bool:
    """Whether retrieval backend failures should abort the pipeline."""
    cfg = _get_retrieval_config()
    return bool(cfg.get("fail_on_backend_error", False))


def _require_backend_credentials() -> bool:
    """Whether configured credential-backed backends must have valid env vars."""
    cfg = _get_retrieval_config()
    return bool(cfg.get("require_backend_credentials", False))


def _get_backend_config(backend_name: str) -> dict:
    """Return config for a specific retrieval backend."""
    cfg = _get_retrieval_config()
    return cfg.get(backend_name, {})


def _format_http_error(e: urllib.error.HTTPError) -> str:
    """Create a compact HTTP error detail string."""
    detail = str(e)
    try:
        body = e.read().decode("utf-8", errors="replace").strip()
    except Exception:
        body = ""
    if body:
        body = " ".join(body.split())
        return f"HTTP {e.code}: {detail}; body={body[:300]}"
    return f"HTTP {e.code}: {detail}"


def _raise_image_source_error(backend_name: str, query_text: str, detail: str) -> None:
    """Raise a standardized backend failure."""
    raise ImageSourceError(
        f"Image backend '{backend_name}' failed for query '{query_text}': {detail}"
    )


def _get_backend_env(
    backend_name: str,
    env_key: str,
    *,
    required: bool = False,
) -> str:
    """Read a backend credential from the configured environment variable."""
    backend_cfg = _get_backend_config(backend_name)
    env_name = str(backend_cfg.get(env_key, "")).strip()
    if not env_name:
        if required:
            _raise_image_source_error(
                backend_name,
                "",
                f"missing config key '{env_key}' for backend credentials",
            )
        return ""

    value = os.environ.get(env_name, "").strip()
    if required and not value:
        _raise_image_source_error(
            backend_name,
            "",
            f"required environment variable '{env_name}' is not set",
        )
    return value


def _strip_html(text: str) -> str:
    """Convert simple HTML snippets into plain text."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(" ".join(text.split()))


def _has_watermark_hint(*values: str) -> bool:
    """Use conservative metadata/url heuristics to avoid likely watermarked files."""
    haystack = " ".join(_strip_html(str(v)).lower() for v in values if v)
    return any(hint in haystack for hint in WATERMARK_HINTS)


def _has_non_photo_hint(*values: str) -> bool:
    """Conservative metadata/url heuristic for graphics, logos, and illustrations."""
    haystack = " ".join(_strip_html(str(v)).lower() for v in values if v)
    return any(hint in haystack for hint in NON_PHOTO_HINTS)


def _normalize_result(
    *,
    url: str,
    caption: str = "",
    width: int = 0,
    height: int = 0,
    similarity: float = 0.0,
    source_page: str = "",
    provider: str = "",
    license_name: str = "",
    license_url: str = "",
    creator: str = "",
    source_name: str = "",
    attribution: str = "",
) -> dict:
    """Normalize provider-specific metadata into a common image result schema."""
    return {
        "url": url,
        "caption": caption,
        "width": width,
        "height": height,
        "similarity": similarity,
        "source_page": source_page,
        "provider": provider,
        "license": license_name,
        "license_url": license_url,
        "creator": creator,
        "source": source_name,
        "attribution": attribution,
    }


def download_image_url(url: str, output_path: str | Path, timeout: int = 30) -> None:
    """Download an image with a stable research user-agent."""
    request = urllib.request.Request(url, headers=DEFAULT_HTTP_HEADERS)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=timeout) as response, open(output_path, "wb") as fout:
        fout.write(response.read())


def passes_download_filter(
    image_path: str | Path,
    min_width: int = 256,
    min_height: int = 256,
) -> bool:
    """Reject obviously low-quality non-photo assets after download."""
    try:
        img = Image.open(image_path)
        width, height = img.size
        if width < min_width or height < min_height:
            return False

        if "A" in img.getbands():
            alpha = np.asarray(img.getchannel("A").resize((128, 128)))
            if float((alpha < 250).mean()) > 0.20:
                return False

        rgb = img.convert("RGB").resize((128, 128))
        arr = np.asarray(rgb)
        flat = arr.reshape(-1, 3)
        quantized = (flat // 32).astype(np.uint8)
        unique_colors = len(np.unique(quantized, axis=0))
        white_ratio = float(((flat > 245).all(axis=1)).mean())
        black_ratio = float(((flat < 10).all(axis=1)).mean())

        if unique_colors < 24:
            return False
        if white_ratio > 0.88 and unique_colors < 48:
            return False
        if black_ratio > 0.88 and unique_colors < 48:
            return False
        return True
    except Exception:
        return False


def _normalize_allowlist(values: list[str] | tuple[str, ...] | None) -> set[str]:
    """Normalize configured allowlists for case-insensitive matching."""
    if not values:
        return set()
    normalized = set()
    for value in values:
        text = str(value).strip().lower()
        if text:
            normalized.add(text)
    return normalized


def _infer_filetype_from_url(url: str) -> str:
    """Infer a lowercase extension from a media URL when metadata omits it."""
    path = urllib.parse.urlparse(url).path
    suffix = Path(path).suffix.lower().lstrip(".")
    return suffix


def _dedupe_normalized_results(results: list[dict], limit: int) -> list[dict]:
    """Deduplicate normalized image results by source URL."""
    unique = []
    seen = set()
    for item in results:
        key = item.get("url", "") or item.get("source_page", "")
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= limit:
            break
    return unique


def _build_openverse_caption(result: dict) -> str:
    """Create a compact text description from Openverse metadata."""
    title = _strip_html(result.get("title", ""))
    if title:
        return title

    tags = []
    seen = set()
    for tag in result.get("tags", []) or []:
        name = _strip_html(tag.get("name", ""))
        if not name:
            continue
        lowered = name.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        tags.append(name)
        if len(tags) >= 4:
            break
    return ", ".join(tags)


def _retrieve_from_clip_retrieval(
    query_text: str,
    num_results: int,
    aesthetic_score_min: float,
    min_width: int,
    min_height: int,
) -> list[dict]:
    """Retrieve images from a clip-retrieval backend."""
    cfg = _get_retrieval_config()
    try:
        from clip_retrieval.clip_client import ClipClient
        try:
            from clip_retrieval.clip_client import Modality
            modality = Modality.IMAGE
        except ImportError:
            modality = "image"

        client = ClipClient(
            url=cfg.get("index_url", DEFAULT_CLIP_RETRIEVAL_URL),
            indice_name=cfg.get("indice_name", "laion5B-L-14"),
            aesthetic_score=aesthetic_score_min,
            aesthetic_weight=0.5,
            modality=modality,
            num_images=num_results,
        )
        results = client.query(text=query_text)
        filtered = []
        for result in results:
            width = int(result.get("width", 0) or 0)
            height = int(result.get("height", 0) or 0)
            url = result.get("url", "")
            caption = result.get("caption", "")
            if width < min_width or height < min_height or not url:
                continue
            if _has_watermark_hint(url, caption):
                continue
            if _has_non_photo_hint(url, caption):
                continue
            filtered.append(_normalize_result(
                url=url,
                caption=caption,
                width=width,
                height=height,
                similarity=float(result.get("similarity", 0) or 0),
                provider="clip-retrieval",
            ))
        return filtered
    except Exception as e:
        detail = str(e)
        logger.error("clip-retrieval failed for '%s': %s", query_text, detail)
        if _strict_backend_errors_enabled():
            _raise_image_source_error("clip-retrieval", query_text, detail)
        return []


def _retrieve_from_wikimedia_commons(
    query_text: str,
    num_results: int,
    min_width: int,
    min_height: int,
) -> list[dict]:
    """Retrieve freely licensed images from Wikimedia Commons."""
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "generator": "search",
        "gsrsearch": query_text,
        "gsrnamespace": "6",
        "gsrlimit": str(min(max(num_results * 4, 20), 50)),
        "prop": "imageinfo|categories",
        "iiprop": "url|size|mime|extmetadata",
        "cllimit": "max",
    }
    url = (
        "https://commons.wikimedia.org/w/api.php?"
        + urllib.parse.urlencode(params)
    )

    max_retries = int(os.environ.get("MIS_WIKIMEDIA_MAX_RETRIES", "4"))
    data = None
    for attempt in range(max_retries):
        _backend_rate_limit("wikimedia_commons")
        try:
            if attempt == 0:
                logger.info(
                    "Wikimedia request: query=%r limit=%s ua=%r min_interval=%ss",
                    query_text,
                    params["gsrlimit"],
                    DEFAULT_HTTP_HEADERS.get("User-Agent", ""),
                    _BACKEND_MIN_INTERVALS.get("wikimedia_commons", 0.0),
                )
            request = urllib.request.Request(url, headers=DEFAULT_HTTP_HEADERS)
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                wait = int(os.environ.get("MIS_WIKIMEDIA_RETRY_BASE_SECONDS", "5")) * (attempt + 1)
                logger.warning(
                    "Wikimedia 429 for '%s' (attempt %d/%d), retrying in %ds",
                    query_text, attempt + 1, max_retries, wait,
                )
                _time_mod.sleep(wait)
                continue
            detail = _format_http_error(e)
            logger.error("Wikimedia Commons retrieval failed for '%s': %s", query_text, detail)
            _disable_backend_for_run("wikimedia_commons", detail)
            if _strict_backend_errors_enabled():
                _raise_image_source_error("wikimedia_commons", query_text, detail)
            return []
        except Exception as e:
            detail = str(e)
            logger.error("Wikimedia Commons retrieval failed for '%s': %s", query_text, detail)
            _disable_backend_for_run("wikimedia_commons", detail)
            if _strict_backend_errors_enabled():
                _raise_image_source_error("wikimedia_commons", query_text, detail)
            return []
    if data is None:
        return []

    pages = data.get("query", {}).get("pages", [])
    results = []
    for page in pages:
        imageinfo = (page.get("imageinfo") or [{}])[0]
        file_url = imageinfo.get("url", "")
        mime = imageinfo.get("mime", "")
        width = int(imageinfo.get("width", 0) or 0)
        height = int(imageinfo.get("height", 0) or 0)
        if not file_url or mime not in {"image/jpeg", "image/png"}:
            continue
        if width < min_width or height < min_height:
            continue

        extmeta = imageinfo.get("extmetadata", {})
        caption = _strip_html(extmeta.get("ImageDescription", {}).get("value", ""))
        creator = _strip_html(extmeta.get("Artist", {}).get("value", ""))
        credit = _strip_html(extmeta.get("Credit", {}).get("value", ""))
        license_name = _strip_html(extmeta.get("LicenseShortName", {}).get("value", ""))
        license_url = extmeta.get("LicenseUrl", {}).get("value", "")
        title = page.get("title", "")
        category_text = " ".join(cat.get("title", "") for cat in page.get("categories", []))
        source_page = ""
        if title:
            source_page = "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(
                title.replace(" ", "_"),
                safe=":/_",
            )

        if _has_watermark_hint(
            title,
            caption,
            creator,
            credit,
            category_text,
            file_url,
        ):
            continue
        if _has_non_photo_hint(
            title,
            caption,
            creator,
            credit,
            category_text,
            file_url,
        ):
            continue

        results.append(_normalize_result(
            url=file_url,
            caption=caption,
            width=width,
            height=height,
            source_page=source_page,
            provider="wikimedia_commons",
            license_name=license_name,
            license_url=license_url,
            creator=creator,
        ))
        if len(results) >= num_results:
            break

    return results


_openverse_token_cache: dict[str, float | str] = {}


def _refresh_openverse_token() -> str:
    """Refresh the Openverse OAuth token using client credentials.

    Returns the new token, or "" if refresh is not possible.
    Updates the OPENVERSE_API_TOKEN env var so subsequent calls pick it up.
    """
    client_id = os.environ.get("OPENVERSE_CLIENT_ID", "").strip()
    client_secret = os.environ.get("OPENVERSE_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        return ""

    try:
        body = urllib.parse.urlencode({
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }).encode("utf-8")
        request = urllib.request.Request(
            "https://api.openverse.org/v1/auth_tokens/token/",
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        new_token = data.get("access_token", "")
        expires_in = int(data.get("expires_in", 43200))
        if new_token:
            os.environ["OPENVERSE_API_TOKEN"] = new_token
            _openverse_token_cache["token"] = new_token
            _openverse_token_cache["expires_at"] = _time_mod.time() + expires_in - 60
            logger.info("Openverse token refreshed, expires in %ds", expires_in)
            return new_token
    except Exception as e:
        logger.warning("Failed to refresh Openverse token: %s", e)
    return ""


def _get_openverse_token() -> str:
    """Get a valid Openverse token, refreshing if expired."""
    cached = _openverse_token_cache.get("token", "")
    expires_at = float(_openverse_token_cache.get("expires_at", 0))
    if cached and _time_mod.time() < expires_at:
        return str(cached)
    # Try env var first
    env_token = os.environ.get("OPENVERSE_API_TOKEN", "").strip()
    if env_token and not cached:
        # First use — trust the env var, set a 12h expiry guess
        _openverse_token_cache["token"] = env_token
        _openverse_token_cache["expires_at"] = _time_mod.time() + 43200 - 60
        return env_token
    # Token expired or missing — try refresh
    refreshed = _refresh_openverse_token()
    return refreshed or env_token


def _retrieve_from_openverse(
    query_text: str,
    num_results: int,
    min_width: int,
    min_height: int,
) -> list[dict]:
    """Retrieve openly licensed images from the official Openverse API."""
    cfg = _get_retrieval_config()
    ov_cfg = cfg.get("openverse", {})
    allowed_licenses = _normalize_allowlist(
        ov_cfg.get("allowed_licenses", ["by", "by-sa", "cc0", "pdm"])
    )
    allowed_sources = _normalize_allowlist(
        ov_cfg.get("allowed_sources", ["flickr"])
    )

    params = {
        "q": query_text,
        "page_size": str(min(int(ov_cfg.get("page_size", 20)), 20)),
    }
    if allowed_licenses:
        params["license"] = ",".join(sorted(allowed_licenses))
    if allowed_sources:
        params["source"] = ",".join(sorted(allowed_sources))

    base_url = ov_cfg.get("api_url", DEFAULT_OPENVERSE_API_URL)
    url = base_url.rstrip("?") + "?" + urllib.parse.urlencode(params)

    headers = dict(DEFAULT_HTTP_HEADERS)
    ov_token = _get_openverse_token()
    if ov_token:
        headers["Authorization"] = f"Bearer {ov_token}"

    import time as _time
    data = None
    max_retries = 3
    for attempt in range(max_retries):
        _backend_rate_limit("openverse")
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 429):
                # Try refreshing the token on auth errors
                if e.code in (401, 403) and attempt == 0:
                    refreshed = _refresh_openverse_token()
                    if refreshed:
                        headers["Authorization"] = f"Bearer {refreshed}"
                        logger.info("Openverse token refreshed after %d, retrying", e.code)
                        continue
                wait = 2 ** (attempt + 1)
                logger.warning(
                    "Openverse API %d for '%s' (attempt %d/%d), retrying in %ds",
                    e.code, query_text, attempt + 1, max_retries, wait,
                )
                _time.sleep(wait)
                continue
            detail = _format_http_error(e)
            logger.error("Openverse retrieval failed for '%s': %s", query_text, detail)
            _disable_backend_for_run("openverse", detail)
            if _strict_backend_errors_enabled():
                _raise_image_source_error("openverse", query_text, detail)
            return []
        except Exception as e:
            detail = str(e)
            logger.error("Openverse retrieval failed for '%s': %s", query_text, detail)
            _disable_backend_for_run("openverse", detail)
            if _strict_backend_errors_enabled():
                _raise_image_source_error("openverse", query_text, detail)
            return []
    if data is None:
        _disable_backend_for_run("openverse", "request exhausted retries or token was rejected")
        if _strict_backend_errors_enabled():
            _raise_image_source_error(
                "openverse",
                query_text,
                "request exhausted retries or token was rejected",
            )
        return []

    exclude_mature = ov_cfg.get("exclude_mature", True)
    results = []
    for item in data.get("results", []):
        file_url = item.get("url", "")
        width = int(item.get("width", 0) or 0)
        height = int(item.get("height", 0) or 0)
        if not file_url or width < min_width or height < min_height:
            continue

        if exclude_mature and item.get("mature") is True:
            continue

        source_name = str(item.get("source", "") or item.get("provider", "")).strip().lower()
        if allowed_sources and source_name not in allowed_sources:
            continue

        license_code = str(item.get("license", "")).strip().lower()
        if allowed_licenses and license_code not in allowed_licenses:
            continue

        filetype = str(item.get("filetype", "") or _infer_filetype_from_url(file_url)).strip().lower()
        if filetype and filetype not in {"jpg", "jpeg", "png"}:
            continue

        category = _strip_html(str(item.get("category", "")))
        if category and "photo" not in category.lower():
            continue

        caption = _build_openverse_caption(item)
        title = _strip_html(item.get("title", ""))
        creator = _strip_html(item.get("creator", ""))
        attribution = _strip_html(item.get("attribution", ""))
        source_page = item.get("foreign_landing_url", "") or item.get("detail_url", "")

        if _has_watermark_hint(title, caption, creator, attribution, file_url):
            continue
        if _has_non_photo_hint(title, caption, creator, attribution, category, file_url):
            continue

        license_name = license_code.upper()
        license_version = str(item.get("license_version", "")).strip()
        if license_name and license_version:
            license_name = f"{license_name} {license_version}"

        results.append(_normalize_result(
            url=file_url,
            caption=caption,
            width=width,
            height=height,
            source_page=source_page,
            provider=f"openverse_{source_name}" if source_name else "openverse",
            license_name=license_name,
            license_url=item.get("license_url", ""),
            creator=creator,
            source_name=source_name,
            attribution=attribution,
        ))
        if len(results) >= num_results:
            break

    return results


def _retrieve_from_nasa_images(
    query_text: str,
    num_results: int,
    min_width: int,
    min_height: int,
) -> list[dict]:
    """Retrieve image candidates from the NASA Image and Video Library."""
    cfg = _get_retrieval_config()
    nasa_cfg = cfg.get("nasa_images", {})
    params = urllib.parse.urlencode({
        "q": query_text,
        "media_type": "image",
        "page": "1",
        "page_size": str(min(int(nasa_cfg.get("page_size", max(num_results * 4, 20))), 100)),
    })
    base_url = nasa_cfg.get("api_url", DEFAULT_NASA_IMAGES_API_URL).rstrip("/")
    url = f"{base_url}/search?{params}"

    try:
        request = urllib.request.Request(url, headers=DEFAULT_HTTP_HEADERS)
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = _format_http_error(e)
        logger.error("NASA retrieval failed for '%s': %s", query_text, detail)
        if _strict_backend_errors_enabled():
            _raise_image_source_error("nasa_images", query_text, detail)
        return []
    except Exception as e:
        detail = str(e)
        logger.error("NASA retrieval failed for '%s': %s", query_text, detail)
        if _strict_backend_errors_enabled():
            _raise_image_source_error("nasa_images", query_text, detail)
        return []

    results = []
    items = data.get("collection", {}).get("items", [])
    for item in items:
        data_items = item.get("data") or []
        if not data_items:
            continue

        meta = data_items[0]
        links = item.get("links") or []
        file_url = ""
        for link in links:
            if str(link.get("render", "")).strip().lower() != "image":
                continue
            href = str(link.get("href", "")).strip()
            if href:
                file_url = href
                break
        if not file_url:
            continue

        title = _strip_html(meta.get("title", ""))
        description = _strip_html(meta.get("description", ""))
        keywords = ", ".join(str(k).strip() for k in meta.get("keywords", []) if str(k).strip())
        nasa_id = str(meta.get("nasa_id", "")).strip()
        creator = _strip_html(
            meta.get("photographer")
            or meta.get("secondary_creator")
            or meta.get("center", "")
        )
        source_page = f"https://images.nasa.gov/details/{urllib.parse.quote(nasa_id)}" if nasa_id else ""

        if _has_watermark_hint(title, description, keywords, file_url):
            continue
        if _has_non_photo_hint(title, description, keywords):
            continue

        # NASA search results do not reliably expose pixel dimensions. We defer
        # size validation to the post-download filter when width/height are unknown.
        width = int(meta.get("width", 0) or 0)
        height = int(meta.get("height", 0) or 0)
        if width and width < min_width:
            continue
        if height and height < min_height:
            continue

        caption = description or title
        results.append(_normalize_result(
            url=file_url,
            caption=caption,
            width=width,
            height=height,
            source_page=source_page,
            provider="nasa_images",
            license_name="NASA Media Usage Guidelines",
            creator=creator,
        ))
        if len(results) >= num_results:
            break

    return results


def _retrieve_from_metmuseum(
    query_text: str,
    num_results: int,
    min_width: int,
    min_height: int,
) -> list[dict]:
    """Retrieve open-access image candidates from The Met Collection API."""
    cfg = _get_retrieval_config()
    met_cfg = cfg.get("metmuseum", {})
    base_url = met_cfg.get("api_url", DEFAULT_METMUSEUM_API_URL).rstrip("/")
    search_params = urllib.parse.urlencode({
        "q": query_text,
        "hasImages": "true",
    })
    search_url = f"{base_url}/search?{search_params}"

    try:
        request = urllib.request.Request(search_url, headers=DEFAULT_HTTP_HEADERS)
        with urllib.request.urlopen(request, timeout=30) as response:
            search_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = _format_http_error(e)
        logger.error("MetMuseum search failed for '%s': %s", query_text, detail)
        if _strict_backend_errors_enabled():
            _raise_image_source_error("metmuseum", query_text, detail)
        return []
    except Exception as e:
        detail = str(e)
        logger.error("MetMuseum search failed for '%s': %s", query_text, detail)
        if _strict_backend_errors_enabled():
            _raise_image_source_error("metmuseum", query_text, detail)
        return []

    object_ids = search_data.get("objectIDs") or []
    max_object_fetch = int(met_cfg.get("max_object_fetch", max(num_results * 6, 24)))
    results = []
    for object_id in object_ids[:max_object_fetch]:
        object_url = f"{base_url}/objects/{object_id}"
        try:
            request = urllib.request.Request(object_url, headers=DEFAULT_HTTP_HEADERS)
            with urllib.request.urlopen(request, timeout=30) as response:
                item = json.loads(response.read().decode("utf-8"))
        except Exception as e:
            detail = str(e)
            logger.debug("MetMuseum object %s failed for '%s': %s", object_id, query_text, detail)
            if _strict_backend_errors_enabled():
                _raise_image_source_error(
                    "metmuseum",
                    query_text,
                    f"object lookup {object_id} failed: {detail}",
                )
            continue

        if item.get("isPublicDomain") is not True:
            continue

        file_url = str(item.get("primaryImage") or item.get("primaryImageSmall") or "").strip()
        if not file_url:
            continue

        title = _strip_html(item.get("title", ""))
        object_name = _strip_html(item.get("objectName", ""))
        medium = _strip_html(item.get("medium", ""))
        artist_name = _strip_html(item.get("artistDisplayName", ""))
        source_page = str(item.get("objectURL", "")).strip()

        if _has_watermark_hint(title, object_name, medium, file_url):
            continue
        if _has_non_photo_hint(title, object_name, medium):
            continue

        # The Met object API exposes image URLs but not pixel dimensions in the
        # object payload, so post-download validation remains the final gate.
        width = int(item.get("imageWidth", 0) or 0)
        height = int(item.get("imageHeight", 0) or 0)
        if width and width < min_width:
            continue
        if height and height < min_height:
            continue

        caption_parts = [part for part in (title, object_name, medium) if part]
        results.append(_normalize_result(
            url=file_url,
            caption=" - ".join(caption_parts[:3]),
            width=width,
            height=height,
            source_page=source_page,
            provider="metmuseum",
            license_name="CC0 / Public Domain",
            creator=artist_name,
        ))
        if len(results) >= num_results:
            break

    return results


def _retrieve_from_pexels(
    query_text: str,
    num_results: int,
    min_width: int,
    min_height: int,
) -> list[dict]:
    """Retrieve images from the Pexels API (requires PEXELS_API_KEY)."""
    api_key = _get_backend_env(
        "pexels",
        "api_key_env",
        required=_strict_backend_errors_enabled() and _require_backend_credentials(),
    )
    if not api_key:
        logger.warning("PEXELS_API_KEY not set — skipping Pexels backend")
        return []

    params = urllib.parse.urlencode({
        "query": query_text,
        "per_page": str(min(num_results * 3, 80)),
        "size": "large",
    })
    url = f"https://api.pexels.com/v1/search?{params}"
    headers = dict(DEFAULT_HTTP_HEADERS)
    headers["Authorization"] = api_key

    _backend_rate_limit("pexels")
    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = _format_http_error(e)
        logger.error("Pexels retrieval failed for '%s': %s", query_text, detail)
        _disable_backend_for_run("pexels", detail)
        if _strict_backend_errors_enabled():
            _raise_image_source_error("pexels", query_text, detail)
        return []
    except Exception as e:
        detail = str(e)
        logger.error("Pexels retrieval failed for '%s': %s", query_text, detail)
        _disable_backend_for_run("pexels", detail)
        if _strict_backend_errors_enabled():
            _raise_image_source_error("pexels", query_text, detail)
        return []

    results = []
    for photo in data.get("photos", []):
        src = photo.get("src", {})
        file_url = src.get("large2x") or src.get("original", "")
        width = int(photo.get("width", 0) or 0)
        height = int(photo.get("height", 0) or 0)
        if not file_url or width < min_width or height < min_height:
            continue

        caption = photo.get("alt", "") or ""
        photographer = photo.get("photographer", "") or ""
        photo_url = photo.get("url", "")

        if _has_watermark_hint(file_url, caption, photographer):
            continue
        if _has_non_photo_hint(file_url, caption):
            continue

        results.append(_normalize_result(
            url=file_url,
            caption=caption,
            width=width,
            height=height,
            source_page=photo_url,
            provider="pexels",
            license_name="Pexels License",
            creator=photographer,
        ))
        if len(results) >= num_results:
            break

    return results


def _retrieve_from_pixabay(
    query_text: str,
    num_results: int,
    min_width: int,
    min_height: int,
) -> list[dict]:
    """Retrieve images from the Pixabay API (requires PIXABAY_API_KEY)."""
    api_key = _get_backend_env(
        "pixabay",
        "api_key_env",
        required=_strict_backend_errors_enabled() and _require_backend_credentials(),
    )
    if not api_key:
        logger.warning("PIXABAY_API_KEY not set — skipping Pixabay backend")
        return []

    params = urllib.parse.urlencode({
        "key": api_key,
        "q": query_text,
        "per_page": str(min(num_results * 3, 200)),
        "image_type": "photo",
        "safesearch": "true",
        "min_width": str(min_width),
        "min_height": str(min_height),
    })
    url = f"https://pixabay.com/api/?{params}"

    _backend_rate_limit("pixabay")
    try:
        request = urllib.request.Request(url, headers=DEFAULT_HTTP_HEADERS)
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = _format_http_error(e)
        logger.error("Pixabay retrieval failed for '%s': %s", query_text, detail)
        _disable_backend_for_run("pixabay", detail)
        if _strict_backend_errors_enabled():
            _raise_image_source_error("pixabay", query_text, detail)
        return []
    except Exception as e:
        detail = str(e)
        logger.error("Pixabay retrieval failed for '%s': %s", query_text, detail)
        _disable_backend_for_run("pixabay", detail)
        if _strict_backend_errors_enabled():
            _raise_image_source_error("pixabay", query_text, detail)
        return []

    results = []
    for hit in data.get("hits", []):
        file_url = hit.get("largeImageURL", "")
        width = int(hit.get("imageWidth", 0) or 0)
        height = int(hit.get("imageHeight", 0) or 0)
        if not file_url or width < min_width or height < min_height:
            continue

        caption = hit.get("tags", "") or ""
        creator = hit.get("user", "") or ""
        page_url = hit.get("pageURL", "")

        if _has_watermark_hint(file_url, caption, creator):
            continue
        if _has_non_photo_hint(file_url, caption):
            continue

        results.append(_normalize_result(
            url=file_url,
            caption=caption,
            width=width,
            height=height,
            source_page=page_url,
            provider="pixabay",
            license_name="Pixabay License",
            creator=creator,
        ))
        if len(results) >= num_results:
            break

    return results


def _retrieve_from_backend(
    backend: str,
    query_text: str,
    num_results: int,
    aesthetic_score_min: float,
    min_width: int,
    min_height: int,
) -> list[dict]:
    """Dispatch retrieval to a single backend."""
    if _is_backend_disabled_for_run(backend):
        return []
    if backend == "clip-retrieval":
        return _retrieve_from_clip_retrieval(
            query_text,
            num_results=num_results,
            aesthetic_score_min=aesthetic_score_min,
            min_width=min_width,
            min_height=min_height,
        )
    if backend == "wikimedia_commons":
        return _retrieve_from_wikimedia_commons(
            query_text,
            num_results=num_results,
            min_width=min_width,
            min_height=min_height,
        )
    if backend == "openverse":
        return _retrieve_from_openverse(
            query_text,
            num_results=num_results,
            min_width=min_width,
            min_height=min_height,
        )
    if backend == "nasa_images":
        return _retrieve_from_nasa_images(
            query_text,
            num_results=num_results,
            min_width=min_width,
            min_height=min_height,
        )
    if backend == "metmuseum":
        return _retrieve_from_metmuseum(
            query_text,
            num_results=num_results,
            min_width=min_width,
            min_height=min_height,
        )
    if backend == "pexels":
        return _retrieve_from_pexels(
            query_text,
            num_results=num_results,
            min_width=min_width,
            min_height=min_height,
        )
    if backend == "pixabay":
        return _retrieve_from_pixabay(
            query_text,
            num_results=num_results,
            min_width=min_width,
            min_height=min_height,
        )

    detail = f"unknown retrieval backend '{backend}'"
    logger.error(detail)
    if _strict_backend_errors_enabled():
        _raise_image_source_error(backend, query_text, detail)
    return []


def retrieve_from_laion(
    query_text: str,
    num_results: int = 40,
    aesthetic_score_min: float = 5.0,
    min_width: int = 256,
    min_height: int = 256,
) -> list[dict]:
    """
    Retrieve images from the configured external image backend.

    Despite the historical name, this now dispatches to one or more configured
    retrieval backends (`clip-retrieval`, `wikimedia_commons`, `openverse`,
    `nasa_images`, `metmuseum`).
    """
    if not _is_laion_enabled():
        return []

    cfg = _get_retrieval_config()
    backend = cfg.get("backend", "clip-retrieval")
    if isinstance(backend, list):
        backends = [str(item).strip() for item in backend if str(item).strip()]
    elif backend == "multi":
        backends = [
            str(item).strip()
            for item in cfg.get("backends", ["wikimedia_commons", "openverse"])
            if str(item).strip()
        ]
    else:
        backends = [str(backend).strip()]

    if not backends:
        detail = "no retrieval backends configured"
        logger.error(detail)
        if _strict_backend_errors_enabled():
            _raise_image_source_error("laion", query_text, detail)
        return []

    all_results = []
    per_backend = max(num_results, (num_results + len(backends) - 1) // len(backends))
    for backend_name in backends:
        all_results.extend(_retrieve_from_backend(
            backend_name,
            query_text,
            num_results=per_backend,
            aesthetic_score_min=aesthetic_score_min,
            min_width=min_width,
            min_height=min_height,
        ))

    if cfg.get("deduplicate", True):
        return _dedupe_normalized_results(all_results, num_results)
    return all_results[:num_results]
