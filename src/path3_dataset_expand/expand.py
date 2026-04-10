"""Main expansion flow: download datasets + run both methods (element decompose + cross pair).

Data sources (3 tiers):
  Tier 1: HF safety datasets with captions (SPA-VL, VLGuard, Hateful Memes) → Method A
  Tier 2: HF safety datasets without captions (UnsafeBench, BeaverTails-V) → Method B
  Tier 3: LAION targeted crawl (reuses Path 5 CATEGORY_QUERIES) → both methods
"""

import logging
import math
import multiprocessing
import os
from pathlib import Path
from typing import Optional

from src.common.schema import SourcePath
from src.common.utils import (
    get_visible_gpu_ids,
    get_hf_home,
    is_english,
    load_prompt_sources,
    save_jsonl,
    DATA_DIR,
)

logger = logging.getLogger(__name__)


# Dataset-specific column name mappings (tried in order)
DATASET_COLUMN_MAP = {
    "spa_vl":        {"image": ["image"], "text": ["question", "caption", "text"], "category": ["category"]},
    "vlguard":       {"image": ["image"], "text": ["prompt", "caption", "text"], "category": ["category"]},
    "hateful_memes": {"image": ["img", "image"], "text": ["text", "caption"], "category": ["label"]},
    "unsafebench":   {"image": ["image"], "text": ["caption", "text", "description"], "category": ["category", "label"]},
    "beavertails_v": {"image": ["image"], "text": ["prompt", "text", "caption"], "category": ["category"]},
}


def probe_dataset_columns(dataset, ds_key: str) -> dict:
    """Inspect first row and log available columns. Return auto-detected column map."""
    try:
        row = dataset[0]
        columns = list(row.keys())
        logger.info(f"Dataset {ds_key} columns: {columns}")

        # Use dataset-specific map if available, else auto-detect
        col_map = DATASET_COLUMN_MAP.get(ds_key, {})
        detected = {}

        # Auto-detect image column
        image_candidates = col_map.get("image", ["image", "img", "pixel_values"])
        for col in image_candidates:
            if col in columns and row[col] is not None:
                detected["image"] = col
                break

        # Auto-detect text column
        text_candidates = col_map.get("text", ["caption", "text", "description", "content", "question", "prompt"])
        for col in text_candidates:
            if col in columns and row[col]:
                detected["text"] = col
                break

        # Auto-detect category column
        cat_candidates = col_map.get("category", ["category", "label", "class", "type"])
        for col in cat_candidates:
            if col in columns and row[col] is not None:
                detected["category"] = col
                break

        logger.info(f"Dataset {ds_key} detected columns: {detected}")
        return detected
    except Exception as e:
        logger.warning(f"Failed to probe {ds_key}: {e}")
        return {}


def download_dataset(dataset_config: dict, split: str = "train") -> Optional[object]:
    """Download a dataset from HuggingFace."""
    from datasets import load_dataset

    hf_repo = dataset_config.get("hf_repo")
    local_path = dataset_config.get("local_path")

    if local_path and Path(local_path).exists():
        logger.info(f"Using local dataset: {local_path}")
        return None  # Handle local datasets separately

    if not hf_repo:
        logger.warning(f"No hf_repo for dataset {dataset_config.get('name')}")
        return None

    try:
        ds = load_dataset(hf_repo, split=split)
        logger.info(f"Downloaded {dataset_config['name']}: {len(ds)} samples")
        return ds
    except Exception as e:
        logger.warning(f"Failed to download {dataset_config['name']}: {e}")
        return None


def extract_images_and_descriptions(
    dataset_name: str,
    dataset,
    output_dir: Path,
    max_images: int = 2000,
    column_map: dict | None = None,
) -> tuple[list[str], list[str], list[dict]]:
    """
    Extract images and their descriptions from a HuggingFace dataset.

    Args:
        column_map: Optional dict with keys "image", "text", "category"
                    mapping to actual column names in the dataset.

    Returns:
        image_paths: List of saved image file paths
        image_sources: List of source dataset names
        image_infos: List of dicts with description, category, etc.
    """
    image_paths = []
    image_sources = []
    image_infos = []

    img_dir = output_dir / "extracted_images" / dataset_name
    img_dir.mkdir(parents=True, exist_ok=True)

    # Determine column names
    img_col = column_map.get("image") if column_map else None
    text_col = column_map.get("text") if column_map else None
    cat_col = column_map.get("category") if column_map else None

    count = 0
    for i, row in enumerate(dataset):
        if count >= max_images:
            break

        # Get image
        image = None
        if img_col and img_col in row and row[img_col] is not None:
            image = row[img_col]
        else:
            for col in ["image", "img", "pixel_values"]:
                if col in row and row[col] is not None:
                    image = row[col]
                    break

        if image is None:
            continue

        # Get text description (skip non-English)
        desc = ""
        if text_col and text_col in row and row[text_col]:
            desc = str(row[text_col])
        else:
            for col in ["caption", "text", "description", "content", "question", "prompt"]:
                if col in row and row[col]:
                    desc = str(row[col])
                    break
        if desc and not is_english(desc):
            desc = ""  # Clear non-English descriptions; image still usable for Method B

        # Get category
        category = ""
        if cat_col and cat_col in row and row[cat_col] is not None:
            category = str(row[cat_col])
        else:
            for col in ["category", "label", "class", "type"]:
                if col in row and row[col] is not None:
                    category = str(row[col])
                    break

        # Save image
        img_path = img_dir / f"{count}.png"
        try:
            if hasattr(image, "save"):
                image.save(img_path)
            else:
                from PIL import Image
                Image.fromarray(image).save(img_path)

            image_paths.append(str(img_path))
            image_sources.append(dataset_name)
            image_infos.append({
                "path": str(img_path),
                "description": desc,
                "category": category,
                "dataset": dataset_name,
            })
            count += 1
        except Exception as e:
            logger.debug(f"Failed to save image {i} from {dataset_name}: {e}")

    logger.info(f"Extracted {count} images from {dataset_name} ({sum(1 for i in image_infos if i['description'])} with descriptions)")
    return image_paths, image_sources, image_infos


def crawl_laion_for_path3(
    output_dir: str | Path,
    max_per_category: int = 200,
) -> tuple[list[str], list[str], list[dict]]:
    """
    Crawl LAION-5B using Path 5's benign-adjacent queries.
    Returns images with captions suitable for both Method A and B.
    """
    from src.path5_embedding_pair.crawl_laion import CATEGORY_QUERIES, crawl_for_category

    output_dir = Path(output_dir)
    image_paths = []
    image_sources = []
    image_infos = []

    # Skip CSEA category
    categories = [cat for cat in CATEGORY_QUERIES.keys() if cat != "CSEA"]

    for cat in categories:
        try:
            crawled = crawl_for_category(cat, output_dir, max_total=max_per_category)
            for item in crawled:
                image_paths.append(item["path"])
                image_sources.append(f"laion_{cat}")
                image_infos.append({
                    "path": item["path"],
                    "description": item.get("caption", item.get("query", "")),
                    "category": cat,
                    "dataset": "laion",
                    "query": item.get("query", ""),
                    "provider": item.get("provider", ""),
                    "source": item.get("source", ""),
                    "license": item.get("license", ""),
                    "license_url": item.get("license_url", ""),
                    "source_page": item.get("source_page", ""),
                    "creator": item.get("creator", ""),
                    "attribution": item.get("attribution", ""),
                })
            logger.info(f"LAION crawl {cat}: {len(crawled)} images")
        except Exception as e:
            logger.warning(f"LAION crawl failed for {cat}: {e}")

    logger.info(f"LAION crawl total: {len(image_paths)} images")
    return image_paths, image_sources, image_infos


def generate_images_from_queries(
    output_dir: str | Path,
    max_per_category: int = 20,
) -> tuple[list[str], list[str], list[dict]]:
    """
    Fallback for retrieval: use the shared T2I backend to generate images from
    CATEGORY_QUERIES.
    The query text itself serves as the image description/caption.
    """
    from src.path5_embedding_pair.crawl_laion import CATEGORY_QUERIES
    from src.common.image_generation import should_force_regenerate_images

    output_dir = Path(output_dir)
    force_regenerate = should_force_regenerate_images()
    gpu_ids = get_visible_gpu_ids()
    if not gpu_ids:
        raise RuntimeError("No GPUs visible for T2I fallback generation")
    categories = [cat for cat in CATEGORY_QUERIES.keys() if cat != "CSEA"]
    tasks = []
    for cat in categories:
        queries = CATEGORY_QUERIES[cat]
        img_dir = output_dir / "generated_images" / cat
        img_dir.mkdir(parents=True, exist_ok=True)
        for index, query in enumerate(queries[:max_per_category]):
            tasks.append({
                "path": str(img_dir / f"{index}.png"),
                "description": query,
                "category": cat,
                "dataset": "t2i_generated",
                "query": query,
            })

    if not tasks:
        return [], [], []

    chunk_size = math.ceil(len(tasks) / len(gpu_ids))
    task_chunks = [tasks[i : i + chunk_size] for i in range(0, len(tasks), chunk_size)]
    worker_args = [
        (gpu_id, chunk, force_regenerate)
        for gpu_id, chunk in zip(gpu_ids, task_chunks)
        if chunk
    ]

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(len(worker_args)) as pool:
        worker_results = pool.map(_generate_query_chunk_on_gpu, worker_args)

    image_infos = [item for chunk in worker_results for item in chunk]
    image_paths = [item["path"] for item in image_infos]
    image_sources = [f"t2i_{item['category']}" for item in image_infos]
    counts_by_category: dict[str, int] = {}
    for item in image_infos:
        counts_by_category[item["category"]] = counts_by_category.get(item["category"], 0) + 1
    for cat in categories:
        logger.info("T2I generated %d images for category %s", counts_by_category.get(cat, 0), cat)
    logger.info(f"T2I generation total: {len(image_paths)} images")
    return image_paths, image_sources, image_infos


def _generate_query_chunk_on_gpu(args: tuple[int, list[dict], bool]) -> list[dict]:
    """Generate a chunk of T2I fallback images on one GPU."""
    gpu_id, tasks, force_regenerate = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("HF_HOME", get_hf_home())

    from src.common.image_generation import ensure_t2i_model_cached, generate_image

    ensure_t2i_model_cached()
    results = []
    for task in tasks:
        img_path = Path(task["path"])
        if img_path.exists() and not force_regenerate:
            results.append(task)
            continue
        try:
            if img_path.exists() and force_regenerate:
                img_path.unlink(missing_ok=True)
            if generate_image(task["query"], img_path):
                results.append(task)
        except Exception as exc:
            logger.warning("GPU %s T2I generation failed for %r: %s", gpu_id, task["query"], exc)
    logger.info("GPU %s generated %d/%d fallback images", gpu_id, len(results), len(tasks))
    return results


def load_mis_images(mis_path: str = "/mnt/hdd/xuran/MIS/mis_train") -> tuple[list[str], list[str], list[dict]]:
    """Load images from the local MIS dataset (used for Method B image pool only)."""
    import json

    mis_path = Path(mis_path)
    json_path = mis_path / "mis_train.json"

    if not json_path.exists():
        logger.warning(f"MIS train JSON not found at {json_path}")
        return [], [], []

    with open(json_path) as f:
        data = json.load(f)

    image_paths = []
    image_sources = []
    image_infos = []

    for sample in data:
        for img_rel_path in sample.get("image", []):
            img_path = mis_path / img_rel_path
            if img_path.exists():
                image_paths.append(str(img_path))
                image_sources.append("mis")
                image_infos.append({
                    "path": str(img_path),
                    "description": "",  # MIS has no per-image descriptions
                    "category": sample.get("category", ""),
                    "sub_category": sample.get("sub_category", ""),
                    "dataset": "mis",
                })

    logger.info(f"Loaded {len(image_paths)} images from MIS (Method B pool only)")
    return image_paths, image_sources, image_infos


def collect_all_data(
    output_dir: Path,
    max_images_per_dataset: int = 2000,
    max_laion_per_category: int = 200,
) -> tuple[list[dict], list[str], list[str], list[dict]]:
    """
    Collect data from all sources. Returns:
        images_with_desc: images with descriptions (for Method A)
        method_b_paths: all image paths (for Method B)
        method_b_sources: corresponding source names
        all_infos: all image info dicts
    """
    os.environ["HF_HOME"] = get_hf_home()

    sources_config = load_prompt_sources()
    safety_datasets = sources_config.get("safety_datasets", {})

    all_image_paths = []
    all_image_sources = []
    all_image_infos = []

    # --- Tier 1 & 2: HF safety datasets ---
    for ds_key, ds_config in safety_datasets.items():
        if ds_config.get("local_path"):
            continue  # Skip MIS here (handled separately below)

        try:
            # Try smaller split first for large datasets
            size = ds_config.get("size", 0)
            if size > 10000:
                split = f"train[:{max_images_per_dataset}]"
            else:
                split = "train"

            ds = download_dataset(ds_config, split=split)
            if ds is None:
                continue

            col_map = probe_dataset_columns(ds, ds_key)
            paths, sources, infos = extract_images_and_descriptions(
                ds_key, ds, output_dir,
                max_images=max_images_per_dataset,
                column_map=col_map,
            )
            all_image_paths.extend(paths)
            all_image_sources.extend(sources)
            all_image_infos.extend(infos)
        except Exception as e:
            logger.warning(f"Skipped {ds_key}: {e}")

    # --- Local safety datasets declared via local_path (e.g. MIS) ---
    mis_config = safety_datasets.get("mis", {})
    if mis_config.get("local_path"):
        try:
            mis_paths, mis_sources, mis_infos = load_mis_images(mis_config["local_path"])
            all_image_paths.extend(mis_paths[:max_images_per_dataset])
            all_image_sources.extend(mis_sources[:max_images_per_dataset])
            all_image_infos.extend(mis_infos[:max_images_per_dataset])
        except Exception as e:
            logger.warning(f"Failed to load MIS local dataset: {e}")

    # --- Tier 3: external targeted crawl (skipped if disabled in config) ---
    from src.common.clip_utils import _is_laion_enabled
    if _is_laion_enabled():
        try:
            laion_paths, laion_sources, laion_infos = crawl_laion_for_path3(
                output_dir, max_per_category=max_laion_per_category,
            )
            all_image_paths.extend(laion_paths)
            all_image_sources.extend(laion_sources)
            all_image_infos.extend(laion_infos)
        except Exception as e:
            logger.warning(f"LAION crawl failed entirely: {e}")
    else:
        logger.info("LAION disabled in config — skipping crawl. T2I fallback available via runner script.")

    # Split: images with descriptions (Method A) vs all (Method B)
    images_with_desc = [i for i in all_image_infos if i.get("description", "").strip()]

    logger.info(
        f"Data collection complete: {len(all_image_infos)} total images, "
        f"{len(images_with_desc)} with descriptions"
    )

    return images_with_desc, all_image_paths, all_image_sources, all_image_infos


def run(output_dir: str | Path | None = None, max_images_per_dataset: int = 2000):
    """
    Main entry point:
    1. Collect images from HF datasets + LAION
    2. Run Method A (element decompose) on images with descriptions
    3. Run Method B (cross-dataset CLIP pairing) on all images
    """
    if output_dir is None:
        output_dir = DATA_DIR / "raw" / "path3"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_with_desc, all_paths, all_sources, all_infos = collect_all_data(
        output_dir, max_images_per_dataset=max_images_per_dataset,
    )

    save_jsonl(all_infos, output_dir / "all_image_infos.jsonl")

    # Method A: Element decompose (for images with descriptions)
    if images_with_desc:
        from src.path3_dataset_expand.element_decompose import prepare_batch_decompose_prompts
        batch_prompts = prepare_batch_decompose_prompts(images_with_desc)
        save_jsonl(
            [{"prompt": p, "info": images_with_desc[i]}
             for i, p in enumerate(batch_prompts)],
            output_dir / "decompose_batch_prompts.jsonl",
        )
        logger.info(f"Prepared {len(batch_prompts)} prompts for element decomposition")

    # Method B: Cross-image pairing prompt preparation
    if len(all_infos) > 100:
        from src.path3_dataset_expand.cross_pair import run as run_cross_pair
        run_cross_pair(all_infos, output_dir)

    return all_paths, all_sources


if __name__ == "__main__":
    from src.common.utils import setup_logging
    setup_logging()
    run()
