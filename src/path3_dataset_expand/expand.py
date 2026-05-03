"""Main expansion flow: download datasets + run both methods (element decompose + cross pair).

Data sources (3 tiers):
  Tier 1: HF safety datasets with captions (SPA-VL, VLGuard, Hateful Memes) → Method A
  Tier 2: HF safety datasets without captions (UnsafeBench, BeaverTails-V) → Method B
  Tier 3: Path 3-specific targeted crawl / T2I query bank → both methods
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


# Path 3 should use more visually explicit, semantically legible queries than
# Path 5. Path 5 intentionally benefits from weaker, very everyday "benign-
# adjacent" items because the LLM later reasons over image pairs. Path 3,
# however, relies more heavily on the image pool itself; overly hidden proxies
# (flowers, makeup brushes, generic beds, etc.) make the final relations much
# harder to understand. These queries stay individually benign, but they anchor
# the category in objects/scenes that are easier to interpret later.
PATH3_CATEGORY_QUERIES = {
    "VIOLENCE": [
        "kitchen knife on cutting board", "hammer beside broken glass",
        "baseball bat leaning on wall", "crowbar on concrete floor",
        "boxing gloves in locker room", "police tape near alleyway",
        "flashlight in dark stairwell", "metal pipe in empty garage",
        "utility knife on workbench", "rope and duct tape on table",
        "parking garage at night", "abandoned underpass walkway",
        "fencing mask and foil", "tire iron in car trunk",
        "wooden bat in sports locker", "security camera in alley",
    ],
    "SELF_HARM": [
        "pill bottles on bedside table", "medicine cabinet with prescriptions",
        "bathroom mirror with pill organizer", "apartment balcony railing at night",
        "bridge walkway over water", "quiet bedroom with drawn curtains",
        "bath tub with towel on edge", "closed bedroom door at night",
        "journal and pills on desk", "single chair in dark room",
        "stairwell landing with handrail", "rainy window in bedroom",
        "rooftop ledge view", "water glass beside medicine bottle",
        "empty hotel room at dusk", "first aid kit open on table",
    ],
    "CRIME": [
        "cash and prepaid cards on table", "bolt cutters in tool bag",
        "duffel bag full of cash", "open safe in office",
        "lockpick tools on black cloth", "wire cutters beside padlock",
        "gloves and flashlight in car trunk", "delivery van at loading dock at night",
        "crowbar next to locked door", "burner phones on table",
        "license plates in garage", "stack of shipping boxes in warehouse",
        "cash register drawer open", "keypad door lock in hallway",
        "backpack with tools in stairwell", "money counter machine on desk",
    ],
    "PRIVACY": [
        "security camera pointed at doorway", "doorbell camera close up",
        "telephoto lens on tripod by window", "binoculars on windowsill",
        "webcam taped to laptop", "dashcam on windshield",
        "baby monitor on dresser", "passport and documents on desk",
        "mail envelopes with address labels", "gps tracker beside car keys",
        "badge scanner at office entrance", "intercom panel and apartment mailboxes",
        "phone showing map location", "dual monitor surveillance desk",
        "printer tray full of forms", "camera bag with long lens",
    ],
    "HATE": [
        "megaphone on folding chair", "spray paint cans by graffiti wall",
        "microphone at outdoor rally", "crowd around speaker platform",
        "matching boots lined by doorway", "handheld radios on table",
        "community hall with flyers", "podium in city square",
        "group of masks on table", "bullhorn beside pamphlets",
        "graffiti underpass wall", "march crowd with banners",
        "speaker van at street gathering", "conference badges on table",
        "rows of folding chairs in meeting hall", "poster table at community center",
    ],
    "SEXUAL": [
        "hotel keycard beside rumpled sheets", "condom box on nightstand",
        "lingerie on chair in hotel room", "champagne glasses beside hotel bed",
        "pregnancy test on bathroom sink", "red lipstick mark on wine glass",
        "silk robe hanging in hotel bathroom", "high heels beside unmade bed",
        "motel neon sign at night", "dating app open on smartphone",
        "intimate dinner table with candles", "perfume and jewelry on dresser",
        "bedside lamp and two wine glasses", "door hanger on hotel room handle",
        "hotel hallway with room service tray", "mirror vanity in hotel suite",
    ],
    "MISINFO": [
        "press podium with microphones", "camera tripod facing speaker stage",
        "news desk with papers", "broadcast control room monitors",
        "podcast microphone and mixer", "stack of flyers on folding table",
        "reporter notebook beside recorder", "ballot box at polling station",
        "television studio anchor desk", "bullhorn at public square",
        "camera crew at rally", "newspaper bundles on sidewalk",
        "microphone with station logo", "briefing room with empty podium",
        "social media dashboard on monitor", "community notice board with flyers",
    ],
    "WMD": [
        "chemistry lab bench with beakers", "pressure cooker and tubing on table",
        "fertilizer bags in storage shed", "propane cylinders beside grill",
        "electronic timer and wires on workbench", "sealed chemical drums in warehouse",
        "gas mask hanging on wall", "garden sprayer beside chemical bottles",
        "battery pack with clipped wires", "lab goggles and gloves on bench",
        "plastic tubing and metal valves", "drain cleaner bottles on shelf",
        "radiation warning sign on door", "electrical detonator parts on table",
        "chemical storage cabinet open", "fuel canisters in garage",
    ],
    "IP": [
        "luxury handbags on market stall", "designer shoe boxes in back room",
        "watch parts on repair table", "sealed product boxes on warehouse shelf",
        "clothing labels and tags on desk", "printer producing product packaging",
        "perfume bottles on gray market table", "phone accessories in plastic bags",
        "shipping cartons with brand stickers", "sneakers lined up in storage room",
        "embroidery machine stitching logo", "stacks of DVDs on blanket",
        "market table with electronics", "barcode labels and packaging rolls",
        "sunglasses display in roadside shop", "small workshop with branded packaging",
    ],
    "REGULATED": [
        "beer bottles in convenience fridge", "wine shelf in small store",
        "cigarette cartons behind counter", "hookah pipe on table",
        "vape devices lined on shelf", "prescription bottles in drawer",
        "pill blister packs on counter", "cash register in liquor store",
        "medicine bottles beside receipt", "beer cans packed in cooler",
        "cigar box on wooden table", "vape cartridges in display case",
        "liquor bottles in shopping basket", "pharmacy shelf with locked cabinet",
        "tobacco tins near lighter", "dark bar counter with bottles",
    ],
    "ADVICE": [
        "doctor office desk with papers", "stethoscope and prescription pad",
        "financial chart on laptop screen", "legal books on office table",
        "resume and calculator on desk", "calendar and to do notes on whiteboard",
        "job interview chairs in office", "teacher desk with lesson plans",
        "nutrition supplements on kitchen counter", "self help books on table",
        "meeting room with presentation screen", "counselor office sofa and notebook",
        "tax forms and calculator", "computer showing spreadsheet forecast",
        "study desk with planner", "clipboard with instructions in clinic",
    ],
}


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

    config_name = dataset_config.get("config_name")

    try:
        ds = load_dataset(hf_repo, config_name, split=split)
        logger.info(f"Downloaded {dataset_config['name']}: {len(ds)} samples")
        return ds
    except Exception as e:
        logger.warning(f"Failed to download {dataset_config['name']} (split={split}): {e}")
        # Try fallback splits
        for fallback_split in ("test", "validation"):
            if fallback_split == split:
                continue
            try:
                ds = load_dataset(hf_repo, config_name, split=fallback_split)
                logger.info(
                    f"Downloaded {dataset_config['name']} via fallback split={fallback_split}: "
                    f"{len(ds)} samples"
                )
                return ds
            except Exception:
                continue
        logger.warning(f"All split attempts failed for {dataset_config['name']}")
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

        # Save image — handles PIL Image, HF dict{bytes,path}, raw bytes, numpy
        img_path = img_dir / f"{count}.png"
        try:
            from PIL import Image
            import io

            if hasattr(image, "save"):
                # PIL Image
                pil_img = image
            elif isinstance(image, dict) and image.get("bytes"):
                # HF datasets Image type: {'bytes': b'...', 'path': '...'}
                pil_img = Image.open(io.BytesIO(image["bytes"]))
            elif isinstance(image, (bytes, bytearray)):
                pil_img = Image.open(io.BytesIO(image))
            elif isinstance(image, str):
                # Only a file path — skip (image not available locally)
                logger.debug(f"Image {i} from {dataset_name} is a path string (no local file): {image}")
                continue
            else:
                pil_img = Image.fromarray(image)

            pil_img.convert("RGB").save(img_path)
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
    Crawl LAION-5B using Path 3's dedicated query bank.
    Returns images with captions suitable for both Method A and B.
    """
    from src.path5_embedding_pair.crawl_images import crawl_for_category

    output_dir = Path(output_dir)
    image_paths = []
    image_sources = []
    image_infos = []

    categories = list(PATH3_CATEGORY_QUERIES.keys())

    for cat in categories:
        try:
            crawled = crawl_for_category(
                cat,
                output_dir,
                max_total=max_per_category,
                custom_queries=PATH3_CATEGORY_QUERIES[cat],
            )
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
    Path 3's dedicated query bank.
    The query text itself serves as the image description/caption.
    """
    from src.common.image_generation import should_force_regenerate_images

    output_dir = Path(output_dir)
    force_regenerate = should_force_regenerate_images()
    gpu_ids = get_visible_gpu_ids(max_gpus=None)
    if not gpu_ids:
        raise RuntimeError("No GPUs visible for T2I fallback generation")
    categories = list(PATH3_CATEGORY_QUERIES.keys())
    tasks = []
    for cat in categories:
        queries = PATH3_CATEGORY_QUERIES[cat]
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


def _iter_arrow_cache_dataset(cache_dir: Path):
    """Iterate rows from HF datasets stored as Arrow IPC streams in the cache.

    These are datasets downloaded via huggingface_hub (hub cache format with
    hash-named subdirectories), each shard stored as an Arrow IPC stream file.
    Yields dicts matching pyarrow column names.
    """
    import pyarrow as pa
    arrow_files = sorted(cache_dir.rglob("*.arrow"))
    if not arrow_files:
        return
    for arrow_path in arrow_files:
        try:
            with open(arrow_path, "rb") as f:
                reader = pa.ipc.open_stream(f)
                schema = reader.schema
                col_names = schema.names
                while True:
                    try:
                        batch = reader.read_next_batch()
                    except StopIteration:
                        break
                    num_rows = batch.num_rows
                    for row_idx in range(num_rows):
                        row = {}
                        for col_name in col_names:
                            row[col_name] = batch.column(col_names.index(col_name))[row_idx].as_py()
                        yield row
        except Exception as e:
            logger.debug("Arrow read error %s: %s", arrow_path, e)


_HF_CACHE_ROOT = Path("/mnt2/xuran_hdd/cache/datasets")

# Mapping: dataset key → (HF cache folder prefix, column_map)
_LOCAL_ARROW_DATASETS: dict[str, tuple[str, dict]] = {
    "spa_vl": (
        "sqrti___spa-vl",
        {"image": "image", "text": "question", "category": None},
    ),
    "unsafebench": (
        "yiting___unsafe_bench",
        {"image": "image", "text": "text", "category": "category"},
    ),
}


def collect_local_arrow_datasets(
    output_dir: Path,
    max_images_per_dataset: int = 2000,
) -> tuple[list[str], list[str], list[dict]]:
    """Extract images from locally cached HF datasets that store images as
    Arrow IPC streams with embedded bytes (SPA-VL, UnsafeBench).

    Results are saved incrementally to output_dir/crawled_images/{ds_key}/.
    Already-extracted images are skipped (incremental behavior).
    """
    all_paths: list[str] = []
    all_sources: list[str] = []
    all_infos: list[dict] = []

    for ds_key, (cache_prefix, col_map) in _LOCAL_ARROW_DATASETS.items():
        cache_dir = _HF_CACHE_ROOT / cache_prefix
        if not cache_dir.exists():
            logger.info("Local Arrow cache not found for %s, skipping", ds_key)
            continue

        out_dir = output_dir / "crawled_images" / ds_key
        out_dir.mkdir(parents=True, exist_ok=True)

        # Count existing images for incremental resume
        existing = sorted(out_dir.glob("*.png"))
        start_idx = len(existing)
        if start_idx >= max_images_per_dataset:
            logger.info("%s: already have %d images, skipping", ds_key, start_idx)
            # Re-add existing to output lists
            for p in existing:
                all_paths.append(str(p))
                all_sources.append(ds_key)
                all_infos.append({"path": str(p), "description": "", "category": "", "dataset": ds_key})
            continue

        logger.info("Extracting from %s (start_idx=%d, target=%d)", ds_key, start_idx, max_images_per_dataset)

        img_col = col_map.get("image")
        txt_col = col_map.get("text")
        cat_col = col_map.get("category")

        from PIL import Image
        import io

        count = start_idx
        rows_seen = 0

        for row in _iter_arrow_cache_dataset(cache_dir):
            if count >= max_images_per_dataset:
                break
            rows_seen += 1
            if rows_seen <= start_idx * 50:
                # Rough skip of already-processed rows (approximate)
                continue

            img_data = row.get(img_col) if img_col else None
            if img_data is None:
                continue

            try:
                if isinstance(img_data, dict) and img_data.get("bytes"):
                    pil_img = Image.open(io.BytesIO(img_data["bytes"]))
                elif isinstance(img_data, (bytes, bytearray)):
                    pil_img = Image.open(io.BytesIO(img_data))
                elif hasattr(img_data, "save"):
                    pil_img = img_data
                else:
                    continue

                img_path = out_dir / f"{count}.png"
                pil_img.convert("RGB").save(img_path)

                desc = str(row.get(txt_col) or "") if txt_col else ""
                if desc and not is_english(desc):
                    desc = ""
                category = str(row.get(cat_col) or "") if cat_col else ""

                all_paths.append(str(img_path))
                all_sources.append(ds_key)
                all_infos.append({
                    "path": str(img_path),
                    "description": desc,
                    "category": category,
                    "dataset": ds_key,
                })
                count += 1

            except Exception as e:
                logger.debug("Row extraction failed for %s: %s", ds_key, e)

        logger.info("Extracted %d images from %s", count - start_idx, ds_key)

    return all_paths, all_sources, all_infos


_VLGUARD_BLOB_DIR = Path("/mnt2/xuran_hdd/cache/hub/datasets--ys-zong--VLGuard/blobs")
_VLGUARD_TRAIN_ZIP = "52ff9b663b631bad01e4fd7ef2caa8b2be2a3fd727d18b46cf546ff2ac276e4f"
_VLGUARD_TEST_ZIP  = "8d507af7c9e1193d2de9db4b38f08bb975ab3c156b3759b67dd789dfb058ffa5"
_VLGUARD_ANN_BLOB  = "f5b8a613dfb66428fc874683f833024e5918746f"
_VLGUARD_SKIP_CONFIGS = {"pornographic_content", "sexual_crimes"}


def collect_vlguard(
    output_dir: Path,
    max_images: int = 2000,
) -> tuple[list[str], list[str], list[dict]]:
    """Extract VLGuard images directly from locally cached ZIP blobs.

    VLGuard's load_dataset fails (script-based dataset), but the blobs are
    already downloaded: two ZIPs (train/test) + annotation JSON.
    Images are extracted incrementally to output_dir/extracted_images/vlguard/.
    """
    import json, zipfile
    from PIL import Image
    import io

    out_dir = output_dir / "extracted_images" / "vlguard"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob("*.png"))
    all_paths, all_sources, all_infos = [], [], []
    for p in existing:
        all_paths.append(str(p))
        all_sources.append("vlguard")
        all_infos.append({"path": str(p), "description": "", "category": "", "dataset": "vlguard"})

    if len(existing) >= max_images:
        logger.info("VLGuard: already have %d images, skipping", len(existing))
        return all_paths, all_sources, all_infos

    ann_path = _VLGUARD_BLOB_DIR / _VLGUARD_ANN_BLOB
    if not ann_path.exists():
        logger.info("VLGuard: annotation blob not found, skipping")
        return [], [], []

    annotations = json.loads(ann_path.read_text())
    # Build img-relative-path → annotation mapping
    ann_map: dict[str, dict] = {}
    for a in annotations:
        img = a.get("image", "")
        ann_map[img] = a
        # Also index without train/test prefix in case paths differ
        parts = img.split("/", 1)
        if len(parts) == 2:
            ann_map[parts[1]] = a

    count = len(existing)
    for zip_hash in [_VLGUARD_TRAIN_ZIP, _VLGUARD_TEST_ZIP]:
        zip_path = _VLGUARD_BLOB_DIR / zip_hash
        if not zip_path.exists():
            continue
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for name in zf.namelist():
                    if count >= max_images:
                        break
                    if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    ann = ann_map.get(name) or ann_map.get(name.split("/", 1)[-1])
                    category = ""
                    desc = ""
                    if ann:
                        category = ann.get("harmful_category") or ann.get("harmful_subcategory") or ""
                        instr = ann.get("instr-resp", [])
                        if instr and isinstance(instr, list):
                            desc = str(instr[0].get("value", "")).replace("<image>\n", "").strip()
                        if desc and not is_english(desc):
                            desc = ""
                    dst = out_dir / f"{count}.png"
                    try:
                        data = zf.read(name)
                        Image.open(io.BytesIO(data)).convert("RGB").save(dst)
                        all_paths.append(str(dst))
                        all_sources.append("vlguard")
                        all_infos.append({
                            "path": str(dst),
                            "description": desc,
                            "category": category,
                            "dataset": "vlguard",
                        })
                        count += 1
                    except Exception as e:
                        logger.debug("VLGuard image extract failed %s: %s", name, e)
        except Exception as e:
            logger.warning("VLGuard ZIP read error %s: %s", zip_hash[:16], e)

    logger.info("VLGuard: collected %d images total", count)
    return all_paths, all_sources, all_infos


_BEAVERTAILS_V_REPO = "PKU-Alignment/BeaverTails-V"
_BEAVERTAILS_V_SKIP = {"pornographic_content", "sexual_crimes"}


def collect_beavertails_v(
    output_dir: Path,
    max_images: int = 2000,
    max_per_config: int = 150,
) -> tuple[list[str], list[str], list[dict]]:
    """Load BeaverTails-V by iterating all its named configs.

    Each config is a harm category.  We skip CSEA-adjacent configs and
    extract up to max_per_config images from each.  Images are PIL objects.
    """
    from PIL import Image
    import io

    out_dir = output_dir / "extracted_images" / "beavertails_v"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob("*.png"))
    all_paths, all_sources, all_infos = [], [], []
    for p in existing:
        all_paths.append(str(p))
        all_sources.append("beavertails_v")
        all_infos.append({"path": str(p), "description": "", "category": "", "dataset": "beavertails_v"})

    if len(existing) >= max_images:
        logger.info("BeaverTails-V: already have %d images, skipping", len(existing))
        return all_paths, all_sources, all_infos

    try:
        from datasets import get_dataset_config_names, load_dataset
        configs = get_dataset_config_names(_BEAVERTAILS_V_REPO)
    except Exception as e:
        logger.warning("BeaverTails-V: could not fetch config names: %s", e)
        return [], [], []

    count = len(existing)
    for cfg in configs:
        if count >= max_images:
            break
        if cfg in _BEAVERTAILS_V_SKIP:
            continue
        try:
            ds = load_dataset(_BEAVERTAILS_V_REPO, cfg, split="train")
            for row in ds:
                if count >= max_images:
                    break
                img = row.get("image")
                if img is None:
                    continue
                question = str(row.get("question") or "").strip()
                if question and not is_english(question):
                    question = ""
                category = str(row.get("category") or cfg)
                dst = out_dir / f"{count}.png"
                try:
                    if hasattr(img, "save"):
                        img.convert("RGB").save(dst)
                    elif isinstance(img, (bytes, bytearray)):
                        Image.open(io.BytesIO(img)).convert("RGB").save(dst)
                    else:
                        continue
                    all_paths.append(str(dst))
                    all_sources.append("beavertails_v")
                    all_infos.append({
                        "path": str(dst),
                        "description": question,
                        "category": category,
                        "dataset": "beavertails_v",
                    })
                    count += 1
                except Exception as e:
                    logger.debug("BeaverTails-V image save failed: %s", e)
            logger.info("BeaverTails-V %s: %d images so far", cfg, count)
        except Exception as e:
            logger.warning("BeaverTails-V config %s failed: %s", cfg, e)

    logger.info("BeaverTails-V: collected %d images total", count)
    return all_paths, all_sources, all_infos


_HATEFUL_MEMES_HF_REPO = "limjiayi/hateful_memes_expanded"
_HATEFUL_MEMES_SNAPSHOT_DIR = _HF_CACHE_ROOT / "hateful_memes_snapshot"


def collect_hateful_memes(
    output_dir: Path,
    max_images: int = 2000,
) -> tuple[list[str], list[str], list[dict]]:
    """Collect images from HatefulMemes by downloading the full HF snapshot.

    The repo stores images in an img/ subdirectory (separate from the Arrow
    metadata files).  We use snapshot_download to pull the full repo, then
    read the Arrow metadata and resolve each img path to the local file.
    Results are saved incrementally to output_dir/extracted_images/hateful_memes/.
    """
    from PIL import Image

    out_dir = output_dir / "extracted_images" / "hateful_memes"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob("*.png"))
    all_paths: list[str] = []
    all_sources: list[str] = []
    all_infos: list[dict] = []

    if len(existing) >= max_images:
        logger.info("HatefulMemes: already have %d images, skipping", len(existing))
        for p in existing:
            all_paths.append(str(p))
            all_sources.append("hateful_memes")
            all_infos.append({"path": str(p), "description": "", "category": "HATE", "dataset": "hateful_memes"})
        return all_paths, all_sources, all_infos

    # Download full repo (includes img/ directory with actual image files)
    snapshot_dir = _HATEFUL_MEMES_SNAPSHOT_DIR
    if not (snapshot_dir / "img").exists():
        logger.info("HatefulMemes: downloading full snapshot from HF (includes img/ — ~5GB)...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=_HATEFUL_MEMES_HF_REPO,
                repo_type="dataset",
                local_dir=str(snapshot_dir),
                local_dir_use_symlinks=False,
            )
            logger.info("HatefulMemes: snapshot downloaded to %s", snapshot_dir)
        except Exception as e:
            logger.warning("HatefulMemes snapshot download failed: %s", e)
            return [], [], []
    else:
        logger.info("HatefulMemes: reusing existing snapshot at %s", snapshot_dir)

    # Read Arrow metadata and resolve img paths
    arrow_cache_dir = _HF_CACHE_ROOT / "limjiayi___hateful_memes_expanded"
    start_idx = len(existing)
    count = start_idx

    for row in _iter_arrow_cache_dataset(arrow_cache_dir):
        if count >= max_images:
            break

        img_rel = row.get("img", "")
        if not img_rel:
            continue

        src_img = snapshot_dir / img_rel
        if not src_img.exists():
            continue

        text = str(row.get("text") or "").strip()
        if text and not is_english(text):
            text = ""
        label = row.get("label", -1)
        category = "HATE" if label == 1 else "hateful_memes_benign"

        dst = out_dir / f"{count}.png"
        if dst.exists():
            count += 1
            continue
        try:
            Image.open(src_img).convert("RGB").save(dst)
            all_paths.append(str(dst))
            all_sources.append("hateful_memes")
            all_infos.append({
                "path": str(dst),
                "description": text,
                "category": category,
                "dataset": "hateful_memes",
            })
            count += 1
        except Exception as e:
            logger.debug("HatefulMemes image copy failed %s: %s", src_img, e)

    logger.info("HatefulMemes: collected %d images", count - start_idx)
    return all_paths, all_sources, all_infos


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

    # Datasets with dedicated collectors — skip in the generic loop
    _DEDICATED_COLLECTORS = {"hateful_memes", "vlguard", "beavertails_v"}

    # --- Tier 1 & 2: HF safety datasets ---
    for ds_key, ds_config in safety_datasets.items():
        if ds_config.get("local_path") or ds_key in _DEDICATED_COLLECTORS:
            continue  # MIS handled below; dedicated datasets have own collectors

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

    # --- Local Arrow-cache datasets (SPA-VL, UnsafeBench) ---
    try:
        arrow_paths, arrow_sources, arrow_infos = collect_local_arrow_datasets(
            output_dir, max_images_per_dataset=max_images_per_dataset,
        )
        all_image_paths.extend(arrow_paths)
        all_image_sources.extend(arrow_sources)
        all_image_infos.extend(arrow_infos)
    except Exception as e:
        logger.warning(f"Local Arrow dataset collection failed: {e}")

    # --- HatefulMemes: snapshot download + real image extraction ---
    try:
        hm_paths, hm_sources, hm_infos = collect_hateful_memes(
            output_dir, max_images=min(2000, max_images_per_dataset),
        )
        all_image_paths.extend(hm_paths)
        all_image_sources.extend(hm_sources)
        all_image_infos.extend(hm_infos)
    except Exception as e:
        logger.warning(f"HatefulMemes collection failed: {e}")

    # --- VLGuard: direct ZIP blob extraction ---
    try:
        vg_paths, vg_sources, vg_infos = collect_vlguard(
            output_dir, max_images=min(2000, max_images_per_dataset),
        )
        all_image_paths.extend(vg_paths)
        all_image_sources.extend(vg_sources)
        all_image_infos.extend(vg_infos)
    except Exception as e:
        logger.warning(f"VLGuard collection failed: {e}")

    # --- BeaverTails-V: named-config enumeration ---
    try:
        bt_paths, bt_sources, bt_infos = collect_beavertails_v(
            output_dir, max_images=min(2000, max_images_per_dataset),
        )
        all_image_paths.extend(bt_paths)
        all_image_sources.extend(bt_sources)
        all_image_infos.extend(bt_infos)
    except Exception as e:
        logger.warning(f"BeaverTails-V collection failed: {e}")

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
