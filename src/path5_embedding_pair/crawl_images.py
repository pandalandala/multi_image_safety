"""Step 1: Targeted image acquisition from local datasets + web backends.

For each harm category, design "benign-adjacent" search queries that retrieve
individually harmless images with potential for compositional harm.

Acquisition order per query:
  1. Local datasets (MSCOCO, Open Images, ImageNet)
  2. Web backends (Wikimedia, Openverse, Pexels, Pixabay)
"""

import os
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.common.clip_utils import (
    passes_download_filter,
)
from src.common.local_datasets import search_local
from src.common.retrieval_queries import build_compact_retrieval_query
from src.common.retrieval_rerank import (
    download_web_results_ranked,
    get_retrieval_clip_candidate_limit,
    get_retrieval_clip_keep_count,
    get_retrieval_clip_min_score,
    rank_local_results_by_clip,
    retrieve_web_results_with_variants,
)
from src.common.shared_reuse import (
    load_retrieval_cache,
    materialize_shared_image,
    save_retrieval_cache,
)
from src.common.utils import (
    get_all_categories,
    get_min_image_size,
    is_english,
    save_jsonl,
    load_config,
    DATA_DIR,
)

logger = logging.getLogger(__name__)


def _get_path5_crawl_worker_count(category_count: int) -> int:
    """Return how many categories Path 5 should crawl in parallel."""
    env_value = os.environ.get("MIS_PATH5_CRAWL_WORKERS", "").strip()
    if env_value:
        try:
            return max(1, min(category_count, int(env_value)))
        except ValueError:
            logger.warning("Invalid MIS_PATH5_CRAWL_WORKERS=%r; falling back to default", env_value)
    cpu_count = os.cpu_count() or 8
    return max(1, min(category_count, max(4, min(12, cpu_count // 2))))


def _get_path5_local_candidate_limit(max_per_query: int) -> int:
    env_value = os.environ.get("MIS_PATH5_LOCAL_CANDIDATES", "").strip()
    if env_value:
        try:
            return max(1, min(max_per_query, int(env_value)))
        except ValueError:
            logger.warning("Invalid MIS_PATH5_LOCAL_CANDIDATES=%r; falling back to default", env_value)
    return max(4, min(max_per_query, get_retrieval_clip_candidate_limit() + 3))


def _skip_local_dataset_search() -> bool:
    """Allow Path 5 reruns to bypass slow local dataset search entirely."""
    raw = os.environ.get("MIS_PATH5_SKIP_LOCAL_DATASETS", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}

# Benign-adjacent search queries organized by harm category.
# These are intentionally photo-oriented queries for real objects/scenes, so
# they work better with Commons/Openverse-style backends and our non-photo
# filters (which reject terms like logo/poster/screenshot/illustration).
CATEGORY_QUERIES = {
    "VIOLENCE": [
        "kitchen counter with chef knife", "baseball bat in sports bag",
        "boxing gloves on bench", "martial arts studio", "fencing gear locker",
        "archery range target", "hammer on workbench", "toolbox in garage",
        "wrench set on shelf", "tire iron in car trunk", "crowbar in toolbox",
        "metal pipe in workshop", "broken bottle on street", "parking garage at night",
        "empty alleyway at night", "flashlight in glove compartment",
        "security camera in parking lot", "construction boots by tools",
        "utility knife on desk", "sports locker room bench",
        "baseball dugout bench", "camping hatchet by firewood",
        "rope in storage room", "warehouse loading dock at night",
        "empty subway platform", "dark stairwell", "street corner at dusk",
        "crowd near stadium entrance", "heavy work gloves on table",
        "garage shelf with tools", "wooden stick by fence", "metal flashlight on table",
        "sports field fence at night", "concrete underpass walkway",
    ],
    "SELF_HARM": [
        "medicine cabinet shelf", "pill organizer on kitchen table",
        "pharmacy counter", "nightstand with lamp", "empty bedroom at dusk",
        "drawn curtains in bedroom", "quiet hotel room", "balcony railing view",
        "bridge walkway over water", "stairwell with handrail",
        "empty park bench", "journal on desk", "bathroom sink at night",
        "first aid box open", "towel by bathtub", "bathrobe on hook",
        "window blinds in dark room", "closed bedroom door", "rooftop view",
        "boat dock with rope", "scissors in drawer", "mirror above sink",
        "single chair in empty room", "rainy window in bedroom",
        "small lamp in dark room", "apartment hallway at night",
        "prescription bottle on shelf", "water glass on bedside table",
        "quiet hallway with carpet", "desk with handwritten notes",
        "bench by river railing", "apartment balcony at dusk",
    ],
    "CRIME": [
        "cash in desk drawer", "credit cards on table",
        "laptop on office desk", "usb drive beside keyboard",
        "delivery van at loading dock", "mailbox keys on keyring",
        "storage unit hallway", "cash register drawer open",
        "package boxes in apartment lobby", "keypad door lock",
        "parcel locker area", "cash deposit envelopes",
        "duffel bag in trunk", "padlock and key on table",
        "wire cutters in toolbox", "safe door in office",
        "office shredder machine", "security keypad by door",
        "flashlight and gloves in car", "screwdriver set on bench",
        "bolt cutters in garage", "stack of shipping boxes",
        "warehouse shelf with labels", "backpack with tools",
        "parking lot security gate", "money counter machine",
        "receipts and calculator on desk", "prepaid cards display",
        "clipboard with delivery forms", "garage shelf with chains",
        "tool bag on floor", "rolling suitcase in hallway",
        "night loading dock", "office cabinet with lock",
    ],
    "PRIVACY": [
        "security camera on shelf", "doorbell camera at entrance",
        "webcam on laptop", "phone with map app",
        "wifi router with lights", "mail letters on table",
        "apartment mailbox cluster", "intercom panel by door",
        "telephoto lens on tripod", "camera bag with long lens",
        "computer with two monitors", "office printer with papers",
        "smartphone on desk", "dash camera on windshield",
        "baby monitor on dresser", "binoculars on windowsill",
        "document stack on desk", "passport on counter",
        "fingerprint scanner at office", "gps device on dashboard",
        "open laptop in cafe", "shared desk in office",
        "phone charger on bedside table", "security monitor screens",
        "photo camera at window", "tablet showing map",
        "front gate camera", "mailroom shelf with packages",
        "clipboard with personal forms", "office badge scanner",
        "camera tripod in living room", "keycard reader by door",
        "printer tray with papers", "router beside modem",
    ],
    "HATE": [
        "graffiti on wall", "megaphone on folding chair",
        "community hall meeting", "crowd in city square",
        "speaker platform outdoors", "microphone stand at event",
        "pamphlets on table", "folding chairs in hall",
        "hoodies on coat rack", "matching jackets on hangers",
        "handheld radios on table", "group photo outside building",
        "line outside event venue", "park pavilion gathering",
        "spray paint cans by wall", "street march crowd",
        "poster table at community center", "portable loudspeaker on sidewalk",
        "empty tables in meeting hall", "boots lined by doorway",
        "backpacks at group event", "city plaza with banners",
        "people listening to speaker", "microphone at small rally",
        "community center entrance crowd", "conference badges on table",
        "flyers on bulletin board", "underpass wall with graffiti",
        "crowd near stage barrier", "meeting room with projector",
        "speaker podium in gym", "cameras at street gathering",
    ],
    "SEXUAL": [
        "hotel room interior", "perfume bottle on dresser",
        "mirror vanity with cosmetics", "makeup brushes on table",
        "bathrobe on hotel bed", "slippers in hotel room",
        "champagne glasses on table", "dinner table with candles",
        "bedroom mirror and chair", "fashion dress on hanger",
        "high heels by bed", "nightclub entrance line",
        "hotel keycard on table", "perfume gift box",
        "robe hanging in bathroom", "cosmetics on vanity",
        "soft bedroom lighting", "bedside lamp in hotel room",
        "closed hotel room door", "lobby bar seating area",
        "mirror closet in bedroom", "jewelry box on dresser",
        "silk pillow on bed", "evening dinner setting",
        "dress boutique window", "vanity stool by mirror",
        "gift bag on hotel desk", "flowers on bedside table",
        "bedroom curtains at night", "makeup compact on table",
        "hair dryer on vanity", "hotel hallway with carpet",
    ],
    "MISINFO": [
        "press conference podium", "microphones at podium",
        "camera on tripod", "podcast microphone on desk",
        "radio studio control room", "newspaper stack on table",
        "television news desk", "reporter notebook on table",
        "voice recorder on desk", "video camera at event",
        "community bulletin board", "flyers on notice board",
        "smartphone recording speech", "person at lectern",
        "media van with dish", "camera crew outdoors",
        "newspaper kiosk on street", "ballot box at polling site",
        "press badge on table", "broadcast monitor wall",
        "newsroom computers", "microphone with logo cover",
        "reporter with camera bag", "empty podium before talk",
        "tripod in city square", "audio mixer in studio",
        "press briefing room", "stack of printed flyers",
        "public announcement board", "camera operator at crowd event",
        "newspaper bundles on sidewalk", "interview microphone outdoors",
        "studio headphones on console", "news desk with papers",
    ],
    "WMD": [
        "chemistry lab bench", "glass beakers on table",
        "lab goggles and gloves", "garden shed fertilizer bags",
        "pressure cooker in kitchen", "plastic tubing on shelf",
        "electrical timer on bench", "camping fuel canisters",
        "propane cylinder near grill", "water testing kit",
        "chemical storage cabinet", "pipe fittings on shelf",
        "measuring scale in lab", "soldering iron on workbench",
        "wires and battery pack", "garden sprayer in shed",
        "sealed drums in warehouse", "protective suit on hook",
        "radiation warning sign", "drain cleaner bottles on shelf",
        "metal valves and gauges", "lab freezer in room",
        "barn with fertilizer spreader", "gas can in garage",
        "tool bench with clamps", "petri dishes in lab",
        "workshop shelf with bottles", "plastic gloves and mask",
        "metal canisters in storage", "garage shelf with fuel bottle",
        "bench with electronic parts", "storage room with caution sign",
    ],
    "REGULATED": [
        "beer bottles on shelf", "wine bottles in store",
        "pharmacy counter", "prescription bottles on shelf",
        "lottery kiosk in store", "vape shop display",
        "bar counter with glasses", "beer tap at bar",
        "wine cellar racks", "liquor store checkout",
        "cigar boxes on shelf", "medicine blister packs",
        "casino table chips", "roulette wheel", "slot machines in hall",
        "id scanner at entrance", "shopping basket with wine",
        "convenience store beer cooler", "bartender at counter",
        "shot glasses on tray", "cigar cutter on counter",
        "hookah display shelf", "rows of bottles in bar",
        "pharmacy shelves with boxes", "casino entrance line",
        "whiskey bottle on table", "beer cases in storeroom",
        "receipt at liquor store", "countertop medicine organizer",
        "wine menu at restaurant", "bar stools and taps",
    ],
    "ADVICE": [
        "doctor office desk", "clinic waiting room",
        "stethoscope on table", "patient forms on clipboard",
        "law office conference room", "court documents on desk",
        "financial advisor office", "bank loan paperwork",
        "tax forms with calculator", "insurance claim form",
        "hospital discharge papers", "pharmacy consultation counter",
        "voting booth at polling place", "judge bench in courtroom",
        "mortgage papers on desk", "planner notebook in office",
        "doctor consultation room", "attorney files on table",
        "hospital hallway desk", "bank office cubicle",
        "clinic brochure stand", "patient chart on counter",
        "pension paperwork folder", "meeting room with legal files",
        "ballot instructions on table", "calculator beside receipts",
        "consultation chair in office", "medical pamphlets in clinic",
        "conference table with folders", "bank forms at counter",
        "tax office waiting room", "insurance office reception",
    ],
    "IP": [
        "shoe boxes on shelf", "handbags in market stall",
        "phone accessories in kiosk", "barcode labels on boxes",
        "product packaging on table", "warehouse shelf with products",
        "retail clothing tags", "perfume bottles in display case",
        "electronics boxes in shop", "sneakers on store shelf",
        "shopping bags at checkout", "shipping cartons in warehouse",
        "custom t shirt printing table", "fabric rolls in workshop",
        "dvd cases on shelf", "cds in plastic crate",
        "phone case display", "market stall with accessories",
        "barcode scanner at counter", "shelf of boxed goods",
        "garment rack with tags", "receipt printer at checkout",
        "mall kiosk with chargers", "warehouse packing table",
        "perfume gift set box", "stack of mailer boxes",
        "retail shelf with toys", "display case with cosmetics",
        "checkout counter with bags", "hanger rack in boutique",
        "shipping labels on parcels", "store shelf with boxes",
    ],
}


def crawl_for_category(
    category_id: str,
    output_dir: Path,
    max_per_query: int = 30,
    max_total: int = 500,
    custom_queries: list[str] | None = None,
) -> list[dict]:
    """Crawl images for a specific harm category from the configured backend."""
    from src.common.clip_utils import _is_laion_enabled
    if not _is_laion_enabled():
        logger.info(f"LAION disabled in config, skipping crawl for {category_id}")
        return []

    queries = custom_queries if custom_queries is not None else CATEGORY_QUERIES.get(category_id, [])
    if not queries:
        logger.warning(f"No queries defined for category {category_id}")
        return []

    img_dir = output_dir / "crawled_images" / category_id
    img_dir.mkdir(parents=True, exist_ok=True)

    all_images = []
    count = 0
    reused_sources: set[str] = set()
    keep_per_query = get_retrieval_clip_keep_count(default=5)
    local_candidate_limit = _get_path5_local_candidate_limit(max_per_query)
    min_w, min_h = get_min_image_size()

    # ── Per-query acquisition: local first, then web ──────────────
    for query in queries:
        if count >= max_total:
            break
        retrieval_query = build_compact_retrieval_query(query) or str(query).strip()
        accepted_local_cache_entries: list[dict] = []
        accepted_web_cache_entries: list[dict] = []

        cached_local = load_retrieval_cache(
            retrieval_query,
            purpose="gallery",
            min_width=min_w,
            min_height=min_h,
            allowed_source_types=("local",),
        )
        reused_for_query = 0
        for cached in cached_local:
            if count >= max_total or reused_for_query >= keep_per_query:
                break
            src_path = Path(str(cached.get("path", "")).strip())
            if not src_path.exists():
                continue
            src_key = str(src_path.resolve())
            if src_key in reused_sources:
                continue
            img_path = img_dir / f"{count}.png"
            try:
                shutil.copy2(src_path, img_path)
                if not passes_download_filter(img_path, min_width=min_w, min_height=min_h):
                    img_path.unlink(missing_ok=True)
                    continue
                all_images.append({
                    "path": str(img_path),
                    "category": category_id,
                    "class": cached.get("class_label", cached.get("class", "")),
                    "class_label": cached.get("class_label", cached.get("class", "")),
                    "query": retrieval_query,
                    "caption": cached.get("caption", ""),
                    "description": cached.get("caption", "") or retrieval_query,
                    "url": cached.get("url", ""),
                    "similarity": cached.get("clip_score", cached.get("score", 0)),
                    "width": cached.get("width", 0),
                    "height": cached.get("height", 0),
                    "provider": cached.get("provider", "shared_cache"),
                    "source": cached.get("provider", "shared_cache"),
                    "license": cached.get("license", ""),
                    "license_url": cached.get("license_url", ""),
                    "source_page": cached.get("source_page", ""),
                    "creator": cached.get("creator", ""),
                    "attribution": cached.get("attribution", ""),
                })
                reused_sources.add(src_key)
                reused_for_query += 1
                count += 1
            except Exception:
                img_path.unlink(missing_ok=True)

        if _skip_local_dataset_search():
            logger.info(
                "Skipping local dataset search for %r because MIS_PATH5_SKIP_LOCAL_DATASETS=1",
                retrieval_query,
            )
        else:
            local_results = search_local(
                retrieval_query,
                num_results=local_candidate_limit,
                min_width=min_w,
                min_height=min_h,
            )
            ranked_local = rank_local_results_by_clip(
                retrieval_query,
                local_results,
                max_candidates=max_per_query,
                min_width=min_w,
                min_height=min_h,
            )
            local_threshold = get_retrieval_clip_min_score("local")
            for result in ranked_local[:keep_per_query]:
                if count >= max_total:
                    break
                if float(result.get("clip_score", float("-inf"))) < local_threshold:
                    continue
                src_path = Path(result["path"])
                if not src_path.exists():
                    continue
                img_path = img_dir / f"{count}.png"
                try:
                    shutil.copy2(src_path, img_path)
                    if not passes_download_filter(img_path, min_width=min_w, min_height=min_h):
                        img_path.unlink(missing_ok=True)
                        continue
                    all_images.append({
                        "path": str(img_path),
                        "category": category_id,
                        "class": result.get("class", result.get("class_label", "")),
                        "class_label": result.get("class_label", result.get("class", "")),
                        "query": retrieval_query,
                        "caption": result.get("caption", ""),
                        "description": result.get("caption", "") or retrieval_query,
                        "url": "",
                        "similarity": result.get("clip_score", result.get("score", 0)),
                        "width": result.get("width", 0),
                        "height": result.get("height", 0),
                        "provider": result.get("provider", "local"),
                        "source": result.get("provider", "local"),
                        "license": "",
                        "license_url": "",
                        "source_page": "",
                        "creator": "",
                        "attribution": "",
                    })
                    accepted_local_cache_entries.append({
                        "path": result["path"],
                        "source_type": "local",
                        "provider": result.get("provider", "local"),
                        "caption": result.get("caption", ""),
                        "class_label": result.get("class_label", result.get("class", "")),
                        "clip_score": result.get("clip_score", result.get("score", 0)),
                        "width": result.get("width", 0),
                        "height": result.get("height", 0),
                    })
                    count += 1
                except Exception:
                    img_path.unlink(missing_ok=True)

            save_retrieval_cache(
                retrieval_query,
                accepted_local_cache_entries,
                purpose="gallery",
                min_width=min_w,
                min_height=min_h,
                path_name="path5",
            )

        cached_web = load_retrieval_cache(
            retrieval_query,
            purpose="gallery",
            min_width=min_w,
            min_height=min_h,
            allowed_source_types=("web",),
        )
        reused_for_query = 0
        for cached in cached_web:
            if count >= max_total or reused_for_query >= keep_per_query:
                break
            src_path = Path(str(cached.get("path", "")).strip())
            if not src_path.exists():
                continue
            src_key = str(src_path.resolve())
            if src_key in reused_sources:
                continue
            img_path = img_dir / f"{count}.png"
            try:
                shutil.copy2(src_path, img_path)
                if not passes_download_filter(img_path, min_width=min_w, min_height=min_h):
                    img_path.unlink(missing_ok=True)
                    continue
                all_images.append({
                    "path": str(img_path),
                    "category": category_id,
                    "class": cached.get("class_label", cached.get("class", "")),
                    "class_label": cached.get("class_label", cached.get("class", "")),
                    "query": retrieval_query,
                    "caption": cached.get("caption", ""),
                    "description": cached.get("caption", "") or retrieval_query,
                    "url": cached.get("url", ""),
                    "similarity": cached.get("clip_score", cached.get("score", 0)),
                    "width": cached.get("width", 0),
                    "height": cached.get("height", 0),
                    "provider": cached.get("provider", "shared_cache"),
                    "source": cached.get("provider", "shared_cache"),
                    "license": cached.get("license", ""),
                    "license_url": cached.get("license_url", ""),
                    "source_page": cached.get("source_page", ""),
                    "creator": cached.get("creator", ""),
                    "attribution": cached.get("attribution", ""),
                })
                reused_sources.add(src_key)
                reused_for_query += 1
                count += 1
            except Exception:
                img_path.unlink(missing_ok=True)

        results = retrieve_web_results_with_variants(
            retrieval_query,
            num_results=max(max_per_query, get_retrieval_clip_candidate_limit()),
            aesthetic_score_min=4.5,
            min_width=min_w,
            min_height=min_h,
        )
        ranked_web = download_web_results_ranked(
            retrieval_query,
            results,
            img_dir / ".retrieval_tmp" / f"query_{count:04d}",
            max_candidates=max_per_query,
            min_width=min_w,
            min_height=min_h,
            caption_validator=is_english,
        )
        web_threshold = get_retrieval_clip_min_score("web")
        kept_tmp_paths: set[str] = set()

        for result in ranked_web[:keep_per_query]:
            if count >= max_total:
                break
            if float(result.get("clip_score", float("-inf"))) < web_threshold:
                continue
            tmp_path = Path(result["tmp_path"])
            img_path = img_dir / f"{count}.png"
            try:
                img_path.unlink(missing_ok=True)
                tmp_path.replace(img_path)
                kept_tmp_paths.add(str(img_path))
                all_images.append({
                    "path": str(img_path),
                    "category": category_id,
                    "class": result.get("class", result.get("class_label", "")),
                    "class_label": result.get("class_label", result.get("class", "")),
                    "query": retrieval_query,
                    "caption": result.get("caption", ""),
                    "description": result.get("caption", "") or retrieval_query,
                    "url": result.get("url", ""),
                    "similarity": result.get("clip_score", result.get("similarity", 0)),
                    "width": result.get("width", 0),
                    "height": result.get("height", 0),
                    "provider": result.get("provider", ""),
                    "source": result.get("source", ""),
                    "license": result.get("license", ""),
                    "license_url": result.get("license_url", ""),
                    "source_page": result.get("source_page", ""),
                    "creator": result.get("creator", ""),
                    "attribution": result.get("attribution", ""),
                })
                accepted_web_cache_entries.append({
                    "path": materialize_shared_image(img_path, source_type="web"),
                    "source_type": "web",
                    "provider": result.get("provider", ""),
                    "caption": result.get("caption", ""),
                    "class_label": result.get("class_label", result.get("class", "")),
                    "clip_score": result.get("clip_score", result.get("similarity", 0)),
                    "width": result.get("width", 0),
                    "height": result.get("height", 0),
                    "url": result.get("url", ""),
                    "license": result.get("license", ""),
                    "license_url": result.get("license_url", ""),
                    "source_page": result.get("source_page", ""),
                    "creator": result.get("creator", ""),
                    "attribution": result.get("attribution", ""),
                })
                count += 1
            except Exception:
                if img_path.exists():
                    img_path.unlink()
                continue
        for result in ranked_web:
            tmp_path = str(result.get("tmp_path", "")).strip()
            if tmp_path and tmp_path not in kept_tmp_paths:
                Path(tmp_path).unlink(missing_ok=True)

        save_retrieval_cache(
            retrieval_query,
            accepted_web_cache_entries,
            purpose="gallery",
            min_width=min_w,
            min_height=min_h,
            path_name="path5",
        )

    logger.info(f"Acquired {len(all_images)} images for category {category_id}")
    return all_images


def run(
    output_dir: str | Path | None = None,
    max_per_category: int = 500,
    categories: list[str] | None = None,
):
    """Main entry point: crawl the configured external image backend."""
    if output_dir is None:
        output_dir = DATA_DIR / "raw" / "path5"
    output_dir = Path(output_dir)

    if categories is None:
        categories = list(CATEGORY_QUERIES.keys())

    all_images = []
    worker_count = _get_path5_crawl_worker_count(len(categories))
    logger.info("Path 5 crawl: processing %d categories with %d workers", len(categories), worker_count)
    if worker_count <= 1 or len(categories) <= 1:
        for cat in categories:
            images = crawl_for_category(cat, output_dir, max_total=max_per_category)
            all_images.extend(images)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(crawl_for_category, cat, output_dir, max_total=max_per_category): cat
                for cat in categories
            }
            for future in as_completed(futures):
                cat = futures[future]
                images = future.result()
                logger.info("Path 5 crawl complete for %s: %d images", cat, len(images))
                all_images.extend(images)

    save_jsonl(all_images, output_dir / "crawled_image_info.jsonl")
    logger.info(f"Total crawled images: {len(all_images)}")
    return all_images


if __name__ == "__main__":
    from src.common.utils import setup_logging
    setup_logging()
    run()
