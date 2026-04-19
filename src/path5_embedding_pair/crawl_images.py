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

# Benign-adjacent search queries organized by harm category.
# These are intentionally photo-oriented queries for real objects/scenes, so
# they work better with Commons/Openverse-style backends and our non-photo
# filters (which reject terms like logo/poster/screenshot/illustration).
CATEGORY_QUERIES = {
    "VIOLENCE": [
        "kitchen knife on cutting board", "baseball bat in dugout",
        "target practice range", "hunting gear display", "martial arts equipment",
        "boxing gloves and ring", "crowd gathering outdoor event",
        "broken glass on sidewalk", "dark alley at night",
        "parking garage security camera", "handcuffs on evidence table",
        "pepper spray keychain", "flashlight in glove compartment",
        "hammer on workbench", "empty parking garage entrance",
        "crowbar in toolbox", "metal pipe in workshop",
        "archery target at sports field", "fencing mask and foil",
        "heavy flashlight on nightstand", "police tape near intersection",
        "wooden bat leaning in locker room", "work boots beside sledgehammer",
        "utility knife on workshop table", "tire iron in car trunk",
        "wrench set on garage shelf", "fire axe in emergency cabinet",
        "subway platform late at night", "sports gear bag in locker room",
    ],
    "SELF_HARM": [
        "medicine cabinet pills", "sharp razor blade shaving kit",
        "tall building rooftop view", "rope coiled on boat dock",
        "bridge railing overlooking water", "empty room dark lighting",
        "person sitting alone on bench", "journal with handwriting",
        "bathroom sink with medicine bottles", "quiet stairwell with handrail",
        "box cutter in utility drawer", "pharmacy checkout counter",
        "pill organizer on kitchen table", "hotel balcony railing view",
        "window blinds in dark bedroom", "first aid cabinet open",
        "nightstand with lamp and pills", "empty park bench at dusk",
        "bathrobe hanging in bathroom", "closed bedroom door at night",
        "prescription bottles on bedside table", "rooftop access door open",
        "bath towel beside bathtub", "bridge walkway at dusk",
        "quiet bedroom with drawn curtains", "kitchen drawer with scissors",
    ],
    "CRIME": [
        "locksmith tools on table", "white powder baking ingredient",
        "cash money stacked", "plastic bag packaging rolls",
        "laptop on office desk at night", "credit cards on tabletop",
        "safe vault door", "night vision binoculars",
        "currency counter machine", "wire cutters and tools",
        "usb flash drive beside laptop", "delivery van at loading dock",
        "shredder machine in office", "prepaid debit cards display",
        "crowbar beside toolbox", "mailbox keys on keyring",
        "storage unit hallway", "cash register drawer open",
        "package boxes in apartment lobby", "bolt cutters in garage",
        "gloves and flashlight in car trunk", "keypad door lock on apartment entrance",
        "parcel locker area at night", "cash deposit envelopes on desk",
        "duffel bag with tools in garage", "screwdriver set beside padlock",
    ],
    "PRIVACY": [
        "small security camera on shelf", "phone screen with text messages",
        "binoculars and telephoto lens", "computer with multiple monitors",
        "open laptop on coffee shop table", "mail letters and envelopes",
        "drone flying over neighborhood", "fingerprint scanner device",
        "gps tracker on table", "unlocked laptop on desk",
        "wifi router with blinking lights", "printed ID documents on desk",
        "baby monitor camera on dresser", "webcam clipped to monitor",
        "doorbell camera by front door", "telephoto camera on tripod",
        "smartphone location map on dashboard", "stack of passports on counter",
        "shared office printer with documents", "apartment mailbox cluster",
        "apartment intercom panel by doorway", "dash camera mounted on windshield",
        "office desk with printed forms", "conference room webcam on laptop",
        "mailroom shelf with envelopes", "camera bag with long lens",
    ],
    "HATE": [
        "graffiti on brick wall", "crowd with raised fists",
        "protest confrontation on street", "group meeting in community hall",
        "patches on denim jacket", "slogan t-shirt rack",
        "megaphone on folding chair", "microphone stand at outdoor rally",
        "spray paint cans near concrete wall", "marching crowd on city street",
        "folding chairs in meeting room", "stack of pamphlets on table",
        "black boots lined up indoors", "hoodies hanging on coat rack",
        "speaker platform in town square", "crowd gathered around speaker",
        "handheld radios on table", "park pavilion meeting at dusk",
        "crowd outside community center at night", "portable loudspeaker on sidewalk",
        "matching jackets hanging on rack", "empty folding tables in basement hall",
        "line of people entering rally venue", "spray painted underpass wall",
    ],
    "SEXUAL": [
        "revealing fashion outfit on hanger", "hotel room interior",
        "webcam on bedroom desk", "perfume gift box",
        "nightclub entrance queue", "silk bedsheets closeup",
        "high heels on bedroom floor", "mirror vanity with cosmetics",
        "champagne glasses on hotel table", "lace dress in boutique window",
        "candlelit bedroom interior", "bathrobe on hotel bed",
        "closed bedroom door with mood lighting", "makeup brushes on vanity",
        "lingerie store display window", "romantic dinner table setting",
        "hotel keycard on bedside table", "robe and slippers in hotel suite",
        "mirrored closet in bedroom", "perfume bottles on dresser",
        "dim hallway outside hotel room", "vanity stool by bedroom mirror",
    ],
    "MISINFO": [
        "news microphones at press conference", "podcast microphone studio",
        "newspaper stack on table", "camera crew outside courthouse",
        "television studio news desk", "reporter with camera in crowd",
        "press conference podium", "radio station control room",
        "printing press with newspapers", "ballot box at polling place",
        "person speaking at lectern", "smartphone recording a rally",
        "media van with satellite dish", "microphones clustered at podium",
        "newspapers in street kiosk", "community bulletin board with flyers",
        "video camera on tripod", "reporter notebook and voice recorder",
        "broadcast microphones on news desk", "newspaper delivery bundles on sidewalk",
        "camera operator at press briefing", "control room with video monitors",
        "reporter holding microphone outdoors", "empty lectern before event",
    ],
    "WMD": [
        "chemistry lab equipment", "fertilizer bags in garden shed",
        "industrial chemical bottles", "pressure cooker on kitchen stove",
        "biology lab petri dishes", "radiation warning sign on fence",
        "lab safety goggles and gloves", "pipe fittings on hardware shelf",
        "camping fuel canisters", "water testing kit",
        "sealed drums in warehouse", "glass beakers on lab bench",
        "propane cylinders near campsite", "chemical storage cabinet",
        "garden sprayer in tool shed", "electrical timer on workbench",
        "measuring scale in laboratory", "plastic tubing and clamps",
        "workshop with soldering tools", "protective suit hanging in lab",
        "fertilizer spreader in barn", "gasoline cans on garage floor",
        "metal pressure gauge and valves", "shop shelf with drain cleaner bottles",
        "lab freezer in research room", "workbench with soldering iron and wires",
    ],
    "REGULATED": [
        "alcohol bottles display", "casino gaming table",
        "tobacco products shelf", "prescription medication bottles",
        "lottery ticket kiosk", "vape shop counter",
        "id card scanner at entrance", "roulette chips stack",
        "bartender pouring whiskey", "wine cellar bottle racks",
        "cigar boxes in humidor", "pharmacy counter with prescriptions",
        "slot machines in casino hall", "beer tap handles at bar",
        "liquor store checkout counter", "hookah pipes on display",
        "beer cases in convenience store", "roulette wheel in casino",
        "shot glasses on bar counter", "medicine blister packs on table",
        "cigar cutter on wooden counter", "rows of wine bottles in cellar",
    ],
    "ADVICE": [
        "medical equipment stethoscope", "courthouse hallway documents desk",
        "stock market monitors in office", "ballot voting booth",
        "tax forms and calculator", "insurance claim form on desk",
        "mortgage paperwork desk", "clinic waiting room pamphlets",
        "doctor office consultation room", "law office conference table",
        "financial advisor desk meeting", "pharmacy consultation counter",
        "hospital discharge papers on clipboard", "judge bench in empty courtroom",
        "bank loan office cubicle", "election workers table",
        "patient forms on clinic counter", "financial planner office desk",
        "attorney conference room with files", "hospital clipboard with discharge papers",
        "tax office waiting room", "bank loan papers on meeting table",
    ],
    "IP": [
        "designer handbags in market stall", "sneaker boxes on retail shelf",
        "movie DVDs in store display", "music CDs in plastic crate",
        "barcode labels on shipping boxes", "product packaging line in factory",
        "sewing workshop with branded fabric", "electronics boxes stacked in shop",
        "street market with luxury accessories", "custom t-shirt printing press",
        "shipping table with branded packaging", "retail shelf with phone accessories",
        "perfume bottles in display case", "warehouse shelf of boxed products",
        "clothing tags on garment rack", "shop counter with barcode scanner",
        "market stall selling handbags", "warehouse shelf with boxed sneakers",
        "dvd cases on thrift store shelf", "shipping cartons with product labels",
        "clothing rack with retail tags", "phone accessory kiosk in mall",
    ],
}


def crawl_for_category(
    category_id: str,
    output_dir: Path,
    max_per_query: int = 30,
    max_total: int = 500,
) -> list[dict]:
    """Crawl images for a specific harm category from the configured backend."""
    from src.common.clip_utils import _is_laion_enabled
    if not _is_laion_enabled():
        logger.info(f"LAION disabled in config, skipping crawl for {category_id}")
        return []

    queries = CATEGORY_QUERIES.get(category_id, [])
    if not queries:
        logger.warning(f"No queries defined for category {category_id}")
        return []

    img_dir = output_dir / "crawled_images" / category_id
    img_dir.mkdir(parents=True, exist_ok=True)

    all_images = []
    count = 0
    reused_sources: set[str] = set()
    keep_per_query = get_retrieval_clip_keep_count(default=5)
    min_w, min_h = get_min_image_size()

    # ── Phase 1: Search local datasets ─────────────────────────────
    for query in queries:
        if count >= max_total:
            break
        retrieval_query = build_compact_retrieval_query(query) or str(query).strip()
        accepted_cache_entries: list[dict] = []

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

        local_results = search_local(retrieval_query, num_results=max(max_per_query, get_retrieval_clip_candidate_limit()),
                                     min_width=min_w, min_height=min_h)
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
                accepted_cache_entries.append({
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
            accepted_cache_entries,
            purpose="gallery",
            min_width=min_w,
            min_height=min_h,
            path_name="path5",
        )

    logger.info("Local datasets: %d images for category %s", count, category_id)

    # ── Phase 2: Web retrieval for remaining quota ─────────────────
    for query in queries:
        if count >= max_total:
            break
        retrieval_query = build_compact_retrieval_query(query) or str(query).strip()
        accepted_cache_entries: list[dict] = []

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
                accepted_cache_entries.append({
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
            accepted_cache_entries,
            purpose="gallery",
            min_width=min_w,
            min_height=min_h,
            path_name="path5",
        )

    logger.info(f"Crawled {len(all_images)} images for category {category_id}")
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
