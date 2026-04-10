"""Step 1: Targeted crawling from the configured image backend.

For each harm category, design "benign-adjacent" search queries that retrieve
individually harmless images with potential for compositional harm.
"""

import logging
from pathlib import Path
from typing import Optional

from PIL import Image

from src.common.clip_utils import (
    retrieve_from_laion,
    download_image_url,
    passes_download_filter,
)
from src.common.utils import (
    get_all_categories,
    is_english,
    save_jsonl,
    load_config,
    DATA_DIR,
)

logger = logging.getLogger(__name__)

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

    for query in queries:
        if count >= max_total:
            break

        results = retrieve_from_laion(
            query,
            num_results=max_per_query,
            aesthetic_score_min=4.5,
            min_width=512,
            min_height=512,
        )

        for result in results:
            if count >= max_total:
                break

            url = result.get("url", "")
            if not url:
                continue

            # Skip non-English captions
            caption = result.get("caption", "")
            if caption and not is_english(caption):
                continue

            img_path = img_dir / f"{count}.png"
            try:
                download_image_url(url, img_path)
                img = Image.open(img_path)
                img.verify()
                if not passes_download_filter(img_path, min_width=512, min_height=512):
                    if img_path.exists():
                        img_path.unlink()
                    continue

                all_images.append({
                    "path": str(img_path),
                    "category": category_id,
                    "query": query,
                    "caption": result.get("caption", ""),
                    "description": result.get("caption", "") or query,
                    "url": url,
                    "similarity": result.get("similarity", 0),
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
                count += 1
            except Exception:
                if img_path.exists():
                    img_path.unlink()
                continue

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
    for cat in categories:
        images = crawl_for_category(cat, output_dir, max_total=max_per_category)
        all_images.extend(images)

    save_jsonl(all_images, output_dir / "crawled_image_info.jsonl")
    logger.info(f"Total crawled images: {len(all_images)}")
    return all_images


if __name__ == "__main__":
    from src.common.utils import setup_logging
    setup_logging()
    run()
