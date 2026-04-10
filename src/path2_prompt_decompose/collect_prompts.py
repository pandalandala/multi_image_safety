"""Step 1: Collect malicious prompts from multiple sources.

Sources: BeaverTails, SorryBench (SillyTilly), ToxicChat (lmsys).
Each prompt is classified into our 12-class harm taxonomy.
"""

import logging
import os
import random
from pathlib import Path

from src.common.schema import HarmCategory
from src.common.utils import (
    get_hf_home,
    is_english,
    save_jsonl,
    DATA_DIR,
)

logger = logging.getLogger(__name__)

# Mapping from BeaverTails category keys to our 12-class taxonomy
BEAVERTAILS_CATEGORY_MAPPING = {
    "animal_abuse": HarmCategory.VIOLENCE,
    "child_abuse": HarmCategory.CSEA,
    "controversial_topics,politics": HarmCategory.REGULATED,
    "discrimination,stereotype,injustice": HarmCategory.HATE,
    "drug_abuse,weapons,banned_substance": HarmCategory.CRIME,
    "financial_crime,property_crime,theft": HarmCategory.CRIME,
    "hate_speech,offensive_language": HarmCategory.HATE,
    "misinformation_regarding_ethics,laws_and_safety": HarmCategory.MISINFO,
    "non_violent_unethical_behavior": HarmCategory.CRIME,
    "privacy_violation": HarmCategory.PRIVACY,
    "self_harm": HarmCategory.SELF_HARM,
    "sexually_explicit,adult_content": HarmCategory.SEXUAL,
    "terrorism,organized_crime": HarmCategory.VIOLENCE,
    "violence,aiding_and_abetting,incitement": HarmCategory.VIOLENCE,
}

# SorryBench numeric categories -> our taxonomy
# SorryBench has 45 fine-grained categories grouped into broader themes
SORRY_BENCH_CATEGORY_MAPPING = {
    # 1-5: Hate speech / discrimination / harassment
    "1": HarmCategory.HATE, "2": HarmCategory.HATE, "3": HarmCategory.HATE,
    "4": HarmCategory.HATE, "5": HarmCategory.HATE,
    # 6-10: Violence / self-harm
    "6": HarmCategory.VIOLENCE, "7": HarmCategory.VIOLENCE,
    "8": HarmCategory.SELF_HARM, "9": HarmCategory.VIOLENCE,
    "10": HarmCategory.VIOLENCE,
    # 11-15: Crime / illegal activities
    "11": HarmCategory.CRIME, "12": HarmCategory.CRIME, "13": HarmCategory.CRIME,
    "14": HarmCategory.CRIME, "15": HarmCategory.CRIME,
    # 16-20: Sexual content
    "16": HarmCategory.SEXUAL, "17": HarmCategory.SEXUAL,
    "18": HarmCategory.SEXUAL, "19": HarmCategory.CSEA, "20": HarmCategory.SEXUAL,
    # 21-25: Privacy / PII
    "21": HarmCategory.PRIVACY, "22": HarmCategory.PRIVACY,
    "23": HarmCategory.PRIVACY, "24": HarmCategory.PRIVACY,
    "25": HarmCategory.PRIVACY,
    # 26-30: Misinformation
    "26": HarmCategory.MISINFO, "27": HarmCategory.MISINFO,
    "28": HarmCategory.MISINFO, "29": HarmCategory.MISINFO,
    "30": HarmCategory.MISINFO,
    # 31-35: Regulated / professional advice
    "31": HarmCategory.ADVICE, "32": HarmCategory.ADVICE,
    "33": HarmCategory.REGULATED, "34": HarmCategory.REGULATED,
    "35": HarmCategory.REGULATED,
    # 36-40: WMD / IP
    "36": HarmCategory.WMD, "37": HarmCategory.WMD,
    "38": HarmCategory.IP, "39": HarmCategory.IP, "40": HarmCategory.IP,
    # 41-45: Misc
    "41": HarmCategory.CRIME, "42": HarmCategory.HATE,
    "43": HarmCategory.VIOLENCE, "44": HarmCategory.MISINFO,
    "45": HarmCategory.REGULATED,
}


def collect_beavertails() -> list[dict]:
    """Collect unsafe prompts from BeaverTails (300k+)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")
        prompts = []
        for row in ds:
            # is_safe is a string "True"/"False" in this dataset
            is_safe = row.get("is_safe", True)
            if isinstance(is_safe, str):
                is_safe = is_safe.lower() == "true"
            if is_safe:
                continue

            text = row.get("prompt", "")
            if not text:
                continue

            # category is a dict like {"animal_abuse": False, "child_abuse": True, ...}
            cat_dict = row.get("category", {})
            if isinstance(cat_dict, str):
                import ast
                try:
                    cat_dict = ast.literal_eval(cat_dict)
                except (ValueError, SyntaxError):
                    cat_dict = {}

            # Find the True categories
            mapped_cat = HarmCategory.CRIME  # default
            original_cats = []
            if isinstance(cat_dict, dict):
                for cat_key, is_true in cat_dict.items():
                    # is_true can also be string
                    if isinstance(is_true, str):
                        is_true = is_true.lower() == "true"
                    if is_true:
                        original_cats.append(cat_key)
                        if cat_key in BEAVERTAILS_CATEGORY_MAPPING:
                            mapped_cat = BEAVERTAILS_CATEGORY_MAPPING[cat_key]

            prompts.append({
                "text": text,
                "source": "beavertails",
                "original_category": ",".join(original_cats) if original_cats else "",
                "category": mapped_cat.value,
            })

        logger.info(f"Collected {len(prompts)} prompts from BeaverTails")
        return prompts
    except Exception as e:
        logger.warning(f"Failed to load BeaverTails: {e}")
        import traceback
        traceback.print_exc()
        return []


def collect_sorry_bench() -> list[dict]:
    """Collect prompts from SorryBench (SillyTilly mirror, 9450 samples)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("SillyTilly/SorryBench", split="train")
        prompts = []
        for row in ds:
            turns = row.get("turns", [])
            text = turns[0] if turns else ""
            cat_num = str(row.get("category", ""))
            if text:
                mapped = SORRY_BENCH_CATEGORY_MAPPING.get(cat_num, HarmCategory.CRIME)
                prompts.append({
                    "text": text,
                    "source": "sorry_bench",
                    "original_category": cat_num,
                    "category": mapped.value,
                })
        logger.info(f"Collected {len(prompts)} prompts from SorryBench")
        return prompts
    except Exception as e:
        logger.warning(f"Failed to load SorryBench: {e}")
        return []


def collect_toxic_chat() -> list[dict]:
    """Collect toxic prompts from lmsys/toxic-chat."""
    try:
        from datasets import load_dataset
        ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
        prompts = []
        for row in ds:
            if row.get("toxicity", 0) != 1:
                continue
            text = row.get("user_input", "")
            if not text or len(text) < 10:
                continue
            # toxic-chat doesn't have fine-grained categories
            # Use CRIME as default, will be re-classified by LLM if needed
            prompts.append({
                "text": text,
                "source": "toxic_chat",
                "original_category": "toxic",
                "category": HarmCategory.CRIME.value,
            })
        logger.info(f"Collected {len(prompts)} prompts from ToxicChat")
        return prompts
    except Exception as e:
        logger.warning(f"Failed to load ToxicChat: {e}")
        return []


def collect_all(
    max_per_source: int = 2000,
    target_total: int = 5000,
) -> list[dict]:
    """
    Collect prompts from all sources, deduplicate, and balance across categories.

    Args:
        max_per_source: Maximum prompts to take from each source
        target_total: Target total number of prompts

    Returns:
        List of prompt dicts with keys: text, source, category, original_category
    """
    os.environ["HF_HOME"] = get_hf_home()

    collectors = [
        collect_beavertails,
        collect_sorry_bench,
        collect_toxic_chat,
    ]

    all_prompts = []
    for collector in collectors:
        prompts = collector()
        if len(prompts) > max_per_source:
            random.shuffle(prompts)
            prompts = prompts[:max_per_source]
        all_prompts.extend(prompts)

    # Filter non-English and deduplicate by text
    seen = set()
    unique_prompts = []
    skipped_lang = 0
    for p in all_prompts:
        if not is_english(p["text"]):
            skipped_lang += 1
            continue
        text_key = p["text"].strip().lower()
        if text_key not in seen:
            seen.add(text_key)
            unique_prompts.append(p)
    if skipped_lang:
        logger.info(f"Skipped {skipped_lang} non-English prompts")

    logger.info(f"Total unique prompts: {len(unique_prompts)}")

    # Limit to target
    if len(unique_prompts) > target_total:
        random.shuffle(unique_prompts)
        unique_prompts = unique_prompts[:target_total]

    return unique_prompts


def run(output_dir: str | Path | None = None, target_total: int = 5000):
    """Main entry point: collect and save prompts."""
    if output_dir is None:
        output_dir = DATA_DIR / "raw" / "path2"

    prompts = collect_all(target_total=target_total)
    save_jsonl(prompts, Path(output_dir) / "collected_prompts.jsonl")

    # Log category distribution
    from collections import Counter
    cat_dist = Counter(p["category"] for p in prompts)
    logger.info(f"Category distribution: {dict(cat_dist)}")
    source_dist = Counter(p["source"] for p in prompts)
    logger.info(f"Source distribution: {dict(source_dist)}")

    return prompts


if __name__ == "__main__":
    from src.common.utils import setup_logging
    setup_logging()
    run()
