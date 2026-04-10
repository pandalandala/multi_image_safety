"""Distribution balancing and sampling across categories, patterns, and difficulty."""

import logging
import random
from collections import Counter, defaultdict

from src.common.schema import HarmCategory, Pattern
from src.common.utils import load_config

logger = logging.getLogger(__name__)


def analyze_distribution(samples: list[dict]) -> dict:
    """Analyze the current distribution of samples."""
    cat_dist = Counter(s.get("category", "") for s in samples)
    pattern_dist = Counter(s.get("pattern", "") for s in samples)
    difficulty_dist = Counter(s.get("difficulty", "medium") for s in samples)
    source_dist = Counter(s.get("source_path", 0) for s in samples)

    stats = {
        "total": len(samples),
        "category_distribution": dict(cat_dist),
        "pattern_distribution": dict(pattern_dist),
        "difficulty_distribution": dict(difficulty_dist),
        "source_distribution": dict(source_dist),
    }

    logger.info(f"Distribution analysis:")
    logger.info(f"  Total: {stats['total']}")
    logger.info(f"  Categories: {stats['category_distribution']}")
    logger.info(f"  Patterns: {stats['pattern_distribution']}")
    logger.info(f"  Difficulty: {stats['difficulty_distribution']}")
    logger.info(f"  Sources: {stats['source_distribution']}")

    return stats


def balance_categories(
    samples: list[dict],
    target_total: int = 10000,
    target_weights: dict | None = None,
) -> list[dict]:
    """
    Balance samples across harm categories.
    Uses SORRY-Bench style equal representation with oversampling for long-tail.
    """
    if target_weights is None:
        config = load_config()
        target_weights = config.get("balance", {}).get("category_weights", {})

    # Group by category
    by_category = defaultdict(list)
    for s in samples:
        cat = s.get("category", "CRIME")
        by_category[cat].append(s)

    # Calculate target count per category
    if target_weights:
        target_per_cat = {
            cat: int(target_total * weight)
            for cat, weight in target_weights.items()
        }
    else:
        # Equal distribution
        n_cats = len(by_category)
        per_cat = target_total // n_cats
        target_per_cat = {cat: per_cat for cat in by_category}

    balanced = []
    for cat, target_count in target_per_cat.items():
        available = by_category.get(cat, [])
        if len(available) == 0:
            logger.warning(f"No samples for category {cat}")
            continue

        if len(available) >= target_count:
            # Downsample
            random.shuffle(available)
            balanced.extend(available[:target_count])
        else:
            # Upsample (repeat with slight shuffling)
            balanced.extend(available)
            remaining = target_count - len(available)
            while remaining > 0:
                take = min(remaining, len(available))
                random.shuffle(available)
                balanced.extend(available[:take])
                remaining -= take
            logger.info(
                f"Category {cat}: upsampled {len(available)} -> {target_count}"
            )

    random.shuffle(balanced)
    logger.info(
        f"Category balancing: {len(samples)} -> {len(balanced)} samples"
    )
    return balanced


def balance_patterns(
    samples: list[dict],
    target_weights: dict | None = None,
) -> list[dict]:
    """Balance samples across harm patterns (A/B/C/D)."""
    if target_weights is None:
        config = load_config()
        target_weights = config.get("balance", {}).get("pattern_weights", {})

    if not target_weights:
        return samples

    total = len(samples)
    by_pattern = defaultdict(list)
    for s in samples:
        pattern = s.get("pattern", "A")
        by_pattern[pattern].append(s)

    balanced = []
    for pattern, weight in target_weights.items():
        target = int(total * weight)
        available = by_pattern.get(pattern, [])
        if available:
            random.shuffle(available)
            balanced.extend(available[:target])

    random.shuffle(balanced)
    return balanced


def ensure_difficulty_spread(
    samples: list[dict],
    min_hard_ratio: float = 0.25,
    min_easy_ratio: float = 0.20,
) -> list[dict]:
    """Ensure a reasonable spread of difficulty levels."""
    by_diff = defaultdict(list)
    for s in samples:
        d = s.get("difficulty", "medium")
        by_diff[d].append(s)

    total = len(samples)
    hard_count = len(by_diff.get("hard", []))
    easy_count = len(by_diff.get("easy", []))

    if hard_count / total < min_hard_ratio:
        logger.warning(
            f"Hard samples ratio too low: {hard_count/total:.2f} < {min_hard_ratio}"
        )
    if easy_count / total < min_easy_ratio:
        logger.warning(
            f"Easy samples ratio too low: {easy_count/total:.2f} < {min_easy_ratio}"
        )

    return samples


def run_balance(
    samples: list[dict],
    target_total: int = 10000,
) -> list[dict]:
    """Run the full balancing pipeline."""
    stats_before = analyze_distribution(samples)

    samples = balance_categories(samples, target_total)
    samples = ensure_difficulty_spread(samples)

    # Final trim to target
    if len(samples) > target_total:
        random.shuffle(samples)
        samples = samples[:target_total]

    stats_after = analyze_distribution(samples)
    return samples
