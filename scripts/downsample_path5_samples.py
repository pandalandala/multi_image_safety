#!/usr/bin/env python3
"""Downsample Path 5 samples by removing samples that overuse frequent images.

Strategy:
1. Count how often each image appears across all samples.
2. Iteratively remove the sample whose two images are currently the most overused.
3. Break ties by preferring removal of lower-confidence samples.

This keeps a smaller set that uses a broader spread of images.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def save_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/mnt/hdd/xuran/multi_image_safety/data/raw/path5/samples_with_prompts.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/hdd/xuran/multi_image_safety/data/raw/path5/samples_with_prompts_balanced_5000.jsonl"),
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=Path("/mnt/hdd/xuran/multi_image_safety/data/raw/path5/samples_with_prompts_balanced_5000.stats.json"),
    )
    parser.add_argument("--target", type=int, default=5000)
    return parser.parse_args()


def sample_score(sample: dict, counts: Counter[str]) -> tuple[int, int, int, int]:
    """Higher score means the sample is a better removal candidate."""
    p1 = sample.get("image1_path", "")
    p2 = sample.get("image2_path", "")
    c1 = counts.get(p1, 0)
    c2 = counts.get(p2, 0)
    confidence = int(sample.get("confidence", 0) or 0)
    return (
        max(c1, c2),
        c1 + c2,
        min(c1, c2),
        -confidence,  # lower confidence gets removed earlier
    )


def build_stats(samples: list[dict]) -> dict:
    image_use = Counter()
    category_counts = Counter()
    for sample in samples:
        image_use[sample.get("image1_path", "")] += 1
        image_use[sample.get("image2_path", "")] += 1
        category_counts[sample.get("category", "<missing>")] += 1

    usage_hist = Counter()
    for count in image_use.values():
        if count <= 2:
            usage_hist["1-2"] += 1
        elif count <= 5:
            usage_hist["3-5"] += 1
        elif count <= 10:
            usage_hist["6-10"] += 1
        elif count <= 20:
            usage_hist["11-20"] += 1
        elif count <= 50:
            usage_hist["21-50"] += 1
        elif count <= 100:
            usage_hist["51-100"] += 1
        else:
            usage_hist["101+"] += 1

    total_appearances = sum(image_use.values())
    return {
        "samples": len(samples),
        "unique_images_used": len(image_use),
        "total_image_appearances": total_appearances,
        "avg_use_per_image": (total_appearances / len(image_use)) if image_use else 0.0,
        "max_image_use": max(image_use.values()) if image_use else 0,
        "usage_histogram": dict(usage_hist),
        "category_counts": dict(category_counts),
        "top_20_most_used_images": [
            {"path": path, "count": count}
            for path, count in image_use.most_common(20)
        ],
    }


def main() -> int:
    args = parse_args()
    samples = load_jsonl(args.input)
    if args.target <= 0:
        raise SystemExit("--target must be positive")
    if len(samples) <= args.target:
        save_jsonl(samples, args.output)
        args.stats_output.write_text(
            json.dumps(
                {
                    "message": "input already at or below target",
                    "before": build_stats(samples),
                    "after": build_stats(samples),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Input already has {len(samples)} samples; copied without downsampling.")
        return 0

    active_samples = list(samples)

    while len(active_samples) > args.target:
        counts: Counter[str] = Counter()
        for sample in active_samples:
            counts[sample.get("image1_path", "")] += 1
            counts[sample.get("image2_path", "")] += 1

        excess = len(active_samples) - args.target
        batch_size = min(excess, max(500, excess // 4))

        scored = []
        for idx, sample in enumerate(active_samples):
            s = sample_score(sample, counts)
            scored.append((s[0], s[1], s[2], s[3], idx))

        # Highest-frequency samples first; ties break toward lower confidence removal.
        scored.sort(reverse=True)
        remove_indices = {idx for *_score, idx in scored[:batch_size]}
        active_samples = [
            sample
            for idx, sample in enumerate(active_samples)
            if idx not in remove_indices
        ]

    kept_samples = active_samples

    save_jsonl(kept_samples, args.output)

    stats = {
        "input": str(args.input),
        "output": str(args.output),
        "target": args.target,
        "before": build_stats(samples),
        "after": build_stats(kept_samples),
    }
    args.stats_output.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "before_samples": len(samples),
                "after_samples": len(kept_samples),
                "before_avg_use_per_image": round(stats["before"]["avg_use_per_image"], 3),
                "after_avg_use_per_image": round(stats["after"]["avg_use_per_image"], 3),
                "before_max_image_use": stats["before"]["max_image_use"],
                "after_max_image_use": stats["after"]["max_image_use"],
                "output": str(args.output),
                "stats_output": str(args.stats_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
