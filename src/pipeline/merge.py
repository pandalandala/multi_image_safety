"""Merge outputs from all paths and run quality control pipeline."""

import logging
from pathlib import Path

from src.common.utils import load_jsonl, save_jsonl, load_config, DATA_DIR

logger = logging.getLogger(__name__)

# Expected output files from each path
PATH_OUTPUT_FILES = {
    1: "path1/validated_samples.jsonl",
    2: "path2/validated_samples.jsonl",
    3: "path3/cross_paired_samples.jsonl",
    4: "path4/samples_with_images.jsonl",
    5: "path5/samples_with_prompts.jsonl",
    6: "path6/validated_samples.jsonl",
}


def collect_from_paths(raw_dir: Path | None = None) -> list[dict]:
    """Collect samples from all path outputs."""
    if raw_dir is None:
        raw_dir = DATA_DIR / "raw"

    all_samples = []
    for path_id, filename in PATH_OUTPUT_FILES.items():
        filepath = raw_dir / filename
        if filepath.exists():
            samples = load_jsonl(filepath)
            logger.info(f"Path {path_id}: loaded {len(samples)} samples from {filepath}")
            # Ensure source_path is set
            for s in samples:
                s.setdefault("source_path", path_id)
            all_samples.extend(samples)
        else:
            logger.warning(f"Path {path_id}: output file not found at {filepath}")

    logger.info(f"Total samples collected: {len(all_samples)}")
    return all_samples


def run_quality_pipeline(
    samples: list[dict],
    target_total: int = 10000,
    use_api_for_verify: bool = True,
) -> list[dict]:
    """Run the full quality control pipeline: safety → verify → covertness → dedup → balance."""
    # Layer 1: Single image safety check
    logger.info("[QC Layer 1] Single image safety check...")
    from src.quality.safety_check import batch_safety_check
    samples = batch_safety_check(samples)

    # Layer 2: Compositional harm verification
    logger.info("[QC Layer 2] Compositional harm verification...")
    from src.quality.composition_verify import batch_verify
    samples = batch_verify(samples, use_api=use_api_for_verify)

    # Layer 3: Covertness scoring
    logger.info("[QC Layer 3] Covertness scoring...")
    from src.quality.covertness_score import batch_score_covertness
    samples = batch_score_covertness(samples)

    # Layer 4: Deduplication
    logger.info("[QC Layer 4] Deduplication...")
    from src.quality.dedup import run_full_dedup
    samples = run_full_dedup(samples)

    # Layer 5: Distribution balancing
    logger.info("[QC Layer 5] Distribution balancing...")
    from src.quality.balance import run_balance
    samples = run_balance(samples, target_total=target_total)

    return samples


def run(
    output_dir: str | Path | None = None,
    target_total: int = 10000,
    use_api: bool = True,
):
    """Main merge and QC entry point."""
    if output_dir is None:
        output_dir = DATA_DIR / "processed"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect from all paths
    all_samples = collect_from_paths()
    save_jsonl(all_samples, output_dir / "all_raw_samples.jsonl")

    # Run quality pipeline
    final_samples = run_quality_pipeline(
        all_samples,
        target_total=target_total,
        use_api_for_verify=use_api,
    )

    save_jsonl(final_samples, output_dir / "final_samples.jsonl")
    logger.info(f"Final dataset: {len(final_samples)} samples")
    return final_samples


if __name__ == "__main__":
    from src.common.utils import setup_logging
    setup_logging()
    run()
