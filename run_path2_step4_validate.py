#!/usr/bin/env python3
"""
Step 4: Validate single-image safety for decomposed samples
Pattern A (compositional covert harm): both images must be individually safe
Runtime: ~10-20 min for ~3989 samples
GPUs: 0 for NSFW classifier
"""

import os
import sys
from pathlib import Path

# Set up environment
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HOME", "/mnt2/xuran_hdd/cache")

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.common.utils import setup_logging
from src.common.utils import clear_step_state, fail_step, finish_step, is_step_complete, jsonl_record_count, start_step
from src.path2_prompt_decompose.validate import run

if __name__ == "__main__":
    setup_logging()
    output_dir = Path("/mnt/hdd/xuran/multi_image_safety/data/raw/path2")
    output_file = output_dir / "validated_samples.jsonl"

    if is_step_complete(
        output_dir,
        "step4_validate_samples",
        expected_outputs=[output_file],
        validator=lambda: jsonl_record_count(output_file) >= 1,
    ):
        print("\n" + "="*80)
        print(f"Step 4 already complete: {output_file}")
        print("="*80 + "\n")
        raise SystemExit(0)

    clear_step_state(output_dir, "step4_validate_samples", stale_paths=[output_file])
    start_step(output_dir, "step4_validate_samples", cleanup_paths=[output_file])
    print("\n" + "="*80)
    print("Step 4: Validating single-image safety (NSFW check)")
    print("="*80 + "\n")

    results = run(nsfw_threshold=0.5)
    if jsonl_record_count(output_file) < 1:
        fail_step(
            output_dir,
            "step4_validate_samples",
            error="Validation finished without producing validated samples",
        )
        raise SystemExit(1)
    finish_step(
        output_dir,
        "step4_validate_samples",
        expected_outputs=[output_file],
        metadata={"records": len(results)},
    )

    print("\n" + "="*80)
    print(f"✓ STEP 4 COMPLETE: Validated {len(results)} samples passed safety check")
    print(f"Output: /mnt/hdd/xuran/multi_image_safety/data/raw/path2/validated_samples.jsonl")
    print("="*80 + "\n")
