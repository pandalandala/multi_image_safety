"""Shared status helpers for the active Path 3 pipeline."""

from __future__ import annotations

from pathlib import Path


def write_path3_status(
    output_dir: Path,
    *,
    method_a_text_only: int,
    method_b_with_images: int,
    merged_total: int,
    method_a_with_images: int = 0,
) -> Path:
    """Write the current active Path 3 status note."""
    status_path = output_dir / "STATUS.md"
    lines = [
        "# Path 3 Status",
        "",
        "Canonical flow:",
        "1. `run_path3_expand.py` builds the active mixed Path 3 output.",
        "2. Method A is text-only at this stage.",
        "3. Method B already has real image paths.",
        "4. `run_path3_acquire_images.py` is the optional next step if you want images for Method A too.",
        "",
        "Current counts:",
        f"- Method A text-only: {method_a_text_only}",
        f"- Method B with images: {method_b_with_images}",
        f"- Method A with images: {method_a_with_images}",
        f"- Mixed merged output: {merged_total}",
        "",
        "Key files:",
        "- `method_a_decomposed.jsonl`: Method A text-only samples",
        "- `method_b_cross_paired.jsonl`: Method B samples with `image1_path` / `image2_path`",
        "- `method_a_with_images.jsonl`: created only after Method A image acquisition",
        "- `cross_paired_samples.jsonl`: current merged Path 3 output",
        "",
        "Active logs:",
        "- `logs/p3_expand.log`: canonical Path 3 expansion log",
        "- `logs/p3_acquire_images.log`: Method A image acquisition log",
        "",
        "Legacy archive:",
        "- `legacy/path3/`: archived historical scripts and logs from older Path 3 iterations",
    ]
    status_path.write_text("\n".join(lines) + "\n")
    return status_path
