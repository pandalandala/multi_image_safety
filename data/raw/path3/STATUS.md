# Path 3 Status

Canonical flow:
1. `run_path3.py` builds the active mixed Path 3 output.
2. Method A LLM decomposition runs first (text-only).
3. Method B LLM cross-image pairing runs (already has real image paths).
4. Method A image acquisition runs automatically (T2I per GPU).
5. Final merged output written to `cross_paired_samples.jsonl`.

Current counts:
- Method A text-only: 6983
- Method B with images: 931
- Method A with images: 6825
- Mixed merged output: 7756

Key files:
- `method_a_decomposed.jsonl`: Method A text-only samples
- `method_b_cross_paired.jsonl`: Method B samples with `image1_path` / `image2_path`
- `method_a_with_images.jsonl`: Method A samples after image acquisition
- `cross_paired_samples.jsonl`: current merged Path 3 output

Active logs:
- `logs/p3_expand.log`: canonical Path 3 expansion log
- `logs/p3_acquire_images.log`: Method A image acquisition log
