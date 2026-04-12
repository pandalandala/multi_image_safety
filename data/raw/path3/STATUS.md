# Path 3 Status

Canonical flow:
1. `run_path3_expand.py` builds the active mixed Path 3 output.
2. Method A is text-only at this stage.
3. Method B already has real image paths.
4. `run_path3_acquire_images.py` is the optional next step if you want images for Method A too.

Current counts:
- Method A text-only: 2688
- Method B with images: 887
- Method A with images: 0
- Mixed merged output: 3575

Key files:
- `method_a_decomposed.jsonl`: Method A text-only samples
- `method_b_cross_paired.jsonl`: Method B samples with `image1_path` / `image2_path`
- `method_a_with_images.jsonl`: created only after Method A image acquisition
- `cross_paired_samples.jsonl`: current merged Path 3 output

Active logs:
- `logs/p3_expand.log`: canonical Path 3 expansion log
- `logs/p3_acquire_images.log`: Method A image acquisition log

Legacy archive:
- `legacy/path3/`: archived historical scripts and logs from older Path 3 iterations
