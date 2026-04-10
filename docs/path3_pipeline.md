# Path 3 Pipeline

## Canonical Flow

1. `run_path3_expand.py`
   Builds the main Path 3 dataset.
   - Method A: text-only decompositions from captioned images
   - Method B: image-backed cross-paired samples
2. `run_path3_acquire_images.py`
   Optional follow-up step.
   - Adds images to Method A
   - Refreshes `cross_paired_samples.jsonl` so the merged output becomes image-backed for both methods

## What Has Images?

- `method_a_decomposed.jsonl`
  Method A only.
  Contains `image1_description` / `image2_description`.
  No `image1_path` / `image2_path` yet.

- `method_b_cross_paired.jsonl`
  Method B only.
  Already contains real `image1_path` / `image2_path`.

- `cross_paired_samples.jsonl`
  Mixed file.
  Before `run_path3_acquire_images.py`, this is:
  - Method A text-only
  - Method B image-backed

## Logs

- `logs/p3_expand.log`
  Canonical Path 3 expansion log.

- `logs/p3_acquire_images.log`
  Canonical Method A image acquisition log.

- `legacy/path3/logs/`
  Historical Path 3 logs from earlier CLIP / refactor / method-B-only iterations.
  Do not use those to infer the current Path 3 state.

## Recommended Commands

Full Path 3 expand:

```bash
source /mnt/hdd/xuran/anaconda3/bin/activate mis_safety
cd /mnt/hdd/xuran/multi_image_safety
python run_path3_expand.py 2>&1 | tee logs/p3_expand.log
```

Add images to Method A only:

```bash
source /mnt/hdd/xuran/anaconda3/bin/activate mis_safety
cd /mnt/hdd/xuran/multi_image_safety
python run_path3_acquire_images.py 2>&1 | tee logs/p3_acquire_images.log
```
