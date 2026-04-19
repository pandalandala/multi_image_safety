#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/mnt2/xuran_hdd/cache/datasets}"
TARGET_DIR="$DATASET_ROOT/imagenet"
METHOD="${IMAGENET_METHOD:-hf_partial}"
IMAGENET_LIMIT_PER_CLASS="${IMAGENET_LIMIT_PER_CLASS:-8}"
EXTRACT_TRAIN_TAR="${EXTRACT_IMAGENET_TRAIN:-0}"

mkdir -p "$TARGET_DIR"

ensure_python_module() {
  local module_name="$1"
  local package_name="$2"
  python - <<PY >/dev/null 2>&1 || python -m pip install "$package_name"
import importlib
importlib.import_module("$module_name")
PY
}

extract_train_tar_if_needed() {
  local train_tar="$TARGET_DIR/ILSVRC2012_img_train.tar"
  local train_root="$TARGET_DIR/ILSVRC/Data/CLS-LOC/train"
  if [[ "$EXTRACT_TRAIN_TAR" != "1" ]]; then
    return 0
  fi
  if [[ ! -f "$train_tar" ]]; then
    echo "ImageNet train tar not found: $train_tar" >&2
    return 0
  fi

  mkdir -p "$train_root"
  tar -xf "$train_tar" -C "$train_root"
  pushd "$train_root" >/dev/null
  shopt -s nullglob
  for f in *.tar; do
    d="${f%.tar}"
    mkdir -p "$d"
    tar -xf "$f" -C "$d"
    rm -f "$f"
  done
  popd >/dev/null
}

case "$METHOD" in
  hf_partial)
    ensure_python_module "datasets" "datasets"
    python - <<PY
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset, load_dataset_builder

target_dir = Path("$TARGET_DIR")
limit_per_class = int("$IMAGENET_LIMIT_PER_CLASS")
train_root = target_dir / "ILSVRC" / "Data" / "CLS-LOC" / "train"
mapping_path = target_dir / "LOC_synset_mapping.txt"
train_root.mkdir(parents=True, exist_ok=True)

builder = load_dataset_builder("ILSVRC/imagenet-1k")
label_names = list(builder.info.features["label"].names)

with mapping_path.open("w", encoding="utf-8") as f:
    for idx, label_name in enumerate(label_names):
        class_id = f"cls_{idx:04d}"
        f.write(f"{class_id} {label_name}\\n")
        (train_root / class_id).mkdir(parents=True, exist_ok=True)

counts = defaultdict(int)
for idx, _ in enumerate(label_names):
    class_id = f"cls_{idx:04d}"
    counts[idx] = len(list((train_root / class_id).glob("*.jpg")))

stream = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
remaining_classes = {idx for idx in range(len(label_names)) if counts[idx] < limit_per_class}

for example in stream:
    if not remaining_classes:
        break
    label = int(example["label"])
    if label not in remaining_classes:
        continue
    class_id = f"cls_{label:04d}"
    out_dir = train_root / class_id
    out_path = out_dir / f"{class_id}_{counts[label]:04d}.jpg"
    if out_path.exists():
        counts[label] += 1
        if counts[label] >= limit_per_class:
            remaining_classes.discard(label)
        continue
    image = example["image"]
    image.save(out_path, format="JPEG", quality=95)
    counts[label] += 1
    if counts[label] >= limit_per_class:
        remaining_classes.discard(label)

covered = sum(1 for idx in range(len(label_names)) if counts[idx] >= 1)
complete = sum(1 for idx in range(len(label_names)) if counts[idx] >= limit_per_class)
print(f"ImageNet partial cache ready: covered_classes={covered}/{len(label_names)} complete_classes={complete}/{len(label_names)} limit_per_class={limit_per_class}")
PY
    ;;
  hf)
    ensure_python_module "datasets" "datasets"
    python - <<PY
from datasets import load_dataset
cache_dir = "$TARGET_DIR/imagenet_hf"
load_dataset("ILSVRC/imagenet-1k", split="train", cache_dir=cache_dir)
print(f"ImageNet HF cache ready: {cache_dir}")
PY
    ;;
  kaggle)
    ensure_python_module "kaggle" "kaggle"
    kaggle competitions download \
      -c imagenet-object-localization-challenge \
      -p "$TARGET_DIR"
    extract_train_tar_if_needed
    ;;
  manual)
    echo "Manual ImageNet mode selected."
    echo "Place ILSVRC2012_img_train.tar under: $TARGET_DIR"
    echo "Then rerun with EXTRACT_IMAGENET_TRAIN=1 IMAGENET_METHOD=manual"
    extract_train_tar_if_needed
    ;;
  *)
    echo "Unsupported IMAGENET_METHOD: $METHOD" >&2
    echo "Use one of: hf_partial, hf, kaggle, manual" >&2
    exit 1
    ;;
esac

echo "ImageNet script complete: $TARGET_DIR"
