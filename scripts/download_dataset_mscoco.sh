#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/mnt2/xuran_hdd/cache/datasets}"
TARGET_DIR="$DATASET_ROOT/coco"
TRAIN_ZIP_URL="http://images.cocodataset.org/zips/train2017.zip"
ANNO_ZIP_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

if ! command -v wget >/dev/null 2>&1; then
  echo "wget is required but not found in PATH" >&2
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "unzip is required but not found in PATH" >&2
  exit 1
fi

if [[ ! -d "$TARGET_DIR/train2017" ]]; then
  wget -c -O train2017.zip "$TRAIN_ZIP_URL"
  unzip -q train2017.zip
  rm -f train2017.zip
else
  echo "MSCOCO train2017 already exists: $TARGET_DIR/train2017"
fi

if [[ ! -f "$TARGET_DIR/annotations/captions_train2017.json" ]]; then
  wget -c -O annotations_trainval2017.zip "$ANNO_ZIP_URL"
  unzip -q annotations_trainval2017.zip
  rm -f annotations_trainval2017.zip
else
  echo "MSCOCO captions already exist: $TARGET_DIR/annotations/captions_train2017.json"
fi

echo "MSCOCO download complete: $TARGET_DIR"
