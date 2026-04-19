#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/mnt2/xuran_hdd/cache/datasets}"
TARGET_DIR="$DATASET_ROOT/open_images"
IMAGES_BASE_DIR="$TARGET_DIR/train"
MODE="${OPEN_IMAGES_MODE:-all_classes}"
OPEN_IMAGES_LIMIT="${OPEN_IMAGES_LIMIT:-20}"
OPEN_IMAGES_BATCH_SIZE="${OPEN_IMAGES_BATCH_SIZE:-50}"

mkdir -p "$TARGET_DIR"
mkdir -p "$IMAGES_BASE_DIR"
cd "$TARGET_DIR"

if ! command -v wget >/dev/null 2>&1; then
  echo "wget is required but not found in PATH" >&2
  exit 1
fi

ensure_python_module() {
  local module_name="$1"
  local package_name="$2"
  python - <<PY >/dev/null 2>&1 || python -m pip install "$package_name"
import importlib
importlib.import_module("$module_name")
PY
}

if [[ ! -f "$TARGET_DIR/oidv7-class-descriptions.csv" ]]; then
  wget -c -O oidv7-class-descriptions.csv \
    "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv"
else
  echo "Open Images class descriptions already exist"
fi

if [[ ! -f "$TARGET_DIR/class-descriptions-boxable.csv" ]]; then
  wget -c -O class-descriptions-boxable.csv \
    "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"
else
  echo "Open Images boxable class descriptions already exist"
fi

if [[ ! -f "$TARGET_DIR/oidv7-train-annotations-human-imagelabels.csv" ]]; then
  wget -c -O oidv7-train-annotations-human-imagelabels.csv \
    "https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-human-imagelabels.csv"
else
  echo "Open Images image-level labels already exist"
fi

resolve_boxable_labels() {
  python - "$@" <<PY
import csv
import sys
from pathlib import Path

boxable_csv = Path("$TARGET_DIR/class-descriptions-boxable.csv")
requested = sys.argv[1:]

boxable = {}
with boxable_csv.open("r", newline="", encoding="utf-8") as f:
    for row in csv.reader(f):
        if len(row) >= 2 and row[1].strip():
            boxable[row[1].strip().lower()] = row[1].strip()

resolved = []
skipped = []
for label in requested:
    canonical = boxable.get(label.strip().lower())
    if canonical:
        resolved.append(canonical)
    else:
        skipped.append(label)

for label in skipped:
    print(f"SKIP::{label}")
for label in resolved:
    print(f"KEEP::{label}")
PY
}

if [[ "$MODE" == "full" ]]; then
  ensure_python_module "awscli" "awscli"
  aws s3 --no-sign-request sync \
    s3://open-images-dataset/train \
    "$TARGET_DIR/train"
elif [[ "$MODE" == "all_classes" ]]; then
  ensure_python_module "openimages" "openimages"
  mapfile -t labels < <(python - <<PY
import csv
from pathlib import Path
labels_csv = Path("$TARGET_DIR/class-descriptions-boxable.csv")
with labels_csv.open("r", newline="") as f:
    for row in csv.reader(f):
        if len(row) >= 2 and row[1].strip():
            print(row[1].strip())
PY
)

  total=${#labels[@]}
  if [[ "$total" -eq 0 ]]; then
    echo "No Open Images labels found in $TARGET_DIR/oidv7-class-descriptions.csv" >&2
    exit 1
  fi

  for ((i=0; i<total; i+=OPEN_IMAGES_BATCH_SIZE)); do
    batch=("${labels[@]:i:OPEN_IMAGES_BATCH_SIZE}")
    echo "Downloading Open Images labels $((i+1))-$((i+${#batch[@]})) / $total"
    python -m openimages.download \
      --base_dir "$IMAGES_BASE_DIR" \
      --format pascal \
      --csv_dir "$TARGET_DIR" \
      --limit "$OPEN_IMAGES_LIMIT" \
      --labels "${batch[@]}"
  done
else
  ensure_python_module "openimages" "openimages"
  if [[ -n "${OPEN_IMAGES_LABELS:-}" ]]; then
    IFS=',' read -r -a labels <<< "$OPEN_IMAGES_LABELS"
  else
    labels=(
      "Knife"
      "Baseball bat"
      "Camera"
      "Kitchen"
      "Bridge"
      "Bottle"
      "Car"
      "Person"
      "Dog"
      "Cat"
      "Chair"
      "Table"
    )
  fi

  mapfile -t resolved_rows < <(resolve_boxable_labels "${labels[@]}")
  labels=()
  for row in "${resolved_rows[@]}"; do
    if [[ "$row" == SKIP::* ]]; then
      echo "Skipping non-boxable Open Images label: ${row#SKIP::}"
    elif [[ "$row" == KEEP::* ]]; then
      labels+=("${row#KEEP::}")
    fi
  done

  if [[ "${#labels[@]}" -eq 0 ]]; then
    echo "No requested labels are supported by the openimages downloader (boxable classes only)." >&2
    exit 1
  fi

  python -m openimages.download \
    --base_dir "$IMAGES_BASE_DIR" \
    --format pascal \
    --csv_dir "$TARGET_DIR" \
    --limit "$OPEN_IMAGES_LIMIT" \
    --labels "${labels[@]}"
fi

echo "Open Images download complete: $TARGET_DIR"
