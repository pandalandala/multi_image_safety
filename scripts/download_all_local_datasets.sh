#!/usr/bin/env bash
set -euo pipefail

PROJ="/mnt/hdd/xuran/multi_image_safety"
DATASET_ROOT="${DATASET_ROOT:-/mnt2/xuran_hdd/cache/datasets}"
INCLUDE_IMAGENET="${INCLUDE_IMAGENET:-0}"

echo "Dataset root: $DATASET_ROOT"

DATASET_ROOT="$DATASET_ROOT" bash "$PROJ/scripts/download_dataset_mscoco.sh"
DATASET_ROOT="$DATASET_ROOT" bash "$PROJ/scripts/download_dataset_open_images.sh"

if [[ "$INCLUDE_IMAGENET" == "1" ]]; then
  DATASET_ROOT="$DATASET_ROOT" bash "$PROJ/scripts/download_dataset_imagenet.sh"
else
  echo "Skipping ImageNet. Set INCLUDE_IMAGENET=1 to include the partial all-class ImageNet downloader."
fi

echo "All requested dataset download scripts finished."
