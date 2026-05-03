#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/hdd/xuran/multi_image_safety"
RAW_DIR="$PROJECT_ROOT/data/raw"
PATH3_DIR="$RAW_DIR/path3"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
BACKUP_DIR="$RAW_DIR/path3_backup_$TIMESTAMP"

cd "$PROJECT_ROOT"

echo "Rebuilding Path 3 from scratch"
echo "Project root: $PROJECT_ROOT"
echo "HF_HOME: ${HF_HOME:-<unset>}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"

if [ -d "$PATH3_DIR" ]; then
  echo "Backing up existing path3 directory -> $BACKUP_DIR"
  mv "$PATH3_DIR" "$BACKUP_DIR"
fi

mkdir -p "$PATH3_DIR"

MIS_FORCE_RERUN_COMPLETED_STEPS=1 python run_path3.py --clean
