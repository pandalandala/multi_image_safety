#!/usr/bin/env bash
set -e

export HF_HOME="${HF_HOME:-/mnt2/xuran_hdd/cache}"
export PYTHONPATH="/mnt/hdd/xuran/multi_image_safety:${PYTHONPATH:-}"

echo "Running Quality Control Pipeline"
python -m src.pipeline.merge "$@"

echo "Exporting to MIS format"
python -m src.pipeline.export "$@"
