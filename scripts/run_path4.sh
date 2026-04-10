#!/bin/bash
set -e
export HF_HOME="/mnt2/xuran_hdd/cache"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="/mnt/hdd/xuran/multi_image_safety:$PYTHONPATH"

echo "Running Path 4: Scenario Construction"
echo "Using GPUs: ${CUDA_VISIBLE_DEVICES}"
python -m src.pipeline.run_path 4 "$@"
