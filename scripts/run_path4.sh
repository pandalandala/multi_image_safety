#!/bin/bash
set -e
export HF_HOME="/mnt2/xuran_hdd/cache"
export MIS_GPU_CANDIDATES="${MIS_GPU_CANDIDATES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
unset CUDA_VISIBLE_DEVICES
export PYTHONPATH="/mnt/hdd/xuran/multi_image_safety:$PYTHONPATH"

echo "Running Path 4: Scenario Construction"
echo "GPU candidates: ${MIS_GPU_CANDIDATES}"
python -m src.pipeline.run_path 4 "$@"
