#!/bin/bash
set -e
export HF_HOME="${HF_HOME:-/mnt2/xuran_hdd/cache}"
export MIS_GPU_CANDIDATES="${MIS_GPU_CANDIDATES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export MIS_PATH6_MAX_MODEL_LEN="${MIS_PATH6_MAX_MODEL_LEN:-4096}"
export MIS_PATH6_GPU_MEMORY_UTILIZATION="${MIS_PATH6_GPU_MEMORY_UTILIZATION:-0.68}"
export PYTHONPATH="/mnt/hdd/xuran/multi_image_safety:$PYTHONPATH"

echo "Running Path 6: TAG+KG Fusion"
echo "GPU candidates: ${MIS_GPU_CANDIDATES}"
python /mnt/hdd/xuran/multi_image_safety/run_path6.py "$@"
