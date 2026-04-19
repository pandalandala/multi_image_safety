#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/mnt/hdd/xuran/multi_image_safety"
LOG_DIR="$PROJECT_ROOT/logs"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/path4_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/_load_local_env.sh"

export HF_HOME="${HF_HOME:-/mnt2/xuran_hdd/cache}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "Running Path 4: Scenario Construction"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "MIS_GPU_CANDIDATES: ${MIS_GPU_CANDIDATES:-<not set>}"
echo "PEXELS_API_KEY: ${PEXELS_API_KEY:+<set>}"
echo "PIXABAY_API_KEY: ${PIXABAY_API_KEY:+<set>}"
echo "Log file: ${LOG_FILE}"

python -m src.pipeline.run_path 4 "$@" 2>&1 | tee "$LOG_FILE"
