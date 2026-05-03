#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/mnt/hdd/xuran/multi_image_safety"
LOG_DIR="$PROJECT_ROOT/logs"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/path3_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/_load_local_env.sh"

export HF_HOME="${HF_HOME:-/mnt2/xuran_hdd/cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export MIS_PATH3_MAX_MODEL_LEN="${MIS_PATH3_MAX_MODEL_LEN:-4096}"
export MIS_PATH3_GPU_MEMORY_UTILIZATION="${MIS_PATH3_GPU_MEMORY_UTILIZATION:-0.68}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ── 是否清除 step 完成标记，重新增量运行 ──────────────────────────────────────
# 修改这里：1 = 清除标记（增量重跑所有步骤），0 = 跳过已完成步骤
CLEAN=1

# 也可以在命令行传 --clean 覆盖此设置
for arg in "$@"; do
  [[ "$arg" == "--clean" ]] && CLEAN=1
done

echo "Running Path 3: expansion (Method A + Method B + Step4 T2I)"
echo "CLEAN=${CLEAN}  (1=re-run all steps incrementally, 0=skip completed steps)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "MIS_GPU_CANDIDATES: ${MIS_GPU_CANDIDATES:-<not set>}"
echo "PEXELS_API_KEY: ${PEXELS_API_KEY:+<set>}"
echo "PIXABAY_API_KEY: ${PIXABAY_API_KEY:+<set>}"
echo "Log file: ${LOG_FILE}"

clean_flag=()
[[ "$CLEAN" == "1" ]] && clean_flag=("--clean")

python "$PROJECT_ROOT/run_path3.py" "${clean_flag[@]}" 2>&1 | tee "$LOG_FILE"
