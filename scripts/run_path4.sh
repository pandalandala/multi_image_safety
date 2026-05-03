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

# ── 是否清除 step 完成标记，重新增量运行 ──────────────────────────────────────
# 修改这里：1 = 清除标记（增量重跑所有步骤），0 = 跳过已完成步骤
# Path 4 需全量重跑：scene categories 40→80，max_per_scene 1→2
CLEAN=1

# 也可以在命令行传 --clean 覆盖此设置
for arg in "$@"; do
  [[ "$arg" == "--clean" ]] && CLEAN=1
done

echo "Running Path 4: Scenario Construction"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "MIS_GPU_CANDIDATES: ${MIS_GPU_CANDIDATES:-<not set>}"
echo "PEXELS_API_KEY: ${PEXELS_API_KEY:+<set>}"
echo "PIXABAY_API_KEY: ${PIXABAY_API_KEY:+<set>}"
echo "Log file: ${LOG_FILE}"

clean_flag=()
[[ "$CLEAN" == "1" ]] && clean_flag=("--clean")

python "$PROJECT_ROOT/run_path4.py" "${clean_flag[@]}" 2>&1 | tee "$LOG_FILE"
