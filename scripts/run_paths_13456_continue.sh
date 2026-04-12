#!/usr/bin/env bash
set -u

PROJ="/mnt/hdd/xuran/multi_image_safety"
LOG_DIR="$PROJ/logs"
CONDA_ACTIVATE="/mnt/hdd/xuran/anaconda3/bin/activate"
CONDA_ENV="mis_safety"

mkdir -p "$LOG_DIR"

source "$CONDA_ACTIVATE" "$CONDA_ENV"
cd "$PROJ"

export HF_HOME="${HF_HOME:-/mnt2/xuran_hdd/cache}"
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export MIS_GPU_CANDIDATES="${MIS_GPU_CANDIDATES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
unset CUDA_VISIBLE_DEVICES

timestamp() {
  date "+%F %T"
}

run_and_log() {
  local path_name="$1"
  local log_file="$2"
  shift 2

  echo "===== ${path_name} START $(timestamp) ====="
  "$@" 2>&1 | tee "$log_file"
  local rc=${PIPESTATUS[0]}
  echo "===== ${path_name} END rc=${rc} $(timestamp) ====="
  return 0
}

echo "============================================"
echo "Sequential Path Runner (Path 1, 2, 6)"
echo "Project: $PROJ"
echo "GPU candidates: ${MIS_GPU_CANDIDATES}"
echo "Started: $(timestamp)"
echo "============================================"

run_and_log "PATH 1" "$LOG_DIR/path1_seq.log" bash "$PROJ/scripts/run_path1.sh"
run_and_log "PATH 2" "$LOG_DIR/path2_seq.log" bash "$PROJ/scripts/run_path2.sh"
run_and_log "PATH 6" "$LOG_DIR/path6_seq.log" bash "$PROJ/scripts/run_path6.sh"

echo "============================================"
echo "Path 1, 2, and 6 have been attempted."
echo "Finished: $(timestamp)"
echo "Logs: $LOG_DIR"
echo "============================================"
