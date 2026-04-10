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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

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
echo "Sequential Path Runner"
echo "Project: $PROJ"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Started: $(timestamp)"
echo "============================================"

run_and_log "PATH 1" "$LOG_DIR/path1_seq.log" bash "$PROJ/scripts/run_path1.sh"
run_and_log "PATH 2" "$LOG_DIR/path2_seq.log" bash "$PROJ/scripts/run_path2.sh"
run_and_log "PATH 3" "$LOG_DIR/path3_seq.log" bash "$PROJ/scripts/run_path3.sh"

if [[ "${MIS_RUN_PATH3_ACQUIRE_IMAGES:-0}" == "1" ]]; then
  run_and_log "PATH 3 IMAGES" "$LOG_DIR/path3_images_seq.log" python "$PROJ/run_path3_acquire_images.py"
fi

run_and_log "PATH 4" "$LOG_DIR/path4_seq.log" bash "$PROJ/scripts/run_path4.sh"
run_and_log "PATH 5" "$LOG_DIR/path5_seq.log" bash "$PROJ/scripts/run_path5.sh"
run_and_log "PATH 6" "$LOG_DIR/path6_seq.log" bash "$PROJ/scripts/run_path6.sh"

echo "============================================"
echo "All requested paths have been attempted."
echo "Finished: $(timestamp)"
echo "Logs: $LOG_DIR"
echo "============================================"
