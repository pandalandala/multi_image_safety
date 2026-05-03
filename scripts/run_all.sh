#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/mnt/hdd/xuran/multi_image_safety"
LOG_DIR="$PROJECT_ROOT/logs"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/run_all_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

normalize_path_name() {
  local raw="$1"
  case "$raw" in
    1|path1) echo "path1" ;;
    2|path2) echo "path2" ;;
    3|path3) echo "path3" ;;
    4|path4) echo "path4" ;;
    5|path5) echo "path5" ;;
    6|path6) echo "path6" ;;
    *)
      echo ""
      ;;
  esac
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_all.sh
  bash scripts/run_all.sh 1 2 3 4 5 6
  bash scripts/run_all.sh 1 3 5 6
  bash scripts/run_all.sh path1 path5 path6

Behavior:
  - If no path arguments are given, runs path1-path6 in order.
  - If a path fails, the script logs the failure and continues to the next path.
  - Final exit code is 0 if all selected paths succeeded, otherwise 1.
EOF
}

SELECTED_PATHS=()
if [[ $# -eq 0 ]]; then
  SELECTED_PATHS=(path1 path2 path3 path4 path5 path6)
else
  for arg in "$@"; do
    if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
      usage
      exit 0
    fi
    normalized="$(normalize_path_name "$arg")"
    if [[ -z "$normalized" ]]; then
      echo "Unknown path selector: $arg" >&2
      usage >&2
      exit 2
    fi
    SELECTED_PATHS+=("$normalized")
  done
fi

run_step() {
  local path_name="$1"
  local script_path="$PROJECT_ROOT/scripts/run_${path_name}.sh"
  local status=0

  echo "============================================================" | tee -a "$LOG_FILE"
  echo "[$(date '+%F %T')] Starting ${path_name}" | tee -a "$LOG_FILE"
  echo "Script: ${script_path}" | tee -a "$LOG_FILE"
  echo "============================================================" | tee -a "$LOG_FILE"

  set +e
  bash "$script_path" 2>&1 | tee -a "$LOG_FILE"
  status=${PIPESTATUS[0]}
  set -e

  if [[ $status -eq 0 ]]; then
    echo "[$(date '+%F %T')] Finished ${path_name} successfully" | tee -a "$LOG_FILE"
  else
    echo "[$(date '+%F %T')] ${path_name} failed with exit code ${status}; continuing to next path" | tee -a "$LOG_FILE"
  fi
  echo | tee -a "$LOG_FILE"
  return "$status"
}

echo "Running selected paths sequentially" | tee "$LOG_FILE"
echo "Selected paths: ${SELECTED_PATHS[*]}" | tee -a "$LOG_FILE"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}" | tee -a "$LOG_FILE"
echo "MIS_GPU_CANDIDATES: ${MIS_GPU_CANDIDATES:-<not set>}" | tee -a "$LOG_FILE"
echo "Master log: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo | tee -a "$LOG_FILE"

FAILED_PATHS=()
SUCCEEDED_PATHS=()

for path_name in "${SELECTED_PATHS[@]}"; do
  if run_step "$path_name"; then
    SUCCEEDED_PATHS+=("$path_name")
  else
    FAILED_PATHS+=("$path_name")
  fi
done

echo "============================================================" | tee -a "$LOG_FILE"
echo "[$(date '+%F %T')] Run summary" | tee -a "$LOG_FILE"
echo "Succeeded: ${SUCCEEDED_PATHS[*]:-<none>}" | tee -a "$LOG_FILE"
echo "Failed: ${FAILED_PATHS[*]:-<none>}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

if [[ ${#FAILED_PATHS[@]} -gt 0 ]]; then
  exit 1
fi

exit 0
