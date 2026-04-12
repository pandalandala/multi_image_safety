#!/usr/bin/env bash
set -euo pipefail

PROJ="/mnt/hdd/xuran/multi_image_safety"
RUNNER="$PROJ/scripts/run_paths_13456_continue.sh"
LOG_DIR="$PROJ/logs"
DELAY_SECONDS="${1:-10800}"

mkdir -p "$LOG_DIR"

if [[ ! -f "$RUNNER" ]]; then
  echo "Runner script not found: $RUNNER" >&2
  exit 1
fi

LOG_FILE="$LOG_DIR/run_paths_126_delayed_$(date +%Y%m%d_%H%M%S).log"

nohup bash -lc "sleep $DELAY_SECONDS; bash '$RUNNER'" > "$LOG_FILE" 2>&1 &
PID=$!

echo "Scheduled run_paths_126"
echo "  Delay seconds: $DELAY_SECONDS"
echo "  PID: $PID"
echo "  Log: $LOG_FILE"
