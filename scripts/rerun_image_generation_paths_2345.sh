#!/usr/bin/env bash
# Re-run image-generation-related stages for Paths 2/3/4/5 with one command.
#
# Default behavior:
# - Forces regeneration of generated images across all paths
# - Preserves retrieved/crawled images when acquisition metadata says they were fetched
# - Re-runs Path 2 image acquisition + validation
# - Re-runs Path 3 expansion + Method A image acquisition
# - Re-runs Path 4 scenario pipeline
# - Re-runs Path 5 image acquisition/pairing pipeline
#
# Notes:
# - Path 5 still follows config/pipeline.yaml. If laion.enabled=true, it will
#   reuse existing crawled images and leave them untouched.
# - To force Path 5 onto T2I generation, set laion.enabled=false in
#   config/pipeline.yaml before running this script.

set -euo pipefail

PROJ=/mnt/hdd/xuran/multi_image_safety
LOG_DIR="$PROJ/logs"
STAMP=$(date '+%Y%m%d_%H%M%S')

mkdir -p "$LOG_DIR"

source /mnt/hdd/xuran/anaconda3/bin/activate mis_safety
cd "$PROJ"

export HF_HOME=/mnt2/xuran_hdd/cache
if [[ -f "$HF_HOME/token" ]]; then
  export HF_TOKEN="$(<"$HF_HOME/token")"
fi
export PYTHONPATH="$PROJ"
export MIS_FORCE_REGENERATE_IMAGES=1

log() {
    echo "[$(date '+%F %T')] $*"
}

run_step() {
    local name=$1
    local cmd=$2
    local logfile=$3

    log "START $name"
    /bin/bash -lc "$cmd" 2>&1 | tee "$logfile"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        log "FAIL  $name (exit code $rc) -- see $logfile"
        exit "$rc"
    fi
    log "DONE  $name"
}

log "============================================================"
log "Re-running Path 2/3/4/5 image-generation stages"
log "Project: $PROJ"
log "Log stamp: $STAMP"
log "MIS_FORCE_REGENERATE_IMAGES=$MIS_FORCE_REGENERATE_IMAGES"
log "============================================================"

run_step \
  "Path2 Step3 Acquire Images" \
  "python run_step3_acquire_images.py" \
  "$LOG_DIR/p2_step3_regen_${STAMP}.log"

run_step \
  "Path2 Step4 Validate" \
  "python run_step4_validate.py" \
  "$LOG_DIR/p2_step4_regen_${STAMP}.log"

run_step \
  "Path3 Expand" \
  "python run_path3_expand.py" \
  "$LOG_DIR/p3_expand_regen_${STAMP}.log"

run_step \
  "Path3 Acquire MethodA Images" \
  "python run_path3_acquire_images.py" \
  "$LOG_DIR/p3_acquire_images_regen_${STAMP}.log"

run_step \
  "Path4 Scenario + Images" \
  "python run_path4_scenario.py" \
  "$LOG_DIR/p4_regen_${STAMP}.log"

run_step \
  "Path5 Embedding + Images" \
  "python run_path5_embedding.py" \
  "$LOG_DIR/p5_regen_${STAMP}.log"

log "============================================================"
log "All requested image-generation stages completed."
log "Logs:"
log "  $LOG_DIR/p2_step3_regen_${STAMP}.log"
log "  $LOG_DIR/p2_step4_regen_${STAMP}.log"
log "  $LOG_DIR/p3_expand_regen_${STAMP}.log"
log "  $LOG_DIR/p3_acquire_images_regen_${STAMP}.log"
log "  $LOG_DIR/p4_regen_${STAMP}.log"
log "  $LOG_DIR/p5_regen_${STAMP}.log"
log "============================================================"
