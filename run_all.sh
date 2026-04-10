#!/usr/bin/env bash
# =============================================================================
# Full pipeline: Path 2 → Path 4 → Path 3 → Path 5 → Path 1 → Path 6
# Estimated total runtime: 15-20 hours (4× RTX A6000 on 4 visible GPUs)
# =============================================================================

set -euo pipefail

PROJ=/mnt/hdd/xuran/multi_image_safety
PYTHON=/mnt/hdd/xuran/anaconda3/envs/mis_safety/bin/python
LOG_DIR=$PROJ/logs
mkdir -p "$LOG_DIR"

cd "$PROJ"
export HF_HOME=/mnt2/xuran_hdd/cache
export PYTHONPATH="$PROJ"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
check_exit() {
    local code=$1 step=$2
    if [ "$code" -ne 0 ]; then
        log "FAIL $step (exit code $code) — check $LOG_DIR/${step}.log"
        # Continue instead of exit for non-critical paths
        return 1
    fi
    log "OK $step completed"
}

log "======================================================================"
log "  Multi-Image Safety Dataset Pipeline (6 Paths)"
log "  Started: $(date)"
log "  GPUs: $(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tr '\n' '|')"
log "  Disk free: $(df -h /mnt/hdd | tail -1 | awk '{print $4}')"
log "======================================================================"

# ── Path 2 Step 2: Decompose (vLLM on 4 visible GPUs) ───────────────────────
log ""
log ">>> PATH 2 STEP 2: Decompose prompts (vLLM)"
"$PYTHON" run_step2_decompose.py \
    --input-file data/raw/path2/collected_prompts.jsonl \
    --output-dir data/raw/path2 \
    --max-prompts 4500 \
    2>&1 | tee "$LOG_DIR/p2_step2.log"
check_exit ${PIPESTATUS[0]} "P2-Step2-Decompose"

# ── Path 2 Step 3: Image acquisition (4-GPU parallel T2I) ────────────────────
log ""
log ">>> PATH 2 STEP 3: Acquire images (4-GPU parallel)"
"$PYTHON" run_step3_acquire_images.py 2>&1 | tee "$LOG_DIR/p2_step3.log"
check_exit ${PIPESTATUS[0]} "P2-Step3-AcquireImages"

# ── Path 2 Step 4: Validate single-image safety ──────────────────────────────
log ""
log ">>> PATH 2 STEP 4: Validate safety"
"$PYTHON" run_step4_validate.py 2>&1 | tee "$LOG_DIR/p2_step4.log"
check_exit ${PIPESTATUS[0]} "P2-Step4-Validate"

# ── Path 4: Scenario generation ──────────────────────────────────────────────
log ""
log ">>> PATH 4: Scenario generation (LLM scenes -> intent inject -> images)"
"$PYTHON" run_path4_scenario.py 2>&1 | tee "$LOG_DIR/p4.log"
check_exit ${PIPESTATUS[0]} "P4-Scenario"

# ── Path 3: Dataset expansion ────────────────────────────────────────────────
log ""
log ">>> PATH 3: Dataset expansion (Method A + Method B)"
"$PYTHON" run_path3_expand.py 2>&1 | tee "$LOG_DIR/p3_expand.log"
check_exit ${PIPESTATUS[0]} "P3-Expand"

# ── Path 3: Acquire images for Method A ──────────────────────────────────────
log ""
log ">>> PATH 3: Acquire images for Method A text-only samples"
"$PYTHON" run_path3_acquire_images.py 2>&1 | tee "$LOG_DIR/p3_images.log"
check_exit ${PIPESTATUS[0]} "P3-AcquireImages"

# ── Path 5: Embedding pair mining ────────────────────────────────────────────
log ""
log ">>> PATH 5: Embedding pair mining (retrieval + LLM cross-pair)"
"$PYTHON" run_path5_embedding.py 2>&1 | tee "$LOG_DIR/p5.log"
check_exit ${PIPESTATUS[0]} "P5-Embedding"

# ── Path 1: KG Concept Pairs (new) ──────────────────────────────────────────
log ""
log ">>> PATH 1: KG concept pair mining (Numberbatch + LLM + CLIP)"
"$PYTHON" run_path1_kg_concept.py 2>&1 | tee "$LOG_DIR/p1.log"
check_exit ${PIPESTATUS[0]} "P1-KGConcept"

# ── Path 6: TAG+KG Fusion (new) ─────────────────────────────────────────────
log ""
log ">>> PATH 6: TAG+KG fusion (chains + fusion + images)"
"$PYTHON" run_path6_tag_kg.py 2>&1 | tee "$LOG_DIR/p6.log"
check_exit ${PIPESTATUS[0]} "P6-TAGFusion"

# ── Summary ──────────────────────────────────────────────────────────────────
log ""
log "======================================================================"
log "  ALL PATHS COMPLETED: $(date)"
log ""
log "  Outputs:"
log "  Path 1: $PROJ/data/raw/path1/validated_samples.jsonl"
log "  Path 2: $PROJ/data/raw/path2/validated_samples.jsonl"
log "  Path 3: $PROJ/data/raw/path3/cross_paired_samples.jsonl"
log "  Path 4: $PROJ/data/raw/path4/samples_with_images.jsonl"
log "  Path 5: $PROJ/data/raw/path5/samples_with_prompts.jsonl"
log "  Path 6: $PROJ/data/raw/path6/validated_samples.jsonl"
log ""
log "  Sample counts:"
for f in \
    data/raw/path1/validated_samples.jsonl \
    data/raw/path2/validated_samples.jsonl \
    data/raw/path3/cross_paired_samples.jsonl \
    data/raw/path4/samples_with_images.jsonl \
    data/raw/path5/samples_with_prompts.jsonl \
    data/raw/path6/validated_samples.jsonl; do
    [ -f "$PROJ/$f" ] && log "    $(wc -l < "$PROJ/$f") samples in $f" || log "    (missing) $f"
done
log ""
log "  Disk free: $(df -h /mnt/hdd | tail -1 | awk '{print $4}')"
log "======================================================================"
