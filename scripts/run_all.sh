#!/bin/bash
# Run the complete multi-image safety dataset construction pipeline
set -e

export HF_HOME="/mnt2/xuran_hdd/cache"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="/mnt/hdd/xuran/multi_image_safety:$PYTHONPATH"

echo "============================================"
echo "Multi-Image Safety Dataset Construction"
echo "============================================"
echo "Using GPUs: ${CUDA_VISIBLE_DEVICES}"

# Parse arguments
USE_API=""
if [ "$1" == "--use-api" ]; then
    USE_API="--use-api"
fi

echo "[1/6] Running Path 2: Prompt Decomposition..."
python -m src.pipeline.run_path 2 $USE_API

echo "[2/6] Running Path 3: Dataset Expansion..."
python -m src.pipeline.run_path 3

echo "[3/6] Running Path 4: Scenario Construction..."
python -m src.pipeline.run_path 4 $USE_API

echo "[4/6] Running Path 5: Embedding Pair Matching..."
python -m src.pipeline.run_path 5 $USE_API

echo "[5/6] Merging and Quality Control..."
python -m src.pipeline.merge

echo "[6/6] Exporting to MIS format..."
python -m src.pipeline.export

echo "============================================"
echo "Pipeline complete!"
echo "Output: /mnt/hdd/xuran/multi_image_safety/data/final/"
echo "============================================"
