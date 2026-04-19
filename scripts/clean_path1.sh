#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/mnt/hdd/xuran/multi_image_safety"

cd "$PROJECT_ROOT"
python run_path1.py --clean "$@"
