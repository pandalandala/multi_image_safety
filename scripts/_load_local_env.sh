#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/mnt/hdd/xuran/multi_image_safety"
ENV_FILE="${MIS_ENV_FILE:-$PROJECT_ROOT/.env.local}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi
