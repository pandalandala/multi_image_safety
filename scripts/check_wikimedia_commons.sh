#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/_load_local_env.sh"

if [[ -f "/mnt/hdd/xuran/anaconda3/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "/mnt/hdd/xuran/anaconda3/bin/activate" mis_safety
fi

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

QUERY="${1:-dog}"

python - "$QUERY" <<'PY'
import json
import os
import sys

from src.common.clip_utils import _retrieve_from_wikimedia_commons

query = sys.argv[1]
result = {
    "query": query,
    "user_agent": os.environ.get("MIS_WIKIMEDIA_USER_AGENT", os.environ.get("MIS_HTTP_USER_AGENT", "")),
    "min_interval": os.environ.get("MIS_WIKIMEDIA_MIN_INTERVAL", ""),
    "max_retries": os.environ.get("MIS_WIKIMEDIA_MAX_RETRIES", ""),
}

try:
    items = _retrieve_from_wikimedia_commons(query, num_results=3, min_width=256, min_height=256)
    result["ok"] = True
    result["count"] = len(items)
    if items:
        sample = items[0]
        result["sample"] = {
            "caption": (sample.get("caption") or "")[:160],
            "source_page": sample.get("source_page", ""),
            "license": sample.get("license", ""),
        }
except Exception as e:
    result["ok"] = False
    result["error"] = f"{type(e).__name__}: {e}"

print(json.dumps(result, ensure_ascii=False, indent=2))
PY
