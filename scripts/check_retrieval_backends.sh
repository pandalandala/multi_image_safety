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
MODE="${2:-configured}"

python - "$QUERY" "$MODE" <<'PY'
import json
import sys

from src.common.clip_utils import _get_retrieval_config, _retrieve_from_backend

query = sys.argv[1]
mode = sys.argv[2]

cfg = _get_retrieval_config()
configured = [str(x).strip() for x in cfg.get("backends", []) if str(x).strip()]
all_backends = ["wikimedia_commons", "openverse", "pexels", "pixabay"]

if mode == "all":
    backends = all_backends
else:
    backends = []
    seen = set()
    for backend in configured + ["openverse"]:
        if backend and backend not in seen:
            backends.append(backend)
            seen.add(backend)

results = []
for backend in backends:
    try:
        items = _retrieve_from_backend(
            backend,
            query,
            num_results=3,
            aesthetic_score_min=5.0,
            min_width=256,
            min_height=256,
        )
        results.append({
            "backend": backend,
            "ok": True,
            "count": len(items),
            "sample_provider": items[0].get("provider") if items else "",
            "sample_caption": (items[0].get("caption") or "")[:120] if items else "",
        })
    except Exception as e:
        results.append({
            "backend": backend,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
        })

print(json.dumps({
    "query": query,
    "mode": mode,
    "configured_backends": configured,
    "results": results,
}, ensure_ascii=False, indent=2))
PY
