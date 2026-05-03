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

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "python is required" >&2
  exit 1
fi

"$PYTHON_BIN" - "$QUERY" <<'PY'
import json
import os
import sys
import urllib.parse
import urllib.request

query = sys.argv[1]
token = os.environ.get("OPENVERSE_API_TOKEN", "").strip()
headers = {
    "User-Agent": "multi-image-safety/1.0",
    "Accept": "application/json",
}
if token:
    headers["Authorization"] = f"Bearer {token}"

url = "https://api.openverse.org/v1/images/?" + urllib.parse.urlencode({
    "q": query,
    "page_size": "1",
})

request = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(request, timeout=30) as response:
    data = json.loads(response.read().decode("utf-8"))

print(json.dumps({
    "query": query,
    "status": response.status,
    "result_count": len(data.get("results", [])),
    "has_token": bool(token),
}, ensure_ascii=False, indent=2))
PY
