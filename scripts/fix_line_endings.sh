#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/mnt/hdd/xuran/multi_image_safety}"

if [[ ! -d "$ROOT" ]]; then
  echo "Directory not found: $ROOT" >&2
  exit 1
fi

echo "Normalizing CRLF to LF under: $ROOT"

find "$ROOT" \
  -type f \
  \( -name "*.sh" -o -name "*.py" -o -name "*.yaml" -o -name "*.yml" -o -name "*.txt" \) \
  -print0 | while IFS= read -r -d '' file; do
    python - "$file" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
data = path.read_bytes()
normalized = data.replace(b"\r\n", b"\n")
if normalized != data:
    path.write_bytes(normalized)
    print(f"fixed {path}")
PY
done

echo "Done."
