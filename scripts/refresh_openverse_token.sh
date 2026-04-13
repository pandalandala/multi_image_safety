#!/usr/bin/env bash
set -euo pipefail

OPENVERSE_API_BASE="${OPENVERSE_API_BASE:-https://api.openverse.org}"
OPENVERSE_TOKEN_URL="${OPENVERSE_TOKEN_URL:-${OPENVERSE_API_BASE}/v1/auth_tokens/token/}"
BASHRC_PATH="${BASHRC_PATH:-$HOME/.bashrc}"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "python is required to update ${BASHRC_PATH}" >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to refresh the Openverse token" >&2
  exit 1
fi

# Load the latest shell exports if they are not already present in this shell.
if [[ -f "$BASHRC_PATH" ]]; then
  # shellcheck disable=SC1090
  source "$BASHRC_PATH" >/dev/null 2>&1 || true
fi

: "${OPENVERSE_CLIENT_ID:?OPENVERSE_CLIENT_ID is not set}"
: "${OPENVERSE_CLIENT_SECRET:?OPENVERSE_CLIENT_SECRET is not set}"

response="$(
  curl -fsS \
    -X POST "$OPENVERSE_TOKEN_URL" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    --data-urlencode "grant_type=client_credentials" \
    --data-urlencode "client_id=${OPENVERSE_CLIENT_ID}" \
    --data-urlencode "client_secret=${OPENVERSE_CLIENT_SECRET}"
)"

token="$(
  printf '%s' "$response" | "$PYTHON_BIN" -c '
import json, sys
data = json.load(sys.stdin)
token = str(data.get("access_token", "")).strip()
if not token:
    raise SystemExit("Openverse token response did not contain access_token")
print(token)
'
)"

token_type="$(
  printf '%s' "$response" | "$PYTHON_BIN" -c '
import json, sys
data = json.load(sys.stdin)
print(str(data.get("token_type", "")).strip())
'
)"

expires_in="$(
  printf '%s' "$response" | "$PYTHON_BIN" -c '
import json, sys
data = json.load(sys.stdin)
print(str(data.get("expires_in", "")).strip())
'
)"

export OPENVERSE_API_TOKEN="$token"

"$PYTHON_BIN" - "$BASHRC_PATH" "$token" <<'PY'
from pathlib import Path
import sys

bashrc = Path(sys.argv[1]).expanduser()
token = sys.argv[2]
line = f'export OPENVERSE_API_TOKEN="{token}"'

text = bashrc.read_text(encoding="utf-8") if bashrc.exists() else ""
lines = text.splitlines()
for idx, existing in enumerate(lines):
    if existing.startswith("export OPENVERSE_API_TOKEN="):
        lines[idx] = line
        break
else:
    if lines and lines[-1] != "":
        lines.append("")
    lines.append("# Openverse API credentials")
    lines.append(line)

bashrc.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo "Openverse token refreshed."
echo "token_type=${token_type:-unknown}"
echo "expires_in=${expires_in:-unknown}"
echo "Updated ${BASHRC_PATH}."
echo "Run: source ${BASHRC_PATH}"
