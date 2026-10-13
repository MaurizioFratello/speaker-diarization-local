#!/usr/bin/env bash
# GUI-safe, repeatable launcher: reuse an existing server or start it with a log file.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PORT="${SPEAKER_APP_PORT:-8765}"
URL="http://127.0.0.1:${PORT}"

if command -v curl >/dev/null 2>&1 && curl -fsS --max-time 2 "$URL/api/health" >/dev/null 2>&1; then
    if command -v xdg-open >/dev/null 2>&1; then
        (xdg-open "$URL" >/dev/null 2>&1 || true) &
    fi
    exit 0
fi

mkdir -p data/logs
exec "$ROOT/run.sh" >> "$ROOT/data/logs/speaker-diarization-app.log" 2>&1
