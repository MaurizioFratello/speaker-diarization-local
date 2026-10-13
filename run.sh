#!/usr/bin/env bash
# Start the local speaker-transcription web app.
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || { echo "Run ./setup.sh first."; exit 1; }
# Keep the model warm while the browser tab is open; after it closes, app.py exits
# once there are no active jobs. Set SPEAKER_IDLE_EXIT=0 to disable auto-exit.
IDLE_EXIT="${SPEAKER_IDLE_EXIT:-120}"
exec .venv/bin/python app.py --preload --open-browser --idle-exit "$IDLE_EXIT" "$@"
