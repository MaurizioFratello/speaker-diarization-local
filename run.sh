#!/usr/bin/env bash
# Start the local speaker-transcription web app.
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || { echo "Run ./setup.sh first."; exit 1; }
exec .venv/bin/python app.py --preload --open-browser "$@"
