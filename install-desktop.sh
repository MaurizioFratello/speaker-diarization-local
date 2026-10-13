#!/usr/bin/env bash
# Install a one-click Linux application launcher for the local browser app.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
DESKTOP_FILE="$APP_DIR/speaker-diarization.desktop"
mkdir -p "$APP_DIR"
cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Speaker Diarization
Comment=Open the local speaker transcription app
Exec="$ROOT/start-app.sh"
Path=$ROOT
Terminal=false
Categories=AudioVideo;Audio;
EOF
chmod +x "$DESKTOP_FILE"
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$APP_DIR" >/dev/null 2>&1 || true
fi
printf 'Installed launcher: %s\n' "$DESKTOP_FILE"
printf 'Start it from the application menu; it opens the browser and preloads Parakeet.\n'
