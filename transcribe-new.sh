#!/usr/bin/env bash
# One click: find recordings in the folder that have no markdown transcript yet,
# transcribe them, save Name.transcript.md beside them, then hand the GPU back.
#
#   transcribe-new.sh                 # the watched NAS folder, output to the terminal
#   transcribe-new.sh /some/dir       # any other folder
#   transcribe-new.sh --gui           # launcher mode: reports in a dialog instead
#
# Safe to click twice: finished work is remembered by content hash, so a second
# click re-checks and exits without loading anything, and the models are only
# loaded at all if something is genuinely missing.
set -uo pipefail

PROJ="$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")" && pwd)"
GUI=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --gui) GUI=1 ;;
    *) ARGS+=("$a") ;;
  esac
done
DIR="${ARGS[0]:-/mnt/nas/IFX/03_Music}"
PY="$PROJ/.venv/bin/python"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$PROJ/data/logs/manual-$STAMP.log"
mkdir -p "$PROJ/data/logs"
export TRANSCRIBE_LOG="$LOG"

line() { printf '%s\n' "--------------------------------------------------------------------"; }
gpu() {
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/  /'
  [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] || echo "  (nothing using the GPU)"
}

line
echo "Transcribe new recordings"
echo "  folder : $DIR"
echo "  log    : $LOG"
line

if [ ! -x "$PY" ]; then
  msg="No virtualenv at $PY. Run ./setup.sh in $PROJ first."
  echo "X $msg"; [ "$GUI" = 1 ] && zenity --error --title="Transcriptions" --text="$msg"
  exit 1
fi

# Two passes, a few seconds apart: the first records each file's size+mtime, the
# second only trusts it once they are unchanged. That is what stops a recording
# that is still being copied in from being transcribed as if it were complete.
run_pass() { ( cd "$PROJ" && "$PY" watcher.py --dir "$DIR" --formats md --min-age 2 --verbose ); }
run_pass 2>&1 | tee -a "$LOG" | sed 's/^/  /'
sleep 5
run_pass 2>&1 | tee -a "$LOG" | sed 's/^/  /'

echo
line
SUMMARY="$(cd "$PROJ" && "$PY" - "$DIR" <<'PY'
import pathlib, sys
root = pathlib.Path(sys.argv[1])
AUDIO = {".mp3", ".wav", ".m4a", ".mp4", ".flac", ".ogg", ".aac", ".webm", ".mkv", ".mov", ".3gp"}
try:
    files = sorted(p for p in root.iterdir() if p.suffix.lower() in AUDIO)
except OSError as exc:
    print(f"Cannot read {root}: {exc}"); raise SystemExit
missing = [p.name for p in files if not p.with_name(p.stem + ".transcript.md").is_file()]
if not files:
    print(f"No audio files in {root.name}."); raise SystemExit
if missing:
    print(f"{len(files) - len(missing)} of {len(files)} transcribed. Still missing: "
          + ", ".join(missing[:6]) + ("…" if len(missing) > 6 else ""))
else:
    print(f"All {len(files)} recordings have a markdown transcript.")
PY
)"
echo "$SUMMARY"
line
echo "GPU after (should show only what was there before):"; gpu

# A click from the launcher has no terminal, so the answer arrives as a dialog;
# from a shell it is already on screen. Log file keeps the full transcript of the run.
if [ "$GUI" = 1 ] && command -v zenity >/dev/null 2>&1; then
  zenity --info --title="Transcriptions — $(basename "$DIR")" \
         --text="$SUMMARY

Full log: $LOG" --width=520 --timeout=120
elif command -v notify-send >/dev/null 2>&1; then
  notify-send -t 8000 "Transcriptions" "$SUMMARY" 2>/dev/null || true
fi

# Stay open when double-clicked (nothing to read otherwise), but never block a script.
if [ -t 0 ]; then printf '\nPress Enter to close… '; read -r _; fi
