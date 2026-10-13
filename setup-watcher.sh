#!/usr/bin/env bash
# Install the automatic transcription timer for watcher.py (systemd user units).
#
#   ./setup-watcher.sh              install + start the timer
#   ./setup-watcher.sh status       timer state, last runs, recent log
#   ./setup-watcher.sh run          transcribe the pending queue now (waits for it)
#   ./setup-watcher.sh once         one-shot scan in the foreground (no timer)
#   ./setup-watcher.sh uninstall    stop + remove the units
#
# Each tick loads the models, works through everything pending, and exits — the
# timer itself costs nothing when there is nothing to do. Runs never overlap
# (watcher.py takes a non-blocking lock), so a queue that takes hours is safe.
set -euo pipefail

# ---------------- configuration (edit here, then re-run this script) ---------
WATCH_DIR="/mnt/nas/IFX/03_Music"           # spaces OK in the folder name; no " or $
FORMATS="md"                                # md,txt,json,srt,vtt
INTERVAL="3"                                  # minutes between scans
MIN_AGE="180"                               # seconds a file must be unchanged
APP_PORT="8765"                             # your web app's port; while it serves, the watcher waits
MOUNT_WAIT="25"                             # seconds to wait for the share at boot before standing down
MIN_FREE_VRAM=""                              # GiB free before models load; empty = derive from ASR
                                           # (raised automatically for the bigger --asr engines)
# Utilisation is deliberately NOT a gate by default (100 = ignore it). On a box with a
# resident inference server it is meaningless: nvidia-smi reads 0 % between requests and
# ~95 % while one is in flight, so gating on it starves the queue for hours without ever
# protecting anything. The VRAM floor below is the guard that actually matters: the card
# is 96 GB and the models fit in the ~3 GB a 93 GB server leaves over, which is how 14 h
# of backlog got transcribed alongside it. Set e.g. 60 to also stand down during heavy use.
MAX_OTHER_UTIL="100"
GPU_TIME_LIMIT="480"                        # minutes of GPU work per run, then give the card back
# --- transcription engine ----------------------------------------------------
# One global choice, not per file. live = the streaming path (partial text while
# it runs, handles overlapping speakers, Nemotron 3.5 ASR 0.6B). The others run
# diarization first and then an offline ASR engine over the speaker cuts: better
# sentences and better German, progress arrives per speaker turn instead of per
# second.   live | fast | balanced | quality | german
ASR="balanced"
# Language for the engines that have no built-in detection (quality/german).
# "auto" keeps the ones that do detect it (live/fast/balanced) and makes
# `quality` refuse to start rather than guess.
ASR_LANG="auto"
# Optional file of canonical spellings (names, product words) to repair in the
# output; every rewrite is logged. See glossary.example.txt.
ASR_GLOSSARY=""
# ----------------------------------------------------------------------------
EXTRA_ARGS=""                               # e.g. "--no-sidecars" or "--max-speakers 4"

PROJ="$(cd "$(dirname "$0")" && pwd)"
PY="$PROJ/.venv/bin/python"
UNITS="$HOME/.config/systemd/user"
SVC="$UNITS/speaker-watcher.service"
TMR="$UNITS/speaker-watcher.timer"

die() { echo "error: $*" >&2; exit 1; }

write_units() {
  [ -x "$PY" ] || die "$PY not found — run ./setup.sh first"
  case "$PROJ$WATCH_DIR" in *'"'*) die 'paths may not contain a double quote (breaks systemd unit quoting)' ;; esac
  [ -f "$PROJ/watcher.py" ] || die "watcher.py missing next to this script"
  case " ${ASR} " in " live "|" fast "|" balanced "|" quality "|" german ") ;; \
    *) die "ASR must be one of: live fast balanced quality german (got '${ASR}')" ;; esac
  mkdir -p "$UNITS"
  # systemd reads '%' in a unit file as a unit specifier, so a percent-encoded
  # URL needs each '%' doubled to survive into the value.
  DOC_PATH="${PROJ// /%20}"; DOC_PATH="${DOC_PATH//%/%%}"
  cat > "$SVC" <<EOF
[Unit]
Description=Transcribe new recordings with speaker separation (${WATCH_DIR})
Documentation=file://${DOC_PATH}/README.md
After=network-online.target remote-fs.target
# Unattended retry job: never let a burst of failed ticks trip the start limit,
# or the timer would keep firing while systemd refuses to start anything.
StartLimitIntervalSec=0

[Service]
Type=oneshot
WorkingDirectory=${PROJ}
ExecStart="${PY}" "${PROJ}/watcher.py" --dir "${WATCH_DIR}" --formats ${FORMATS} --min-age ${MIN_AGE} ${EXTRA_ARGS}
Environment=PYTHONUNBUFFERED=1
Environment=SPEAKER_APP_PORT=${APP_PORT}
# The share is autofs/CIFS over a VPN address: watcher.py probes it in a
# bounded daemon thread instead of hanging in an uninterruptible stat(), and
# the GPU thresholds below keep it off the card while you are using it.
Environment=SPEAKER_MOUNT_WAIT=${MOUNT_WAIT}
${MIN_FREE_VRAM:+Environment=SPEAKER_MIN_FREE_VRAM=${MIN_FREE_VRAM}}
Environment=SPEAKER_MAX_OTHER_UTIL=${MAX_OTHER_UTIL}
Environment=SPEAKER_GPU_TIME_LIMIT=${GPU_TIME_LIMIT}
Environment=SPEAKER_ASR=${ASR}
Environment=SPEAKER_LANG=${ASR_LANG}
Environment=SPEAKER_GLOSSARY=${ASR_GLOSSARY}
# Be gentle: this should not compete with an interactive session or the NAS.
Nice=10
IOSchedulingClass=best-effort
IOSchedulingPriority=7
# A long backlog runs for hours; never time it out, but stop promptly on demand.
TimeoutStartSec=infinity
TimeoutStopSec=60
SendSIGKILL=no
EOF
  case "$INTERVAL" in
    ''|*[!0-9]*) die "INTERVAL must be a whole number of minutes (got '$INTERVAL')" ;;
  esac
  # Wall-clock schedule, not OnUnit{Active,Inactive}Sec: a oneshot that is skipped
  # by its condition (share unmounted) neither activates nor deactivates, which
  # leaves those relative timers stuck in state 'elapsed' and silently dead. An
  # absolute OnCalendar is independent of the service's state and always re-arms.
  # Long runs are harmless: a start job for an already-running oneshot is a no-op,
  # and watcher.py's own lock covers the overlap.
  cat > "$TMR" <<EOF
[Unit]
Description=Look for new recordings to transcribe every ${INTERVAL} min

[Timer]
OnCalendar=*:0/${INTERVAL}
AccuracySec=10s
RandomizedDelaySec=15
Persistent=true

[Install]
WantedBy=timers.target
EOF
}

systemctl_user() { systemctl --user "$@"; }

case "${1:-install}" in
  install)
    write_units
    systemctl_user daemon-reload
    systemctl_user enable --now speaker-watcher.timer
    # Re-arm even when the timer was already active (or stuck 'elapsed'): re-running
    # this after editing the config above must actually take effect.
    systemctl_user restart speaker-watcher.timer
    # Timers must survive logout: without linger the watcher only runs while
    # this user has a live session.
    if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null || echo yes)" = "no" ]; then
      echo "enabling linger (so the timer keeps running after logout)…"
      loginctl enable-linger "$USER" || echo "  (failed — run: sudo loginctl enable-linger $USER)"
    fi
    systemctl_user --no-pager --full list-timers speaker-watcher.timer || true
    echo
    echo "Installed. Check with: ./setup-watcher.sh status"
    echo "Nothing is transcribed before two runs have seen a file settled (~2 ticks.)"
  echo "Prefer to trigger it yourself? ./setup-watcher.sh pause"
    ;;
  pause)
    # Keep the units installed but stop the schedule: use `run` (or ask your agent
    # to) whenever you actually want the queue worked. `install` turns it back on.
    systemctl_user disable --now speaker-watcher.timer 2>/dev/null || true
    echo "Timer paused. Transcribe on demand with: ./setup-watcher.sh run  (or: ./setup-watcher.sh install)"
    ;;
  uninstall)
    systemctl_user disable --now speaker-watcher.timer 2>/dev/null || true
    systemctl_user stop speaker-watcher.service 2>/dev/null || true
    rm -f "$SVC" "$TMR"
    systemctl_user daemon-reload
    echo "Removed the timer and units."
    ;;
  status)
    systemctl_user --no-pager --full list-timers speaker-watcher.timer || echo "timer not installed"
    echo "--- last runs ---"
    systemctl_user --no-pager --full status speaker-watcher.service 2>&1 | sed -n '1,12p' || true
    echo "--- log (last 30 lines) ---"
    journalctl --user -u speaker-watcher.service -n 30 --no-pager 2>/dev/null || true
    echo "--- watcher log tail ---"
    [ -f "$PROJ/data/logs/watcher.log" ] && tail -n 20 "$PROJ/data/logs/watcher.log"
    ;;
  run)
    # --wait keeps the shell attached so you watch progress; drop it if you want
    # to fire and forget (the timer does that every ${INTERVAL} minutes anyway).
    write_units
    systemctl_user daemon-reload
    systemctl_user start speaker-watcher.service --wait
    ;;
  once)
    # Foreground, no systemd, no locking surprises: good for a first pilot.
    [ -x "$PY" ] || die "$PY not found — run ./setup.sh first"
  case "$PROJ$WATCH_DIR" in *'"'*) die 'paths may not contain a double quote (breaks systemd unit quoting)' ;; esac
    exec "$PY" "$PROJ/watcher.py" --dir "$WATCH_DIR" --formats "$FORMATS" \
         --min-age "$MIN_AGE" $EXTRA_ARGS "$@"
    ;;
  *)
    die "usage: $0 [install|uninstall|status|run|once] [extra watcher.py args]"
    ;;
esac
