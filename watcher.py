#!/usr/bin/env python3
"""Watch a folder and transcribe new audio/video with speaker separation.

One-shot by design: scan the folder, transcribe everything that is pending
(loading the models exactly once for the whole batch), write the transcripts,
and exit. Nothing stays resident and no VRAM is pinned between runs — a systemd
timer (``systemd/speaker-watcher.timer``, every 3 min) is what makes it
automatic. Install it with ``./setup-watcher.sh``.

    .venv/bin/python watcher.py --dry-run     # show the queue; no transcripts, state or ledger
    .venv/bin/python watcher.py               # do the work
    .venv/bin/python watcher.py --limit 1 --formats md,srt

The watched folder is a NAS mount that FreeFileSync pushes into, so a "new"
file can be half-copied, can reappear under another name, and can exist twice
as an mp3/wav pair. Three rules keep that from producing junk:

* a file is eligible only once it has stopped changing (see ``scan``)
* transcripts are recorded in ``data/watcher-ledger.json`` keyed by content
  hash, and only after the outputs are on disk — so an interrupted run resumes
  by itself and a truncated or empty transcript is never treated as finished
* files that normalize to the same recording name (``X.mp3`` + ``X.wav``)
  compete, and only the cheapest-to-read member of the pair is transcribed
"""
from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import unicodedata
import wave
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

DEFAULT_DIR = Path(os.environ.get("SPEAKER_WATCH_DIR", "/mnt/nas/IFX/03_Music"))
DATA_DIR = BASE / "data"
RESULTS_DIR = DATA_DIR / "results"        # canonical, web-UI-readable transcripts
MIRROR_DIR = DATA_DIR / "transcripts"     # local copies of every exported format
STATE_PATH = DATA_DIR / "watcher-state.json"
LEDGER_PATH = DATA_DIR / "watcher-ledger.json"
LOCK_PATH = DATA_DIR / "watcher.lock"
SCRATCH_DIR = DATA_DIR / "scratch"
LOG_PATH = DATA_DIR / "logs" / "watcher.log"

# Same input list the web app accepts (app.py).
AUDIO_EXTS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".amr",
    ".aiff", ".aif", ".mp4", ".mkv", ".mov", ".webm", ".m4b", ".mp2", ".3gp",
}
FORMATS = ("md", "txt", "json", "srt", "vtt")
# Staging files left by copy/sync tools (never transcribe a half-written file).
TRANSIENT = re.compile(
    r"(?:\.part\d*|\.tmp[\w.]*|\.temp|\.crdownload|\.download|\.file|\.etp$|\.ffs_\w+"
    r"|\.~lock\.|~)$",
    re.IGNORECASE,
)
# Transcripts we (or earlier runs) wrote next to the audio — not audio, not new work.
SIDECAR = re.compile(r"\.(?:transcript|nemotron)\.(?:md|txt|json|srt|vtt)$", re.IGNORECASE)

MIN_SIZE = 1024          # ignore empty placeholders
MAX_TRIES = 3            # failures per content hash before quarantining


# ------------------------------------------------------------------ utilities

def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass  # logging must never sink a transcription run


def _log_tail() -> str | None:
    """Message of the last log line (without its timestamp), so a quiet tick can
    stay quiet — see the idle check in ``main``.

    Read as bytes and decoded with 'replace': seeking to an arbitrary byte offset
    can land inside a multi-byte character (the messages contain em dashes and
    ellipses), which would raise UnicodeDecodeError on every tick.
    """
    try:
        with LOG_PATH.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 4096))
            blob = fh.read().decode("utf-8", "replace")
    except OSError:
        return None
    lines = [ln for ln in blob.splitlines() if ln.strip()]
    if not lines:
        return None
    last = lines[-1]
    if len(lines) == 1:  # we started mid-line; it is still the tail of that message
        return last
    return last.split(" | ", 1)[-1]


def _idle_is_quiet(summary: str) -> bool:
    """True when the previous line was this same 'nothing to do' summary."""
    tail = _log_tail()
    return bool(tail) and (tail == summary or tail.endswith(summary))


def _atomic_write(path: Path, data: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    text = isinstance(data, str)
    with tmp.open("w" if text else "wb", encoding="utf-8" if text else None) as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())  # land the bytes before the rename, on the NAS too
    os.replace(tmp, path)


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        # A torn file from a kill mid-write is recoverable; don't wipe real data
        # when a *different* corruption is at fault — keep the backup and requeue.
        bad = path.with_suffix(path.suffix + ".bad")
        try:
            path.replace(bad)
            log(f"WARN {path.name} was unreadable; kept as {bad.name}, starting fresh")
        except OSError:
            log(f"WARN {path.name} unreadable; starting fresh")
        return default
    except OSError as exc:
        log(f"WARN cannot read {path.name}: {exc}")
        return default


class Store:
    """JSON file kept in memory, flushed atomically after every file processed."""

    def __init__(self, path: Path):
        self.path = path
        self.data = _load_json(path, {})
        if not isinstance(self.data, dict):
            log(f"WARN {path.name} is not an object; ignoring its contents")
            self.data = {}

    def save(self) -> None:
        try:
            _atomic_write(self.path, json.dumps(self.data, ensure_ascii=False, indent=1))
        except OSError as exc:
            log(f"WARN could not save {self.path.name}: {exc}")


def acquire_lock():
    """Non-blocking run lock; None if another watcher holds it. Kept for process life."""
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = LOCK_PATH.open("w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        fh.close()
        if exc.errno in (errno.EACCES, errno.EAGAIN):
            return None
        raise
    fh.write(str(os.getpid()))
    fh.flush()
    return fh


def web_app_is_busy(port: int | None = None) -> bool:
    """True when app.py is live: it owns the GPU, so the watcher stands down.

    The port probe alone was not enough — ``app.py --port 9000`` is a documented
    option and would have let both run at once — so an interpreter running app.py
    counts as busy too.
    """
    if port is None:
        try:
            port = int(os.environ.get("SPEAKER_APP_PORT", "8765"))
        except ValueError:
            log("WARN SPEAKER_APP_PORT is not a number; falling back to 8765")
            port = 8765
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.4):
            return True
    except OSError:
        pass
    return app_process_running()


def app_process_running() -> bool:
    """Detect an interactive ``python app.py`` by its /proc command line.

    argv[0] must look like a Python/uv interpreter, so an editor merely opened on
    app.py does not hold the watcher off forever.
    """
    me = os.getpid()
    try:
        pids = [int(d) for d in os.listdir("/proc") if d.isdigit()]
    except OSError:
        return False
    for pid in pids:
        if pid == me:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                argv = [a.decode("utf-8", "replace") for a in fh.read().split(b"\0") if a]
        except OSError:
            continue
        if len(argv) >= 2 and Path(argv[0]).name.startswith(("python", "uv")) \
                and any(Path(a).name == "app.py" for a in argv[1:]):
            return True
    return False


# Set when systemd (or Ctrl-C) asks us to stop, so the file loop finishes its
# bookkeeping instead of being frozen mid-write by the default SIGTERM action.
_STOP = {"requested": False}


class ShutdownRequested(BaseException):
    """Deliberately not an Exception: an `except Exception` somewhere inside NeMo
    must not be able to swallow our own stop request and keep holding the card."""


def install_signal_handlers() -> None:
    def _handler(signum, _frame):
        _STOP["requested"] = True
        raise ShutdownRequested(f"received {signal.Signals(signum).name}")

    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (OSError, ValueError):  # not the main thread: nothing to install
            pass


def sweep_stale_tmp() -> None:
    """Remove `.name.tmp<pid>` leftovers from a run that was killed mid-write."""
    stale = re.compile(r"^\..+\.tmp\d+$")
    for d in (RESULTS_DIR, MIRROR_DIR, DATA_DIR):
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for p in entries:
            try:
                if stale.match(p.name) and time.time() - p.stat().st_mtime > 3600:
                    p.unlink(missing_ok=True)
            except OSError:
                pass


# ------------------------------------------------------- file identity / health

def file_hash(path: Path, chunk: int = 8 * 2**20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def duration_seconds(path: Path) -> float | None:
    """Media duration from the bundled ffmpeg (no ffprobe / system ffmpeg needed)."""
    try:
        import imageio_ffmpeg

        proc = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", os.fspath(path)],
            capture_output=True, text=True, timeout=180,
        )
    except Exception:
        return None
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr or "")
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def has_writer(path: Path) -> bool:
    """Best-effort: does a *local* process hold this file open for writing?

    FreeFileSync copies from the Windows side and never appears here, so the
    mtime/size checks in ``scan`` do most of the gating; this additionally
    covers a local rsync/scp/curl still filling the file. Unknown means "no":
    a missing lsof must not freeze the queue.
    """
    if shutil.which("lsof") is None:
        return False
    try:
        proc = subprocess.run(
            ["lsof", "-F", "a", os.fspath(path)],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return False
    # -F field output: one line per access mode, e.g. "ar" / "aw" / "auw".
    return any(ln.startswith("a") and "w" in ln[1:] for ln in proc.stdout.splitlines())


def norm_stem(name: str) -> str:
    """Recording identity within a folder: the file stem, ignoring case and extension only.

    Deliberately strict. Deleting every non-alphanumeric used to make `Rec1_2.mp3` and
    `Rec12.mp3` the same recording, so one transcript was transcribed and then written
    beside both \u2014 a transcript of the wrong audio sitting next to a file it never
    came from, which is worse than doing the work twice. Two encodings of one recording
    still share a stem (X.mp3 / X.wav), and identical bytes under different names are
    still deduplicated by content hash, which cannot be wrong by construction.
    """
    stem = unicodedata.normalize("NFKD", Path(name).stem)
    stem = re.sub(SIDECAR, "", stem)
    return re.sub(r"\s+", " ", stem).strip().casefold()


def scan(root: Path, min_age: float, state: Store, *, check_open: bool,
         only: str | None = None,
         dry_run: bool = False) -> tuple[list[dict], list[tuple[str, str]]]:
    """Eligible audio files, with a content hash (reused from ``state`` when unchanged).

    Returns (ready, waiting_items) where waiting_items is [(filename, reason)].
    ``state`` maps path -> {size, mtime, digest}: the digest is why a tick over an
    unchanged folder costs nothing — the hash of a 450 MB file on the NAS is
    computed once, never again while it sits there.
    """
    now = time.time()
    pat = re.compile(only, re.IGNORECASE) if only else None
    ready: list[dict] = []
    waiting_items: list[tuple[str, str]] = []
    seen: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "~", "$RECYCLE", "System Volume"))]
        for fn in filenames:
            if fn.startswith("."):
                continue
            p = Path(dirpath) / fn
            key = str(p)
            seen.add(key)
            try:
                st = p.stat()
            except OSError:
                continue  # path disappeared mid-walk (a sync still settling)
            if not p.is_file() or p.suffix.lower() not in AUDIO_EXTS:
                continue
            if TRANSIENT.search(fn) or SIDECAR.search(fn) or st.st_size < MIN_SIZE:
                continue
            if pat and not pat.search(str(p.relative_to(root))):
                continue  # --only: never hashed, never queued, state left as it was
            mtime = round(st.st_mtime, 3)
            prev = state.data.get(key) or {}
            first_sighting = not prev
            unchanged = bool(prev) and prev.get("size") == st.st_size and prev.get("mtime") == mtime
            # Two consecutive identical observations are required, never just an
            # old mtime: sync tools copy with the *source* timestamp, so a file
            # that is still arriving from the Windows box can look days old.
            # First sighting and any size/mtime change both mean "not settled".
            waiting = None
            if first_sighting:
                if not dry_run:
                    waiting = "seen for the first time — waiting for it to settle"
            elif not unchanged:
                waiting = "size/mtime changed since the last run — still copying"
            elif now - st.st_mtime < min_age:
                waiting = f"modified within the last {min_age:.0f}s"
            elif check_open and has_writer(p):
                waiting = "held open for writing by a local process"
            entry = {"size": st.st_size, "mtime": mtime,
                     "digest": prev.get("digest") if unchanged else None}
            if waiting:
                state.data[key] = entry  # remembered so the next run can compare
                waiting_items.append((fn, waiting))
                continue
            if not entry["digest"]:
                try:
                    entry["digest"] = file_hash(p)
                except OSError as exc:
                    log(f"WARN cannot read {fn}: {exc}")
                    continue
            state.data[key] = entry
            ready.append({"path": p, "size": st.st_size, "mtime": entry["mtime"],
                          "digest": entry["digest"]})
    if dry_run:
        # A preview must not touch the bookkeeping: if it recorded first
        # sightings, the next real run would treat them as settled and skip the
        # two-observation safety interval.
        return ready, waiting_items
    # Forget only files under *this* root: an earlier run's entries for a different
    # --dir (or a test folder) must survive, or switching folders would discard
    # every cached hash and force a multi-gigabyte rehash plus a fresh settle wait.
    under_root = str(root).rstrip(os.sep) + os.sep
    for key in [k for k in state.data if k.startswith(under_root) and k not in seen]:
        del state.data[key]  # files that moved away, so the state stays small
    state.save()
    return ready, waiting_items


def buckets_by_recording(items: list[dict]) -> list[list[dict]]:
    """Group encodings of one recording (same folder + normalized name)."""
    buckets: dict[tuple[str, str], list[dict]] = {}
    for it in items:
        buckets.setdefault((str(it["path"].parent), norm_stem(it["path"].name)), []).append(it)
    return list(buckets.values())


def cheapest(group: list[dict]) -> dict:
    """Member with the fewest bytes per second of audio: same talk, least to read.

    Duration matters — the smaller file could just be a truncated copy. Missing
    duration metadata falls back to comparing sizes.
    """
    for it in group:
        if "bps" not in it:
            dur = duration_seconds(it["path"])
            it["duration"] = round(dur, 1) if dur else None
            it["bps"] = it["size"] / dur if dur else float(it["size"])
    return min(group, key=lambda it: (it["bps"], it["size"], str(it["path"])))


def plan(root: Path, args, state: Store,
         ledger: Store) -> tuple[list[dict], list[tuple[str, str]]]:
    """Turn the folder into a work queue: stable, unseen, one member per recording."""
    ready, waiting_items = scan(root, args.min_age, state, check_open=not args.ignore_open,
                                only=args.only, dry_run=args.dry_run)
    todo: list[dict] = []
    scheduled: dict[str, dict] = {}   # digest -> queued job, stops same-tick duplicates
    for group in buckets_by_recording(ready):
        known = [ledger.data.get(it["digest"]) for it in group]
        done = next((e for e in known if e and e.get("status") == "done"), None)
        if done is None and not args.force:
            # The ledger is written just after the canonical result, so a missing
            # entry can only mean we were killed in that window. The result file is
            # the durable proof of the work: trust it, heal the sidecars, and never
            # pay the GPU twice for a transcript that already exists.
            for it in group:
                orphan = result_path(it["digest"])
                if orphan.is_file() and orphan.stat().st_size > 2:
                    done = {"name": it["path"].name, "result": str(orphan), "formats": {}}
                    break
        if done and not args.force:
            if _state_ok(done):
                if not _sidecars_missing(group, args) or getattr(args, "restore_missing", False):
                    _heal_sidecars(group, done, args)
                    continue
                # The folder is the ground truth. A transcript that is not in it is
                # treated as work that needs doing, not as something to reinstate from
                # data/: if the user deleted it, quietly copying a stored copy back is
                # both surprising and the route by which the wrong file's text could
                # reappear beside this audio. --restore-missing opts into the free path
                # (e.g. after a mirror sync wiped the folder's transcripts).
                log(f"{done.get('name')}: no transcript in the folder — transcribing it "
                    "(use --restore-missing to reinstate the stored copy instead)")
            log(f"stored transcript for {done.get('name')} is missing on disk — requeueing")
        if not args.force:
            adopted = _adopt_existing(group, args)
            if adopted is not None:
                ledger.data[next(it["digest"] for it in group)] = adopted
                ledger.save()
                continue
        rep = cheapest(group)
        if len(group) > 1:
            others = sorted(str(g["path"].name) for g in group if g is not rep)
            log(f"same recording twice: using {rep['path'].name}, not {', '.join(others)}")
        prior = ledger.data.get(rep["digest"]) or {}
        stalled = prior.get("status") == "stalled"
        quarantined = prior.get("status") == "failed" and prior.get("tries", 0) >= MAX_TRIES
        if (stalled or quarantined) and not args.force:
            why = (f"quarantined after {prior.get('tries', MAX_TRIES)} failures"
                   if quarantined
                   else f"parked after {prior.get('env_strikes', 0)} environmental errors")
            log(f"skipping {rep['path'].name} ({why}: {prior.get('error')}) — "
                "fix the cause, then --retry-failed")
            continue
        rep["name"] = rep["path"].name
        twin = scheduled.get(rep["digest"])
        if twin is not None:
            # Identical bytes that fell into *different* groups (one recording
            # copied under two names) must not cost a second run in this batch:
            # queue one job and let it write the transcripts for all of them.
            twin.setdefault("copies", []).extend([it["path"] for it in group])
            log(f"identical content: "
                f"{', '.join(sorted(g['path'].name for g in group))} share the work "
                f"already queued for {twin['path'].name}")
            continue
        scheduled[rep["digest"]] = rep
        todo.append(rep)
    if args.limit:
        todo = todo[: args.limit]
    return todo, waiting_items


def result_path(digest: str) -> Path:
    """Canonical transcript JSON for a content hash (durable 'done' marker)."""
    return RESULTS_DIR / f"{digest[:12]}.json"


def _state_ok(entry: dict) -> bool:
    """A ledger 'done' is only real while its output still exists on disk."""
    if entry.get("adopted"):
        # An adopted entry has no canonical JSON: the transcript beside the audio is
        # the evidence. Without this it would fail the check every tick, log
        # "missing on disk \u2014 requeueing" and be re-adopted forever.
        want = entry.get("formats") or {}
        return bool(want) and all(Path(p).is_file() for p in want.values())
    return bool(entry.get("result")) and Path(entry["result"]).is_file()


def _sidecar_text(entry: dict, fmt: str) -> str | None:
    """Render a sidecar from the canonical JSON belonging to *this* content hash.

    The stored mirror copies under data/transcripts/ are archives, not a source to
    write from: they are named after the file that produced them, and any doubt about
    which recording a name belonged to is exactly how a transcript of one recording
    could end up beside another. The canonical result is keyed by the SHA-256 of the
    audio that is sitting in the folder, so re-rendering from it cannot be mismatched.
    """
    if entry.get("result") and Path(entry["result"]).is_file():
        try:
            job = json.loads(Path(entry["result"]).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(job, dict) or "segments" not in job:
            return None
        from exports import render_exports

        return render_exports(job)[fmt]
    return None


def _adopt_existing(group: list[dict], args) -> dict | None:
    """Trust a transcript that is already in the folder instead of re-running the GPU.

    The ledger is normally the source of truth, but it can be lost (data/ wiped,
    machine reimaged) while the markdown exported beside the audio survives. Re-
    transcribing ~14 h of recordings to rediscover what is already on disk is the
    one outcome this watcher must never produce, so a plausible sidecar is enough:
    it must be named after this exact file, be non-empty, carry our header, and be
    at least as new as the audio (which rules out the stale transcript of an
    earlier recording that was later replaced under the same name).
    """
    fmt = args.format_list[0]
    names = {it["path"].name for it in group}          # a twin pair shares one title
    for it in group:
        side = it["path"].with_name(f"{it['path'].stem}.transcript.{fmt}")
        try:
            st = side.stat()
        except OSError:
            continue
        if st.st_size < 20 or st.st_mtime + 2 < it["mtime"]:
            continue
        try:
            head = side.open("r", encoding="utf-8").read(400)
        except (OSError, UnicodeDecodeError):
            continue
        title = (head.splitlines() or [""])[0]
        if not title.startswith("# Transcript"):
            continue
        # render_exports titles its output `# Transcript \u2014 <file name>`, so insist
        # that the text in our slot is about *this* recording. Without this, any file
        # someone renamed to fit would be believed, and the recording would end up
        # "done" carrying somebody else's words.
        titled = title.partition("\u2014")[2].strip()
        if titled not in names:
            log(f"not adopting {side.name}: it is titled {titled!r}, not one of "
                f"{sorted(names)}")
            continue
        log(f"adopting existing {side.name} ({st.st_size} bytes) — not re-transcribing "
            f"{it['path'].name}")
        return {"name": it["path"].name, "path": str(it["path"]), "status": "done",
                "size": it["size"], "mtime": it["mtime"], "adopted": True,
                "segments": None, "speakers": [], "result": None,
                "formats": {fmt: str(side)},
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    return None


def _sidecars_missing(group: list[dict], args) -> bool:
    """True if this recording has no transcript at all next to it in the folder.

    Asks whether *nothing* is there rather than whether every member has one: a pair
    like X.mp3 / X.wav is one recording, its transcript is shared by stem, and a gap
    on one member is healed from the canonical JSON (safe); an empty folder entry is
    a deletion, and that means doing the work.
    """
    return not any(it["path"].with_name(f"{it['path'].stem}.transcript.{fmt}").is_file()
                   for it in group for fmt in args.format_list)


def _heal_sidecars(group: list[dict], entry: dict, args) -> None:
    """Re-write transcripts that were deleted/overwritten next to the audio."""
    if args.no_sidecars:
        return
    for it in group:
        for fmt in args.format_list:
            want = it["path"].with_name(f"{it['path'].stem}.transcript.{fmt}")
            if want.is_file():
                continue
            text = _sidecar_text(entry, fmt)
            if text is not None:
                _atomic_write(want, text)
                log(f"restored {want.name} from the stored transcript")


def transcribe_queue(todo: list[dict], args, ledger: Store) -> tuple[int, int]:
    """Load the models once, then run the queue serially. Returns (done, failed).

    Reached only when `todo` is non-empty, and the import below is the first line
    in the program that touches torch/CUDA: an idle tick never gets here.
    """
    reason = gpu_deferral_reason(args)
    if reason:
        log(f"GPU not free: {reason} — standing down without loading anything; "
            "the next tick will try again")
        return 0, 0

    from pipeline import Transcriber, prepare_audio  # heavy: pulls in the GPU stack
    try:
        from pipeline import _is_silent_wav as is_silent
    except ImportError:  # older vendored copy: assume non-silent, so empties are retried
        is_silent = lambda wav: False  # noqa: E731

    run_dir = SCRATCH_DIR / f"run-{os.getpid()}-{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    transcriber = None
    done = failed = 0
    aborted = False
    env_skips = 0
    watchdog = _gpu_watchdog(getattr(args, "gpu_time_limit", 0))
    try:
        for n, job in enumerate(todo, 1):
            # Ask again before every file. Whoever opened the gate cannot be trusted
            # for a batch that may run for hours: the card can be taken in between.
            # Utilisation is left out here on purpose \u2014 it swings between 0 and 95 %
            # every second on a box with a resident server, and the backlog provably
            # completes alongside it; only the VRAM floor is a hard reason to stop.
            if _STOP["requested"]:
                log("  stop requested \u2014 leaving the rest of the queue for the next run")
                aborted = True
                break
            reason = gpu_deferral_reason(args, include_util=False)
            if reason:
                log(f"  VRAM taken over mid-queue: {reason} \u2014 leaving the rest "
                    "of the queue for the next run")
                aborted = True
                break
            src: Path = job["path"]
            digest: str = job["digest"]
            entry = dict(ledger.data.get(digest) or {})
            entry.update(name=job["name"], path=str(src), status="running",
                         size=job["size"], mtime=job["mtime"])
            t0 = time.time()
            log(f"[{n}/{len(todo)}] {src.name}"
                + (f" (~{job['duration'] / 60:.0f} min audio)" if job.get("duration") else ""))
            wav = run_dir / f"{n:02d}-{src.stem}.16k.wav"
            try:
                prepare_audio(src, wav)
                with wave.open(str(wav)) as w:
                    dur = max(w.getnframes() / w.getframerate(), 0.1)
                if transcriber is None:
                    log("loading models…")
                    transcriber = Transcriber(device=args.device, max_speakers=args.max_speakers)
                    log(f"models ready ({time.time() - t0:.0f}s)")
                t_work = time.time()  # per-file throughput excludes the one-off model load

                beat = [0.0]

                def publish(step, partial, _dur=dur):
                    now = time.monotonic()
                    if now - beat[0] < args.progress_every:
                        return
                    beat[0] = now
                    cur = max((p["end"] for p in partial), default=0.0)
                    log(f"    …{min(100 * cur / _dur, 100):.0f}% "
                        f"({cur / 60:.0f}/{_dur / 60:.0f} min, "
                        f"{len({p['speaker'] for p in partial})} speakers)")

                segs = [s for s in transcriber.transcribe(wav, publish=publish)
                        if (s.get("text") or "").strip()]
                if not segs:
                    # Empty output means silence only if the audio actually *is*
                    # silent: otherwise a file the models could not read would be
                    # marked done with a blank transcript and never looked at again.
                    try:
                        silent = is_silent(wav)
                    except Exception:
                        silent = False
                    if not silent:
                        raise EmptyResult(
                            f"no segments for {dur:.0f}s of non-silent audio")
                    raise NoSpeech("silent recording")
                write_transcript(src, digest, segs, entry, args,
                                 extra_paths=job.get("copies"))
                entry.update(duration=round(dur, 1), elapsed=round(time.time() - t_work, 1))
                done += 1
                log(f"  ok: {len(segs)} segments, {len(entry['speakers'])} speakers, "
                    f"{dur / 60:.1f} min audio in {entry['elapsed'] / 60:.1f} min")
            except NoSpeech as exc:
                # Silence is a valid answer, not a fault: record it and move on,
                # otherwise the queue retries this file until the end of time.
                write_transcript(src, digest, [], entry, args, extra_paths=job.get("copies"))
                entry.update(duration=None, elapsed=round(time.time() - t_work, 1))
                done += 1
                log(f"  ok: {exc} — wrote an empty transcript")
            except ShutdownRequested as exc:
                # systemd asked us to stop. Save what we know and leave the rest
                # queued: the default SIGTERM action would have frozen this
                # process wherever it happened to be, mid-write at worst.
                saved = entry.get("status") == "done"
                if not saved:
                    entry["status"] = "pending"
                    entry["interrupted"] = str(exc)
                log(f"  {exc} — "
                    + ("transcript already saved" if saved
                       else f"leaving {src.name} queued") + "; stopping cleanly")
                aborted = True
                break
            except Exception as exc:
                entry["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                entry["error"] = f"{type(exc).__name__}: {exc}"
                if _environmental(exc):
                    # GPU busy / share glitch: never charged to the file, so no
                    # 'tries' and no 'failed' status — the README promises a busy
                    # GPU cannot mark recordings failed, and quarantine is reserved
                    # for files that are actually broken.
                    entry["status"] = "pending"
                    strikes = int(entry.get("env_strikes", 0)) + 1
                    entry["env_strikes"] = strikes
                    log(f"  could not process {src.name}: {entry['error']}")
                    if strikes > ENV_GIVEUP_MAX:
                        # Not a file fault, but endlessly hammering a dead share or a
                        # permanently busy GPU helps nobody: park it visibly. Distinct
                        # from 'failed' so --retry-failed (or the GPU freeing up after
                        # a reboot) brings it back without any file being blamed.
                        entry["status"] = "stalled"
                        log(f"  giving up on {src.name} for now after {strikes} "
                            "environmental errors — run --retry-failed once the GPU/share "
                            "is healthy")
                        continue
                    if strikes <= ENV_STRIKES_MAX:
                        # Almost always global, so stop here: the rest of the queue
                        # would only fail the same way and burn the GPU for nothing.
                        log(f"  environmental, not a bad file (strike {strikes}/"
                            f"{ENV_STRIKES_MAX}) — retrying on the next tick, "
                            "the rest of the queue is left alone")
                        transcriber = None
                        aborted = True
                        break
                    # This one file keeps failing on the environment while its
                    # neighbours get through: set it aside for this run without
                    # blaming it, and keep the queue moving.
                    log(f"  {src.name} is still failing after {strikes} environmental "
                        "strikes — skipping it for now (it stays queued, not failed)")
                    env_skips += 1
                    if env_skips >= 3:
                        log("  three files skipped on environmental errors in one run — "
                            "the GPU/share looks unavailable, stopping until the next tick")
                        transcriber = None
                        aborted = True
                        break
                    continue
                failed += 1
                entry["status"] = "failed"
                entry["tries"] = int(entry.get("tries", 0)) + 1
                log(f"  FAILED ({entry['tries']}/{MAX_TRIES}): {entry['error']}")
                if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
                    transcriber = None  # rebuild the model from a clean GPU state
                    log("  VRAM exhausted — models will be reloaded for the next file")
            finally:
                wav.unlink(missing_ok=True)
                ledger.data[digest] = entry
                ledger.save()  # after every file, so a kill cannot lose finished work
        if aborted:
            log(f"stopped early: {done} transcribed, {len(todo) - done - failed} left for the next tick")
        return done, failed
    finally:
        if watchdog is not None:
            watchdog.cancel()
        if transcriber is not None:
            transcriber = None           # our reference goes first, or the models
            release_gpu()                # stay alive through gc.collect()
            log("GPU released (models dropped, allocator cache flushed)")
        shutil.rmtree(run_dir, ignore_errors=True)


class NoSpeech(RuntimeError):
    pass


class EmptyResult(RuntimeError):
    """No segments, but the audio was not silent: a transcription failure."""


# GPU/share problems are not the file's fault. Retrying them belongs to the next
# tick, not to the per-file quarantine, or a busy GPU would eventually mark every
# recording in the folder "failed" and stop trying them altogether.
ENVIRONMENTAL = re.compile(
    r"out of memory|\boom\b|cuda|cublas|cudnn|nccl|device-side|no kernel image|"
    r"tensor core|not compiled with|gpu|driver|cannot allocate|"
    # the share itself misbehaving (CIFS/NFS blips, often surfaced by ffmpeg)
    r"stale file handle|transport endpoint|input/output error|i/o error|remote i/o error|"
    r"errno (5|16|18|28|110|111|116|122|125)\b|no route to host|network is (unreachable|down)|"
    r"connection reset|timed out|host is down|read-only file system|too many open files",
    re.IGNORECASE,
)
# Same idea, caught from the errno instead of the message: EIO/ESTALE/ENOLINK/
# ENETUNREACH/ETIMEDOUT/EHOSTDOWN/EHOSTUNREACH/EAGAIN/ENOMEM.
TRANSIENT_ERRNOS = {errno.EIO, errno.ESTALE, errno.ENOLINK, errno.ENETUNREACH,
                   errno.ENETDOWN, errno.ETIMEDOUT, errno.EHOSTDOWN,
                   errno.EHOSTUNREACH, errno.EAGAIN, errno.ENOMEM, errno.ENODEV}
ENV_STRIKES_MAX = 5   # strikes on one file before it is skipped for a run (still retried)
ENV_GIVEUP_MAX = 30   # ~1.5 h of ticks, then stop hammering it until --retry-failed


def _environmental(exc: BaseException) -> bool:
    seen = set()
    while exc is not None and id(exc) not in seen:  # cause/context chain: an OSError
        seen.add(id(exc))                            # from the NAS often arrives wrapped
        if isinstance(exc, MemoryError):
            return True
        if isinstance(exc, OSError) and exc.errno in TRANSIENT_ERRNOS:
            return True
        if ENVIRONMENTAL.search(f"{type(exc).__name__}: {exc}"):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def write_transcript(src: Path, digest: str, segs: list[dict], entry: dict, args,
                     extra_paths: list[Path] | None = None) -> None:
    """Canonical JSON in data/results (plus local mirrors and sidecars by the audio).

    The canonical file is named with the first 12 hash digits and the web app's
    own result schema, so watcher output opens in the browser UI and can be
    re-exported or speaker-renamed there without transcribing again.

    'done' is recorded in the ledger entry as soon as that file is on disk, and
    plan() also accepts the file itself as proof: between the write and the
    ledger flush a SIGKILL must not be able to make us pay the GPU twice.
    """
    from exports import render_exports  # shared with the web app; stdlib-only on purpose

    job = {"name": src.name, "segments": segs, "labels": {}}
    rendered = render_exports(job)
    canon = result_path(digest)
    _atomic_write(canon, json.dumps(
        {"name": src.name, "segments": segs, "labels": {}}, ensure_ascii=False, indent=1))
    entry.update(result=str(canon), status="done", error=None, tries=0, env_strikes=0,
                 segments=len(segs), speakers=sorted({s["speaker"] for s in segs}),
                 ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
    formats = {}
    # Copies of the same bytes under other names (and the twin of an mp3/wav pair)
    # get their transcripts from this one run instead of a second transcription.
    targets = [src] + [p for p in (extra_paths or []) if p != src]
    for fmt in args.format_list:
        # Content-addressed: a flat, name-keyed mirror let two different recordings
        # that share a stem (in this or another watched folder) overwrite each other.
        mirror = MIRROR_DIR / digest[:12] / f"{src.stem}.transcript.{fmt}"
        _atomic_write(mirror, rendered[fmt])
        formats[fmt] = str(mirror)
        if args.no_sidecars:
            continue
        for target in targets:
            _atomic_write(target.with_name(f"{target.stem}.transcript.{fmt}"), rendered[fmt])
    entry["formats"] = formats
    entry["copies"] = [str(p) for p in targets[1:]]



# --------------------------------------------------------------- GPU courtesy
#
# The card belongs to the user, not to this watcher. Three rules follow from
# that: no CUDA is touched at all unless a file is actually waiting (torch is
# imported lazily, only inside transcribe_queue), the run stands down instead of
# queuing behind someone else's work, and the models are dropped before exiting.

def gpu_state() -> tuple[float | None, int | None]:
    """(worst-case free GiB, peak utilization %) from nvidia-smi; (None, None) if unreadable.

    Sampled three times and reduced pessimistically (min free, max busy): one reading
    of utilization.gpu is the busy fraction of the last ~1 s window, so a resident
    inference server between requests reads 0 % and would let us walk onto a card
    that is busy a moment later.
    """
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None, None
    free = util = None
    try:
        for attempt in range(3):
            proc = subprocess.run(
                [smi, "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15)
            row = next(ln for ln in proc.stdout.splitlines() if ln.strip())
            f_mb, u = [float(x) for x in row.replace(" ", "").split(",")[:2]]
            free = f_mb / 1024.0 if free is None else min(free, f_mb / 1024.0)
            util = int(u) if util is None else max(util, int(u))
            if attempt < 2:
                time.sleep(0.12)
        return free, util
    except Exception:
        return None, None


def gpu_deferral_reason(args, *, include_util: bool = True) -> str | None:
    """Why the GPU must not be taken right now; None means go ahead.

    `include_util=False` keeps only the hard VRAM floor, which is what a running
    batch re-checks between files; utilisation is an advisory gate for the start.
    """
    if getattr(args, "device", None) == "cpu":
        return None                      # no GPU involved at all
    if shutil.which("nvidia-smi") is None:
        return None                      # no NVIDIA tooling: nothing to be polite about
    free_gb, util = gpu_state()
    if free_gb is None:
        # Fail closed. A courtesy guard that assumes an idle card whenever it cannot
        # see the card has the priority exactly backwards for this watcher.
        return ("nvidia-smi did not answer, so the state of the card is unknown "
                "(driver busy, or the query timed out)")
    if free_gb < args.min_free_vram:
        return f"only {free_gb:.1f} GiB of VRAM free, {args.min_free_vram:.1f} GiB needed"
    if include_util and util is not None and util > args.max_other_util:
        return f"GPU is {util}% busy with another job (limit {args.max_other_util}%) " \
               f"and {free_gb:.1f} GiB is free"
    return None


def release_gpu(_unused=None) -> None:
    """Hand the card back: flush the caching allocator, collect what is unreferenced.

    The caller must drop its own reference first; rebinding a parameter here would
    have left the models alive through the collection and made this log a lie.
    """
    import gc

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass                             # exiting releases the context anyway
    return None


def _gpu_watchdog(minutes: float):
    """Never hold the card forever, even if a decoder wedges: SIGTERM ourselves,
    which the installed handler turns into a clean ShutdownRequested."""
    if minutes <= 0:
        return None

    def _fire():
        log(f"GPU time limit of {minutes:.0f} min reached — giving the card back")
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except OSError:
            pass

    timer = threading.Timer(minutes * 60, _fire)
    timer.daemon = True
    timer.start()
    return timer


def _path_is_dir_bounded(path: Path, timeout: float):
    """is_dir() that cannot hang forever.

    The watched folder is an autofs/CIFS mount reached over a VPN: touching the
    path can block in an uninterruptible kernel call while the network comes up,
    so the probe runs in a daemon thread and a wedged probe is a clean no-op.
    """
    box: dict = {}

    def _probe():
        try:
            box["v"] = path.is_dir()
        except OSError as exc:
            box["v"], box["err"] = None, exc

    t = threading.Thread(target=_probe, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None
    return box.get("v")


# ------------------------------------------------------------------- entrypoint

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Transcribe new files in a folder with speaker separation, then exit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="folder to watch")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the queue and change nothing (no transcripts, state or ledger; "
                         "--reset/--retry-failed are reported but not applied; a log line is still appended)")
    ap.add_argument("--limit", type=int, default=0, help="max files this run (0 = all pending)")
    ap.add_argument("--only", default=None,
                    help="only consider files whose path matches this regex (e.g. 'recording0011')")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--max-speakers", type=int, default=None, help="cap speaker streams (1-8)")
    ap.add_argument("--formats", default="md", help=f"comma list from {'|'.join(FORMATS)}")
    ap.add_argument("--min-age", type=float, default=180.0,
                    help="seconds a file must be unchanged before it is touched")
    ap.add_argument("--no-sidecars", action="store_true",
                    help="do not write transcripts next to the audio (use if a sync mirrors the folder)")
    ap.add_argument("--force", action="store_true", help="re-transcribe what is already done")
    ap.add_argument("--retry-failed", action="store_true", help="clear the quarantine first")
    ap.add_argument("--ignore-open", action="store_true", help="skip the lsof write-holder check")
    ap.add_argument("--reset", action="store_true", help="forget all remembered state/ledger (re-scan from zero)")
    ap.add_argument("--progress-every", type=float, default=30.0, help="seconds between progress lines")
    ap.add_argument("--restore-missing", action="store_true",
                    help="write back the copy kept in data/ instead of re-transcribing a "
                         "transcript that is gone from the folder (free, no GPU; meant for "
                         "after a mirror sync wiped the folder)")
    ap.add_argument("--verbose", action="store_true", default=None,
                    help="print the summary even when there was nothing to do; a manual "
                         "run should never be silent, an unattended tick should be")
    ap.add_argument("--min-free-vram", type=float, default=float(os.environ.get("SPEAKER_MIN_FREE_VRAM", "2.0")),
                    help="GiB that must be free before the models are loaded")
    ap.add_argument("--max-other-util", type=int, default=int(os.environ.get("SPEAKER_MAX_OTHER_UTIL", "100")),
                    help="stand down instead of loading models when the GPU is busier than this % "
                         "(100 = never defer on utilization)")
    ap.add_argument("--gpu-time-limit", type=float, default=float(os.environ.get("SPEAKER_GPU_TIME_LIMIT", "480")),
                    help="minutes of GPU work per run before giving the card back (0 = unlimited)")
    ap.add_argument("--mount-wait", type=float, default=float(os.environ.get("SPEAKER_MOUNT_WAIT", "25")),
                    help="seconds to wait for the watched folder to answer before standing down")
    ap.add_argument("--app-port", type=int,
                    default=int(os.environ.get("SPEAKER_APP_PORT", "8765")),
                    help="port the web app serves on; while it answers, the watcher "
                         "stands down (an app.py process is detected even on other ports)")
    args = ap.parse_args(argv)

    args.format_list = [f.strip().lower().lstrip(".") for f in args.formats.split(",") if f.strip()]
    if not args.format_list or set(args.format_list) - set(FORMATS):
        bad = sorted(set(args.format_list) - set(FORMATS))
        ap.error(f"unknown format(s): {', '.join(bad) or 'none given'} (have: {', '.join(FORMATS)})")

    root = args.dir.expanduser()
    probe = _path_is_dir_bounded(root, args.mount_wait)
    if probe is None:
        log(f"WARN {root} is not answering (share still coming up at boot?) — "
            f"no GPU work this tick")
        sys.stdout.flush()
        os._exit(0)                      # the wedged stat thread cannot be joined
    is_dir = probe
    if not is_dir:
        if not root.exists():
            # Normal at boot on an automounted VPN share: the path is simply not
            # there yet, so this is a clean no-op that the timer retries — not a
            # failure, and no red unit every few minutes during network bring-up.
            log(f"WARN {root} is not there (yet) -- share not mounted? nothing to do")
            return 0
        log(f"ERROR {root} exists but is not a folder -- check WATCH_DIR")
        return 2

    lock = acquire_lock()
    if lock is None:
        log("another watcher is already running — nothing to do")
        return 0
    if not args.dry_run and web_app_is_busy(args.app_port):
        log("the web app holds the GPU — skipping this tick, the timer will retry")
        return 0

    install_signal_handlers()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sweep_stale_tmp()
    if args.reset:
        # --dry-run means "tell me, change nothing", and it has to hold even when
        # combined with flags that are destructive on their own.
        if args.dry_run:
            log("--reset ignored: dry run does not touch state or ledger")
        else:
            for f in (STATE_PATH, LEDGER_PATH):
                f.unlink(missing_ok=True)
            log("cleared watcher state and ledger")
    state = Store(STATE_PATH)
    ledger = Store(LEDGER_PATH)
    if args.retry_failed:
        cleared = [k for k, e in ledger.data.items() if e.get("status") != "done"]
        if args.dry_run:
            log(f"--retry-failed ignored: dry run would clear the quarantine on {len(cleared)} file(s)")
        else:
            for k in cleared:
                ledger.data[k].update(tries=0, env_strikes=0, status="pending")
                ledger.data[k].pop("error", None)
            ledger.save()
            log(f"cleared the quarantine on {len(cleared)} file(s)")

    todo, waiting_items = plan(root, args, state, ledger)
    n_done = len([e for e in ledger.data.values() if e.get("status") == "done"])
    summary = (f"{root}: {len(todo)} to transcribe, {len(waiting_items)} waiting to settle, "
               f"{n_done} already done")
    # A 3-minute timer is 480 ticks a day; when nothing has changed, repeating the
    # same idle line turns the log into noise where a real signal should stand out.
    # An unattended tick that found nothing must not fill the journal; a run somebody
    # asked for (--verbose, or a terminal in front of it) always says something, even
    # when the answer is "nothing to do".
    loud = args.verbose if args.verbose is not None else sys.stdout.isatty()
    if todo or waiting_items or args.dry_run or loud or not _idle_is_quiet(summary):
        log(summary)
    for it in todo:
        log(f"  queued: {it['path'].name} ({it['size'] / 2**20:.0f} MiB)"
            + (f" ~{it['duration'] / 60:.0f} min" if it.get("duration") else ""))
    for fn, reason in waiting_items:
        log(f"  waiting: {fn} — {reason}")
    if args.dry_run:
        log("dry run — nothing transcribed")
        return 0
    if not todo:
        return 0

    done, failed = transcribe_queue(todo, args, ledger)
    rc = 0 if _STOP["requested"] else (1 if failed else 0)
    if _STOP["requested"]:
        log(f"stopped on request after {done} file(s); the rest stay queued for the next tick")
    else:
        log(f"finished: {done} transcribed, {failed} failed")
    strays = [t for t in threading.enumerate()
              if t is not threading.main_thread() and not t.daemon]
    if strays:
        # NeMo/torch can leave a housekeeping thread alive; abandoning it with a
        # hard exit is the only way to be sure the process — and with it the CUDA
        # context and its memory — really goes back to the system.
        log(f"{len(strays)} non-daemon thread(s) still alive "
            f"({', '.join(t.name for t in strays[:3])}); exiting hard to release everything")
        sys.stdout.flush()
        os._exit(rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
