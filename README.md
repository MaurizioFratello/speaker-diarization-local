# Speaker Transcript

Fully local, GPU-accelerated **who said what, when** transcription. Drop an
audio or video file into the browser; get back a speaker-attributed
transcript with a timeline, colored speaker labels, an audio player, and
exports (Markdown / Text / JSON / SRT / VTT).

Built on the official NVIDIA stack:

- Diarization: [`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization)
  (Sortformer-family, up to 8 speakers, arrival-time ordered labels)
- ASR: [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
  (cache-aware streaming, 32 language locales, automatic language detection)
- Coupled via NeMo Speech's `SpeakerTaggedASR` multitalker streaming pipeline
  (per the model's [ASR integration guide](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/main/ASR_INTEGRATION_GUIDE.md)),
  vendored into [`pipeline.py`](pipeline.py).

Nothing leaves your machine after the model weights are on disk (≈2.6 GB
total, no HF token needed — both models are open-weight). If
`models/Nemotron-3-Diarization.nemo` and
`models/nemotron-3.5-asr-streaming-0.6b.nemo` exist, they are loaded locally;
otherwise they are fetched once from the Hugging Face Hub and cached.

## Requirements

- NVIDIA GPU (Ampere or newer: RTX 30/40/50, RTX PRO, A/L/H series) — tested with
  ~2.5 GB VRAM. CPU fallback: `--device cpu`.
- Linux or Windows 10/11, 64-bit. Internet only needed for setup and once for
  model weights (see `models/` above).
- No system ffmpeg, CUDA toolkit, or Python install needed (uv provides a
  Python 3.12; ffmpeg is bundled via the `imageio-ffmpeg` wheel).

## Quick start

### Linux

```bash
./setup.sh    # one-time: installs uv, Python 3.12 venv, CUDA torch, NeMo
./run.sh      # starts the app and preloads models
```

### Windows (PowerShell)

```powershell
.\setup.ps1   # one-time
.\run.bat     # starts the app and preloads models
```

The launcher opens **http://127.0.0.1:8765** in your default browser once the server and preloaded models are ready. If it cannot open a browser, navigate there manually and drop files.

Supported inputs: WAV, MP3, M4A, FLAC, OGG/Opus, AAC, WMA, AMR, AIFF, and
video containers (MP4, MKV, MOV, WEBM) — the audio track is extracted locally.

Jobs get a browser player as soon as the audio is converted — even while
transcription is still running. It streams the internal 16 kHz
transcode (never the uploaded container), so playback works for every input
format regardless of browser codec support. Click a speaker badge (✎) in the
legend to rename a speaker: names flow into the timeline, segment list, copy
button and every download (txt/md/srt/vtt; JSON adds `speaker_name`).
Clearing a name restores "Speaker N".

Playback is linked across the page: a playhead line sweeps the colored
speaker timeline, and the transcript — rendered in its own scrollable panel —
highlights and auto-scrolls to the row being spoken while the player,
legend, timeline and download buttons stay pinned at the top. Clicking the
timeline or any transcript row jumps the audio there; seeking the player
re-centers the transcript panel. Scroll the transcript by hand to pause
auto-follow — the next click or seek resumes it.

While the GPU works, the transcript, timeline, speaker badges and a
progress bar update live ("Transcribing… 34 % (21:04 / 62:04)" — the
percentage is how far into the audio the text has reached, published from
non-mutating snapshots of the streaming decoder between chunks). Downloads,
copy and speaker renaming stay disabled until the job finishes; playback in
the meantime is fully available.

One job at a time: while a transcription runs, the drop zone locks (new
drops and clicks are rejected until it finishes), so the single transcript
view never tracks two jobs at once.

Uploads are single-use: the uploaded file is deleted as soon as the audio
conversion has consumed it, and only the small player transcode remains while
the job's result is on screen; the upload directory is wiped on every server
start. Transcripts (`data/results/*.json`) persist, so exports and renames
still work after restarts.

## Options

```
python app.py --port 8765 --host 127.0.0.1 --device cuda|cpu --preload --open-browser
```


- `NEMODIAR_ASR_BF16=0|1` pins the ASR weight precision; by default the app
  auto-selects bf16 (eager decoder, ~1.2 GB less VRAM) when the GPU has less
  than ~6 GB free, and fp32 (CUDA-graph decoder) otherwise.
- `NEMODIAR_MAX_SPKS=1..8` env var caps speaker streams (lower = less VRAM).
- `--host 0.0.0.0` to serve your LAN (keep it private; there is no auth).

## CLI (no browser)

```bash
.venv/bin/python pipeline.py meeting.mp4 --json-out meeting.json
```

Prints `[{"text", "start", "end", "speaker"}, ...]` segments.

## Files

| Path | Role |
|---|---|
| `app.py` | FastAPI server: upload, jobs, exports, media playback |
| `pipeline.py` | NeMo diarization+ASR pipeline + ffmpeg pre-converter |
| `exports.py` | Transcript renderers (md/txt/json/srt/vtt), stdlib-only by design |
| `watcher.py` | One-shot folder scanner: transcribes what is new, then exits |
| `setup-watcher.sh` | Installs/removes the systemd timer that runs `watcher.py` |
| `transcribe-new.sh` | One click: transcribe what has no markdown yet, then release the GPU |
| `static/index.html` | Single-page drag & drop UI (no external assets) |
| `data/` | `uploads/` single-use scratch (wiped on boot); `results/` per-job transcripts (persist) |
| `setup.sh` / `run.sh` / `setup.ps1` / `run.bat` | Linux/Windows launchers |

## Automatic transcription of a watched folder

Drop recordings into `/mnt/nas/IFX/03_Music` (or any folder) and get
`Name.transcript.md` back beside them, with speaker labels, no browser
involved:

```bash
./setup-watcher.sh              # install the timer (every 3 minutes)
./setup-watcher.sh status       # timer state + recent runs
./setup-watcher.sh run          # process the pending queue now (blocks until done)
./setup-watcher.sh once --dry-run   # foreground preview: no transcripts, state or ledger (log only)
./setup-watcher.sh uninstall    # stop and remove the timer
```

`watcher.py` is a **one-shot**: each tick scans, loads the models once, works
through everything pending, and exits — it is not a server and pins no VRAM
between runs. The systemd timer (`speaker-watcher.timer`) is what makes it
automatic. It uses a wall-clock `OnCalendar=*:0/<INTERVAL>` schedule rather than
`OnUnitActiveSec`/`OnUnitInactiveSec`: a oneshot that is skipped by its condition
(share unmounted) neither activates nor deactivates, which leaves those relative
timers stuck in `SubState=elapsed` and silently dead — verified live. A long run
is never interrupted and ticks never pile up: a start job for an already-running
oneshot is a no-op, and `watcher.py` takes a non-blocking lock so a second process
exits at once instead of competing for the GPU. Config lives at the top of
`setup-watcher.sh` (`WATCH_DIR`, `FORMATS`, `INTERVAL`, `MIN_AGE`, `EXTRA_ARGS`).

Because the watched folder is a NAS share that FreeFileSync pushes into, a
"new" file is not necessarily finished arriving, so the queue is gated:

- **Settled, not merely present.** A file needs two consecutive runs to see the
  same size *and* mtime, plus `MIN_AGE` seconds since it was modified. Sync
  tools copy with the source timestamp, so an age check alone would happily
  transcribe a half-copied 450 MB recording. A local process holding the file
  open for writing also defers it (lsof, when available).
- **One transcript per recording.** `X.mp3` and `X.wav` are one recording
  (identity = the file stem, ignoring case and extension only — deliberately
  strict, so `Rec1_2.mp3` and `Rec12.mp3` are *not* merged and can never share a
  transcript); only the member with the fewest bytes per second of audio is read
  (the same words, less traffic over the share). Identical content under
  different names is recognised by its SHA-256 and not re-transcribed.
- **A transcript exists only when it is complete.** `data/watcher-ledger.json`
  records per content hash what was produced, and is flushed after every file;
  transcripts are written to a temp file and renamed, and a result with zero
  segments from a *non-silent* file counts as a failure rather than success.
  Kilobyte-scale files are ignored as placeholders. Kill the process or crash
  the GPU mid-run and nothing is lost, nothing is duplicated.
- **The folder is the ground truth.** A recording with no `Name.transcript.md`
  beside it is work to be done, full stop — the watcher never writes back a copy
  it kept in `data/` behind your back, because restoring is the direction in which
  a file could end up carrying words that were never spoken on it. Delete a
  transcript and the next run makes it again from the audio. If a *sync* wiped them
  and you want the free path, ask for it explicitly with `--restore-missing`
  (`./transcribe-new.sh --restore`): it re-renders from `data/results/<hash>.json`,
  keyed by the SHA-256 of the audio, so it cannot mismatch either.
  The reverse holds too: lose `data/` entirely and a `Name.transcript.md` still
  beside the audio is *adopted* as done instead of re-running hours of work —
  provided it is non-empty, is not older than the audio, and its title line names
  this exact recording (`render_exports` writes `# Transcript — <file name>`), so a
  foreign file dropped into that slot can never mark real work as finished.
- **The GPU is borrowed, never occupied.** Before loading anything it asks
  `nvidia-smi` and stands down — exit 0, no models, work stays queued — if
  another process has taken the card (`--min-free-vram 2.0`, `--max-other-util
  25`), so a timer tick cannot queue behind or squeeze an inference server,
  Blender, or a training run. `--device cpu` skips the guard entirely, and
  `--max-other-util 100` disables only the utilisation half of it (use that if
  your resident server idles at 100% and you still want transcription).
  `--gpu-time-limit 480` is a hard ceiling on one run so a wedged model can never
  hold the card indefinitely. When work is done the models are dropped, the
  CUDA caching allocator is flushed, and any thread still keeping the process
  alive is cut with a hard exit — verified by `nvidia-smi` showing no leftover
  compute process, and by an idle tick importing neither `torch` nor `nemo`.
- **Bad files are set aside, the queue keeps moving.** Three failures on one
  file quarantines it (see `--retry-failed`). GPU or share errors (OOM while
  something else owns the VRAM, a CIFS `Input/output error`, a wrapped `EIO`)
  are blamed on the environment instead, never on the file: no retry is spent,
  the run stops early and retries on the next tick, so a busy GPU cannot mark
  your recordings "failed". A file that never gets past such errors is *parked*
  as `stalled` after ~30 strikes rather than retried forever — still recovered by
  `--retry-failed`, never counted as broken. The watcher also stands down while
  the web app is live — probed on its port (`APP_PORT`, default 8765) *and* by
  looking for a running `python app.py`, so a `--port 9000` server is caught too
  — and it releases the GPU between files.
- **Silence is a result.** A recording with no speech gets an empty transcript
  and is marked done, instead of being retried forever.

Transcripts are written as GitHub-flavoured markdown (`**Speaker 1**
`[00:00:04.6]`  text`) through `exports.py`, the same renderer the web app uses,
so a watcher file and a browser download are byte-identical text. It lives in a
separate stdlib-only module on purpose: `app.py` imports NeMo at module level,
and a 3-minute timer that merely restores a sidecar must not drag the GPU stack
into memory. The canonical copy is
`data/results/<content-hash-12>.json` in the web app's job format — those
transcripts can be opened and speaker-renamed in the UI as well.

Useful flags: `--only REGEX` (one recording), `--limit N`, `--formats md,srt`,
`--device cpu`, `--max-speakers 4` (or `NEMODIAR_MAX_SPKS=4`: fewer streams,
less VRAM — useful when another GPU job is resident), `--force`, `--reset`.
The GPU courtesy thresholds are set in `setup-watcher.sh` as `MIN_FREE_VRAM`,
`MAX_OTHER_UTIL`, `GPU_TIME_LIMIT`, `MOUNT_WAIT` and reach `watcher.py` as
`Environment=` variables.

It is safe at boot: the timer is installed as a *user* unit with lingering
enabled (`loginctl enable-linger`), so it runs before you log in, and
`Persistent=true` catches up on ticks missed while the machine was off. The NAS
is `x-systemd.automount`-managed and `watcher.py` probes the folder in a daemon
thread with a `--mount-wait` deadline, so an unreachable share costs one quiet
tick instead of a job stuck in an uninterruptible CIFS stat forever.

Those failure rules are covered by `test_watcher.py` (stubbed GPU stack, throwaway
folder; the real `data/` and GPU are untouched):

```bash
.venv/bin/python test_watcher.py
```

## Troubleshooting

- **CUDA OOM** on first transcribe: ASR weights drop to bf16 automatically on
  tight GPUs (`NEMODIAR_ASR_BF16=0|1` pins the choice). Still OOMing? Set
  `NEMODIAR_MAX_SPKS=4` (or 2) — one ASR stream per detected speaker.
- **Slow start**: first run downloads both checkpoints; afterwards the UI
  shows "Loading models" only until weights are cached.
- **Wrong/merged speakers**: labels are per-file, ordered by first appearance;
  they are not identities. Max 8 speakers per file.
- **Windows firewall prompt** on first run: allow it for localhost, or run with
  `--host 127.0.0.1` (default).

### One-click script

`transcribe-new.sh` is the whole job in one click — looks for audio in the folder
that has no `Name.transcript.md` yet, transcribes those, writes the markdown
beside them, and shows the GPU afterwards to prove it kept nothing:

```bash
./transcribe-new.sh                  # /mnt/nas/IFX/03_Music
./transcribe-new.sh /path/to/folder  # or anywhere else
```

It is installed as an application launcher ("Transcribe New Recordings" in the
app grid, and on the Desktop), so clicking it runs the same thing and reports the
result in a dialog. `Terminal=false` on purpose: there is no terminal emulator on
this machine. Each click logs to `data/logs/manual-<timestamp>.log`.

### Manual mode (what is installed here)

No timer or systemd unit is installed on this machine — nothing transcribes by
itself, and nothing holds the GPU between runs. When new recordings land, work
the queue by hand:

```bash
.venv/bin/python watcher.py --dir /mnt/nas/IFX/03_Music --formats md   # twice: the
# first pass records the file's size+mtime, the second trusts it as settled and
# transcribes it (that two-observation rule is what keeps a half-copied 450 MB
# recording from being read as complete)
```

It loads the models only if something is actually pending, stands down if the
card has less than `--min-free-vram` GiB spare, and releases everything before
exiting. To opt back into the schedule: `./setup-watcher.sh install` (undo with
`./setup-watcher.sh uninstall`).
