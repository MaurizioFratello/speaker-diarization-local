# Speaker Transcript

Fully local, GPU-accelerated **who said what, when** transcription. Drop an
audio or video file into the browser; get back a speaker-attributed
transcript with a timeline, colored speaker labels, an audio player, and
exports (Markdown / Text / JSON / SRT / VTT).

Built on the official NVIDIA stack:

- Diarization: [`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization)
  (Sortformer-family, up to 8 speakers, arrival-time ordered labels)
- ASR: one of the NVIDIA checkpoints in [`engines.py`](engines.py) — see
  [Transcription engines](#transcription-engines)

Nothing leaves your machine after the model weights are on disk (no HF token
needed, all open-weight). Anything in `models/*.nemo` is loaded from there;
anything missing is fetched once from the Hugging Face Hub and cached.

## Transcription engines

Two ways to get from audio to speaker-attributed text. Pick one globally with
`--asr` (or `SPEAKER_ASR` for the watcher); it is one setting, not a per-file choice.

| `--asr` | ASR checkpoint | Diarization → ASR | Language | VRAM to be admitted |
|---|---|---|---|---|
| `live` | Nemotron 3.5 ASR Streaming 0.6B (2.3 GB) | coupled streaming ([official multitalker rail](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/main/ASR_INTEGRATION_GUIDE.md), vendored in [`pipeline.py`](pipeline.py)) | detected (101 prompts) | 2.5 GB |
| `fast` | same weights, decoded offline | diarize first, ASR on the speaker cuts ([`segment_asr.py`](segment_asr.py)) | detected | 3.1 GB |
| **`balanced`** *(default)* | Parakeet TDT 0.6B v3 (2.5 GB, 25 languages) | diarize first | detected by the model | 3.1 GB |
| `quality` | Canary 1B v2 (6.4 GB) | diarize first | **must be given** (`--lang de`) | 5.5 GB |

A fifth engine, `--asr german` (German-only FastConformer hybrid, 3.2 GB to be
admitted), is implemented but deliberately left out of the comparison below: it is
German-only, which `balanced` already covers, and its checkpoint is another 3.8 GB
download.

VRAM is what an engine asks the card for before it will load — bf16 weights, a
CUDA context and the activations of a `batch_size=16` decode, measured from a real
run rather than read off the checkpoint size (`engines.vram_floor_gib()` derives it,
so a smaller batch lowers the bar). An engine that is refused does not fail: it runs
on CPU and says so.

The `live` row is the original path: it emits partial text while the file is
still being read, and because diarization and ASR share one streaming geometry
it can put two simultaneous speakers in one line. Everything else runs
**diarization first**, cuts the audio at speaker turns and transcribes each cut
with a full-context offline model. That is what makes the stronger models
usable — Parakeet and Canary are not cache-aware — and it buys cleaner sentences,
far fewer 1–3 word fragments and lower word error rates (see
[Benchmarks](#benchmarks)). What it costs: a word said by two people at once is
given to the dominant speaker only, and progress arrives per speaker turn
instead of per second. Both modes return the same
`[{"text", "start", "end", "speaker"}]` contract, so the UI, the exports and the
watched folder are unaffected by the choice.

`fast` exists for the case "I already have the live weights, but I would rather
have whole sentences than a live feed": the same checkpoint, decoded with
context. It does not report word timestamps offline, so the timing of individual
words is reconstructed from their position in the cut.

**Automatic language detection**: `live`, `fast` and `balanced` need no
language at all — the model detects it per utterance, which is what you want
when a German meeting contains English technical terms. `quality` (Canary) is
prompt-conditioned and refuses to guess: it exits with a message telling you to
pass `--lang de` (or to use `--asr balanced`). A spoken-language identifier for
the `auto`+`quality` combination is planned, see
[docs/ASR-BACKENDS-PLAN.md](docs/ASR-BACKENDS-PLAN.md).

**Vocabulary (all engines)**: `--glossary FILE` repairs domain words and names
in the *text*, after decoding, with an audit trail (`grep` it in the job JSON
under `corrections`). Greedy decoding in Parakeet/Canary has no acoustic word
boost, so this is the lever for `Dehnungsmessstreifen` and company; see
[glossary.example.txt](glossary.example.txt) and
[docs/GLOSSARY.md](docs/GLOSSARY.md). Canary additionally accepts a
`--hotwords` context hint, which is a soft biasing rather than a guarantee.

## Requirements

- NVIDIA GPU (Ampere or newer: RTX 30/40/50, RTX PRO, A/L/H series) — 1.7–4.4 GB
  VRAM depending on the engine (see the table). No GPU, or the card is busy?
  Everything runs on CPU too (`--device cuda` degrades to CPU by itself when the
  card has no room; see [Troubleshooting](#troubleshooting)).
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
              [--asr balanced] [--lang auto] [--glossary glossary.txt]
```

`--asr`, `--lang` and `--glossary` are the global engine choice (see
[Transcription engines](#transcription-engines)); they are read from
`SPEAKER_ASR` / `SPEAKER_LANG` / `SPEAKER_GLOSSARY` when not given.


Deployment settings are environment variables, read in this order: CLI flag →
`SPEAKER_*` → `NEMODIAR_*` (the older prefix, still honoured) → built-in default.
`setup-watcher.sh` writes `SPEAKER_ASR`, `SPEAKER_LANG`, `SPEAKER_GLOSSARY` into
the systemd unit, so there is one place to change the engine on a server.

- `SPEAKER_ASR=live|fast|balanced|quality|german` — the global engine choice.
- `SPEAKER_LANG=de` — required by `quality`, optional for the others. On `live` it
  conditions the model's prompt, and a language the audio does not speak can make
  the output collapse to nothing there, so leave it at `auto` for mixed meetings.
- `SPEAKER_GLOSSARY=/path/glossary.txt`, `SPEAKER_HOTWORDS="…"`.
- `SPEAKER_MAX_SPKS=1..8` caps speaker streams (lower = less VRAM; `live` mode only).
- `SPEAKER_MIN_FREE_VRAM` / `--min-free-vram` is **unset by default**, which means
  "whatever the selected engine needs" (2.5 GiB for `live`, 3.1 for `fast`/`balanced`,
  3.2 for `german`, 5.5 for `quality`). Set a number to override that judgement, and `0`
  to load regardless of the card — a guard you cannot turn off is not a courtesy. The
  floor is weights + context + activations, so a smaller `--batch-size` lowers it.
- `SPEAKER_ASR_BF16=0|1` pins the ASR weight precision on the `live` path; by
  default the app auto-selects bf16 (eager decoder, ~1.2 GB less VRAM) when the
  GPU has less than ~6 GB free, and fp32 (CUDA-graph decoder) otherwise. The
  offline engines always use bf16 on CUDA (fp16 where bf16 is missing).
- `--host 0.0.0.0` to serve your LAN (keep it private; there is no auth).

## CLI (no browser)

```bash
.venv/bin/python pipeline.py meeting.mp4 --json-out meeting.json     --asr quality --lang de --glossary glossary.txt
```

Prints `[{"text", "start", "end", "speaker"}, ...]` segments. `--asr live`
keeps the streaming path; `--device cpu` and `--batch-size N` are there for
machines without a free GPU.

## Files

| Path | Role |
|---|---|
| `app.py` | FastAPI server: upload, jobs, exports, media playback |
| `pipeline.py` | both transcriber modes + ffmpeg pre-converter |
| `engines.py` | offline ASR engines: checkpoint registry, timestamps, language prompts |
| `segment_asr.py` | diarize-first: speaker runs, cuts, leak filtering, segment assembly |
| `model_store.py` | loads `models/*.nemo` when present, otherwise the Hub |
| `glossary.py` | text-side vocabulary repair (stdlib-only, no ML dependencies) |
| `test_glossary.py` / `test_engines.py` | dependency-light test suites (`python test_engines.py`) |
| `bench_asr.py` | engine benchmark: time, VRAM, fragmentation, CER (see [Benchmarks](#benchmarks)) |
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
`setup-watcher.sh` (`WATCH_DIR`, `FORMATS`, `INTERVAL`, `MIN_AGE`, `ASR`,
`ASR_LANG`, `ASR_GLOSSARY`, `EXTRA_ARGS`). The watcher raises its own
`MIN_FREE_VRAM` courtesy floor to whatever the selected engine needs, so
`ASR=quality` waits for the card instead of loading 6 GB onto it and OOMing.

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

## Benchmarks

`bench_asr.py` measures every engine on the same file, each in its own process so
one engine's crash or VRAM reservation cannot distort another's numbers:

```bash
.venv/bin/python bench_asr.py /mnt/nas/IFX/03_Music/Recording0024_ManuelDillinger_MEMS.mp3 \
    --lang de --glossary glossary.example.txt \
    --groundtruth samples/groundtruth/recording0024.json --wait-gpu 1200 --out bench/full
.venv/bin/python bench_asr.py MEDIA --rows balanced:cuda,quality:cuda --out bench/one
```

It reports wall time and real-time factor (RTF: seconds of wall per second of
audio, `<1` means faster than playback), the allocator's high-water mark, how the
transcript is chopped up (words per segment, and the share of segments with ≤ 2
words — the number that describes the one-word-line problem), and a character
error rate against a hand-corrected reference window. Per-engine transcripts land
in `bench/<out>/<config>_<device>/transcript.txt` for reading;
`bench/full/summary.json` holds the rows.

Read CER with care: it compares the reference text with whatever the engine put
inside that time window, so it also measures *where the engine chose to cut*, not
only what it heard. It is directly comparable between engines that share the
diarization (fast/balanced/quality/german); against `live`, which runs its own
joint diarization, a CER difference partly reflects different segment boundaries.
The qualitative score below is the tiebreaker.

**Qualitative score (1–7).** Objective metrics only go so far, so each engine's
transcript of the same passage was read against the reference correction and
scored: 7 = publishable as it stands, 6 = a few wrong words, 5 = readable with
visible effort, 4 = meaning intact but several mishearings and fragments,
3 = patchy, 2 = mostly broken, 1 = unusable. Scores are the maintainer's judgement
on one 53-minute German technical interview (two speakers, domain vocabulary,
mixed English terms) — treat them as a strong prior, not a universal ranking.

### Results — one 53-minute interview, four engines

Media: `Recording0024_ManuelDillinger_MEMS.mp3` — 3222 s, two speakers, German with
English technical vocabulary, the room reverberant and the vocabulary hostile
(`Dehnungsmessstreifen`, `stresskompensiert`, `E-Modul`). Glossary on for every row,
`--lang` given only to engines that require it, one fresh process per row, and a
60-second hand-corrected window (`samples/groundtruth/recording0024.json`) for CER.

| engine | asked | actually ran | wall | RTF | load | peak VRAM | segs | words/seg | ≤2-word | CER | score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `balanced` | cpu | cpu | 126 s | 0.039 | 6 s | – | 404 | 12.3 | 16 % | **0.054** | **6** |
| `quality` | cpu | cpu | 427 s | 0.133 | 10 s | – | 572 | 8.2 | 23 % | 0.058 | 6 |
| `fast` | cpu | cpu | 207 s | 0.064 | 22 s | – | 293 | 16.5 | 9 % | 0.074 | 5 |
| `live` | cpu | cpu | 886 s | 0.275 | 22 s | – | 400 | 12.7 | 17 % | 0.088 | 4 |
| `live` | cuda | cuda | **64 s** | **0.020** | 25 s | 1.69 GiB | 393 | 12.9 | 19 % | 0.085 | 4 |
| `balanced` | cuda | **cuda** (3-min clip) | 9.3 s | 0.052 | 6 s | **3.15 GiB** | 34 | 10.0 | 18 % | 0.030 | – |
| `balanced` | cuda | → cpu (53-min file, 4 attempts) | 97–229 s | 0.030–0.071 | 6 s | evicted | 404 | 12.3 | 16 % | 0.054 | – |

RTF is wall seconds per audio second, so `balanced` on CPU turns a 53-minute
interview into a transcript in two minutes — 25× faster than playback on cores that
were nobody else's. The `live`/cuda row is the only one that kept the card; see the
co-tenancy note below, because that is the interesting part of this table.

**What the scores were judging.** Same 60 seconds of reference text, one reading per
engine, 7 = publishable as is, 4 = meaning intact but fragments and mishearings.
`balanced` wrote `Dehnungsmessstreifen` correctly in both places, and `quality` did
not write it once (`den Weststreifen`, `Demosstreifen`) — yet `quality` invented less
elsewhere, produced `Ja.` where the others produced `Yeah.`, and kept sentence
punctuation disciplined. `fast` and `live` misheard the domain vocabulary repeatedly
(`demerst Streifen`, `Scholdungstechnik`, `Meshstreifen`, `Akowsensorke`) and
`live` also breaks mid-sentence at chunk boundaries. Both engines that got the word
wrong are also the two that produced the highest word-per-second counts, so the
glossary is load-bearing, not decorative.

**The `≤ 2-word` column is the honest one.** Diarize-first did *not* by itself cure the
one-word lines: it replaced the streaming path's chunk-boundary cuts with the
speaker-turn and punctuation boundaries of a much better model, and `quality` — which
reports real word timestamps — fragments the most (8.2 words per line). The fix is
`SegCfg` (`merge_gap`, `sent_break`), which is the open tuning pass in
[docs/ASR-BACKENDS-PLAN.md](docs/ASR-BACKENDS-PLAN.md); `fast` shows the direction, at
16.5 words per line.

**Co-tenancy, measured rather than assumed.** This box normally runs an inference
server holding 90–92 GiB of the 98 GiB card, and the offline engines were refused,
evicted, or both:

- `fast` and `balanced` asked for cuda were admitted (2.3–3.7 GiB resident), then died
  mid-decode on a 198 MiB activation tensor with 155 MiB left, because the neighbour's
  reservation *grew* after our check. Those rows completed on CPU at 164–317 s, i.e.
  they paid 40–110 s for the failed attempt.
- **Length decides it.** `balanced` on the GPU: a 3-minute clip loaded (3.15 GiB
  resident), decoded and finished on the card, producing the same 34 segments and
  within 4 words of the same transcript as the CPU run — a useful sanity check on the
  bf16 path. The 53-minute file was attempted four times at two batch sizes and landed
  on CPU every time, at 97–229 s versus 126 s for a run that never tried the card: the
  failed attempt is pure overhead, and the window for the neighbour to grow into the gap
  grows with the file.
- The arithmetic is tight rather than broken: the resident LLM leaves ~4.5 GiB of the
  98 GiB card and dips to ~2.9 GiB while it serves, and `balanced` occupies 3.15 GiB.
- The floor each engine needs is therefore weights + activations, not weights alone:
  3.1 GiB for `fast`/`balanced`, 3.2 for `german`, 5.5 for `quality`, and 2.5 measured
  for `live` — which is why `live` is the only engine that coexisted.
- `BENCH_BATCH=4` was tried to shrink the activations. It did not rescue co-tenancy;
  the neighbour expanded into whatever was left.
- `quality` (Canary, 4.4 GiB of bf16 weights) was never measurable on the GPU here, so
  its row is CPU-only and its GPU number is extrapolated, not verified.

Practical consequence: on a machine where a large model holds the card, run these
engines on CPU (RTF 0.03–0.13) or run the transcript before starting that model. The
code no longer cares which way it goes — the engine that loses the card mid-file
finishes the file on CPU and reports the device that answered, along with a `cuda_note`
saying whether it was `declined at the gate`, `refused at load` or `evicted mid-decode`.
Those three want different responses (a bigger gap, a smaller batch, a shorter job), and
for two rounds of this benchmark a row that only said `cpu` could not tell them apart.

## Troubleshooting

- **CUDA OOM / card full**: the offline engines check the free VRAM before
  loading and fall back to CPU (with a log line) rather than failing the job; on
  `--asr live` the ASR weights drop to bf16 automatically on tight GPUs
  (`NEMODIAR_ASR_BF16=0|1` pins the choice). Still OOMing? Set
  `NEMODIAR_MAX_SPKS=4` (or 2) — one ASR stream per detected speaker.
- **`quality` refuses to start**: Canary has no language detection, so it wants
  `--lang de` (etc.) instead of guessing. `--asr balanced` detects the language.
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
