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

Then open **http://127.0.0.1:8765** and drop files.

Supported inputs: WAV, MP3, M4A, FLAC, OGG/Opus, AAC, WMA, AMR, AIFF, and
video containers (MP4, MKV, MOV, WEBM) — the audio track is extracted locally.

Finished jobs get an in-browser player. It streams the internal 16 kHz
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

Uploads are single-use: the uploaded file is deleted as soon as the audio
conversion has consumed it, and only the small player transcode remains while
the job's result is on screen; the upload directory is wiped on every server
start. Transcripts (`data/results/*.json`) persist, so exports and renames
still work after restarts.

## Options

```
python app.py --port 8765 --host 127.0.0.1 --device cuda|cpu --preload
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
| `static/index.html` | Single-page drag & drop UI (no external assets) |
| `data/` | `uploads/` single-use scratch (wiped on boot); `results/` per-job transcripts (persist) |
| `setup.sh` / `run.sh` / `setup.ps1` / `run.bat` | Linux/Windows launchers |

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
