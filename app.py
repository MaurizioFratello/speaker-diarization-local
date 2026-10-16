"""Local speaker-transcription web app.

Drop an audio/video file in the browser, get a speaker-attributed transcript
("who said what, when"), rendered on a timeline with exports.
Diarization and ASR run locally via NeMo; model choice is configurable with
``--asr``. Start with:  python app.py  [--port 8765] [--device cuda]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import threading
import time
import traceback
import wave
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response

from pipeline import (
    Transcriber,
    gate_open,
    load_transcriber,
    map_spliced_time,
    prepare_audio,
    splice_speech_regions,
    trim_edge_silence,
)

BASE_DIR = Path(__file__).resolve().parent
# SPEAKER_DATA_DIR relocates uploads/results/queue/app-settings.json (a throwaway
# run, a second instance on this checkout, or the device-switch tests).
DATA_DIR = Path(os.environ.get("SPEAKER_DATA_DIR") or BASE_DIR / "data")
UPLOAD_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
for d in (UPLOAD_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
# uploads/ is single-use scratch: files live only for conversion + transient
# playback; every boot wipes it, finished jobs keep only the 16k preview wav.
for _p in UPLOAD_DIR.iterdir():
    if _p.is_file():
        _p.unlink(missing_ok=True)

# Persistent pending queue: uploads land here and survive restarts until a
# worker claims them; transcode leftovers from before a crash do not.
QUEUE_DIR = DATA_DIR / "queue"
QUEUE_DIR.mkdir(parents=True, exist_ok=True)
for _p in QUEUE_DIR.glob("*.16k.wav"):  # pending originals stay for requeue
    _p.unlink(missing_ok=True)

_shutdown = threading.Event()
_browser_seen_lock = threading.Lock()
_browser_last_seen: float | None = None


def _mark_browser_seen() -> None:
    global _browser_last_seen
    with _browser_seen_lock:
        _browser_last_seen = time.monotonic()


def _monitor_browser_idle(server: "uvicorn.Server", idle_seconds: float) -> None:
    """Exit after the browser disappears and all submitted jobs have finished."""
    while not server.started and not server.should_exit:
        time.sleep(0.2)
    poll_s = min(5.0, max(0.5, idle_seconds / 10))
    while not server.should_exit:
        time.sleep(poll_s)
        with _browser_seen_lock:
            last_seen = _browser_last_seen
        # Arm only after a real browser page has connected; slow model preload or a
        # failed automatic browser launch must not kill a freshly started app.
        if last_seen is None or time.monotonic() - last_seen < idle_seconds:
            continue
        with jobs_lock:
            busy = any(j.get("status") in ("queued", "running") for j in jobs.values())
        if busy:
            continue
        print(f"[app] browser absent and idle for {idle_seconds:.0f}s; shutting down", flush=True)
        server.should_exit = True
        return


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Wipe upload scratch on exit so no large media files survive the app."""
    yield
    _shutdown.set()
    gpu_executor.shutdown(wait=False, cancel_futures=True)
    _sweep_uploads(None, force=True)
    threads = list(getattr(gpu_executor, "_threads", ()))
    deadline = time.time() + 15
    while time.time() < deadline and any(th.is_alive() for th in threads):
        time.sleep(0.5)
    if any(th.is_alive() for th in threads):
        os._exit(0)  # a stuck GPU worker must not hold the app open
    global transcriber
    if transcriber is not None:
        try:
            transcriber.unload()
        except Exception:
            pass
        transcriber = None
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


app = FastAPI(title="Speaker Transcription", lifespan=_lifespan)

# Global ASR choice (one setting, not per-upload): --asr / --lang / --glossary,
# also readable from SPEAKER_ASR / SPEAKER_LANG / SPEAKER_GLOSSARY.
ASR = {"mode": os.environ.get("SPEAKER_ASR", "balanced"),
       "lang": os.environ.get("SPEAKER_LANG", "auto"),
       "glossary": os.environ.get("SPEAKER_GLOSSARY", "")}
DEVICE_SETTINGS_FILE = DATA_DIR / "app-settings.json"


def _read_device_preference() -> str:
    env_device = os.environ.get("SPEAKER_DEVICE", "").lower()
    if env_device in {"cuda", "cpu"}:
        return env_device
    try:
        saved = json.loads(DEVICE_SETTINGS_FILE.read_text(encoding="utf-8")).get("device")
        if saved in {"cuda", "cpu"}:
            return saved
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    return "cuda"  # pipeline safely falls back to CPU if CUDA is unavailable/full


def _save_device_preference(device: str) -> None:
    tmp = DEVICE_SETTINGS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"device": device}, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, DEVICE_SETTINGS_FILE)


DEVICE_PREFERENCE = _read_device_preference()
transcriber: Transcriber | None = None
transcriber_error: str | None = None
#: The device a caller *asked* for, and the device the loaded object *is*. They
#: differ whenever the VRAM gate declines CUDA or a co-resident process evicts us;
#: keeping only the request made the **Use CPU** switch look dead, because a run that
#: had once degraded to CPU answered every later "cuda" request from the cache.
transcriber_device_preference: str | None = None
transcriber_actual_device: str | None = None
#: Why the two differ ("declined at the gate: …", "evicted mid-decode: …"), so a
#: transcript can say whether CPU was chosen or forced.
transcriber_note: str | None = None
transcriber_lock = threading.Lock()
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
gpu_executor = ThreadPoolExecutor(max_workers=1)  # serialize GPU work

# Silero VAD leading/trailing silence trim (set False via --no-vad-trim).
VAD_TRIM = True

# FIFO of submitted job ids awaiting their turn on the GPU.
_queue: list[str] = []
MAX_QUEUED = 8
_convert_lock = threading.Lock()  # one ffmpeg conversion at a time
_gpu_lock = threading.Lock()  # one transcription on the GPU at a time
# Set once the first model-load attempt (preload or first job) finishes;
# concurrent jobs wait on it instead of starting a second racing load.
model_ready_event = threading.Event()
# Set once a Transcriber load has actually begun (preload thread or first job);
# jobs only wait on model_ready_event when someone else is mid-load.
_model_load_active = threading.Event()


# --------------------------------------------------------------------------- models

def _release_loaded_transcriber() -> None:
    global transcriber, transcriber_device_preference
    global transcriber_actual_device, transcriber_note
    import torch
    old = transcriber
    transcriber = None
    transcriber_device_preference = None
    transcriber_actual_device = None
    transcriber_note = None
    if old is not None:
        try:
            old.unload()
        finally:
            del old
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _sync_transcriber_state(tr: Transcriber) -> None:
    """Read the device back off the loaded object instead of trusting the request.

    Runs under ``transcriber_lock``, or from the single worker right after a job, so
    a mid-decode eviction that moved the models to CPU is reported by the next
    /api/settings, /api/health and by that job's own provenance line.
    """
    global transcriber_actual_device, transcriber_note
    dev = getattr(tr, "device", None)
    transcriber_actual_device = str(getattr(dev, "type", dev) or "")
    note = getattr(tr, "cuda_note", None)
    if not note and transcriber_device_preference and (
            transcriber_actual_device != transcriber_device_preference):
        note = (f"requested {transcriber_device_preference}, loaded on "
                f"{transcriber_actual_device} (CUDA unavailable or insufficient VRAM;"
                " retried on every drop)")
    transcriber_note = note or None


def get_transcriber(device: str | None = None) -> Transcriber:
    global transcriber, transcriber_error, transcriber_device_preference
    device = device or DEVICE_PREFERENCE
    with transcriber_lock:
        if transcriber is not None and transcriber_device_preference == device:
            _sync_transcriber_state(transcriber)
            # "already loaded on cpu" is not "already loaded on cuda": keep the
            # degraded models only while the card genuinely cannot take us, so the
            # moment a neighbour's VRAM is freed the next drop gets CUDA back without
            # restarting the app -- and without paying a reload per drop meanwhile.
            if transcriber_actual_device == device or not gate_open(device, ASR["mode"])[0]:
                return transcriber
        _model_load_active.set()
        model_ready_event.clear()
        try:
            if transcriber is not None:
                _release_loaded_transcriber()
            # A failed load is retryable, e.g. after the user changes the device.
            transcriber_error = None
            transcriber = load_transcriber(
                ASR["mode"], device=device, lang=ASR["lang"],
                glossary=ASR["glossary"] or None,
            ).load()
            transcriber_device_preference = device
            _sync_transcriber_state(transcriber)
            return transcriber
        except Exception as exc:  # surface load failure to clients
            transcriber_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _model_load_active.clear()
            model_ready_event.set()


# --------------------------------------------------------------------------- jobs

from exports import render_exports
def _result_path(job_id: str) -> Path:
    return RESULTS_DIR / f"{job_id}.json"


def _save_result(job: dict) -> None:
    _result_path(job["id"]).write_text(
        json.dumps({"name": job.get("name"), "segments": job["segments"], "labels": job.get("labels") or {},
                    "vad_full": bool(job.get("vad_full")), "engine": job.get("engine"),
                    "corrections": job.get("corrections") or []}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    # Human-readable markdown copy saved alongside the machine-readable result;
    # rewritten on rename so the on-disk transcript always shows current names.
    _result_path(job["id"]).with_suffix(".md").write_text(render_exports(job)["md"], encoding="utf-8")


def _lookup_job(job_id: str) -> dict | None:
    """In-memory job; rehydrate a finished one from disk after a server restart."""
    job = jobs.get(job_id)
    if job is not None:
        return job
    if not re.fullmatch(r"[0-9a-f]{12}", job_id):
        return None
    p = _result_path(job_id)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    segs = data["segments"] if isinstance(data, dict) else data
    labels = data.get("labels", {}) if isinstance(data, dict) else {}
    vad_full = bool(data.get("vad_full")) if isinstance(data, dict) else False
    if not isinstance(segs, list):
        return None
    name = data.get("name") if isinstance(data, dict) else None
    if not name:
        src = [q for q in UPLOAD_DIR.glob(f"{job_id}.*") if not q.name.endswith(".16k.wav")]
        name = src[0].name[len(job_id) + 1:] if src else (f"{job_id}.wav" if (UPLOAD_DIR / f"{job_id}.16k.wav").is_file() else f"{job_id}.audio")
    job = {
        "id": job_id, "name": name, "status": "done", "stage": None, "error": None,
        "elapsed": None, "segments": segs, "labels": labels,
        "vad_full": vad_full,
        # provenance: which card answered, and the reason when not the asked-for one.
        # Dropping this lost the device line from every transcript after a restart.
        "engine": data.get("engine") if isinstance(data, dict) else None,
        "corrections": data.get("corrections") if isinstance(data, dict) else None,
        "duration": round(max((s.get("end", 0.0) for s in segs), default=0.0), 1),
        "speakers": sorted({s.get("speaker", "speaker_0") for s in segs}),
    }
    jobs[job_id] = job
    return job


def _sweep_uploads(keep, force: bool = False) -> None:
    """Delete upload/transcode scratch except files of queued/running jobs and `keep`.

    `keep` may be a Path, a list of Paths, or None. With force=True (shutdown),
    live jobs' files go too."""
    if keep is None:
        keep_names = set()
    elif isinstance(keep, (list, tuple, set)):
        keep_names = {str(p) for p in keep}
    else:
        keep_names = {str(keep)}
    live = set()
    if not force:
        with jobs_lock:
            for j in jobs.values():
                if j["status"] in ("queued", "running"):
                    live.add(str(j["id"]))
    for p in UPLOAD_DIR.iterdir():
        if str(p) in keep_names or p.name.split(".")[0] in live:
            continue
        p.unlink(missing_ok=True)


def _check_shutdown() -> None:
    if _shutdown.is_set():
        raise RuntimeError("Server is shutting down")


def run_job(job_id: str, src_path: Path, device: str | None, vad_full: bool = False) -> None:
    job = jobs[job_id]
    device = device or DEVICE_PREFERENCE
    wav = None
    model_wav = None
    with jobs_lock:
        if job_id in _queue:
            _queue.remove(job_id)
    try:
        t0 = time.time()
        if src_path.parent != UPLOAD_DIR:  # claimed from the persistent queue dir
            dst = UPLOAD_DIR / src_path.name
            shutil.move(str(src_path), dst)
            src_path = dst
        wav = src_path.with_name(src_path.stem + ".16k.wav")
        job["stage"] = "converting"
        with _convert_lock:
            prepare_audio(src_path, wav)
        # the uploaded file is single-use: consumed by the conversion
        src_path.unlink(missing_ok=True)
        _check_shutdown()
        with wave.open(str(wav)) as _w:
            job["duration"] = round(_w.getnframes() / _w.getframerate(), 1)
        # Wait for a shared model load (preload thread or a racing job) before
        # touching the GPU: two concurrent Transcriber loads OOM the laptop GPU.
        job["stage"] = "loading-models"
        if _model_load_active.is_set() and not model_ready_event.is_set():
            model_ready_event.wait(timeout=600)
        tr = get_transcriber(device)
        job["engine"] = tr.describe()
        _check_shutdown()
        with _gpu_lock:
            job["stage"] = "transcribing"
            total = job["duration"] or 1.0
            quantum = getattr(tr, "chunk_step_sec", 0.8)
            mapping = None
            vad_off = 0.0
            model_wav = str(wav)
            if vad_full:
                model_wav, mapping = splice_speech_regions(wav, quantum=quantum)
            elif VAD_TRIM:
                model_wav, vad_off = trim_edge_silence(wav, quantum=quantum)
            model_dur = (mapping[-1][0] + mapping[-1][1]) if mapping else total

            def _remap(partial):
                for p in partial:
                    p["start"] = round(map_spliced_time(mapping, p["start"]), 3)
                    p["end"] = round(map_spliced_time(mapping, p["end"]), 3)
                return partial

            def _publish(step, partial):
                if mapping:
                    _remap(partial)
                job["step"] = step
                job["partial"] = partial
                job["partial_speakers"] = sorted({p["speaker"] for p in partial})
                if mapping:  # spliced stream: remapped ends plateau, drive by step
                    job["progress"] = min(1.0, round(step * max(quantum, 0.05) / max(model_dur, 1.0), 3))
                else:
                    job["progress"] = min(1.0, round(max((p["end"] for p in partial), default=0.0) / total, 3))

            segs = tr.transcribe(model_wav, publish=_publish, offset=vad_off)
        if mapping:
            _remap(segs)
        _check_shutdown()
        # drop empty texts
        segs = [s for s in segs if s["text"]]
        job["labels"] = {}
        job.update(
            status="done",
            segments=segs,
            elapsed=round(time.time() - t0, 1),
            duration=round(max((s["end"] for s in segs), default=0.0), 1),
            speakers=sorted({s["speaker"] for s in segs}),
            stage=None,
            partial=None, partial_speakers=None, progress=None, step=None,
        )
        job["corrections"] = list(getattr(tr, "last_corrections", []))
        # The same provenance shape the watcher writes, so exports.py renders one line
        # for both paths and a downloaded .md says which card or cores answered.
        # The device comes from the loaded object, not from what was asked for, and
        # the reason travels with it: "on cpu" without it is untraceable.
        with transcriber_lock:
            _sync_transcriber_state(tr)
            actual, note = transcriber_actual_device, transcriber_note
        job["engine"] = {**(job.get("engine") or {}),
                         **getattr(tr, "describe", lambda: {})(),
                         "device": actual or str(getattr(tr, "device", "")),
                         "requested_device": device,
                         "repairs": len(job["corrections"])}
        if note and actual != device:
            job["engine"]["device_note"] = note
        _save_result(job)
        _sweep_uploads(wav)
        (QUEUE_DIR / f"{job_id}.json").unlink(missing_ok=True)
    except Exception as exc:
        traceback.print_exc()
        (QUEUE_DIR / f"{job_id}.json").unlink(missing_ok=True)
        if model_wav is not None and wav is not None and str(model_wav) != str(wav):
            Path(model_wav).unlink(missing_ok=True)
        if wav is not None:
            wav.unlink(missing_ok=True)
        src_path.unlink(missing_ok=True)
        job.update(status="error", error=f"{type(exc).__name__}: {exc}", stage=None, partial=None)


def _requeue_pending() -> None:
    """Re-enqueue jobs that were submitted but never claimed (server restart).

    A job whose queued file was already claimed (moved into uploads, which the
    boot sweep wipes) cannot be replayed — its leftover metadata is dropped and
    the client sees "Unknown job" and re-uploads.
    """
    items = []
    for meta in QUEUE_DIR.glob("*.json"):
        try:
            d = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta.unlink(missing_ok=True)
            continue
        jid = str(d.get("id", ""))
        if not re.fullmatch(r"[0-9a-f]{12}", jid):
            meta.unlink(missing_ok=True)
            continue
        src = QUEUE_DIR / str(d.get("src", ""))
        if not src.is_file():
            alt = UPLOAD_DIR / str(d.get("src", ""))
            src = alt if alt.is_file() else src
        if not src.is_file():
            meta.unlink(missing_ok=True)  # source vanished mid-flight
            continue
        items.append((float(d.get("submitted_at", 0.0)), jid, d, src))
    for _, jid, d, src in sorted(items):
        if jid in jobs:
            continue
        jobs[jid] = {"id": jid, "name": d.get("name", f"{jid}.audio"), "status": "queued",
                     "stage": None, "segments": None,
                     "vad_full": bool(d.get("vad_full"))}
        _queue.append(jid)
        gpu_executor.submit(run_job, jid, src, d.get("device"),
                            bool(d.get("vad_full")))
        print(f"[queue] requeued job {jid} ({jobs[jid]['name']})", flush=True)


# --------------------------------------------------------------------------- routes

SAFE_EXT = re.compile(r"^\.[a-z0-9]{1,5}$")


@app.post("/api/transcribe")
async def api_transcribe(request: Request):
    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "filename"):
        raise HTTPException(400, "No file uploaded (multipart field 'file').")
    name = Path(upload.filename or "upload").name
    ext = Path(name).suffix.lower()
    if not SAFE_EXT.match(ext or ""):
        ext = ".bin"
    job_id = uuid.uuid4().hex[:12]
    src = QUEUE_DIR / f"{job_id}{ext}"
    with src.open("wb") as fh:
        shutil.copyfileobj(upload.file, fh)
    if src.stat().st_size == 0:
        src.unlink(missing_ok=True)
        raise HTTPException(400, "Empty file.")

    vad_full = "vad_full" in form
    job = {"id": job_id, "name": name, "status": "queued", "stage": None,
           "segments": None, "vad_full": vad_full}
    with jobs_lock:
        device = DEVICE_PREFERENCE
        if len(_queue) >= MAX_QUEUED:
            src.unlink(missing_ok=True)
            raise HTTPException(429, f"Queue full ({MAX_QUEUED} jobs) — try again later.")
        jobs[job_id] = job
        _queue.append(job_id)
        pos = len(_queue) - 1
        (QUEUE_DIR / f"{job_id}.json").write_text(
            json.dumps({"id": job_id, "name": name, "src": src.name, "device": device,
                        "vad_full": vad_full,
                        "submitted_at": time.time()}, ensure_ascii=False),
            encoding="utf-8")
    gpu_executor.submit(run_job, job_id, src, device, vad_full)
    return {"job_id": job_id, "queue_pos": pos}


@app.get("/api/jobs/{job_id}")
async def api_job(job_id: str):
    job = _lookup_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    keep = {k: job.get(k) for k in ("id", "name", "status", "stage", "error", "elapsed", "duration", "speakers", "labels", "vad_full")}
    with jobs_lock:
        keep["queue_pos"] = _queue.index(job_id) if job_id in _queue else None
    keep["has_media"] = (UPLOAD_DIR / f"{job_id}.16k.wav").is_file() or any(
        UPLOAD_DIR.glob(f"{job_id}.*")
    )
    keep["progress"] = job.get("progress")
    keep["engine"] = job.get("engine")
    keep["glossary_fixes"] = len(job.get("corrections") or [])
    if job["status"] == "done":
        keep["segments"] = job["segments"]
    elif job.get("partial") is not None:
        keep["segments"] = job["partial"]
        keep["speakers"] = job.get("partial_speakers") or []
        keep["partial"] = True
    return keep

@app.get("/media/{job_id}")
async def api_media(job_id: str):
    if not re.fullmatch(r"[0-9a-f]{12}", job_id):
        raise HTTPException(400, "Bad id.")
    wav = UPLOAD_DIR / f"{job_id}.16k.wav"
    if wav.is_file():
        return FileResponse(wav)
    hits = [p for p in UPLOAD_DIR.glob(f"{job_id}.*") if not p.name.endswith(".16k.wav")]
    if not hits:
        raise HTTPException(404, "No media.")
    return FileResponse(hits[0])

@app.get("/api/export/{job_id}")
async def api_export(job_id: str, fmt: str = "txt"):
    job = _lookup_job(job_id)
    if job is None or job["status"] != "done":
        raise HTTPException(404, "Job not done.")
    exports = job.setdefault("exports", render_exports(job))
    if fmt not in exports:
        raise HTTPException(400, f"fmt must be one of {sorted(exports)}")
    stem = re.sub(r"[^\w.\- ]+", "_", Path(job["name"]).stem)[:60]
    return _raw(exports[fmt].encode("utf-8"), f"{stem}.{fmt}")


@app.post("/api/jobs/{job_id}/rename")
async def api_rename(job_id: str, request: Request):
    job = _lookup_job(job_id)
    if job is None or job["status"] != "done":
        raise HTTPException(404, "Job not done.")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body.")
    speaker = str(body.get("speaker", ""))
    label = re.sub(r"[\x00-\x1f\x7f]+", " ", str(body.get("label", "")))
    label = re.sub(r"\s+", " ", label).strip()[:64]
    if speaker not in (job.get("speakers") or []):
        raise HTTPException(400, "Unknown speaker.")
    with jobs_lock:
        labels = job.setdefault("labels", {})
        if label:
            labels[speaker] = label
        else:
            labels.pop(speaker, None)
        job.pop("exports", None)
        _save_result(job)
    return {"ok": True, "labels": job["labels"]}


def _raw(payload: bytes, filename: str):
    return Response(
        content=payload,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/session/heartbeat")
async def api_session_heartbeat():
    _mark_browser_seen()
    return {"ok": True}


@app.get("/api/settings")
async def api_settings():
    return {
        "device_preference": DEVICE_PREFERENCE,
        "actual_device": transcriber.device.type if transcriber is not None else None,
        "device_note": transcriber_note,
        "model_loaded": transcriber is not None,
    }


@app.post("/api/settings/device")
async def api_set_device(request: Request):
    global DEVICE_PREFERENCE
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body.")
    if not isinstance(body, dict):
        raise HTTPException(400, "JSON body must be an object.")
    device = str(body.get("device", "")).lower()
    if device not in {"cuda", "cpu"}:
        raise HTTPException(400, "device must be 'cuda' or 'cpu'.")
    if _model_load_active.is_set():
        raise HTTPException(409, "Models are currently loading; try again shortly.")
    with jobs_lock:
        if any(j.get("status") in ("queued", "running") for j in jobs.values()):
            raise HTTPException(409, "Wait for queued/running transcriptions before switching devices.")
        DEVICE_PREFERENCE = device
        _save_device_preference(device)
    try:
        tr = await asyncio.to_thread(get_transcriber, device)
    except Exception as exc:
        raise HTTPException(500, f"Could not load models on {device}: {exc}")
    return {"device_preference": device, "actual_device": tr.device.type,
            "device_note": transcriber_note, "model_loaded": True}


@app.get("/api/health")
async def api_health():
    import torch

    return {
        "model_loaded": transcriber is not None,
        "model_error": transcriber_error,
        "cuda": torch.cuda.is_available(),
        "device": transcriber.device.type if transcriber is not None else None,
        "device_preference": DEVICE_PREFERENCE,
        "device_note": transcriber_note,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "asr": (transcriber.describe() if transcriber is not None
                else {"engine": ASR["mode"], "language": ASR["lang"], "loaded": False}),
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    _mark_browser_seen()
    html = BASE_DIR / "static" / "index.html"
    return HTMLResponse(html.read_text(encoding="utf-8"))


def _open_browser_when_ready(
    url: str, server: "uvicorn.Server", wait_models: bool, timeout: float = 900.0
) -> None:
    """Open the UI in the default browser once the server is listening -- and,
    when models are being preloaded, once they are actually loaded (or the
    load failed, so the user sees the error banner instead of a dead page).
    """
    deadline = time.monotonic() + timeout
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.1)
    if wait_models:
        while (
            not model_ready_event.is_set()
            and not transcriber_error
            and time.monotonic() < deadline
        ):
            time.sleep(0.25)
    if time.monotonic() >= deadline:
        print(f"[app] browser launch skipped: startup unconfirmed after {timeout:.0f}s", flush=True)
        return
    try:
        if webbrowser.open_new(url):
            print(f"[app] opened {url} in your browser", flush=True)
        else:
            print(f"[app] no browser available - open {url} manually", flush=True)
    except Exception as exc:
        print(f"[app] browser launch failed ({exc}) - open {url} manually", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Speaker transcription web app")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default=None, help="cuda (default) or cpu")
    ap.add_argument("--asr", default=ASR["mode"],
                    help="transcription engine: live | fast | balanced | quality | german")
    ap.add_argument("--lang", default=ASR["lang"],
                    help="language for engines that require one (e.g. de); auto elsewhere")
    ap.add_argument("--glossary", default=ASR["glossary"],
                    help="file of canonical spellings to repair in the output")
    ap.add_argument("--preload", action="store_true", help="load models at startup")
    ap.add_argument("--idle-exit", type=float, default=0.0, metavar="SECONDS",
                    help="shut down after no browser heartbeat for this long and no jobs remain (0 disables)")
    ap.add_argument("--open-browser", action="store_true",
                    help="open the UI in the default browser once startup (and --preload "
                         "model loading) is finished")
    ap.add_argument("--no-vad-trim", action="store_true",
                    help="disable Silero VAD trimming of leading/trailing silence")
    args = ap.parse_args()

    global VAD_TRIM, DEVICE_PREFERENCE
    if args.no_vad_trim:
        VAD_TRIM = False
    if args.device:
        DEVICE_PREFERENCE = args.device
    ASR.update(mode=args.asr, lang=args.lang, glossary=args.glossary)

    _requeue_pending()

    if args.preload:
        threading.Thread(target=lambda: get_transcriber(DEVICE_PREFERENCE), daemon=True).start()

    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="warning")
    server = uvicorn.Server(config)

    browser_host = "127.0.0.1" if args.host in ("0.0.0.0", "::", "") else args.host
    url = f"http://{browser_host}:{args.port}"
    if args.open_browser:
        threading.Thread(
            target=_open_browser_when_ready,
            args=(url, server, args.preload),
            daemon=True,
        ).start()
    if args.idle_exit > 0:
        threading.Thread(
            target=_monitor_browser_idle,
            args=(server, args.idle_exit),
            daemon=True,
        ).start()
    print(f"[app] speaker transcript app on {url}", flush=True)

    server.run()


if __name__ == "__main__":
    main()
