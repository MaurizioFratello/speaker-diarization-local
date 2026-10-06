"""Local speaker-transcription web app.

Drop an audio/video file in the browser, get a speaker-attributed transcript
("who said what, when"), rendered on a timeline with exports.
Everything runs locally on GPU via NeMo (Nemotron-3-Diarization +
Nemotron-3.5-ASR). Start with:  python app.py  [--port 8765] [--device cuda]
"""

from __future__ import annotations

import argparse
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
    map_spliced_time,
    prepare_audio,
    splice_speech_regions,
    trim_edge_silence,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
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


app = FastAPI(title="Speaker Transcription", lifespan=_lifespan)

transcriber: Transcriber | None = None
transcriber_error: str | None = None
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

def get_transcriber(device: str | None) -> Transcriber:
    global transcriber, transcriber_error
    with transcriber_lock:
        if transcriber_error:
            raise RuntimeError(transcriber_error)
        if transcriber is None:
            _model_load_active.set()
            try:
                transcriber = Transcriber(device=device)
            except Exception as exc:  # surface load failure to clients
                transcriber_error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                model_ready_event.set()
        elif device is not None and transcriber.device.type != device:
            raise HTTPException(
                409,
                f"Models are already loaded on '{transcriber.device.type}'; "
                f"restart the server to switch to '{device}'.",
            )
        return transcriber


# --------------------------------------------------------------------------- jobs

from exports import render_exports, spk_display
def _result_path(job_id: str) -> Path:
    return RESULTS_DIR / f"{job_id}.json"


def _save_result(job: dict) -> None:
    _result_path(job["id"]).write_text(
        json.dumps({"name": job.get("name"), "segments": job["segments"], "labels": job.get("labels") or {},
                    "vad_full": bool(job.get("vad_full"))}, ensure_ascii=False, indent=1),
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

    device = None
    if "cpu" in form:
        device = "cpu"
    vad_full = "vad_full" in form
    job = {"id": job_id, "name": name, "status": "queued", "stage": None,
           "segments": None, "vad_full": vad_full}
    with jobs_lock:
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


@app.get("/api/health")
async def api_health():
    import torch

    return {
        "model_loaded": transcriber is not None,
        "model_error": transcriber_error,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.get("/", response_class=HTMLResponse)
async def index():
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
    ap.add_argument("--preload", action="store_true", help="load models at startup")
    ap.add_argument("--open-browser", action="store_true",
                    help="open the UI in the default browser once startup (and --preload "
                         "model loading) is finished")
    ap.add_argument("--no-vad-trim", action="store_true",
                    help="disable Silero VAD trimming of leading/trailing silence")
    args = ap.parse_args()

    global VAD_TRIM
    if args.no_vad_trim:
        VAD_TRIM = False

    _requeue_pending()

    if args.preload:
        threading.Thread(target=lambda: get_transcriber(args.device), daemon=True).start()

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
    print(f"[app] speaker transcript app on {url}", flush=True)

    server.run()


if __name__ == "__main__":
    main()
