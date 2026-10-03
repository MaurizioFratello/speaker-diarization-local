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
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response

from pipeline import Transcriber, prepare_audio

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


# --------------------------------------------------------------------------- models

def get_transcriber(device: str | None) -> Transcriber:
    global transcriber, transcriber_error
    with transcriber_lock:
        if transcriber_error:
            raise RuntimeError(transcriber_error)
        if transcriber is None:
            try:
                transcriber = Transcriber(device=device)
            except Exception as exc:  # surface load failure to clients
                transcriber_error = f"{type(exc).__name__}: {exc}"
                raise
        elif device is not None and transcriber.device.type != device:
            raise HTTPException(
                409,
                f"Models are already loaded on '{transcriber.device.type}'; "
                f"restart the server to switch to '{device}'.",
            )
        return transcriber


# --------------------------------------------------------------------------- jobs

def fmt_ts(t: float, sep: str = ",") -> str:
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{int(s):02d}{sep}{int((s % 1) * 1000):03d}"


def spk_display(job: dict, speaker) -> str:
    lab = (job.get("labels") or {}).get(str(speaker))
    if lab:
        return lab
    return re.sub(r"^(?:speaker|spk)[_ ]", "Speaker ", str(speaker))


def render_exports(job: dict) -> dict[str, str]:
    segs = job["segments"]
    name = job["name"]
    def spk_name(s): return spk_display(job, s)

    txt = "\n".join(
        f"[{fmt_ts(s['start'], '.')} {spk_name(s['speaker'])}] {s['text']}" for s in segs
    )
    md = [f"# Transcript — {name}", ""]
    for s in segs:
        md.append(f"**{spk_name(s['speaker'])}** `[{fmt_ts(s['start'], '.')}]`  \n{s['text']}\n")
    srt = []
    for i, s in enumerate(segs, 1):
        srt.append(f"{i}\n{fmt_ts(s['start'])} --> {fmt_ts(s['end'])}\n{spk_name(s['speaker'])}: {s['text']}\n")
    vtt = ["WEBVTT\n"] + [
        f"{i}\n{fmt_ts(s['start'], '.')} --> {fmt_ts(s['end'], '.')}\n{spk_name(s['speaker'])}: {s['text']}\n"
        for i, s in enumerate(segs, 1)
    ]
    return {
        "txt": txt,
        "md": "\n".join(md),
        "json": json.dumps(
            [{**s, "speaker_name": spk_display(job, s["speaker"])} for s in segs],
            ensure_ascii=False, indent=1,
        ),
        "srt": "\n".join(srt),
        "vtt": "\n".join(vtt),
    }


def _result_path(job_id: str) -> Path:
    return RESULTS_DIR / f"{job_id}.json"


def _save_result(job: dict) -> None:
    _result_path(job["id"]).write_text(
        json.dumps({"name": job.get("name"), "segments": job["segments"], "labels": job.get("labels") or {}}, ensure_ascii=False, indent=1),
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
    if not isinstance(segs, list):
        return None
    name = data.get("name") if isinstance(data, dict) else None
    if not name:
        src = [q for q in UPLOAD_DIR.glob(f"{job_id}.*") if not q.name.endswith(".16k.wav")]
        name = src[0].name[len(job_id) + 1:] if src else (f"{job_id}.wav" if (UPLOAD_DIR / f"{job_id}.16k.wav").is_file() else f"{job_id}.audio")
    job = {
        "id": job_id, "name": name, "status": "done", "stage": None, "error": None,
        "elapsed": None, "segments": segs, "labels": labels,
        "duration": round(max((s.get("end", 0.0) for s in segs), default=0.0), 1),
        "speakers": sorted({s.get("speaker", "speaker_0") for s in segs}),
    }
    jobs[job_id] = job
    return job


def _sweep_uploads(keep: Path | None, force: bool = False) -> None:
    """Delete upload/transcode scratch except files of queued/running jobs and `keep`.

    With force=True (shutdown), live jobs' files go too."""
    live = set()
    if not force:
        with jobs_lock:
            for j in jobs.values():
                if j["status"] in ("queued", "running"):
                    live.add(str(j["id"]))
    for p in UPLOAD_DIR.iterdir():
        if p == keep or p.name.split(".")[0] in live:
            continue
        p.unlink(missing_ok=True)


def _check_shutdown() -> None:
    if _shutdown.is_set():
        raise RuntimeError("Server is shutting down")


def run_job(job_id: str, src_path: Path, device: str | None) -> None:
    job = jobs[job_id]
    try:
        t0 = time.time()
        wav = src_path.with_name(src_path.stem + ".16k.wav")
        job["stage"] = "converting"
        prepare_audio(src_path, wav)
        # the uploaded file is single-use: consumed by the conversion
        src_path.unlink(missing_ok=True)
        _check_shutdown()
        job["stage"] = "loading-models"
        tr = get_transcriber(device)
        _check_shutdown()
        job["stage"] = "transcribing"
        segs = tr.transcribe(wav)
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
        )
        _save_result(job)
        _sweep_uploads(wav)
    except Exception as exc:
        traceback.print_exc()
        wav.unlink(missing_ok=True)
        src_path.unlink(missing_ok=True)
        job.update(status="error", error=f"{type(exc).__name__}: {exc}", stage=None)


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
    src = UPLOAD_DIR / f"{job_id}{ext}"
    with src.open("wb") as fh:
        shutil.copyfileobj(upload.file, fh)
    if src.stat().st_size == 0:
        src.unlink(missing_ok=True)
        raise HTTPException(400, "Empty file.")

    device = None
    if "cpu" in form:
        device = "cpu"
    job = {"id": job_id, "name": name, "status": "queued", "stage": None, "segments": None}
    with jobs_lock:
        jobs[job_id] = job
    gpu_executor.submit(run_job, job_id, src, device)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def api_job(job_id: str):
    job = _lookup_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    keep = {k: job.get(k) for k in ("id", "name", "status", "stage", "error", "elapsed", "duration", "speakers", "labels")}
    if job["status"] == "done":
        keep["segments"] = job["segments"]
        keep["has_media"] = (UPLOAD_DIR / f"{job_id}.16k.wav").is_file() or any(
            UPLOAD_DIR.glob(f"{job_id}.*")
        )
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Speaker transcription web app")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default=None, help="cuda (default) or cpu")
    ap.add_argument("--preload", action="store_true", help="load models at startup")
    args = ap.parse_args()

    if args.preload:
        threading.Thread(target=lambda: get_transcriber(args.device), daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
