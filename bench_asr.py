#!/usr/bin/env python
"""Benchmark the ASR engines: time, memory, fragmentation, and transcript dumps.

    python bench_asr.py MEDIA --configs balanced,quality --lang de --out bench

One subprocess per config (own process = an honest peak-VRAM reading, and a
crash in one engine cannot poison the others). For each config it records

  wall_s    load + transcribe, whole file
  rtf       wall / audio duration   (<1 = faster than real time)
  peak_gib  high-water mark of PyTorch's allocator on the device used
  segs/words, and how the text is chopped up: mean words per segment and the
          share of segments with <= 2 words (what the app renders as one-line
          fragments; the number to watch when comparing against `live`)

and writes <out>/<config>.json (the segment list), <out>/<config>.txt (plain
transcript for reading) and <out>/summary.json (+ a table on stdout).

Quality (1-7) is assigned by a human reading the transcripts against a
ground-truth correction — see --groundtruth for the objective part: character
error rate over the ground-truth window, using jiwer when installed.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = os.environ.get("BENCH_PYTHON", str(ROOT / ".venv" / "bin" / "python"))

#: (config, device, note) rows for the default sweep
SWEEP = [
    ("live", "cpu", "current fallback: coupled streaming path on CPU"),
    ("live", "cuda", "current default on GPU"),
    ("fast", "cuda", "diarize-first, Nemotron weights, no download"),
    ("fast", "cpu", "diarize-first, Nemotron weights"),
    ("balanced", "cuda", "diarize-first, Parakeet TDT v3"),
    ("balanced", "cpu", "diarize-first, Parakeet TDT v3"),
    ("quality", "cuda", "diarize-first, Canary 1B v2"),
    ("german", "cuda", "diarize-first, German FastConformer hybrid"),
]


def _wait_for_gpu(cfg: str, seconds: float) -> float:
    """Poll until the card has room for this engine; return the GiB free at exit."""
    from engines import vram_floor_gib
    need = vram_floor_gib(cfg)
    t0 = time.time()
    free = 0.0
    while time.time() - t0 < seconds:
        try:
            import torch
            free = torch.cuda.mem_get_info()[0] / 2**30 if torch.cuda.is_available() else 0.0
        except Exception:
            free = 0.0
        if free >= need:
            return free
        print(f"    …{free:.1f} GiB free, {need:.1f} needed — waiting", flush=True)
        time.sleep(20)
    return free


def _child(cfg: str, device: str, media: str, lang: str, workdir: Path,
           self_glossary: str = "") -> dict:
    """Run one config in a subprocess; return its measurement row."""
    script = r"""
import json, os, sys, time, torch
sys.path.insert(0, os.getcwd())
import soundfile as sf
from pathlib import Path
from pipeline import load_transcriber, prepare_audio, trim_edge_silence
t0 = time.time()
tr = load_transcriber(sys.argv[1], device=sys.argv[2], lang=sys.argv[4],
                      glossary=sys.argv[6] or None).load()
load = time.time() - t0
wav = prepare_audio(sys.argv[3], Path(sys.argv[5]) / (Path(sys.argv[3]).stem + ".16k.wav"))
audio_s = len(sf.read(wav, dtype="float32")[0]) / 16000
work, off = trim_edge_silence(wav)
t1 = time.time()
segs = tr.transcribe(work, offset=off)
wall = time.time() - t1
peak = torch.cuda.max_memory_allocated(tr.device) / 2**30 if tr.device.type == "cuda" else 0.0
free = torch.cuda.mem_get_info()[0] / 2**30 if tr.device.type == "cuda" else 0.0
json.dump({"segments": segs, "load_s": load, "wall_s": wall, "peak_gib": peak,
           "device_used": tr.device.type,
           "free_gib_after": free, "describe": tr.describe(), "audio_s": audio_s},
          open(Path(sys.argv[5]) / "row.json", "w"))
"""
    workdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    if device == "cpu":
        # Hide the card for CPU rows. Not tidiness: on a card somebody else has
        # filled, a CPU run can still die in cuDevicePrintCtxRetain ("CUDA error: out
        # of memory") because torch.compile/NeMo probe the device once — and that
        # error arrives inside dynamo, where no load-time fallback can catch it.
        env["CUDA_VISIBLE_DEVICES"] = ""
    t0 = time.time()
    proc = subprocess.run(
        [PYTHON, "-c", script, cfg, device, media, lang, str(workdir), self_glossary],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    row_file = workdir / "row.json"
    if proc.returncode != 0 or not row_file.is_file():
        tail = [l for l in (proc.stderr or "").splitlines() if l.strip()][-6:]
        return {"config": cfg, "device": device, "device_used": "-",
                "error": " | ".join(tail)[-600:], "note": NOTE.get((cfg, device), ""),
                "wall_s": time.time() - t0}
    row = json.loads(row_file.read_text())
    row.update(config=cfg, device=device, total_s=time.time() - t0)
    return row


NOTE = {(c, d): n for c, d, n in SWEEP}   # keyed by the pair: the notes are device-specific


def _stats(row: dict) -> dict:
    segs = row.get("segments") or []
    words = [len(s["text"].split()) for s in segs]
    return {
        "config": row["config"], "device": row["device"],
        "device_used": row.get("device_used", row["device"]),
        "audio_s": round(row.get("audio_s", 0), 1),
        "load_s": round(row.get("load_s", 0), 1),
        "wall_s": round(row.get("wall_s", 0), 1),
        "rtf": round(row["wall_s"] / max(row.get("audio_s", 1), 1), 3),
        "peak_gib": round(row.get("peak_gib", 0), 2),
        "segs": len(segs),
        "words": sum(words),
        "words_per_seg": round(statistics.mean(words), 1) if words else 0,
        "pct_short": round(100 * sum(1 for w in words if w <= 2) / len(words)) if words else 0,
        "speakers": len({s["speaker"] for s in segs}),
        "note": NOTE.get((row["config"], row["device"]), ""),
    }


def _cer(row: dict, truth_path: Path | None) -> float | None:
    """Character error rate over the ground-truth window (jiwer if present)."""
    if not truth_path or not truth_path.is_file():
        return None
    truth = json.loads(truth_path.read_text())
    ref = truth["text"]
    win = (truth["start"], truth["end"])
    # Overlap, not containment: engines disagree about where a segment starts (the
    # diarization cuts differ by a few hundred ms), and a containment rule would
    # silently delete a sentence from whichever engine has a boundary there.
    got = " ".join(
        s["text"] for s in (row.get("segments") or [])
        if s["start"] < win[1] and s["end"] > win[0]
    )
    def norm(t: str) -> str:
        return " ".join(t.lower().replace("—", " ").replace("–", " ").split())
    try:
        from jiwer import cer
        return round(cer(norm(ref), norm(got)), 4)
    except ImportError:
        import difflib
        ref, got = norm(ref), norm(got)
        # autojunk=False matters: with the default, characters appearing in >1% of
        # a long string are treated as junk, and CER came out at 0.7 for text that
        # was 6% wrong.
        return round(1 - difflib.SequenceMatcher(None, ref, got, autojunk=False).ratio(), 4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("media")
    ap.add_argument("--rows", default=",".join(f"{c}:{d}" for c, d, _ in SWEEP),
                    help="comma-separated config:device pairs (default: the whole sweep)")
    ap.add_argument("--lang", default="de", help="language for engines that require one")
    ap.add_argument("--out", default="bench")
    ap.add_argument("--glossary", default="", help="apply this glossary in every config")
    ap.add_argument("--wait-gpu", type=float, default=0.0,
                    help="seconds to wait for the card to have room before a cuda row")
    ap.add_argument("--groundtruth", default=None,
                    help="JSON {start,end,text}: adds a CER column")
    ap.add_argument("--existing", action="store_true",
                    help="only re-summarise rows already in --out")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = []
    for spec in args.rows.split(","):
        cfg, _, device = spec.partition(":")
        pairs.append((cfg.strip(), (device or "cuda").strip()))

    rows = []
    for cfg, device in pairs:
        # Hand the language only to engines that require one: live/fast/balanced
        # detect per utterance, and forcing German on them would lose mixed English
        # terms and measure a configuration nobody would deploy.
        try:
            from engines import resolve_spec
            row_lang = args.lang if not resolve_spec(cfg).native_auto else "auto"
        except Exception:
            row_lang = args.lang
        d = out / f"{cfg}_{device}"
        if args.existing and not (d / "row.json").is_file():
            continue
        print(f"=== {cfg} on {device} ...", flush=True)
        if device == "cuda" and args.wait_gpu:
            print(f"    card: {time.strftime('%H:%M:%S')}", flush=True)
            _wait_for_gpu(cfg, args.wait_gpu)
        row = (json.loads((d / "row.json").read_text()) if (d / "row.json").is_file()
               else _child(cfg, device, args.media, row_lang, d,
                           str(ROOT / args.glossary) if args.glossary else ""))
        (d / "row.json").write_text(json.dumps(row))
        segs = row.get("segments") or []
        (d / "transcript.txt").write_text("\n\n".join(
            f"[{s['start']:8.2f}] {s['speaker']}\n{s['text']}" for s in segs))
        st = _stats(row)
        st["cer"] = _cer(row, Path(args.groundtruth) if args.groundtruth else None)
        if row.get("error"):
            st["error"] = row["error"]
        rows.append(st)
        print(json.dumps(st), flush=True)

    (out / "summary.json").write_text(json.dumps(rows, indent=1))
    cols = ("config", "device", "device_used", "audio_s", "load_s", "wall_s", "rtf",
            "peak_gib", "segs", "words", "words_per_seg", "pct_short", "cer", "speakers")
    print("\n" + "  ".join(f"{c[:9]:>9}" for c in cols))
    for r in rows:
        print("  ".join(
            f"{('-' if r.get(c) is None else r.get(c, '')):>9.2f}" if isinstance(r.get(c), float)
            else f"{str(r.get(c, '-')):>9}" for c in cols),
            "  <-- FAILED" if r.get("error") else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
