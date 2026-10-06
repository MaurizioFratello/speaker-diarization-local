"""Transcript renderers, shared by the web app and the folder watcher.

Deliberately free of NeMo/torch imports: the watcher restores a sidecar on an
idle tick and renders transcripts after a run, and pulling the GPU stack in for
that would make a 3-minute timer load hundreds of megabytes of CUDA-capable
libraries and risk touching the card. Keep this module stdlib-only; app.py
imports these names, so the browser exports are unchanged.
"""
from __future__ import annotations

import json
import re


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
