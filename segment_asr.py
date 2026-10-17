"""Diarize-first transcription: the diarizer decides WHO and WHEN, an offline
ASR engine (:mod:`engines`) decides WHAT.

Second pipeline mode next to the coupled streaming path in :mod:`pipeline`. The
streaming path can only ever run a cache-aware streaming ASR (Nemotron 3.5), which
is a latency trade — 8.3 % WER on German FLEURS against 4.4–5.0 % for the offline
Parakeet/Canary checkpoints — and it fragments sentences at chunk boundaries
(this project's own transcripts: median 2.7–6.6 s between segment breaks, 11–26 %
of segments one or two words long). See ``docs/ASR-BACKENDS-PLAN.md``.

Flow
----
1. ``Nemotron-3-Diarization`` runs **offline** (30.4 s buffer configuration) and
   returns speaker-activity probabilities at 10 ms resolution.
2. Frames collapse to a *dominant speaker* track; contiguous runs become audio
   **cuts** (context-padded, dropout-absorbing, length-capped).
3. The engine transcribes the cuts in batches and returns timed words.
4. Words outside a cut's dominance window are dropped: that is what keeps a
   neighbour's leaking word out of your line (turn-taking overlap).
5. Identical words claimed by two speakers at the same instant go to the more
   dominant one; the rest are assembled into segments.

Speaker labels stay ``speaker_N``, ordered by first appearance *in the output*, so
a channel that produced no words never appears as an empty speaker. Progress is
published after every batch, so the UI keeps painting while the GPU works.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from engines import SAMPLE_RATE, Cut, NemoOfflineEngine, Word
from model_store import load_nemo_model

DIAR_MODEL_ID = "nvidia/Nemotron-3-Diarization"

# Nemotron-3-Diarization's documented "very high latency (offline)" configuration,
# in 80 ms encoder frames: 340 frames + 40 right context = a 30.4 s input buffer.
DIAR_OFFLINE = {
    "spkcache_len": 264,
    "fifo_len": 40,
    "chunk_len": 340,
    "chunk_right_context": 40,
    "spkcache_update_period": 300,
}


@dataclass
class SegCfg:
    """Cut geometry and segment assembly (defaults tuned for meeting audio)."""

    threshold: float = 0.5        # speaker-activity probability that counts as speaking
    fill_gap: float = 0.35        # silence inside one speaker's run that gets absorbed
    merge_gap: float = 0.6        # same-speaker runs closer than this share one cut
    min_cut: float = 0.35         # shorter runs are not worth an ASR call
    max_cut: float = 30.0         # longer runs are split evenly (keeps batches cheap)
    pad: float = 0.15             # context around a cut that is *not* attributed to it
    sent_break: float = 0.8       # pause that ends a transcript segment
    soft_break: float = 0.3       # shorter pause, but only after end-of-sentence punctuation
    max_words: int = 90           # hard cap per segment
    batch: int = 16               # cuts per ASR call = publish cadence
    words_per_sec: float = 30.0   # above this a cut counts as a hallucination burst
    repeat_collapse: int = 5      # >= this many identical words in a row are one word
    dedupe_tol: float = 0.35      # time window for cross-speaker duplicate words
    max_speakers: int = 8


@dataclass
class _Run:
    """One speaker's contiguous stretch of the timeline."""

    speaker: int
    t0: float
    t1: float


class DiarizeFirst:
    """``Transcriber``-compatible: ``transcribe(wav_path, publish=…, offset=…)``."""

    #: callers floor VAD cut points to this grid; this mode has no streaming encoder
    #: to stay aligned with, and ``transcribe`` publishes *seconds processed* as its
    #: step, so ``step * chunk_step_sec / duration`` remains the right progress.
    chunk_step_sec = 1.0

    def __init__(
        self,
        engine: NemoOfflineEngine,
        *,
        diar_model: str = DIAR_MODEL_ID,
        device: str = "cpu",
        cfg: SegCfg | None = None,
        diar_offline: bool = True,
    ) -> None:
        self.engine = engine
        self.diar_model_id = diar_model
        self.device = device
        self.cfg = cfg or SegCfg()
        self.diar_offline = diar_offline
        self.diar_model = None
        self._probs: np.ndarray | None = None

    # ------------------------------------------------------------ lifecycle
    def load(self):
        import torch
        from nemo.collections.asr.models import SortformerEncLabelModel

        if self.diar_model is None:
            model = load_nemo_model(self.diar_model_id, SortformerEncLabelModel)
            self.diar_model = model.to(device=torch.device(self.device)).eval()
            if torch.device(self.device).type == "cuda":
                from gpu_features import BoundedMelFeatures
                preprocessor = self.diar_model.preprocessor
                preprocessor.featurizer = BoundedMelFeatures(preprocessor.featurizer)
            self.frame_len = float(model.cfg.preprocessor.get("window_stride", self.frame_len))
            if self.diar_offline:
                modules = self.diar_model.sortformer_modules
                for attr, value in DIAR_OFFLINE.items():
                    if hasattr(modules, attr):
                        setattr(modules, attr, value)
                self.diar_model._check_streaming_parameters()
        self.engine.load()
        return self

    def unload(self) -> None:
        self.engine.unload()
        self.diar_model = None

    def describe(self) -> dict:
        info = dict(self.engine.describe())
        info.update(
            pipeline="diarize-first",
            diarization=self.diar_model_id,
            diarization_mode="offline (30.4 s buffer)" if self.diar_offline else "checkpoint defaults",
        )
        return info

    #: Speaker index → display number, fixed by the diarization before any ASR runs.
    _label_order: dict[int, int] = {}

    #: Seconds per diarization output frame. Read from the checkpoint at load();
    #: the default is this model's own value (10 ms — the label grid is upsampled
    #: from the 80 ms encoder frames), so the geometry stays honest without weights.
    frame_len = 0.01

    # ------------------------------------------------------------- main path
    def transcribe(self, wav_path: str, publish: Callable | None = None, offset: float = 0.0) -> list[dict]:
        import soundfile as sf

        self.load()
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            raise ValueError(f"diarize-first expects {SAMPLE_RATE} Hz mono wav, got {sr} Hz from {wav_path}")

        self._probs = self._activity(audio)
        try:
            runs = self._runs(self._probs)
            # Speaker numbering is decided by the diarization alone, before any ASR:
            # a partial publish must not renumber someone halfway through the file
            # just because a long cut happened to be transcribed first.
            self._label_order = {}
            for run in sorted(runs, key=lambda r: r.t0):
                self._label_order.setdefault(run.speaker, len(self._label_order))
            cuts = self._cuts(audio, runs)
            if os.environ.get("NEMODIAR_DEBUG"):
                speech = float((self._probs > self.cfg.threshold).any(axis=1).mean()) * len(audio) / SAMPLE_RATE
                print(
                    f"[debug] {len(self._probs)} frames × {self._probs.shape[1]} spk, "
                    f"{speech:.1f}s speech, {len(runs)} runs, {len(cuts)} cuts",
                    flush=True,
                )
            if not cuts:
                return []
            words_by_run: dict[int, list[Word]] = {index: [] for index in range(len(runs))}
            processed = 0.0
            for batch in _chunks(cuts, self.cfg.batch):
                for (cut, run_index), words in zip(batch, self.engine.transcribe([c for c, _ in batch])):
                    if os.environ.get("NEMODIAR_DEBUG"):
                        print(
                            f"[debug] cut spk{cut.speaker} {cut.t0:7.2f}-{cut.end:7.2f} "
                            f"({cut.duration:5.2f}s) -> {len(words)} words",
                            flush=True,
                        )
                    words_by_run[run_index].extend(self._guard(cut, words))
                processed = max(cut.end for cut, _ in batch)
                if publish is not None:
                    publish(int(processed), self._finalize(words_by_run, offset))
            return self._finalize(words_by_run, offset)
        finally:
            self._probs = None

    # ------------------------------------------------- 1. speaker activity
    def _activity(self, audio: np.ndarray) -> np.ndarray:
        """Speaker-activity probabilities as ``[frames, speakers]``."""
        _, probs = self.diar_model.diarize(
            audio=[np.asarray(audio, dtype=np.float32)],
            batch_size=1,
            sample_rate=SAMPLE_RATE,
            include_tensor_outputs=True,
            verbose=False,
        )
        p = probs[0]
        if p.ndim == 3:
            p = p[0]
        p = p.detach().float().cpu().numpy() if hasattr(p, "detach") else np.asarray(p, dtype=np.float32)
        return p[:, : self.cfg.max_speakers]

    # -------------------------------------------- 2. dominance → speaker runs
    def _runs(self, p: np.ndarray) -> list[_Run]:
        if p.size == 0:
            return []
        fl = self.frame_len
        frames = p.shape[0]
        active = p > self.cfg.threshold
        dominant = np.where(active.any(axis=1), np.argmax(p, axis=1), -1)

        # Absorb dropouts shorter than fill_gap that sit inside one speaker's run:
        # without this, one clipped syllable splits a sentence into two cuts.
        fill = max(1, int(round(self.cfg.fill_gap / fl)))
        for i in range(1, frames - 1):
            if dominant[i] != -1:
                continue
            back = next((dominant[j] for j in range(i - 1, max(-1, i - fill - 1), -1) if dominant[j] >= 0), -1)
            if back < 0:
                continue
            ahead = next((dominant[j] for j in range(i + 1, min(frames, i + fill + 1)) if dominant[j] >= 0), -1)
            if ahead == back:
                dominant[i] = back

        runs: list[_Run] = []
        start = 0
        for i in range(1, frames + 1):
            if i == frames or dominant[i] != dominant[start]:
                if dominant[start] >= 0:
                    runs.append(_Run(int(dominant[start]), start * fl, i * fl))
                start = i

        merged: list[_Run] = []
        for run in runs:
            if merged and merged[-1].speaker == run.speaker and run.t0 - merged[-1].t1 <= self.cfg.merge_gap:
                merged[-1].t1 = run.t1
            else:
                merged.append(run)

        out: list[_Run] = []
        for run in merged:
            if run.t1 - run.t0 < self.cfg.min_cut:
                continue
            if run.t1 - run.t0 <= self.cfg.max_cut:
                out.append(run)
                continue
            parts = int(np.ceil((run.t1 - run.t0) / self.cfg.max_cut))
            width = (run.t1 - run.t0) / parts
            out.extend(_Run(run.speaker, run.t0 + k * width, run.t0 + (k + 1) * width) for k in range(parts))
        return out

    def _cuts(self, audio: np.ndarray, runs: Sequence[_Run]) -> list[tuple[Cut, int]]:
        duration = len(audio) / SAMPLE_RATE
        cuts: list[tuple[Cut, int]] = []
        for index, run in enumerate(runs):
            a = max(0.0, run.t0 - self.cfg.pad)
            b = min(duration, run.t1 + self.cfg.pad)
            i0, i1 = int(a * SAMPLE_RATE), int(b * SAMPLE_RATE)
            if i1 - i0 < 160:
                continue
            audio_slice = np.ascontiguousarray(audio[i0:i1])
            cuts.append((Cut(audio_slice, a, run.speaker, (run.t0, run.t1)), index))
        return cuts

    # ---------------------------------------------------------- 3. safeguards
    def _guard(self, cut: Cut, words) -> list[Word]:
        """Drop hallucination bursts and collapse repeat loops before they reach the text."""
        words = list(words)
        if not words:
            return []
        if len(words) / max(cut.duration, 0.05) > self.cfg.words_per_sec:
            return []
        keep: list[Word] = []
        repeat = 0
        for word in words:
            if keep and _norm(word.text) == _norm(keep[-1].text) and _norm(word.text):
                repeat += 1
                if repeat >= self.cfg.repeat_collapse:
                    continue
            else:
                repeat = 1
            keep.append(word)
        punctuated = [w for w in keep if not _PUNCT_ONLY.match(w.text)]
        return punctuated or keep

    # ----------------------------------------------- 4/5. dedupe and assembly
    def _dominance(self, speaker: int, t0: float, t1: float) -> float:
        """Own probability minus the best rival's, averaged over the word's span."""
        p = self._probs
        if p is None or p.size == 0 or speaker >= p.shape[1]:
            return 0.0
        fl = self.frame_len
        i0 = max(0, int(t0 / fl))
        i1 = min(p.shape[0], max(i0 + 1, int(t1 / fl)))
        window = p[i0:i1]
        if window.shape[0] == 0:
            return 0.0
        rivals = np.delete(window, speaker, axis=1)
        return float(window[:, speaker].mean() - (rivals.max(axis=1).mean() if rivals.size else 0.0))

    def _finalize(self, words_by_run: dict[int, list[Word]], offset: float) -> list[dict]:
        words = _dedupe([w for group in words_by_run.values() for w in group], self.cfg, self._dominance)
        if not words:
            return []
        segments: list[tuple[str, float, float, int]] = []
        for speaker in sorted({w.speaker for w in words}):
            timeline = sorted((w for w in words if w.speaker == speaker), key=lambda w: w.start)
            segments.extend(
                (text, t0, t1, speaker) for text, t0, t1 in _sentences(timeline, self.cfg) if text.strip()
            )
        segments.sort(key=lambda s: (s[1], s[2]))
        order: dict[int, int] = dict(self._label_order)
        for _, _, _, speaker in segments:
            order.setdefault(speaker, len(order))
        return [
            {
                "text": _tidy(text),
                "start": round(t0 + offset, 3),
                "end": round(max(t1, t0 + 0.02) + offset, 3),
                "speaker": f"speaker_{order[speaker]}",
            }
            for text, t0, t1, speaker in segments
        ]


# --------------------------------------------------------------------------- helpers


def _chunks(items: Sequence, size: int):
    for i in range(0, len(items), max(1, size)):
        yield list(items[i : i + size])


_PUNCT_ONLY = re.compile(r"^\W+$", re.UNICODE)
_WORD_STRIP = re.compile(r"[^\w]+", re.UNICODE)


def _norm(text: str) -> str:
    return _WORD_STRIP.sub("", text.lower())


def _dedupe(words: list[Word], cfg: SegCfg, dominance: Callable[[int, float, float], float]) -> list[Word]:
    """One word spoken once but transcribed by two overlapping cuts → the dominant speaker keeps it."""
    if len(words) < 2:
        return words
    timeline = sorted(words, key=lambda w: (w.start, w.speaker))
    dropped: set[int] = set()
    for i, word in enumerate(timeline):
        if i in dropped or not _norm(word.text):
            continue
        for j in range(i + 1, len(timeline)):
            other = timeline[j]
            if other.start - word.start > cfg.dedupe_tol:
                break
            if j in dropped or other.speaker == word.speaker or _norm(other.text) != _norm(word.text):
                continue
            if dominance(word.speaker, word.start, word.end) >= dominance(other.speaker, other.start, other.end):
                dropped.add(j)
            else:
                dropped.add(i)
                break
    return [w for i, w in enumerate(timeline) if i not in dropped]


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?", "…"))


def _sentences(timeline: Sequence[Word], cfg: SegCfg):
    """Group one speaker's words into segments by pause, punctuation and length."""
    buf: list[Word] = []
    for word in timeline:
        if buf:
            gap = word.start - buf[-1].end
            if (
                gap > cfg.sent_break
                or (gap > cfg.soft_break and _ends_sentence(buf[-1].text))
                or len(buf) >= cfg.max_words
            ):
                yield _flush(buf)
                buf = []
        buf.append(word)
    if buf:
        yield _flush(buf)


def _flush(buf: list[Word]) -> tuple[str, float, float]:
    return " ".join(w.text for w in buf), buf[0].start, max(w.end for w in buf)


_SPACE_BEFORE = re.compile(r"\s+([,.;:!?„“»')%])")
_SPACE_AFTER = re.compile(r"([(«\"'])(\S)")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def _tidy(text: str) -> str:
    """Close up punctuation spacing so the transcript reads like text, not tokens."""
    text = _SPACE_BEFORE.sub(r"\1", text)
    text = _SPACE_AFTER.sub(r"\1\2", text)
    return _MULTI_SPACE.sub(" ", text).strip()
