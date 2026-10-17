"""ASR engines for the diarize-first pipeline: audio cuts in, timed words out.

An engine knows nothing about speakers — attribution, overlap filtering and
sentence assembly live in :mod:`segment_asr`.  Everything here is an *offline*
(full-context) checkpoint driven through NeMo, which is exactly why it beats the
streaming model on sentence-level accuracy and why it cannot ride the coupled
multitalker rail (see ``docs/ASR-BACKENDS-PLAN.md`` §1).

Tiers (the ``--asr`` values the app and the watcher accept)::

    live        coupled streaming path (pipeline.py, unchanged behaviour)
    fast        nvidia/nemotron-3.5-asr-streaming-0.6b decoded offline — no download
    balanced    nvidia/parakeet-tdt-0.6b-v3     (2.5 GB, 25 langs, native language detection)
    quality     nvidia/canary-1b-v2             (6.1 GB, 25 langs, needs a language)
    german      nvidia/stt_de_fastconformer_hybrid_large_pc (German-only, strong prior)

Local mirrors (``models/<name>.nemo``) are honoured by :mod:`model_store`, so an
offline NAS box never needs the Hub at run time.
"""
from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import model_store

SAMPLE_RATE = 16_000

# ``<de-DE>``/``<en>`` language tags the prompt-conditioned model appends to its text.
_SPACE2 = re.compile(r" {2,}")
_LANG_TAG = re.compile(r"\s*<[a-z]{2}(?:-[A-Za-z]{2})?>\s*")   # Nemotron prompt markers


class EngineError(RuntimeError):
    """Raised for a model/option combination that cannot be honoured."""


@dataclass(frozen=True)
class Cut:
    """One stretch of audio handed to the ASR, plus the bookkeeping to place it back.

    ``keep`` is the absolute window (seconds, original timeline) inside which word
    positions are trusted: the cut itself may be padded with surrounding context, and
    words that belong to the neighbouring speaker's leak are dropped by ``keep``.
    """

    audio: np.ndarray
    t0: float
    speaker: int
    keep: tuple[float, float]

    @property
    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE

    @property
    def end(self) -> float:
        return self.t0 + self.duration


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    speaker: int


@dataclass(frozen=True)
class EngineSpec:
    """Declarative description of one ASR checkpoint."""

    key: str
    label: str
    model_id: str
    language_slot: str | None            # None | "target_lang" | "prompt_lang"
    native_auto: bool                    # detects the language without being told
    weights_gib: float                   # fp32 .nemo size, for the GPU-courtesy check
    word_timestamps: bool = True         # False -> timings are synthesised from text
    slots: dict[str, str] = field(default_factory=dict)
    model_class: str = "ASR"             # "ASR" | "MultiTask"
    hint: str = ""

    @property
    def short(self) -> str:
        return self.model_id.split("/")[-1]


SPECS: dict[str, EngineSpec] = {
    spec.key: spec
    for spec in (
        EngineSpec(
            key="fast",
            label="Nemotron 3.5 ASR (offline)",
            model_id="nvidia/nemotron-3.5-asr-streaming-0.6b",
            language_slot="target_lang",
            native_auto=True,
            weights_gib=2.2,
            word_timestamps=False,
            hint="same weights as the live path, decoded with full context — no download, "
                 "but this checkpoint exposes no word timestamps offline",
        ),
        EngineSpec(
            key="balanced",
            label="Parakeet TDT 0.6B v3",
            model_id="nvidia/parakeet-tdt-0.6b-v3",
            language_slot=None,
            native_auto=True,
            weights_gib=2.3,
            hint="25 languages, language detected by the model, word timestamps",
        ),
        EngineSpec(
            key="quality",
            label="Canary 1B v2",
            model_id="nvidia/canary-1b-v2",
            language_slot="prompt_lang",
            native_auto=False,
            weights_gib=7.0,  # the .nemo bundles a second model for timestamps
            slots={"pnc": "true", "itn": "false", "timestamp": "true", "emotion": "<|emo:undefined|>"},
            model_class="MultiTask",
            hint="most accurate, needs a language; supports a decodercontext hotword hint",
        ),
        EngineSpec(
            key="german",
            label="FastConformer German hybrid (large, PnC)",
            model_id="nvidia/stt_de_fastconformer_hybrid_large_pc",
            language_slot=None,
            native_auto=False,
            weights_gib=2.4,
            word_timestamps=False,
            hint="German-only: strongest pure-German prior, no timestamps",
        ),
    )
}

ALIASES = {
    "nemotron": "fast",
    "nemotron-offline": "fast",
    "parakeet": "balanced",
    "parakeet-tdt": "balanced",
    "canary": "quality",
    "canary-1b-v2": "quality",
    "de": "german",
}


def resolve_spec(key: str) -> EngineSpec:
    """``--asr`` value → spec, accepting the short aliases. Unknown keys raise."""
    name = (key or "balanced").strip().lower().replace("_", "-")
    if name in SPECS:
        return SPECS[name]
    if name in ALIASES:
        return SPECS[ALIASES[name]]
    raise EngineError(
        f"unknown ASR engine {key!r}; choose from "
        + ", ".join(sorted(SPECS))
        + " (plus 'live' for the coupled streaming path)"
    )


# --------------------------------------------------------------------------- engine


class NemoOfflineEngine:
    """One offline NeMo ASR checkpoint.

    ``transcribe()`` takes cuts in any order, writes them as 16 kHz PCM wav to a
    scratch directory, runs one batched greedy decode (longest-first so padding
    waste stays low) and returns word lists in the input order.
    """

    def __init__(
        self,
        spec: EngineSpec,
        *,
        device: str = "cpu",
        dtype=None,
        batch_size: int = 16,
        language: str = "auto",
        scratch_dir: str | Path | None = None,
        hotwords: str | None = None,
    ) -> None:
        self.spec = spec
        self.device = device
        self.dtype = dtype
        self.batch_size = max(1, int(batch_size))
        self.language = (language or "auto").lower()
        self.hotwords = (hotwords or "").strip() or None
        self.scratch = Path(scratch_dir).resolve() if scratch_dir else None
        self.model: Any = None

    # -- lifecycle ---------------------------------------------------------
    @property
    def name(self) -> str:
        return self.spec.key

    def describe(self) -> dict:
        return {
            "engine": self.spec.key,
            "label": self.spec.label,
            "model": self.spec.model_id,
            "language": self.language,
            "language_detection": "model" if self.spec.native_auto else ("pinned" if self.language != "auto" else "none"),
            "timestamps": self.spec.word_timestamps,
            "loaded": self.model is not None,
        }

    def load(self):
        if self.model is not None:
            return self.model

        import torch
        from nemo.collections.asr.models import ASRModel

        cls = ASRModel
        if self.spec.model_class == "MultiTask":
            from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel as cls  # type: ignore[assignment]

        model = model_store.load_nemo_model(self.spec.model_id, cls)
        dtype = self.dtype or (torch.bfloat16 if str(self.device).startswith("cuda") else torch.float32)
        self.model = model.to(device=torch.device(self.device), dtype=dtype).eval()
        # Offline batches change shape across cuts/files. NeMo's captured RNNT
        # decoder can retain stale buffers (illegal CUDA access on a later file).
        # Eager decoding still runs on CUDA; disable graphs in the config too,
        # since requesting timestamps can reconstruct the decoder on first use.
        inner = getattr(getattr(self.model, "decoding", None), "decoding", None)
        if str(self.device).startswith("cuda") and hasattr(inner, "disable_cuda_graphs"):
            from omegaconf import open_dict
            decoding_cfg = self.model.cfg.decoding
            with open_dict(decoding_cfg):
                decoding_cfg.greedy.use_cuda_graph_decoder = False
            inner.disable_cuda_graphs()
        return self.model

    def unload(self) -> None:
        self.model = None

    # -- decode ------------------------------------------------------------
    def _call_kwargs(self, language: str) -> dict[str, Any]:
        """How to ask this checkpoint for one greedy decode, given a resolved language.

        Measured against the installed NeMo 3.1: the prompt-conditioned RNNT only
        honours its language through ``override_config`` (passing ``target_lang=``
        or calling ``set_inference_prompt`` first raises ``Unknown prompt key:
        'None'``), and Canary takes the prompt slots as keyword arguments.
        """
        slot = self.spec.language_slot
        if slot == "target_lang":
            from nemo.collections.asr.models.rnnt_bpe_models_prompt import RNNTPromptTranscribeConfig

            value = language if language in ("auto", "") else locale_tag(language)
            return {
                "override_config": RNNTPromptTranscribeConfig(
                    # use_lhotse=False: with the lhotse path the prompt index is drawn
                    # per cut from the *cut's* language, which a bare file list does not
                    # carry — half the cuts then die on "Unknown prompt key: 'None'".
                    use_lhotse=False,
                    batch_size=self.batch_size,
                    timestamps=self.spec.word_timestamps,
                    target_lang=value,
                )
            }
        kwargs: dict[str, Any] = dict(
            self.spec.slots, batch_size=self.batch_size, timestamps=self.spec.word_timestamps
        )
        if slot == "prompt_lang":
            if language in ("auto", ""):
                raise EngineError(
                    f"{self.spec.label} has no built-in language detection, so it needs an "
                    "explicit language (--lang de). Use --asr balanced for auto-detection."
                )
            # Canary's prompt slots take the bare code ("de"), which its formatter
            # turns into <|de|>; the locale form ("de-DE") is not a known slot value
            # and is dropped, and the checkpoint's own default is English — measured:
            # passing "de-DE" produced a Swedish-flavoured transcript of German speech.
            kwargs["source_lang"] = kwargs["target_lang"] = language
        elif language not in ("auto", "", self.language):
            raise EngineError(f"{self.spec.label} is not prompt-conditioned; cannot switch to {language!r}")
        if self.hotwords and self.spec.model_class == "MultiTask":
            kwargs["decodercontext"] = self.hotwords[:400]  # Canary reads this as context biasing
        return kwargs

    def transcribe(self, cuts: Sequence[Cut], language: str | None = None) -> list[list[Word]]:
        """Transcribe cuts; word lists come back in the same order as ``cuts``."""
        if self.model is None:
            self.load()
        if not len(cuts):
            return []
        kwargs = self._call_kwargs(self.language if language is None else language)
        order = sorted(range(len(cuts)), key=lambda i: -cuts[i].duration)
        tmp = Path(tempfile.mkdtemp(prefix="cuts-", dir=self.scratch))
        try:
            import soundfile as sf

            paths = []
            for i in order:
                path = tmp / f"{i:06d}.wav"
                sf.write(path, np.asarray(cuts[i].audio, dtype=np.float32), SAMPLE_RATE, subtype="PCM_16")
                paths.append(str(path))
            hyps = _as_hypotheses(self.model.transcribe(paths, verbose=False, **kwargs))
            out: list[list[Word]] = [[] for _ in cuts]
            for i, hyp in zip(order, hyps):
                out[i] = _words(hyp, cuts[i], _frame_seconds(self.model), self.spec.label)
            return out
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- helpers


def _as_hypotheses(result) -> list:
    """NeMo returns (best, [nbest]) or a list of strings/hypotheses depending on model."""
    if isinstance(result, tuple):
        result = result[0]
    out = []
    for item in result:
        if isinstance(item, (list, tuple)):
            item = item[0]
        out.append(item)
    return out


def _frame_seconds(model) -> float:
    """Seconds per timestamp tick (window_stride × encoder subsampling factor)."""
    try:
        stride = float(model.cfg.preprocessor.window_stride)
    except Exception:
        stride = 0.01
    try:
        sub = int(model.cfg.encoder.subsampling_factor)
    except Exception:
        sub = 8
    return stride * max(1, sub)


def _words(hyp, cut: Cut, frame_seconds: float, spec_label: str = "ASR") -> list[Word]:
    """Turn one hypothesis into timed words inside ``cut.keep``.

    TDT/RNNT report frame offsets, Canary reports offsets *and* seconds; a model
    without timestamps (the German hybrid) gets its timings synthesised from the
    character distribution, which is still good enough to attribute a word to a
    speaker because attribution is decided over 80 ms windows.
    """
    text = getattr(hyp, "text", hyp)
    if not isinstance(text, str):
        text = str(text)
    # Unanchored on purpose: a prompt-tagged hypothesis carries the tag inline, e.g.
    # "… zwischen Kraft und Stress <de-DE> Was ist denn …", and an end-anchored
    # pattern left it in the transcript as a word of its own.
    text = _SPACE2.sub(" ", _LANG_TAG.sub(" ", text)).strip()
    if not text:
        return []

    entries = []
    ts = getattr(hyp, "timestamp", None) or {}
    for item in ts.get("word") or []:
        word = _LANG_TAG.sub("", item.get("word") or "").strip()
        if not word:
            continue  # a bare language tag is not a word
        if "start" in item and "end" in item:
            t0, t1 = float(item["start"]), float(item["end"])
        else:
            t0 = float(item.get("start_offset", 0)) * frame_seconds
            t1 = float(item.get("end_offset", 0)) * frame_seconds
        entries.append((word, max(0.0, t0), max(0.0, t1)))

    if entries and max(t1 for _, _, t1 in entries) > cut.duration + 1.0:
        raise EngineError(
            f"{spec_label} returned a word at {max(t1 for _, _, t1 in entries):.1f}s inside a "
            f"{cut.duration:.1f}s cut: the batch came back out of order, so word timings "
            "cannot be trusted. Transcribe with batch_size=1."
        )

    # The dominance window is in file time; the ASR's word offsets are local to the
    # cut — compare in one coordinate system or every cut silently loses its head.
    k0 = max(cut.keep[0], cut.t0) - cut.t0
    k1 = min(cut.keep[1], cut.end) - cut.t0

    if not entries:
        # No timestamps from this checkpoint: spread the text over what the speaker
        # was actually dominant for (not the padded cut), proportional to length.
        span, pos = max(k1 - k0, 0.05), k0
        chars = sum(len(w) for w in text.split()) or 1
        for w in text.split():
            width = span * len(w) / chars
            entries.append((w, pos, pos + width))
            pos += width

    words = []
    for w, t0, t1 in entries:
        if t1 < t0:
            t0, t1 = t1, t0
        mid = (t0 + t1) / 2  # a word straddling the dominance edge stays with its owner
        if not (k0 - 0.05 <= mid <= k1 + 0.05):
            continue  # leaked word from the speaker who is not being transcribed here
        words.append(Word(w, cut.t0 + t0, cut.t0 + max(t1, t0 + 0.02), cut.speaker))
    return words


def locale_tag(code: str) -> str:
    """``de`` → ``de-DE`` (prompt tables are keyed by locale); longer codes pass through."""
    c = code.strip()
    if "-" in c:
        return c
    regions = {
        "de": "de-DE", "en": "en-US", "es": "es-ES", "fr": "fr-FR", "it": "it-IT",
        "pt": "pt-PT", "nl": "nl-NL", "pl": "pl-PL", "cs": "cs-CZ", "sk": "sk-SK",
        "hr": "hr-HR", "sl": "sl-SI", "bg": "bg-BG", "ro": "ro-RO", "hu": "hu-HU",
        "ru": "ru-RU", "uk": "uk-UA", "da": "da-DK", "sv": "sv-SE", "fi": "fi-FI",
        "no": "nb-NO", "el": "el-GR", "lv": "lv-LV", "lt": "lt-LT", "et": "et-EE",
        "mt": "mt-MT", "tr": "tr-TR", "ar": "ar-AR", "hi": "hi-IN", "ja": "ja-JP",
        "ko": "ko-KR", "vi": "vi-VN", "zh": "zh-CN",
    }
    return regions.get(c.lower(), c)


def vram_floor_gib(asr: str | None) -> float:
    """VRAM an engine needs before it should take the card (bf16 weights + slack).

    The watcher's courtesy gate uses this so ``quality`` does not load a 6 GB
    checkpoint onto a card with 2 GiB free and OOM on the way in.
    """
    try:
        spec = resolve_spec(asr_mode(asr))   # None/'' -> SPEAKER_ASR -> default
    except EngineError:
        return 2.5   # `live`: measured 1.69 GiB allocator peak plus its CUDA context
    # + 2.0, not + 0.9: the fast engine loaded its 1.1 GiB of bf16 weights onto a card
    # with 2.5 GiB free, and then died mid-decode moving a 198 MiB activation tensor
    # while another tenant grew into the rest. Weights alone under-prices a batch of
    # 16 cuts; lower ``batch_size`` if you need the floor lower.
    return round(spec.weights_gib / 2 + 2.0, 1)


def env(name: str, default: str | None = None) -> str | None:
    """One deployment setting, read from ``SPEAKER_<NAME>``.

    ``NEMODIAR_<NAME>`` is still honoured so an existing installation keeps working
    unchanged. This lives in the light module (no torch, no NeMo) because the
    watcher's GPU courtesy check has to consult it before importing anything heavy.
    """
    import os

    return os.environ.get(f"SPEAKER_{name}") or os.environ.get(f"NEMODIAR_{name}") or default


def asr_mode(asr_or_args=None) -> str:
    """The engine to use: explicit value or ``--asr`` flag, then SPEAKER_ASR, then
    ``balanced``. One precedence rule for app, watcher, CLI and library."""
    value = getattr(asr_or_args, "asr", asr_or_args)
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return (env("ASR", "balanced") or "balanced").strip().lower()
