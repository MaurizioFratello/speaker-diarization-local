"""Speaker-attributed transcription: who said WHAT, WHEN.

Two pipeline modes, one output contract ``[{"text", "start", "end", "speaker"}]``:

``live``
  NeMo Speech's official coupled multitalker *streaming* path, vendored and
  simplified from:
    - NVIDIA-NeMo/Speech examples/asr/asr_cache_aware_streaming/
        speech_to_text_multitalker_streaming_infer.py
    - nvidia/Nemotron-3-Diarization ASR_INTEGRATION_GUIDE.md (Option 2)
  Only a cache-aware streaming ASR fits that rail, so this mode is tied to
  ``nvidia/nemotron-3.5-asr-streaming-0.6b``. It emits partial text while the
  file is still being read and merges words spoken by two people at once.

``fast`` / ``balanced`` / ``quality`` / ``german``  (see :mod:`engines`)
  Diarization first (offline), then an offline ASR engine over the speaker cuts
  (:mod:`segment_asr`). Better sentences, better accuracy, fewer parameters —
  at the cost of overlap handling and of progress arriving per speaker turn
  instead of per second.

Input: mono 16 kHz wav (convert upstream with `prepare_audio`).
Output: list of {"text", "start", "end", "speaker"} dicts, sorted by time.
"""
from __future__ import annotations

import os
import threading
from math import ceil as _ceil, floor as _floor

# Reduce allocator fragmentation when sharing a GPU with other processes.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.models import ASRModel, SortformerEncLabelModel
from nemo.collections.asr.parts.submodules.subsampling import FeatureStacking
from nemo.collections.asr.parts.utils.multispk_transcribe_utils import (
    SpeakerTaggedASR,
    configure_diar_streaming,
    validate_feature_frame_strides,
)
from nemo.collections.asr.parts.utils.streaming_utils import (
    CacheAwareStreamingAudioBuffer,
)
from nemo.utils import logging as nemo_logging

from dataclasses import replace

from engines import EngineSpec, NemoOfflineEngine, asr_mode, env, locale_tag, resolve_spec
from glossary import correct_segments, load_glossary_if_exists
from model_store import MODELS_DIR, mirror_path  # re-exported: callers import these from here
from model_store import load_nemo_model as _load_model
from segment_asr import DIAR_MODEL_ID, DiarizeFirst, SegCfg

ASR_MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"  # the ``live`` mode's ASR


# Officially recommended settings for the Nemotron-3.5-ASR pairing
# (masked_asr=true). Do not tune ASR/diar chunk geometry independently.
PIPELINE_CFG: dict[str, Any] = {
    "device": "cuda",
    "sample_rate": 16000,
    "deploy_mode": True,
    "streaming_mode": True,
    "max_num_of_spks": 8,
    "batch_size": 1,
    "parallel_speaker_strategy": True,
    "masked_asr": True,
    "mask_preencode": False,
    "single_speaker_mode": False,
    "cache_gating": True,
    "cache_gating_buffer_size": 2,
    "binary_diar_preds": True,
    "spkcache_len": None,  # set by configure_diar_streaming
    "spkcache_update_period": 222,
    "fifo_len": 264,
    "diar_right_context": 0,
    "att_context_size": [56, 13],
    "use_amp": True,
    "precision": "bf16",
    "online_normalization": False,
    "pad_and_drop_preencoded": False,
    "feat_len_sec": 0.01,
    "discarded_frames": 8,
    "word_window": 50,
    "sent_break_sec": 1.0,
    "fix_prev_words_count": 5,
    "update_prev_words_sentence": 5,
    "left_frame_shift": -1,
    "right_frame_shift": 0,
    "min_sigmoid_val": 1e-2,
    "ignored_initial_frame_steps": 5,
    "generate_realtime_scripts": False,
    "print_sample_indices": [0],
    "colored_text": False,
    "verbose": False,
    "print_time": False,
    "log": False,
    "real_time_mode": False,
    "spk_supervision": "diar",
    "target_lang": "auto",
    "strip_lang_tags": True,
    "lang_tag_pattern": None,
}


def prepare_audio(src: str | os.PathLike, dst_wav: str | os.PathLike) -> str:
    """Convert any audio/video file to 16 kHz mono 16-bit PCM wav.

    Uses the statically-linked ffmpeg binary shipped by the ``imageio-ffmpeg``
    wheel, so no system ffmpeg install is needed on Linux or Windows.
    """
    import imageio_ffmpeg
    import subprocess

    src = os.fspath(src)
    dst_wav = os.fspath(dst_wav)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", src,
        "-vn",                       # drop video streams
        "-ac", "1", "-ar", "16000",  # mono, 16 kHz
        "-c:a", "pcm_s16le",
        dst_wav,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not Path(dst_wav).is_file():
        raise RuntimeError(f"audio conversion failed (ffmpeg rc={proc.returncode}): {proc.stderr.strip() or src}")
    return dst_wav


def _is_silent_wav(path: str, threshold: float = 3e-4) -> bool:
    import soundfile as sf

    data, _sr = sf.read(path, dtype="float32", always_2d=True)
    if data.size == 0:  # empty input: treat as silent rather than crashing
        nemo_logging.warning(f"{path} contains no audio frames")
        return True
    return float(np.max(np.abs(data))) < threshold


_vad_model = None
_vad_model_lock = threading.Lock()


def _get_vad_model():
    """Lazy Silero VAD singleton (CPU; model ships inside the silero-vad wheel)."""
    global _vad_model
    with _vad_model_lock:
        if _vad_model is None:
            from silero_vad import load_silero_vad

            _vad_model = load_silero_vad(onnx=False)
        return _vad_model


def _vad_regions(data, sr, threshold: float = 0.5):
    """Silero speech regions as [(start_s, end_s)]; [] when VAD is unavailable."""
    try:
        from silero_vad import get_speech_timestamps

        return [(t["start"], t["end"]) for t in get_speech_timestamps(
            torch.from_numpy(data), _get_vad_model(), sampling_rate=sr,
            threshold=threshold, return_seconds=True
        )]
    except Exception as exc:
        nemo_logging.warning(f"Silero VAD unavailable ({type(exc).__name__}: {exc})")
        return []


def splice_speech_regions(
    wav_path: str | os.PathLike,
    gap_merge: float = 1.0,
    pad: float = 0.3,
    quantum: float = 0.8,
) -> tuple[str, list[tuple[float, float, float]] | None]:
    """Remove INTERIOR silence: concatenate VAD-positive regions into one file.

    Much faster on sparse recordings, but every splice re-anchors the streaming
    encoders, so wording/speaker handoffs right at a cut can drift (measured
    ~5% word delta on a 20-min recording; use as opt-in fast mode).
    Returns ``(model_path, mapping)`` with mapping triples
    ``(spliced_pos, length, original_pos)`` for :func:`map_spliced_time`;
    ``(original path, None)`` when nothing is worth removing or on any failure
    (never fatal).
    """
    wav_path = os.fspath(wav_path)
    try:
        import soundfile as sf

        data, sr = sf.read(wav_path, dtype="float32")
        duration = len(data) / sr
        ts = _vad_regions(data, sr)
        if not ts:
            return wav_path, None
        regions: list[list[float]] = []
        for s, e in ts:
            s, e = max(0.0, s - pad), min(duration, e + pad)
            if regions and s - regions[-1][1] <= gap_merge:
                regions[-1][1] = max(regions[-1][1], e)
            else:
                regions.append([s, e])
        q = max(0.01, float(quantum))
        cuts: list[tuple[float, float]] = []
        prev_end = 0.0
        for s, e in regions:
            cs = min(max(_floor(s / q) * q, 0.0), duration)
            ce = min(_ceil(e / q) * q, duration)
            cs = max(cs, prev_end)  # snap can widen cuts into neighbours: never overlap
            if ce <= cs:
                continue
            cuts.append((cs, ce))
            prev_end = ce
        if not cuts:
            return wav_path, None
        kept = sum(e - s for s, e in cuts)
        if kept <= 0 or duration - kept < 1.0:
            return wav_path, None  # degenerate or too little silence to justify splicing
        pieces, mapping, cursor = [], [], 0.0
        for s, e in cuts:
            i0, i1 = int(s * sr), int(e * sr)
            pieces.append(data[i0:i1])
            seg_len = (i1 - i0) / sr
            mapping.append((cursor, seg_len, s))
            cursor += seg_len
        out = Path(wav_path).with_name(Path(wav_path).stem + ".skip16k.wav")
        sf.write(out, np.concatenate(pieces), sr, subtype="PCM_16")
        nemo_logging.info(
            f"VAD full-file: kept {kept:.1f}s of {duration:.1f}s in {len(cuts)} "
            f"speech regions ({duration - kept:.1f}s silence skipped)"
        )
        return str(out), mapping
    except Exception as exc:
        nemo_logging.warning(f"VAD full-file splice skipped ({type(exc).__name__}: {exc})")
        return wav_path, None


def map_spliced_time(mapping: list[tuple[float, float, float]], pos: float) -> float:
    """Map a position from the spliced timeline back onto the original file."""
    if not mapping:
        return pos
    for spliced_pos, length, orig in mapping:
        if spliced_pos - 1e-6 <= pos < spliced_pos + length:
            return orig + (pos - spliced_pos)
    last_sp, last_len, last_orig = mapping[-1]
    if pos >= last_sp:  # exact end / tiny float overshoot -> clamp to region end
        return last_orig + last_len
    return pos


def trim_edge_silence(
    wav_path: str | os.PathLike, pad: float = 0.3, quantum: float = 0.8
) -> tuple[str, float]:
    """Strip leading/trailing silence so diarization+ASR never pays for dead air.

    ``quantum`` is the streaming chunk step in seconds: cut points are floored
    to multiples of it, which keeps every retained audio frame on the same
    chunk grid the untrimmed file would have used (verified output-identical).
    Returns ``(model_path, leading_offset)``: feed ``model_path`` to
    :meth:`Transcriber.transcribe` with ``offset=leading_offset`` so segment
    timestamps stay valid on the ORIGINAL (untrimmed playback) timeline.
    Always safe: any failure falls back to the untouched input.
    """
    wav_path = os.fspath(wav_path)
    try:
        import soundfile as sf

        data, sr = sf.read(wav_path, dtype="float32")
        duration = len(data) / sr
        ts = _vad_regions(data, sr)
        if not ts:
            return wav_path, 0.0
        q = max(0.01, float(quantum))
        # Floor to the chunk grid: retained frames keep their grid alignment.
        start = min(max(_floor((ts[0][0] - pad) / q) * q, 0.0), duration)
        end = min(max(_ceil((ts[-1][1] + pad) / q) * q, 0.0), duration)
        lead, trail = start, duration - end
        if lead + trail < 1.0:  # not worth a rewrite
            return wav_path, 0.0
        out = Path(wav_path).with_name(Path(wav_path).stem + ".trim16k.wav")
        sf.write(out, data[int(start * sr) : int(end * sr)], sr, subtype="PCM_16")
        nemo_logging.info(f"VAD trimmed {lead:.1f}s leading / {trail:.1f}s trailing silence")
        return str(out), float(start)
    except Exception as exc:
        nemo_logging.warning(f"VAD edge-silence trim skipped ({type(exc).__name__}: {exc})")
        return wav_path, 0.0


class GlossaryFix:
    """Text-side correction shared by both transcriber modes.

    Subclasses set ``self._g`` (a :class:`glossary.Glossary`, or ``None``). A
    glossary is a *text* concern, so it behaves identically no matter which
    engine decoded the audio, and it is applied to published partials as well as
    to the final list: the UI must never show a spelling the finished transcript
    will disagree with.
    """

    _g = None

    def _begin_fix_log(self) -> None:
        """Fresh audit trail; called at the top of every transcribe()."""
        self.last_corrections: list[dict] = []

    def _record(self, changes: list[dict]) -> None:
        """Append repairs, once each.

        Accumulated rather than replaced, and deduplicated: the streaming path
        corrects every partial it publishes, so the same repair is seen on each
        update of one line and again in the final list. Replacing here is what made
        a job report `corrections: null` for a transcript whose spelling *had* been
        repaired — the publish pass fixed the text, so the final pass found nothing.
        """
        seen = {(c.get("start"), c["from"], c["to"]) for c in self.last_corrections}
        for c in changes:
            key = (c.get("start"), c["from"], c["to"])
            if key not in seen:
                seen.add(key)
                self.last_corrections.append(c)

    def _publish(self, cb):
        if cb is None or self._g is None:
            return cb

        def send(step, partial):
            fixed, changes = correct_segments(partial, self._g)
            self._record(changes)
            return cb(step, fixed)

        return send

    def _fix(self, segs: list[dict]) -> list[dict]:
        """Correct finished segments; the audit trail lands in ``last_corrections``."""
        if self._g is None:
            return segs
        segs, changes = correct_segments(segs, self._g)
        self._record(changes)
        return segs


def _open_glossary(path: str | os.PathLike | None):
    """Glossary for a CLI/env path, with a loud word when the path is wrong."""
    if not path:
        return None
    g = load_glossary_if_exists(path)
    if g is None:
        nemo_logging.warning(f"glossary {path}: missing or empty, no corrections applied")
    else:
        nemo_logging.info(f"Glossary: {len(g)} entries from {path}")
    return g


class Transcriber(GlossaryFix):
    """Loads both models once; call .transcribe(wav_path) per file.

    Each call is an independent session: speaker labels (Speaker 0..7) are
    ordered by first appearance within that file only.
    """

    def __init__(
        self,
        diar_model: str = DIAR_MODEL_ID,
        asr_model: str = ASR_MODEL_ID,
        device: str | None = None,
        max_speakers: int | None = None,
        asr_bf16: bool | None = None,
        glossary: str | os.PathLike | None = None,
        lang: str | None = None,
    ) -> None:
        torch.set_float32_matmul_precision("high")
        # Resolved once here: the prompt is set later in this method, and describe()
        # has to answer "which language did we condition on?" without loading again.
        self.lang = (lang or env("LANG", "auto") or "auto").strip().lower()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        map_location = torch.device(device)
        if max_speakers is None and env("MAX_SPKS"):
            max_speakers = int(env("MAX_SPKS"))

        # Load weights on CPU first, then move/cast: avoids fp32 GPU spikes
        # and (with NEMODIAR_ASR_BF16=1) halves resident ASR memory.
        nemo_logging.info(f"Loading diarization model {diar_model}")
        self.diar_model = _load_model(diar_model, SortformerEncLabelModel)
        nemo_logging.info(f"Loading ASR model {asr_model}")
        self.asr_model: ASRModel = _load_model(asr_model, ASRModel)
        self._asr_id = asr_model
        self._g = _open_glossary(glossary)
        self.last_corrections = []

        self.use_bf16 = (
            PIPELINE_CFG["use_amp"]
            and PIPELINE_CFG["precision"].lower().startswith("bf16")
            and map_location.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )
        dtype = torch.bfloat16 if self.use_bf16 else torch.float32
        self.diar_model = self.diar_model.to(device=map_location, dtype=dtype).eval()
        # fp32 weights match the official guide (CUDA-graph RNNT decoder is not
        # mixed-precision safe: see TODO in NeMo label_looping_base). bf16 weights
        # halve VRAM but require disabling the CUDA-graph decoder.
        if asr_bf16 is None:
            pin = env("ASR_BF16")
            if pin is not None:
                asr_bf16 = pin == "1"
            elif map_location.type == "cuda":
                # fp32 ASR weights (~2.4 GB) plus activations/graphs need ~6 GB free;
                # on shared/constrained GPUs use bf16 + eager decoder instead.
                free_b = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
                asr_bf16 = free_b < 6 * 2**30
            else:
                asr_bf16 = False
        asr_dtype = dtype if (asr_bf16 and self.use_bf16) else torch.float32
        if map_location.type == "cuda":
            free_gb = torch.cuda.mem_get_info()[0] / 2**30
            nemo_logging.info(
                f"ASR weights: {'bf16 (eager decoder)' if asr_dtype != torch.float32 else 'fp32 (CUDA-graph decoder)'}, "
                f"{free_gb:.1f} GB VRAM free at load"
            )
        self.asr_model = self.asr_model.to(device=map_location, dtype=asr_dtype).eval()
        if asr_dtype != torch.float32:
            inner = getattr(self.asr_model.decoding, "decoding", None)
            if inner is not None and hasattr(inner, "disable_cuda_graphs"):
                inner.disable_cuda_graphs()
            comp = getattr(inner, "decoding_computer", None)
            if comp is not None and getattr(comp, "cuda_graphs_mode", None) is not None:
                raise RuntimeError("CUDA-graph RNNT decoder still active; refusing mixed-precision decode")

        att = PIPELINE_CFG["att_context_size"]
        if hasattr(self.asr_model.encoder, "set_default_att_context_size"):
            self.asr_model.encoder.set_default_att_context_size(att_context_size=att)
        else:
            raise ValueError("ASR model does not support multiple lookaheads.")

        if hasattr(self.asr_model, "set_inference_prompt"):
            # ``--lang de`` conditions the prompt instead of letting the model decide;
            # auto (the default) is what a mixed-language meeting wants.
            prompt = (PIPELINE_CFG["target_lang"] if self.lang == "auto"
                      else locale_tag(self.lang))
            self.asr_model.set_inference_prompt(prompt or "auto")
            self.asr_model.decoding.set_strip_lang_tags(
                PIPELINE_CFG["strip_lang_tags"],
                lang_tag_pattern=PIPELINE_CFG["lang_tag_pattern"],
            )

        self.autocast = torch.amp.autocast(
            device_type=map_location.type, dtype=torch.bfloat16, enabled=self.use_bf16
        )

        validate_feature_frame_strides(
            asr_model=self.asr_model, diar_model=self.diar_model
        )
        scfg = self.asr_model.encoder.streaming_cfg
        diar_chunk_len = scfg.valid_out_len + scfg.cache_drop_size
        # Streaming chunk step in seconds; VAD cut points are floored to it so
        # the encoder sees the exact same frames it would have untrimmed.
        self.chunk_step_sec = (
            diar_chunk_len * self.asr_model.encoder.subsampling_factor
            * PIPELINE_CFG["feat_len_sec"]
        )

        cfg = OmegaConf.create(dict(PIPELINE_CFG, device=str(map_location)))
        if max_speakers:
            cfg.max_num_of_spks = int(max_speakers)
        self.cfg = cfg

        self.diar_out_sub = configure_diar_streaming(
            diar_model=self.diar_model,
            cfg=cfg,
            output_subsampling_factor=self.asr_model.encoder.subsampling_factor,
            diar_chunk_len=diar_chunk_len,
        )
        cfg.spkcache_len = int(self.diar_model.sortformer_modules.spkcache_len)

        if isinstance(self.diar_model.encoder.pre_encode, FeatureStacking) and not cfg.pad_and_drop_preencoded:
            # Diarization needs cache-free aligned chunks; padded pre-encode.
            cfg.pad_and_drop_preencoded = True

        self.device = map_location

    def load(self):
        """No-op for symmetry with :class:`OfflineTranscriber` (this path loads eagerly)."""
        return self

    def unload(self) -> None:
        """No-op: the streaming path holds its weights for the process lifetime."""

    def describe(self) -> dict:
        """What this transcriber is made of (app status line + JSON metadata)."""
        return {
            "engine": "live",
            "label": "Nemotron 3.5 ASR Streaming 0.6B",
            "model": self._asr_id,
            "language": getattr(self, "lang", "auto"),
            "language_detection": "model" if getattr(self, "lang", "auto") == "auto" else "prompt",
            "timestamps": True,
            "pipeline": "coupled streaming (multitalker)",
            "diarization": DIAR_MODEL_ID,
            "diarization_mode": "streaming (30.4 s buffer, 6.8 s hop)",
        }

    @torch.inference_mode()
    def transcribe(self, wav_path: str | os.PathLike, publish=None, offset: float = 0.0) -> list[dict]:
        self._begin_fix_log()
        wav_path = os.fspath(wav_path)
        if _is_silent_wav(wav_path):
            return []
        publish = self._publish(publish)

        cfg = self.cfg
        samples = [{"audio_filepath": wav_path}]
        buffer = CacheAwareStreamingAudioBuffer(
            model=self.asr_model,
            online_normalization=cfg.online_normalization,
            pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
        )
        if self.device.type == "cuda":
            # Keep the whole-file waveform and mel features in CPU RAM: running the
            # mel front-end over long files on the GPU asks for gigabyte-scale
            # one-shot allocations. Only per-step chunks move to the GPU below.
            buffer.get_model_device = lambda: torch.device("cpu")
            buffer.preprocessor = buffer.preprocessor.to("cpu")
        buffer.append_audio_file(audio_filepath=wav_path, stream_id=-1)
        streamer = SpeakerTaggedASR(cfg, self.asr_model, self.diar_model)

        diar_right_offset = (
            0 if cfg.spk_supervision == "rttm" else cfg.diar_right_context * self.diar_model.encoder.subsampling_factor
        )
        import time as _time
        last_pub = 0.0
        for step_num, (chunk_audio, chunk_lengths, diar_chunk_audio, diar_chunk_lengths) in enumerate(
            buffer.iter_with_right_context(diar_right_offset)
        ):
            chunk_audio = chunk_audio.to(self.device)
            chunk_lengths = chunk_lengths.to(self.device)
            diar_chunk_audio = diar_chunk_audio.to(self.device)
            diar_chunk_lengths = diar_chunk_lengths.to(self.device)
            drop_extra_pre_encoded = (
                0
                if step_num == 0 and not cfg.pad_and_drop_preencoded
                else self.asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
            )
            with self.autocast:
                streamer.perform_parallel_streaming_stt_spk(
                    step_num=step_num,
                    chunk_audio=chunk_audio,
                    chunk_lengths=chunk_lengths,
                    diar_chunk_audio=diar_chunk_audio,
                    diar_chunk_lengths=diar_chunk_lengths,
                    is_buffer_empty=buffer.is_buffer_empty(),
                    drop_extra_pre_encoded=drop_extra_pre_encoded,
                )
            if publish is not None and _time.monotonic() - last_pub >= 0.5:
                last_pub = _time.monotonic()
                try:
                    # Non-mutating snapshot: seglsts is rebuilt per step from the
                    # speaker-wise sentence lists; copy each dict so the next step's
                    # _update_last_sentence can never tear a published row.
                    partial = [
                        self._normalize(dict(s), offset)
                        for st in streamer.instance_manager.batch_asr_states
                        for s in st.seglsts
                        if (s.get("words") or "").strip()
                    ]
                    partial.sort(key=lambda x: x["start"])
                    publish(step_num, partial)
                except Exception:
                    pass  # progress must never sink the run

        seglsts = streamer.generate_seglst_dicts_from_parallel_streaming(samples=samples)
        return self._fix([self._normalize(seg, offset) for seg in seglsts])

    @staticmethod
    def _normalize(seg: dict, offset: float = 0.0) -> dict:
        # SegLST sentence dicts from NeMo: keys 'words', 'start_time', 'end_time', 'speaker'
        text = (seg.get("words") or seg.get("text") or "").strip()
        start = seg.get("start_time", seg.get("start", 0.0))
        end = seg.get("end_time", seg.get("end", start))
        speaker = seg.get("speaker", seg.get("spk", seg.get("label", "speaker_0")))
        if isinstance(speaker, (int, np.integer)):
            speaker = f"speaker_{int(speaker)}"
        speaker = str(speaker).replace("spk_", "speaker_")
        return {
            "text": text,
            "start": round(float(start) + offset, 3),
            "end": round(float(end) + offset, 3),
            "speaker": speaker,
        }


class OfflineTranscriber(GlossaryFix):
    """Diarize-first mode: same call signature as :class:`Transcriber`.

    ``asr`` picks the checkpoint (see ``engines.SPECS``); ``lang`` is ``auto`` or
    a language code for the engines that cannot detect one. On a shared GPU the
    engine runs in bf16 and drops to CPU when the card has no room for it — an
    unattended watcher should degrade quietly, not OOM whatever else is resident.
    """

    #: seconds of processed audio per progress step (see ``DiarizeFirst.transcribe``)
    chunk_step_sec = 1.0

    def __init__(
        self,
        asr: str | None = None,
        *,
        lang: str | None = None,
        device: str | None = None,
        max_speakers: int | None = None,
        hotwords: str | None = None,
        glossary: str | os.PathLike | None = None,
        batch_size: int = 16,
        seg: SegCfg | None = None,
        diar_model: str = DIAR_MODEL_ID,
    ) -> None:
        spec = resolve_spec(asr_mode(asr))
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if max_speakers is None and env("MAX_SPKS"):
            max_speakers = int(env("MAX_SPKS"))

        device, dtype = _pick_device(spec, device)
        self._impl = DiarizeFirst(
            NemoOfflineEngine(
                spec,
                device=device,
                dtype=dtype,
                batch_size=batch_size,
                language=lang or env("LANG", "auto"),
                hotwords=hotwords or env("HOTWORDS"),
            ),
            diar_model=diar_model,
            device=device,
            cfg=replace(seg or SegCfg(), max_speakers=int(max_speakers or 8)),
        )
        self._g = _open_glossary(glossary or env("GLOSSARY"))
        self.last_corrections = []
        self.device = torch.device(device)

    def load(self):
        """Load diarization + ASR, degrading to CPU if the GPU has no room for it.

        An OOM here would otherwise cost a whole unattended watch cycle, and this
        box shares one card with whoever else is on it. NeMo wraps the OOM of a
        multi-model checkpoint in a plain RuntimeError, so the message is matched.
        """
        try:
            self._impl.load()
        except Exception as exc:
            if "out of memory" not in str(exc).lower():
                raise
            nemo_logging.warning(
                f"{self._impl.engine.spec.label}: CUDA out of memory, reloading on CPU"
            )
            self._reload_on_cpu()
            self._impl.load()
        return self

    def unload(self) -> None:
        self._impl.unload()

    def describe(self) -> dict:
        return self._impl.describe()

    @torch.inference_mode()
    def transcribe(self, wav_path, publish=None, offset: float = 0.0) -> list[dict]:
        self._begin_fix_log()
        if _is_silent_wav(os.fspath(wav_path)):
            return []
        try:
            segs = self._impl.transcribe(
                os.fspath(wav_path), publish=self._publish(publish), offset=offset
            )
        except Exception as exc:
            # A model that fitted at load time can still be pushed over by somebody
            # else's process while it decodes: measured with NeMo moving a
            # normalisation tensor to the card mid-run ("Tried to allocate 198.00
            # MiB", 155 MiB free, another process holding 90 GiB). A load-time
            # fallback alone would leave an unattended watcher failing the same file
            # on every tick, so the run itself degrades: drop the card, redo it on
            # CPU, and tell the caller which device answered.
            if self.device.type != "cuda" or "out of memory" not in str(exc).lower():
                raise
            nemo_logging.warning(
                f"{self._impl.engine.spec.label}: CUDA out of memory while decoding, "
                f"retrying this file on CPU")
            self._reload_on_cpu()
            segs = self._impl.transcribe(
                os.fspath(wav_path), publish=self._publish(publish), offset=offset
            )
        return self._fix(segs)

    def _reload_on_cpu(self) -> None:
        """Release the card and move this engine's models to CPU."""
        import torch as _torch

        self._impl.unload()
        _torch.cuda.empty_cache()
        self._impl.engine.device, self._impl.engine.dtype = "cpu", _torch.float32
        self._impl.device = "cpu"
        self.device = _torch.device("cpu")


def _pick_device(spec: EngineSpec, device: str) -> tuple[str, Any]:
    """Return ``(device, dtype)``, degrading cuda -> cpu rather than fighting for VRAM."""
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return "cpu", torch.float32
    try:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        free_gb = torch.cuda.mem_get_info()[0] / 2**30
    except Exception as exc:  # a saturated card cannot even give us a context
        nemo_logging.warning(f"CUDA unusable ({exc.__class__.__name__}: {exc}) — running on CPU")
        return "cpu", torch.float32
    need_gb = spec.weights_gib / (2 if dtype == torch.bfloat16 else 1) + 2.0
    if free_gb < need_gb:
        nemo_logging.warning(
            f"{spec.label}: {free_gb:.1f} GiB free, {need_gb:.1f} GiB needed — running on CPU"
        )
        return "cpu", torch.float32
    return "cuda", dtype


def load_transcriber(
    asr: str | None = None,
    *,
    device: str | None = None,
    lang: str | None = None,
    max_speakers: int | None = None,
    hotwords: str | None = None,
    glossary: str | os.PathLike | None = None,
    batch_size: int = 16,
):
    """Build the transcriber for an ``--asr`` value (``live`` keeps the streaming path)."""
    mode = asr_mode(asr)
    if mode in ("live", "streaming", "coupled"):
        return Transcriber(device=device, max_speakers=max_speakers, glossary=glossary,
                           lang=lang)
    return OfflineTranscriber(
        mode,
        device=device,
        lang=lang,
        max_speakers=max_speakers,
        hotwords=hotwords,
        glossary=glossary,
        batch_size=batch_size,
    )


if __name__ == "__main__":  # CLI smoke test: python pipeline.py audio.wav
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Speaker-attributed transcription CLI")
    ap.add_argument("audio", help="path to audio/video file")
    ap.add_argument("--device", default=None, help="cuda / cpu")
    ap.add_argument("--max-speakers", type=int, default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--asr", default=None,
                    help="live | fast | balanced | quality | german "
                         "(default: $SPEAKER_ASR, alias $NEMODIAR_ASR)")
    ap.add_argument("--lang", default=None,
                    help="language code for engines that require one (e.g. de); default: auto")
    ap.add_argument("--hotwords", default=None,
                    help="context-biasing hint for engines that support it (Canary)")
    ap.add_argument("--glossary", default=None,
                    help="file of canonical spellings to repair in the output")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-vad-trim", action="store_true",
                    help="disable Silero VAD trimming of leading/trailing silence")
    args = ap.parse_args()

    t = load_transcriber(
        args.asr, device=args.device, lang=args.lang, max_speakers=args.max_speakers,
        hotwords=args.hotwords, glossary=args.glossary, batch_size=args.batch_size,
    )
    print(f"pipeline: {t.describe()}", flush=True)
    work = Path(args.audio).stem + ".16k.wav"
    converted = not str(args.audio).endswith(".16k.wav")
    wav = prepare_audio(args.audio, work) if converted else args.audio
    model_wav, vad_off = (wav, 0.0) if args.no_vad_trim else trim_edge_silence(
        wav, quantum=getattr(t, "chunk_step_sec", 0.8)
    )
    try:
        segs = t.transcribe(model_wav, offset=vad_off)
    finally:
        if os.fspath(model_wav) != os.fspath(wav):
            Path(model_wav).unlink(missing_ok=True)
        if converted:
            Path(wav).unlink(missing_ok=True)
    print(json.dumps(segs, ensure_ascii=False, indent=1))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(segs, ensure_ascii=False, indent=1), encoding="utf-8")
