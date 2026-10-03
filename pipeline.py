"""Speaker-attributed transcription: who said WHAT, WHEN.

Pipeline = NeMo Speech's official coupled streaming multitalker path,
vendored/simplified from:
  - NVIDIA-NeMo/Speech examples/asr/asr_cache_aware_streaming/
      speech_to_text_multitalker_streaming_infer.py
  - nvidia/Nemotron-3-Diarization ASR_INTEGRATION_GUIDE.md (Option 2)

Models (public, no HF token required):
  - Diarization: nvidia/Nemotron-3-Diarization (Sortformer-family, 8 spk)
  - ASR:         nvidia/nemotron-3.5-asr-streaming-0.6b (32 locales,
                 language auto-detected via prompt)

Input: mono 16 kHz wav (convert upstream with `prepare_audio`).
Output: list of {"text", "start", "end", "speaker"} dicts, sorted by time.
"""
from __future__ import annotations

import os

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

DIAR_MODEL_ID = "nvidia/Nemotron-3-Diarization"
ASR_MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"

# Optional local mirror: models/<repo-basename>.nemo used instead of Hub download.
MODELS_DIR = Path(__file__).resolve().parent / "models"


def _load_model(name: str, cls):
    """Load a NeMo model (on CPU) from local ``models/`` mirror, explicit path, or HF Hub."""
    if str(name).endswith(".nemo"):
        path, src = Path(name), str(name)
    else:
        local = MODELS_DIR / f"{name.rstrip('/').split('/')[-1]}.nemo"
        if local.is_file():
            path, src = local, f"local mirror {local}"
        else:
            nemo_logging.info(f"Downloading {name} from Hugging Face Hub")
            return cls.from_pretrained(model_name=name, map_location="cpu")
    nemo_logging.info(f"Loading {src}")
    return cls.restore_from(restore_path=str(path), map_location="cpu")


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
        raise RuntimeError(f"audio conversion failed: {proc.stderr.strip() or src}")
    return dst_wav


def _is_silent_wav(path: str, threshold: float = 3e-4) -> bool:
    import soundfile as sf

    data, _sr = sf.read(path, dtype="float32", always_2d=True)
    return float(np.max(np.abs(data))) < threshold


class Transcriber:
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
    ) -> None:
        torch.set_float32_matmul_precision("high")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        map_location = torch.device(device)
        if max_speakers is None and os.environ.get("NEMODIAR_MAX_SPKS"):
            max_speakers = int(os.environ["NEMODIAR_MAX_SPKS"])

        # Load weights on CPU first, then move/cast: avoids fp32 GPU spikes
        # and (with NEMODIAR_ASR_BF16=1) halves resident ASR memory.
        nemo_logging.info(f"Loading diarization model {diar_model}")
        self.diar_model = _load_model(diar_model, SortformerEncLabelModel)
        nemo_logging.info(f"Loading ASR model {asr_model}")
        self.asr_model: ASRModel = _load_model(asr_model, ASRModel)

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
            env = os.environ.get("NEMODIAR_ASR_BF16")
            if env is not None:
                asr_bf16 = env == "1"
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
            self.asr_model.set_inference_prompt(PIPELINE_CFG["target_lang"] or "auto")
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

    @torch.inference_mode()
    def transcribe(self, wav_path: str | os.PathLike, publish=None) -> list[dict]:
        wav_path = os.fspath(wav_path)
        if _is_silent_wav(wav_path):
            return []

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
                        self._normalize(dict(s))
                        for st in streamer.instance_manager.batch_asr_states
                        for s in st.seglsts
                        if (s.get("words") or "").strip()
                    ]
                    partial.sort(key=lambda x: x["start"])
                    publish(step_num, partial)
                except Exception:
                    pass  # progress must never sink the run

        seglsts = streamer.generate_seglst_dicts_from_parallel_streaming(samples=samples)
        return [self._normalize(seg) for seg in seglsts]

    @staticmethod
    def _normalize(seg: dict) -> dict:
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
            "start": round(float(start), 3),
            "end": round(float(end), 3),
            "speaker": speaker,
        }


if __name__ == "__main__":  # CLI smoke test: python pipeline.py audio.wav
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Speaker-attributed transcription CLI")
    ap.add_argument("audio", help="path to audio/video file")
    ap.add_argument("--device", default=None, help="cuda / cpu")
    ap.add_argument("--max-speakers", type=int, default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    t = Transcriber(device=args.device, max_speakers=args.max_speakers)
    work = Path(args.audio).stem + ".16k.wav"
    converted = not str(args.audio).endswith(".16k.wav")
    wav = prepare_audio(args.audio, work) if converted else args.audio
    try:
        segs = t.transcribe(wav)
    finally:
        if converted:
            Path(wav).unlink(missing_ok=True)
    print(json.dumps(segs, ensure_ascii=False, indent=1))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(segs, ensure_ascii=False, indent=1), encoding="utf-8")
