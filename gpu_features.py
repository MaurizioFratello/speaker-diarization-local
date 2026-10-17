"""Bounded-memory CUDA mel extraction for long, single-recording diarization.

NeMo streams the diarization encoder, but its default mel extractor first builds
an entire recording's complex STFT on the GPU. Keep that large intermediate
bounded without cutting the recording into independent speaker-label sessions.
"""
import math

import torch
from torch.nn import functional as F


class BoundedMelFeatures(torch.nn.Module):
    """Wrap NeMo FilterbankFeatures; retain global normalization and frame timing."""

    def __init__(self, base, chunk_seconds=60):
        super().__init__()
        self.base = base
        self.chunk_frames = max(1, int(chunk_seconds * base.sample_rate / base.hop_length))

    @property
    def filter_banks(self):
        return self.base.filter_banks

    def forward(self, x, seq_len, linear_spec=False):
        base = self.base
        hop = base.hop_length
        if x.shape[-1] <= self.chunk_frames * hop or base.training:
            return base(x, seq_len, linear_spec=linear_spec)
        if x.shape[0] != 1 or base.frame_splicing != 1 or linear_spec:
            raise ValueError("bounded diarization features require one recording and unspliced mel features")

        from nemo.collections.asr.parts.preprocessing.features import normalize_batch

        lengths = base.get_seq_len(seq_len)
        frames = int(lengths[0].item())
        # Hop-aligned overlap includes the FFT window AND preemphasis's previous
        # sample. Interior chunk boundaries must behave like the full waveform.
        context = math.ceil((base.n_fft + 1) / hop) * hop
        normalize, pad_to = base.normalize, base.pad_to
        features = None
        try:
            base.normalize, base.pad_to = None, 0
            for start in range(0, frames, self.chunk_frames):
                end = min(frames, start + self.chunk_frames)
                left = max(0, start * hop - context)
                right = min(x.shape[-1], end * hop + context)
                chunk = x[:, left:right]
                part, _ = base(chunk, seq_len.new_tensor([chunk.shape[-1]]))
                if features is None:
                    # NeMo's STFT has one extra (masked) frame before pad_to.
                    features = part.new_full((1, part.shape[1], frames + 1), base.pad_value)
                offset = start - left // hop
                features[:, :, start:end] = part[:, :, offset:offset + end - start]
                del part, chunk
        finally:
            base.normalize, base.pad_to = normalize, pad_to
        if normalize:
            features, _, _ = normalize_batch(features, lengths, normalize_type=normalize)
        features[:, :, frames:] = base.pad_value
        if pad_to == "max":
            features = F.pad(features, (0, base.max_length - features.shape[-1]), value=base.pad_value)
        elif pad_to > 0:
            features = F.pad(features, (0, (-features.shape[-1]) % pad_to), value=base.pad_value)
        return features, lengths
