"""Check bounded CUDA preprocessing against NeMo's original whole-file features."""
import unittest
from unittest.mock import patch

import torch
from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures
from gpu_features import BoundedMelFeatures


class BoundedFeatureTests(unittest.TestCase):
    def test_matches_whole_recording_at_chunk_boundaries(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for exact_pad in (False, True):
                with self.subTest(device=device, exact_pad=exact_pad):
                    torch.manual_seed(12)
                    base = FilterbankFeatures(nfilt=128, dither=0, exact_pad=exact_pad).to(device).eval()
                    x = torch.randn(1, 16000 * 4 + 73, device=device)
                    length = torch.tensor([x.shape[-1]], device=device)
                    expected, expected_len = base(x, length)
                    bounded = BoundedMelFeatures(base, chunk_seconds=1).eval()
                    with patch.object(base, "stft", wraps=base.stft) as stft:
                        actual, actual_len = bounded(x, length)
                    torch.testing.assert_close(actual_len, expected_len)
                    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
                    self.assertGreater(stft.call_count, 1)
                    self.assertLess(max(call.args[0].shape[-1] for call in stft.call_args_list), 18000)
                    self.assertEqual(base.normalize, "per_feature")
                    self.assertEqual(base.pad_to, 16)

    def test_settings_restored_after_failed_chunk(self):
        base = FilterbankFeatures(dither=0).eval()
        bounded = BoundedMelFeatures(base, chunk_seconds=1).eval()
        with patch.object(base, "stft", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                bounded(torch.ones(1, 32000), torch.tensor([32000]))
        self.assertEqual(base.normalize, "per_feature")
        self.assertEqual(base.pad_to, 16)


if __name__ == "__main__":
    unittest.main()
