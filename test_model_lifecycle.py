"""Model lifecycle regressions. Run with .venv/bin/python test_model_lifecycle.py.

Imports the inference stack but mocks model loading: no downloads or GPU allocations.
"""
import gc
import os
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import engines
import pipeline


class ModelLifecycleTests(unittest.TestCase):
    def test_engine_load_is_idempotent_until_unloaded(self):
        loads = []

        class Model:
            def to(self, **kwargs):
                return self

            def eval(self):
                return self

        def restore(*args):
            model = Model()
            loads.append(model)
            return model

        engine = engines.NemoOfflineEngine(engines.resolve_spec("balanced"), device="cpu")
        with patch.object(engines.model_store, "load_nemo_model", side_effect=restore):
            first = engine.load()
            self.assertIs(engine.load(), first)
            self.assertEqual(len(loads), 1, "preload + transcribe must not restore twice")
            engine.unload()
            self.assertIsNot(engine.load(), first)
            self.assertEqual(len(loads), 2)

    def test_cuda_decoder_graph_setting_survives_decoder_recreation(self):
        from omegaconf import OmegaConf

        disable = Mock()
        model = SimpleNamespace(
            cfg=OmegaConf.create({"decoding": {"greedy": {"use_cuda_graph_decoder": True}}}),
            decoding=SimpleNamespace(decoding=SimpleNamespace(disable_cuda_graphs=disable)),
        )
        model.to = Mock(return_value=model)
        model.eval = Mock(return_value=model)
        engine = engines.NemoOfflineEngine(engines.resolve_spec("balanced"), device="cuda")
        with patch.object(engines.model_store, "load_nemo_model", return_value=model):
            engine.load()
        disable.assert_called_once_with()
        self.assertFalse(model.cfg.decoding.greedy.use_cuda_graph_decoder)
        self.assertEqual(model.to.call_args.kwargs["device"].type, "cuda")

    def _transcriber(self, fail_during):
        refs = []
        test = self

        class Token:
            pass

        class Impl:
            engine = SimpleNamespace(spec=SimpleNamespace(label="stub"), device="cuda", dtype=None)
            device = "cuda"
            calls = 0

            def fail_or_return(self):
                self.calls += 1
                if self.calls == 1:
                    # Like a failed CUDA operation's frame holding an activation.
                    activation = Token()
                    refs.append(weakref.ref(activation))
                    raise RuntimeError("CUDA out of memory. Tried to allocate 198 MiB.")
                return []

            def load(self):
                return self.fail_or_return()

            def transcribe(self, *args, **kwargs):
                return self.fail_or_return()

            def unload(self):
                gc.collect()
                test.assertIsNone(refs[0](), "failed traceback still holds the activation during CPU recovery")

        tr = pipeline.OfflineTranscriber.__new__(pipeline.OfflineTranscriber)
        tr._impl = Impl()
        tr._g = None
        tr.last_corrections = []
        tr.device = torch.device("cuda")
        tr.cuda_note = None
        with patch.dict(os.environ, {"SPEAKER_CUDA_ONLY": "0"}), patch.object(torch.cuda, "empty_cache"):
            if fail_during == "load":
                tr.load()
            else:
                with patch.object(pipeline, "_is_silent_wav", return_value=False):
                    tr.transcribe("unused.wav")
        self.assertEqual(tr._impl.calls, 2)
        self.assertEqual(tr.device.type, "cpu")

    def test_load_oom_releases_failed_frame_before_retry(self):
        self._transcriber("load")

    def test_decode_oom_releases_failed_frame_before_retry(self):
        self._transcriber("decode")


if __name__ == "__main__":
    unittest.main()
