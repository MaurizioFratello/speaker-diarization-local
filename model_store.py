"""Where model weights come from: a local ``models/`` mirror first, the HF Hub second.

Kept separate from ``pipeline.py`` so that ``engines.py`` can load additional ASR
checkpoints without importing (and re-importing) the whole streaming pipeline.

A mirror is simply ``models/<repo-basename>.nemo`` — the same layout
``pipeline.py`` has always used, so ``models/Nemotron-3-Diarization.nemo`` and
``models/nemotron-3.5-asr-streaming-0.6b.nemo`` keep working untouched.
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import tarfile
from pathlib import Path
from typing import Any, Type

MODELS_DIR = Path(__file__).resolve().parent / "models"


def mirror_path(name: str) -> Path:
    """Path a checkpoint would be mirrored to: ``models/<repo-basename>.nemo``."""
    return MODELS_DIR / f"{str(name).rstrip('/').split('/')[-1]}.nemo"


def resolve(name: str) -> tuple[str, str]:
    """Return ``(restore_path_or_repo, description)`` for a model name.

    ``name`` may be an explicit ``.nemo`` file, a mirrored Hub id, or a Hub id
    that still has to be downloaded.
    """
    if str(name).endswith(".nemo"):
        return str(name), str(name)
    local = mirror_path(name)
    if local.is_file():
        return str(local), f"local mirror {local}"
    return str(name), "Hugging Face Hub"


def load_nemo_model(name: str, cls: Type[Any], map_location: str = "cpu"):
    """Load a NeMo model from an explicit path, the local mirror, or the Hub.

    A local mirror is normalised first (see ``_prefer_cpu_submodels``): Canary
    bundles a second ASR whose own config pins it to CUDA, which is what makes
    ``device="cpu"`` impossible for that engine without the rewrite.
    """
    from nemo.utils import logging as nemo_logging

    source, why = resolve(name)
    if why != "Hugging Face Hub":
        source = _prefer_cpu_submodels(source)
    with _nested_restores_on(map_location):
        if why == "Hugging Face Hub":
            nemo_logging.info(f"Downloading {name} from Hugging Face Hub")
            return cls.from_pretrained(model_name=name, map_location=map_location)
        nemo_logging.info(f"Loading {why}")
        return cls.restore_from(restore_path=source, map_location=map_location)


@contextlib.contextmanager
def _nested_restores_on(map_location: str):
    """Make a ``restore_from`` nested inside another restore use ``map_location``.

    The bundled sub-model of a multi-model checkpoint is restored with a bare
    ``ASRModel.restore_from(path)``. With ``map_location=None``, ``torch.load`` puts
    every tensor back on the device it was saved from — CUDA — so the allocation
    happens before the caller's choice of device is ever consulted, and a busy card
    turns it into an OOM inside NeMo. Config rewriting alone does not help (measured:
    ``device: cpu`` in the sub-config still lands on the GPU); the state dict has to
    be mapped. Shadowing the classmethod for the duration of our own restore is the
    smallest fix that does not need a NeMo fork.
    """
    try:
        from nemo.collections.asr import models as asr_models_pkg

        ASRModel = asr_models_pkg.ASRModel           # re-exported; the defining module
        owner = next(c for c in ASRModel.__mro__ if "restore_from" in vars(c))  # moves with
    except Exception as exc:  # unexpected layout: stock behaviour, and say so
        from nemo.utils import logging as _nl
        _nl.warning(f"cannot pin nested restores ({exc}); a bundled sub-model may still "
                    f"load onto the GPU")
        yield
        return
    descriptor = vars(owner)["restore_from"]
    raw = getattr(descriptor, "__func__", descriptor)   # the function, not the bound method
    target = map_location

    def patched(cls, *args, **kwargs):
        # Signature-agnostic on purpose: callers use ``restore_path=`` here and a
        # positional path in other NeMo versions, and a nested restore must not fail
        # just because our wrapper guessed the parameter name wrong.
        kwargs.setdefault("map_location", target)
        if kwargs.get("map_location") in (None, ""):
            kwargs["map_location"] = target
        return raw(cls, *args, **kwargs)

    owner.restore_from = classmethod(patched)
    try:
        yield
    finally:
        owner.restore_from = descriptor


_SUBCONFIG_DEVICE = re.compile(r"^(\s*)device:\s*cuda\w*\s*$", re.M)


def _prefer_cpu_submodels(path: str) -> str:
    """Pin the *bundled sub-models* of a local ``.nemo`` mirror to CPU, once, in place.

    Canary 1B v2 carries a second, smaller ASR in the same archive for forced
    alignment (``timestamps_asr_model_config.yaml``). NeMo restores it with a bare
    ``ASRModel.restore_from(path)`` — no ``map_location``, no overrides — and that
    config's own ``device: cuda`` decides where the modules get built. On a CPU run,
    or on a card somebody else is filling, the result is a CUDA OOM raised inside
    someone else's constructor *after* the outer model already loaded, so the
    reservation is held and the caller's device=cpu fallback cannot even start.
    Measured here: Canary with 2.2 GiB free dies this way on both paths.

    Building on CPU and letting the caller move the whole model to the target device
    in one step is what the outer model already does, so the rewrite is free on the
    GPU path and unblocks the CPU one. Only ``*_config.yaml`` members that already
    say ``device: cuda`` are touched; anything else in the archive is copied as is.
    """
    p = Path(path)
    if not p.is_file():
        return str(p)
    try:
        with tarfile.open(p) as tf:
            rewrite: dict[str, bytes] = {}
            for m in tf.getmembers():
                if not (m.isfile() and m.name.endswith("_config.yaml") and "model" in m.name):
                    continue
                text = tf.extractfile(m).read().decode("utf-8")
                fixed = _SUBCONFIG_DEVICE.sub(lambda mo: mo.group(1) + "device: cpu", text)
                if fixed != text:
                    rewrite[m.name] = fixed.encode("utf-8")
            if not rewrite:
                return str(p)
            tmp = p.with_name(p.name + ".tmp")
            with tarfile.open(tmp, "w") as out:
                for m in tf.getmembers():
                    if m.name in rewrite:
                        info = tarfile.TarInfo(m.name)
                        info.size = len(rewrite[m.name])
                        info.mtime = m.mtime
                        out.addfile(info, io.BytesIO(rewrite[m.name]))
                    elif m.isdir():
                        out.addfile(m)
                    else:
                        out.addfile(m, tf.extractfile(m))
        os.replace(tmp, p)
        from nemo.utils import logging as nemo_logging
        nemo_logging.info(
            f"{p.name}: pinned bundled sub-model config(s) to CPU "
            f"({', '.join(sorted(rewrite))}); the caller chooses the device once")
    except OSError as exc:
        # A mirror we cannot rewrite is still a mirror we can load: worst case the
        # caller gets the usual out-of-memory message on a full card.
        import warnings
        warnings.warn(f"could not normalise {p}: {exc}")
        tmp = p.with_name(p.name + ".tmp")
        if tmp.exists():
            tmp.unlink()
    return str(p)
