"""Where model weights come from: a local ``models/`` mirror first, the HF Hub second.

Kept separate from ``pipeline.py`` so that ``engines.py`` can load additional ASR
checkpoints without importing (and re-importing) the whole streaming pipeline.

A mirror is simply ``models/<repo-basename>.nemo`` — the same layout
``pipeline.py`` has always used, so ``models/Nemotron-3-Diarization.nemo`` and
``models/nemotron-3.5-asr-streaming-0.6b.nemo`` keep working untouched.
"""
from __future__ import annotations

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
    """Load a NeMo model on CPU from an explicit path, the local mirror, or the Hub."""
    from nemo.utils import logging as nemo_logging

    source, why = resolve(name)
    if why == "Hugging Face Hub":
        nemo_logging.info(f"Downloading {name} from Hugging Face Hub")
        return cls.from_pretrained(model_name=name, map_location=map_location)
    nemo_logging.info(f"Loading {why}")
    return cls.restore_from(restore_path=source, map_location=map_location)
