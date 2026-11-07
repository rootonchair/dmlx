"""Filesystem locations. Overridable with DMLX_HOME for tests and portable installs."""

from __future__ import annotations

import os
from pathlib import Path


DATA_FOLDER_MARKER = ".dmlx-data"


def home() -> Path:
    override = os.environ.get("DMLX_HOME")
    base = Path(override) if override else Path.home() / "Library" / "Application Support" / "dMLX"
    if not base.exists():
        base.mkdir(parents=True)
        # Only a folder dMLX created gets the marker; the app's Uninstall deletes no other (Store.swift).
        (base / DATA_FOLDER_MARKER).touch()
    return base


def results_dir() -> Path:
    path = home() / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def uploads_dir() -> Path:
    path = home() / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    """Model specs added from the app (one YAML file per model)."""
    path = home() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def logs_dir() -> Path:
    path = home() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def hf_cache_dir() -> Path:
    """dMLX's own Hugging Face cache: every model it downloads, so Uninstall can delete them all."""
    return home() / "models-cache"


def use_own_caches() -> None:
    """Point huggingface_hub, hf_xet and mflux at folders inside the data folder instead of the caches
    they share with other tools. Must run before any of them is imported: they read these at import.

    The shared caches are never read: models found there are downloaded again into dMLX's own, so other
    tools keep theirs. An HF_HUB_CACHE that is already set (tests, a developer's own setup) is kept.
    """
    if os.environ.get("HF_HUB_CACHE"):
        return
    for name, folder in (("HF_HUB_CACHE", "models-cache"), ("HF_XET_CACHE", "xet-cache"), ("MFLUX_CACHE_DIR", "mflux-cache")):
        path = home() / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
