"""Filesystem locations. Overridable with DMLX_HOME for tests and portable installs."""

from __future__ import annotations

import os
from pathlib import Path


def home() -> Path:
    override = os.environ.get("DMLX_HOME")
    base = Path(override) if override else Path.home() / "Library" / "Application Support" / "dMLX"
    base.mkdir(parents=True, exist_ok=True)
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


# Where huggingface_hub and mflux kept their caches before the engine pointed them at its own folders:
# (hub cache, mflux cache), or None when nothing was redirected (tests, or HF_HUB_CACHE set on purpose).
_shared_caches: tuple[Path, Path] | None = None


def hf_cache_dir() -> Path:
    """dMLX's own Hugging Face cache: every model it downloads, so Uninstall can delete them all."""
    return home() / "models-cache"


def use_own_caches() -> None:
    """Point huggingface_hub, hf_xet and mflux at folders inside the data folder instead of the caches
    they share with other tools. Must run before any of them is imported: they read these at import.

    An HF_HUB_CACHE that is already set (tests, a developer's own setup) is kept, and nothing moves.
    """
    global _shared_caches
    if os.environ.get("HF_HUB_CACHE"):
        return
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    hf_home = Path(os.environ.get("HF_HOME") or cache / "huggingface")
    mflux = Path(os.environ.get("MFLUX_CACHE_DIR") or Path.home() / "Library" / "Caches" / "mflux")
    _shared_caches = (hf_home / "hub", mflux)
    for name, folder in (("HF_HUB_CACHE", "models-cache"), ("HF_XET_CACHE", "xet-cache"), ("MFLUX_CACHE_DIR", "mflux-cache")):
        path = home() / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)


def shared_caches() -> tuple[Path, Path] | None:
    return _shared_caches
