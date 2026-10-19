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
