"""Model download and local resolution (Hugging Face cache layout, shared with other tools)."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

from dmlx.registry import ModelSpec

log = logging.getLogger("dmlx.downloads")


def _api():
    from huggingface_hub import HfApi

    return HfApi()


def local_path(spec: ModelSpec) -> Path | None:
    """Snapshot directory if every file of the repo is already downloaded."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        path = Path(snapshot_download(spec.repo, local_files_only=True))
    except (LocalEntryNotFoundError, FileNotFoundError, OSError):
        return None
    for name in spec.components:
        if not any((path / name).glob("*.safetensors")):
            return None
    return path


def fill_component_sizes(spec: ModelSpec, path: Path) -> None:
    for name, comp in spec.components.items():
        on_disk = sum(f.stat().st_size for f in (path / name).glob("*.safetensors"))
        comp.size = max(comp.size, on_disk)
    spec.local_path = path


@dataclass
class DownloadState:
    model_id: str
    status: str = "idle"  # idle | downloading | done | error
    total: int = 0
    done: int = 0
    error: str | None = None
    files: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "model": self.model_id,
            "status": self.status,
            "total_bytes": self.total,
            "done_bytes": self.done,
            "progress": round(self.done / self.total, 4) if self.total else 0.0,
            "error": self.error,
        }


class Downloader:
    def __init__(self):
        self._states: dict[str, DownloadState] = {}
        self._lock = threading.Lock()

    def state(self, model_id: str) -> DownloadState:
        with self._lock:
            return self._states.setdefault(model_id, DownloadState(model_id))

    def start(self, spec: ModelSpec, on_done=None) -> DownloadState:
        state = self.state(spec.id)
        with self._lock:
            if state.status == "downloading":
                return state
            state.status, state.error, state.done = "downloading", None, 0
        threading.Thread(target=self._run, args=(spec, state, on_done), name=f"dl-{spec.id}", daemon=True).start()
        return state

    def _run(self, spec: ModelSpec, state: DownloadState, on_done) -> None:
        from huggingface_hub import hf_hub_download

        try:
            info = _api().model_info(spec.repo, files_metadata=True)
            files = [(s.rfilename, s.size or 0) for s in info.siblings]
            state.total = sum(size for _, size in files)
            completed = 0
            for name, size in files:
                done_event = threading.Event()
                watcher = threading.Thread(
                    target=self._watch_partial, args=(spec, name, size, completed, state, done_event), daemon=True
                )
                watcher.start()
                hf_hub_download(spec.repo, name)
                done_event.set()
                completed += size
                state.done = completed
            path = local_path(spec)
            if path is None:
                raise RuntimeError("Download finished but files are incomplete")
            fill_component_sizes(spec, path)
            state.status = "done"
            if on_done:
                on_done(spec)
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            log.exception("download failed for %s", spec.id)
            state.status, state.error = "error", str(exc)

    @staticmethod
    def _watch_partial(spec, name, size, completed, state, done_event) -> None:
        """Track the .incomplete blob huggingface_hub writes, for byte-level progress."""
        from huggingface_hub.constants import HF_HUB_CACHE

        repo_dir = Path(HF_HUB_CACHE) / f"models--{spec.repo.replace('/', '--')}" / "blobs"
        while not done_event.wait(0.5):
            partial = max((p.stat().st_size for p in repo_dir.glob("*.incomplete")), default=0) if repo_dir.exists() else 0
            state.done = completed + min(partial, size)
