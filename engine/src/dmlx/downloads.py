"""Model download and local resolution, in dMLX's own Hugging Face cache (paths.hf_cache_dir()).

The layout is the standard Hugging Face one; only the folder is dMLX's, so Uninstall can delete every model.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path

from dmlx import paths
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
        path = Path(snapshot_download(spec.repo, revision=spec.revision, local_files_only=True))
    except (LocalEntryNotFoundError, FileNotFoundError, OSError):
        return None
    import json

    for name, component in spec.components.items():
        if component.external:
            continue
        comp = path / component.folder
        if not any(comp.glob("*.safetensors")):
            return None
        index = comp / "model.safetensors.index.json"
        if index.exists():
            shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
            if any(not (comp / shard).exists() for shard in shards):
                return None
    return path


def fill_component_sizes(spec: ModelSpec, path: Path) -> None:
    for name, comp in spec.components.items():
        if comp.external:
            continue
        on_disk = sum(f.stat().st_size for f in (path / comp.folder).glob("*.safetensors"))
        comp.size = max(comp.size, on_disk)
        if comp.stream and on_disk:
            from dmlx.adapters.streaming import stream_peak_bytes

            comp.stream_size = stream_peak_bytes(path / comp.folder, comp.stream)
    spec.local_path = path


def delete_local(spec: ModelSpec) -> int:
    """Delete this model's repo from dMLX's Hugging Face cache. Returns the bytes freed; raises if the
    files could not be deleted."""
    freed = delete_repos([spec.repo])
    spec.local_path = None
    for comp in spec.components.values():
        comp.stream_size = 0
    return freed


# Fetched by mflux for models with an external text encoder (Qwen Image 2.1's Qwen3.5-0.8B + adapter).
MFLUX_AUX_REPOS = ("AiArtLab/zen-image-edit",)
MFLUX_AUX_FOLDERS = ("zen-image-edit",)  # what mflux keeps of it in its own cache


def delete_mflux_aux() -> int:
    """Delete what mflux fetched for external text encoders. Returns the bytes freed."""
    freed = delete_repos(list(MFLUX_AUX_REPOS))
    mflux = os.environ.get("MFLUX_CACHE_DIR")
    for folder in MFLUX_AUX_FOLDERS if mflux else ():
        path = Path(mflux) / folder
        if path.is_dir():
            freed += _size_on_disk(path)
            shutil.rmtree(path)
    return freed


def _repo_folder(repo: str) -> str:
    return "models--" + repo.replace("/", "--")


def _size_on_disk(folder: Path) -> int:
    """Bytes the folder's own files take, links not followed (a link into the shared store counts 0)."""
    total = 0
    for path in folder.rglob("*"):
        if path.is_file() and not path.is_symlink():
            total += path.lstat().st_blocks * 512
    return total


def delete_repos(repos: list[str]) -> int:
    """Delete repos from dMLX's Hugging Face cache. Returns the bytes freed; raises if deleting failed,
    leaving the repo folder in place so its files are not orphaned in the shared store.

    Goes through huggingface_hub's own deletion: recent versions keep files in a shared, deduplicated store
    that repo folders only link to, and only the library knows which shared files no other repo still uses.
    It selects revisions by commit hash across the whole cache, and a duplicated repo keeps the commits of
    the one it copies, so a commit another repo also has is left out of it; that repo's folder is removed
    directly instead (its shared files then stay in the store).
    """
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.constants import HF_HUB_CACHE

    cache_dir = Path(HF_HUB_CACHE)
    folders = [cache_dir / _repo_folder(repo) for repo in repos]
    if not any(f.exists() for f in folders):
        return 0
    wanted = {repo.lower() for repo in repos}
    cache = scan_cache_dir(cache_dir)
    ours = [r for r in cache.repos if r.repo_type == "model" and r.repo_id.lower() in wanted]
    elsewhere = {rev.commit_hash for r in cache.repos if r not in ours for rev in r.revisions}
    revisions = [rev.commit_hash for r in ours for rev in r.revisions if rev.commit_hash not in elsewhere]
    freed = 0
    if revisions:
        strategy = cache.delete_revisions(*revisions)
        freed = strategy.expected_freed_size
        strategy.execute()
    for folder in folders:
        if folder.exists():
            freed += _size_on_disk(folder)  # what huggingface_hub left: refs, a duplicated repo's own files
            shutil.rmtree(folder)
    return freed


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
            info = _api().model_info(spec.repo, revision=spec.revision, files_metadata=True)
            external = {name for name, component in spec.components.items() if component.external}
            files = [
                (s.rfilename, s.size or 0)
                for s in info.siblings
                if s.rfilename.split("/", 1)[0] not in external
            ]
            state.total = sum(size for _, size in files)
            completed = 0
            for name, size in files:
                done_event = threading.Event()
                watcher = threading.Thread(
                    target=self._watch_partial, args=(spec, name, size, completed, state, done_event), daemon=True
                )
                watcher.start()
                hf_hub_download(spec.repo, name, revision=spec.revision)
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

        # Older hub versions keep blobs per repo; newer ones share a cache-wide blobs folder.
        dirs = [Path(HF_HUB_CACHE) / f"models--{spec.repo.replace('/', '--')}" / "blobs", Path(HF_HUB_CACHE) / "blobs"]
        while not done_event.wait(0.5):
            partial = 0
            for d in dirs:
                if d.exists():
                    for p in d.rglob("*.incomplete"):
                        try:
                            partial = max(partial, p.stat().st_size)
                        except FileNotFoundError:
                            pass
            state.done = completed + min(partial, size)
