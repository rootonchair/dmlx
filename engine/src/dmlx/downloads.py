"""Model download and local resolution, in dMLX's own Hugging Face cache (paths.hf_cache_dir()).

The layout is the standard Hugging Face one; only the folder is dMLX's, so Uninstall can delete every model.
"""

from __future__ import annotations

import ctypes
import logging
import os
import shutil
import threading
import time
from collections.abc import Iterable
from contextlib import contextmanager
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
    # Downloaded means every repo it needs is complete: the job itself never downloads (task 19).
    if any(extra_path(extra) is None for extra in extra_repos(spec)):
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
    spec.disk_size = _files_size(path) + sum(_extra_size_on_disk(extra) for extra in extra_repos(spec))
    spec.local_path = path


def _files_size(snapshot: Path) -> int:
    """Bytes of every file of a snapshot (its links point at the blobs)."""
    total = 0
    for item in snapshot.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            pass
    return total


def delete_local(spec: ModelSpec, keep_repos: Iterable[str] = ()) -> int:
    """Delete this model's repo, and the extra repos it downloaded with it, from dMLX's Hugging Face cache.
    Returns the bytes freed, counting an unfinished download's files too (the cache scan sees only finished
    revisions). `keep_repos`: extra repos another model still uses; they stay."""
    keep = {repo.lower() for repo in keep_repos}
    freed = _delete_repo(spec.repo)
    for extra in extra_repos(spec):
        if extra.repo.lower() in keep:
            continue
        freed += _delete_repo(extra.repo)
        if extra.tensors:
            dest = _mflux_cache() / extra.tensors[2]
            for path in (dest, dest.with_name(dest.name + ".incomplete")):
                try:
                    freed += path.stat().st_size
                    path.unlink()
                except OSError:
                    pass
    spec.local_path = None
    spec.disk_size = 0
    for comp in spec.components.values():
        comp.stream_size = 0
    return freed


def _delete_repo(repo: str) -> int:
    folder = _hub_cache() / _repo_folder(repo)
    own = 0 if folder.is_symlink() else sum(
        p.lstat().st_size for p in folder.rglob("*") if p.is_file() and not p.is_symlink())
    return max(_delete_from_cache(None, [repo]), own)


def repo_on_disk(repo: str) -> bool:
    """Whether any file of the repo is in the cache (a download, finished or not)."""
    blobs = _blobs(repo)
    return blobs.is_dir() and any(blobs.iterdir())


@dataclass(frozen=True)
class ExtraRepo:
    """Another repo a model needs, downloaded with it by the same resumable downloader, so the model's
    first job never downloads (task 19: mflux fetched Qwen Image 2.1's 2.1 GB text encoder itself, with
    huggingface_hub and hf_xet, inside the first job)."""

    repo: str
    revision: str  # a commit
    allow: tuple[str, ...]  # the files the adapter reads (mflux's allow_patterns)
    required: tuple[str, ...]  # files that must be there for the repo to count as downloaded
    size: int  # bytes of the allowed files and the tensor file, shown before the download
    # Tensors kept as a file of their own: (index file in the repo, tensor-name prefix, path under mflux's
    # cache). mflux fetches Qwen Image 2.1's 0.3 GB text_fusion adapter as one byte range of a 14.5 GB shard.
    tensors: tuple[str, str, str] | None = None


_ZEN_REVISION = "3d5717559ff5bd8b431e66032b285a3fa4a1b20c"
# Qwen3.5-0.8B and the text_fusion adapter, read by mflux's Qwen21SmallTextEncoder.load() (zen_image_edit_weights).
ZEN_IMAGE_EDIT = ExtraRepo(
    repo="AiArtLab/zen-image-edit",
    revision=_ZEN_REVISION,
    allow=("processor/*", "text_encoder/config.json", "text_encoder/model.safetensors", "transformer/config.json",
           "transformer/diffusion_pytorch_model.safetensors.index.json"),
    required=("processor/tokenizer.json", "processor/tokenizer_config.json", "processor/preprocessor_config.json",
              "text_encoder/config.json", "text_encoder/model.safetensors", "transformer/config.json",
              "transformer/diffusion_pytorch_model.safetensors.index.json"),
    size=1_769_886_998 + 319_497_084,  # the files, and the text_fusion tensors (measured 2026-11-10)
    tensors=("transformer/diffusion_pytorch_model.safetensors.index.json", "text_fusion.",
             f"zen-image-edit/{_ZEN_REVISION}/text_fusion.safetensors"),
)
# By adapter, for models whose text encoder is external (built-in or added from the app).
EXTRA_REPOS: dict[str, tuple[ExtraRepo, ...]] = {"mflux.qwen_image_21": (ZEN_IMAGE_EDIT,)}


def extra_repos(spec: ModelSpec) -> tuple[ExtraRepo, ...]:
    if not any(c.external for c in spec.components.values()):
        return ()
    return EXTRA_REPOS.get(spec.adapter, ())


def _mflux_cache() -> Path:
    """mflux's cache folder, as mflux works it out (mflux.cli.defaults.defaults.MFLUX_CACHE_DIR)."""
    if os.environ.get("MFLUX_CACHE_DIR"):
        return Path(os.environ["MFLUX_CACHE_DIR"]).resolve()
    import platformdirs

    return Path(platformdirs.user_cache_dir(appname="mflux"))


def extra_path(extra: ExtraRepo) -> Path | None:
    """The extra repo's snapshot when all its files (and its tensor file) are there; never downloads."""
    snapshot = _hub_cache() / _repo_folder(extra.repo) / "snapshots" / extra.revision
    if not all((snapshot / name).is_file() for name in extra.required):
        return None
    if extra.tensors and not (_mflux_cache() / extra.tensors[2]).is_file():
        return None
    return snapshot


def _extra_size_on_disk(extra: ExtraRepo) -> int:
    total = _blobs_size(extra.repo)
    if extra.tensors:
        dest = _mflux_cache() / extra.tensors[2]
        for path in (dest, dest.with_name(dest.name + ".incomplete")):
            try:
                total += path.stat().st_size
            except OSError:
                pass
    return total


def extras_download_size(spec: ModelSpec) -> int:
    return sum(extra.size for extra in extra_repos(spec))


# The extra repos (above): owned by dMLX, moved in from the shared cache like its models.
MFLUX_AUX_REPOS = tuple(dict.fromkeys(e.repo for extras in EXTRA_REPOS.values() for e in extras))
MFLUX_AUX_FOLDERS = ("zen-image-edit",)  # what mflux keeps of it in its own cache


def _repo_folder(repo: str) -> str:
    return "models--" + repo.replace("/", "--")


_clonefile = getattr(ctypes.CDLL(None, use_errno=True), "clonefile", None)


def _clone(src: Path, dst: Path) -> None:
    """APFS clone: instant, and the two copies share their blocks, so it takes no extra space."""
    if _clonefile is None or _clonefile(os.fsencode(src), os.fsencode(dst), 0) != 0:
        raise OSError(ctypes.get_errno(), f"cannot clone {src}")


def _clone_repo(src: Path, dst: Path) -> None:
    """Rebuild a cached repo at dst: refs and snapshot links as they are, every blob as a file of its own.

    A blob may be a link into the cache-wide shared store of recent huggingface_hub versions, which stays
    behind; the clone takes the file it points to.
    """
    staging = dst.with_name(dst.name + ".moving")
    shutil.rmtree(staging, ignore_errors=True)
    for folder, dirs, files in os.walk(src):
        rel = Path(folder).relative_to(src)
        (staging / rel).mkdir(parents=True, exist_ok=True)
        for name in dirs:
            if (Path(folder) / name).is_symlink():
                os.symlink(os.readlink(Path(folder) / name), staging / rel / name)
        for name in files:
            path, target = Path(folder) / name, staging / rel / name
            if name.endswith(".incomplete"):
                continue
            if rel.parts[:1] == ("blobs",):
                _clone(path.resolve(strict=True), target)
            elif path.is_symlink():
                os.symlink(os.readlink(path), target)
            else:
                _clone(path, target)
    os.rename(staging, dst)


def _delete_from_cache(cache_dir: Path | None, repos: list[str]) -> int:
    """Delete repos from a Hugging Face cache (None: the current one). Returns the bytes freed.

    Goes through huggingface_hub's own deletion: recent versions keep files in a shared, deduplicated store
    that repo folders only link to, and only the library knows which shared files no other repo still uses.
    It selects revisions by commit hash across the whole cache, and a duplicated repo keeps the commits of
    the one it copies, so a commit another repo also has is left out of it; that repo's folder is removed
    directly instead (its shared files then stay in the store).
    """
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.constants import HF_HUB_CACHE

    cache_dir = Path(cache_dir or HF_HUB_CACHE)
    wanted = {repo.lower() for repo in repos}
    freed = 0
    try:
        cache = scan_cache_dir(cache_dir)
        ours = [r for r in cache.repos if r.repo_type == "model" and r.repo_id.lower() in wanted]
        elsewhere = {rev.commit_hash for r in cache.repos if r not in ours for rev in r.revisions}
        revisions = [rev.commit_hash for r in ours for rev in r.revisions if rev.commit_hash not in elsewhere]
        if revisions:
            strategy = cache.delete_revisions(*revisions)
            freed = strategy.expected_freed_size
            strategy.execute()
    except Exception:  # noqa: BLE001 - a corrupt cache entry must not stop the engine
        log.exception("could not delete %s from the Hugging Face cache", ", ".join(repos))
    for repo in repos:
        shutil.rmtree(cache_dir / _repo_folder(repo), ignore_errors=True)
    return freed


def adopt(repos: Iterable[str]) -> list[str]:
    """Move dMLX's models out of the shared Hugging Face cache into dMLX's own, so they are not downloaded
    again. A repo moves only when dMLX's cache does not have it yet. Returns the repos moved.

    The copy is an APFS clone, so it is instant; a repo that cannot be cloned (another volume) stays where
    it is and is downloaded again when needed.
    """
    shared = paths.shared_caches()
    if shared is None:
        return []
    old_hub, old_mflux = shared
    new_hub = Path(os.environ["HF_HUB_CACHE"])
    moved = []
    for repo in dict.fromkeys(repos):
        src, dst = old_hub / _repo_folder(repo), new_hub / _repo_folder(repo)
        if src.resolve() == dst.resolve() or not src.is_dir() or dst.exists():
            continue
        try:
            _clone_repo(src, dst)
            moved.append(repo)
            log.info("moved %s from the shared Hugging Face cache into dMLX's", repo)
        except OSError as exc:
            log.warning("kept %s in the shared cache (%s); it is downloaded again if needed", repo, exc)
            shutil.rmtree(dst.with_name(dst.name + ".moving"), ignore_errors=True)
    if moved:
        _delete_from_cache(old_hub, moved)
    new_mflux = Path(os.environ["MFLUX_CACHE_DIR"])
    for folder in MFLUX_AUX_FOLDERS:
        src, dst = old_mflux / folder, new_mflux / folder
        if src.is_dir() and not dst.exists() and src.resolve() != dst.resolve():
            try:
                os.rename(src, dst)
            except OSError:
                pass  # another volume: mflux fetches it again (a few hundred MB)
    return moved


CHUNK = 1024 * 1024
RETRIES = 3


class DownloadCancelled(Exception):
    pass


# One writer per file (fix round 2 of task 19): two models that share an extra repo, pulled at once, both
# appended to the same `.incomplete`, and one accepted a damaged blob. A download takes the file's lock;
# the other waits, then finds the file done and reuses it (or continues the part the first one left).
_file_locks: dict[str, threading.Lock] = {}
_file_locks_guard = threading.Lock()


def _file_lock(path: Path) -> threading.Lock:
    with _file_locks_guard:
        return _file_locks.setdefault(str(path), threading.Lock())


@contextmanager
def _owning(path: Path, cancel: threading.Event):
    """Hold the file's lock while writing it; a cancel stops the wait too."""
    lock = _file_lock(path)
    while not lock.acquire(timeout=0.1):
        if cancel.is_set():
            raise DownloadCancelled
    try:
        yield
    finally:
        lock.release()


def _being_written(path: Path) -> bool:
    lock = _file_lock(path.with_name(path.name.removesuffix(".incomplete")))
    if lock.acquire(blocking=False):
        lock.release()
        return False
    return True


def _file_sha256(path: Path, cancel: threading.Event) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * CHUNK):
            if cancel.is_set():
                raise DownloadCancelled
            digest.update(chunk)
    return digest.hexdigest()


def _hub_cache() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def _blobs(repo: str) -> Path:
    return _hub_cache() / _repo_folder(repo) / "blobs"


def _is_orphan(path: Path) -> bool:
    """A partial file no download can resume: huggingface_hub names its own `<etag>.<random>.incomplete`
    and starts a new one each time; dMLX's are `<etag>.incomplete`."""
    return path.name.endswith(".incomplete") and path.name.count(".") > 1


def clean_partials(repo: str, keep: Iterable[str] | None = None, everything: bool = False) -> int:
    """Delete partial files of a repo: orphans always, then those not named in `keep` (a fresh download
    keeps its own, to resume), or all of them. Returns the bytes freed. A repo folder that is a link
    (a test setup pointing at another cache) is left alone."""
    keep = None if keep is None else set(keep)
    freed = 0
    blobs = _blobs(repo)
    if blobs.parent.is_symlink() or not blobs.is_dir():
        return 0
    for path in blobs.glob("*.incomplete"):
        if _being_written(path):
            continue  # another model's download is filling it now (a shared extra repo)
        if everything or _is_orphan(path) or (keep is not None and path.name not in keep):
            try:
                freed += path.stat().st_size
                path.unlink()
            except FileNotFoundError:
                pass
    if freed:
        log.info("deleted %.2f GB of unfinished download files of %s", freed / 1e9, repo)
    return freed


def partial_bytes(spec: ModelSpec) -> int:
    """Bytes already on disk for this model: its repo and its extra repos (finished files and partial ones)."""
    return _blobs_size(spec.repo) + sum(_extra_size_on_disk(extra) for extra in extra_repos(spec))


def _blobs_size(repo: str) -> int:
    blobs = _blobs(repo)
    total = 0
    if blobs.is_dir():
        for path in blobs.iterdir():
            try:
                total += path.stat().st_size
            except OSError:
                pass
    return total


def _stream(url: str, headers: dict[str, str]):
    """A streamed GET (a context manager with `.status_code`, `.headers` and `.iter_bytes()`)."""
    import httpx
    from huggingface_hub.utils import get_session

    return get_session().stream("GET", url, headers=headers, follow_redirects=True,
                                timeout=httpx.Timeout(30.0, connect=15.0))


def _abort(response) -> None:
    """Shut the connection under a response another thread is reading, so a stalled read ends now
    instead of at the 30 s read timeout. Best effort: a response without a socket is only closed."""
    import socket

    try:
        stream = (getattr(response, "extensions", None) or {}).get("network_stream")
        sock = stream.get_extra_info("socket") if stream is not None else None
        if sock is not None:
            # The plain socket's shutdown, also for a TLS socket: SSLSocket.shutdown would drop its TLS
            # state under the reading thread. The reader then gets an error or the end of the stream.
            socket.socket.shutdown(sock, socket.SHUT_RDWR)
            return
    except OSError:
        return  # already closed
    except Exception:  # noqa: BLE001 - never fail a cancel over this
        log.debug("could not shut a download's connection", exc_info=True)
    close = getattr(response, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001
            log.debug("could not close a download's response", exc_info=True)


@dataclass
class DownloadState:
    model_id: str
    status: str = "idle"  # idle | downloading | done | error (idle again after a cancel)
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


@dataclass
class _File:
    name: str
    size: int
    etag: str
    sha256: str | None  # LFS files: checked once the file is complete


@dataclass
class _Repo:
    """One repo of a download: the model's own, or an extra one."""

    repo: str
    revision: str | None
    commit: str
    files: list[_File]


@dataclass
class _Tensors:
    """A byte range of a remote safetensors file, saved as a safetensors file of its own (mflux's layout)."""

    url: str
    dest: Path
    header: bytes  # the new file's header (length and JSON); the range follows it
    start: int  # file-absolute byte range in the remote file
    end: int

    @property
    def size(self) -> int:
        return len(self.header) + self.end - self.start

    def on_disk(self) -> int:
        for path in (self.dest, self.dest.with_name(self.dest.name + ".incomplete")):
            try:
                return min(path.stat().st_size, self.size)
            except OSError:
                continue
        return 0


class HTTPStatusError(RuntimeError):
    def __init__(self, status: int, name: str):
        super().__init__(f"HTTP {status} for {name}")
        self.status = status


def plain_error(exc: BaseException) -> str:
    """What the app shows when a download stops on an error: a plain sentence, not `[Errno 28] ...`."""
    import errno

    again = "Download again to continue from where it stopped."
    status = getattr(exc, "status", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return "The disk is full. Free some space, then download again; it continues from where it stopped."
    if status in (401, 403):
        return ("Hugging Face refused the download. The model may need you to sign in or accept its license "
                "on huggingface.co.")
    if status == 404:
        return "Hugging Face no longer has this model's files."
    if status == 429:
        return f"Hugging Face is limiting downloads right now. Wait a few minutes. {again}"
    if isinstance(status, int) and status >= 500:
        return f"Hugging Face is having trouble right now. Try again later. {again}"
    if "damaged in transfer" in str(exc):
        return f"A file was damaged in transfer and was deleted. {again}"
    import httpx

    if isinstance(exc, (httpx.TransportError, ConnectionError, TimeoutError)) or "connection" in str(exc).lower():
        return f"The connection to Hugging Face was lost. Check your internet connection. {again}"
    return f"The download stopped because of an error ({exc}). {again}"


def _retryable(exc: BaseException) -> bool:
    """A dropped connection or a server hiccup is tried again; a full disk or a refusal is not."""
    import errno

    if isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EACCES, errno.EROFS, errno.EDQUOT):
        return False
    if isinstance(exc, HTTPStatusError):
        return exc.status in (408, 429) or exc.status >= 500
    return True


class Downloader:
    """Downloads a model's repo, and the extra repos it needs, file by file into the Hugging Face cache
    layout, over plain HTTP.

    Not hf_hub_download (task 19): it writes each attempt to a new `<etag>.<random>.incomplete`, so an
    interrupted download (a quit, a crash) started again from zero and left the old file behind, and it
    cannot be stopped from another thread. Here each file goes to `blobs/<etag>.incomplete`; a later pull
    continues it with a Range request, and a cancel stops between two chunks.
    """

    def __init__(self):
        self._states: dict[str, DownloadState] = {}
        self._cancels: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._responses: dict[threading.Event, object] = {}  # the response each download is reading
        self._lock = threading.Lock()

    def state(self, model_id: str) -> DownloadState:
        with self._lock:
            return self._states.setdefault(model_id, DownloadState(model_id))

    def start(self, spec: ModelSpec, on_done=None) -> DownloadState:
        state = self.state(spec.id)
        with self._lock:
            if state.status == "downloading":
                return state
            # A cancelled download may still be finishing its last chunk: the new one waits for it, so a
            # Resume right after a Cancel is never lost and two threads never write the same file.
            previous = self._threads.get(spec.id)
            # What is on disk already counts from the first answer: a resumed download does not show 0.
            state.status, state.error, state.done = "downloading", None, partial_bytes(spec)
            cancel = self._cancels[spec.id] = threading.Event()
            thread = threading.Thread(target=self._run, args=(spec, state, on_done, cancel, previous),
                                      name=f"dl-{spec.id}", daemon=True)
            self._threads[spec.id] = thread
        thread.start()
        return state

    def cancel(self, model_id: str, wait: float = 0.0) -> bool:
        """Stop a running download, keeping its files so a later pull resumes. False when none runs.

        The download counts as stopped at once (status idle): it writes no more progress, and a pull
        started now (or a Remove) waits for its thread. The connection it is reading is shut, so the
        thread ends at once even when the network stalls (fix round 3 of task 19: this used to wait up
        to 10 s for a stalled read). It does not wait for the thread unless `wait` is given.
        """
        with self._lock:
            state = self._states.get(model_id)
            if state is None or state.status != "downloading":
                return False
            cancel = self._cancels[model_id]
            cancel.set()
            state.status, state.error = "idle", None
            thread = self._threads.get(model_id)
            response = self._responses.get(cancel)
        if response is not None:
            _abort(response)
        if wait and thread is not None and thread is not threading.current_thread():
            thread.join(wait)
        return True

    @contextmanager
    def _reading(self, cancel: threading.Event, response):
        """While a response is read, a cancel can shut its connection."""
        with self._lock:
            self._responses[cancel] = response
        try:
            if cancel.is_set():  # cancelled just before the response was known
                raise DownloadCancelled
            yield response
        finally:
            with self._lock:
                if self._responses.get(cancel) is response:
                    del self._responses[cancel]

    def wait_stopped(self, model_id: str, timeout: float = 30.0) -> bool:
        """Wait for a cancelled download's thread to end (before its files are deleted)."""
        with self._lock:
            thread = self._threads.get(model_id)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
            return not thread.is_alive()
        return True

    def _run(self, spec: ModelSpec, state: DownloadState, on_done, cancel: threading.Event | None = None,
             previous: threading.Thread | None = None) -> None:
        cancel = cancel or threading.Event()
        if previous is not None and previous.is_alive() and previous is not threading.current_thread():
            previous.join()

        def current() -> bool:
            """Still this download's thread: a cancelled one must not overwrite a newer download's state."""
            return self._cancels.get(spec.id, cancel) is cancel

        def check() -> None:
            if cancel.is_set():
                raise DownloadCancelled

        try:
            check()
            external = {name for name, component in spec.components.items() if component.external}
            repos = [self._list(spec.repo, spec.revision, lambda name: name.split("/", 1)[0] not in external)]
            check()
            tensors: list[_Tensors] = []
            for extra in extra_repos(spec):
                from huggingface_hub.utils import filter_repo_objects

                def allowed(name: str, patterns=list(extra.allow)) -> bool:
                    return bool(list(filter_repo_objects([name], allow_patterns=patterns)))

                repos.append(self._list(extra.repo, extra.revision, allowed))
                check()
                if extra.tensors and not (_mflux_cache() / extra.tensors[2]).is_file():
                    tensors.append(self._plan_tensors(extra, repos[-1].commit))
                    check()
            for repo in repos:
                blobs = _blobs(repo.repo)
                blobs.mkdir(parents=True, exist_ok=True)
                # A fresh start: only this download's own partial files are kept (to resume).
                clean_partials(repo.repo, keep={f"{f.etag}.incomplete" for f in repo.files})
            if current():
                state.total = sum(f.size for r in repos for f in r.files) + sum(t.size for t in tensors)

            def on_disk(after: int = 0) -> int:
                """Bytes on disk of the files from the `after`-th on (done files count as whole)."""
                items = [(r, f) for r in repos for f in r.files]
                return (sum(self._on_disk(_blobs(r.repo), f) for r, f in items[after:])
                        + sum(t.on_disk() for t in tensors))

            if current():
                state.done = on_disk()
            completed, index = 0, 0
            for repo in repos:
                blobs = _blobs(repo.repo)
                snapshot = blobs.parent / "snapshots" / repo.commit
                for f in repo.files:
                    blob = blobs / f.etag
                    if not (blob.exists() and blob.stat().st_size == f.size):
                        # Later files' partial parts keep counting, so the progress never goes back.
                        self._fetch(repo, f, blob, state, completed + on_disk(index + 1), cancel, current)
                    completed += f.size
                    index += 1
                    if current():
                        state.done = completed + on_disk(index)
                    check()
                    pointer = snapshot / f.name
                    pointer.parent.mkdir(parents=True, exist_ok=True)
                    _replace_link(pointer, os.path.relpath(blob, pointer.parent))
                if not _is_commit(repo.revision):
                    refs = blobs.parent / "refs"
                    refs.mkdir(exist_ok=True)
                    _replace_text(refs / (repo.revision or "main"), repo.commit)
            for i, t in enumerate(tensors):
                self._fetch_tensors(t, state, completed + sum(u.on_disk() for u in tensors[i + 1:]), cancel, current)
                completed += t.size
            for repo in repos:
                clean_partials(repo.repo, everything=True)
            path = local_path(spec)
            if path is None:
                raise RuntimeError("Download finished but files are incomplete")
            fill_component_sizes(spec, path)
            if current():
                state.done = state.total
                state.status = "done"
            if on_done:
                on_done(spec)
        except DownloadCancelled:
            log.info("download of %s cancelled at %.2f of %.2f GB; its files are kept to resume",
                     spec.id, state.done / 1e9, state.total / 1e9)
            if current() and state.status == "downloading":
                state.status, state.error = "idle", None
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            log.exception("download failed for %s", spec.id)
            if current() and not cancel.is_set():
                state.status, state.error = "error", plain_error(exc)

    @staticmethod
    def _list(repo: str, revision: str | None, wanted) -> _Repo:
        info = _api().model_info(repo, revision=revision, files_metadata=True)
        files = [
            _File(s.rfilename, s.size or 0,
                  (getattr(s.lfs, "sha256", None) or getattr(s, "blob_id", None) or ""),
                  getattr(s.lfs, "sha256", None))
            for s in info.siblings
            if wanted(s.rfilename)
        ]
        commit = getattr(info, "sha", None) or revision
        if not commit or any(not f.etag for f in files):
            raise RuntimeError("Hugging Face did not list the files of this model")
        return _Repo(repo, revision, commit, files)

    @staticmethod
    def _get(url: str, headers: dict[str, str] | None = None) -> bytes:
        from huggingface_hub.utils import build_hf_headers

        with _stream(url, {**build_hf_headers(), **(headers or {})}) as response:
            if response.status_code >= 400:
                raise HTTPStatusError(response.status_code, url.rsplit("/", 1)[-1])
            return b"".join(response.iter_bytes(CHUNK))

    def _plan_tensors(self, extra: ExtraRepo, commit: str) -> _Tensors:
        """Where the tensors are in the remote shard, and the header of the file they become (as mflux's
        ZenImageEditWeights._extract builds it): three small requests."""
        import json
        import struct

        from huggingface_hub import hf_hub_url

        index_name, prefix, rel = extra.tensors
        index = json.loads(self._get(hf_hub_url(extra.repo, index_name, revision=commit)))
        shards = {shard for key, shard in index["weight_map"].items() if key.startswith(prefix)}
        if len(shards) != 1:
            raise RuntimeError(f"Expected the {prefix} tensors in one file of {extra.repo}")
        folder = index_name.rsplit("/", 1)[0] + "/" if "/" in index_name else ""
        url = hf_hub_url(extra.repo, folder + shards.pop(), revision=commit)
        header_size = struct.unpack("<Q", self._get(url, {"Range": "bytes=0-7"})[:8])[0]
        header = json.loads(self._get(url, {"Range": f"bytes=8-{8 + header_size - 1}"})[:header_size])
        wanted = {key: entry for key, entry in header.items() if key.startswith(prefix)}
        first = min(entry["data_offsets"][0] for entry in wanted.values())
        last = max(entry["data_offsets"][1] for entry in wanted.values())
        rebased = {key: {**entry, "data_offsets": [entry["data_offsets"][0] - first, entry["data_offsets"][1] - first]}
                   for key, entry in wanted.items()}
        encoded = json.dumps(rebased, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        return _Tensors(url, _mflux_cache() / rel, struct.pack("<Q", len(encoded)) + encoded,
                        8 + header_size + first, 8 + header_size + last)

    @staticmethod
    def _on_disk(blobs: Path, f: _File) -> int:
        """Bytes of this file already downloaded (whole or partial)."""
        for path in (blobs / f.etag, blobs / f"{f.etag}.incomplete"):
            try:
                return min(path.stat().st_size, f.size)
            except OSError:
                continue
        return 0

    def _fetch(self, repo: _Repo, f: _File, blob: Path, state: DownloadState, completed: int,
               cancel: threading.Event, current=lambda: True) -> None:
        """Download one file to `<blob>.incomplete`, continuing what is there, then move it in place."""
        from huggingface_hub import hf_hub_url

        partial = blob.with_name(blob.name + ".incomplete")
        url = hf_hub_url(repo.repo, f.name, revision=repo.commit)
        with _owning(blob, cancel):
            if blob.exists() and blob.stat().st_size == f.size:
                return  # another download (a model sharing this repo) finished it while this one waited
            self._fetch_bytes(url, partial, f.size, f.name, state, completed, cancel, current, sha256=f.sha256)
            os.replace(partial, blob)

    def _fetch_tensors(self, t: _Tensors, state: DownloadState, completed: int, cancel: threading.Event,
                       current=lambda: True) -> None:
        partial = t.dest.with_name(t.dest.name + ".incomplete")
        partial.parent.mkdir(parents=True, exist_ok=True)
        with _owning(t.dest, cancel):
            if t.dest.is_file() and t.dest.stat().st_size == t.size:
                return  # another download finished it while this one waited
            have = partial.stat().st_size if partial.exists() else 0
            if have < len(t.header) or have > t.size or _head(partial, len(t.header)) != t.header:
                partial.write_bytes(t.header)  # a new file, or one planned differently: start it again
            self._fetch_bytes(t.url, partial, t.size, t.dest.name, state, completed, cancel, current,
                              skip=len(t.header), start=t.start)
            if _head(partial, len(t.header)) != t.header:
                partial.unlink(missing_ok=True)
                raise RuntimeError(f"{t.dest.name} was damaged on disk; download it again")
            os.replace(partial, t.dest)

    def _fetch_bytes(self, url: str, partial: Path, size: int, name: str, state: DownloadState, completed: int,
                     cancel: threading.Event, current, sha256: str | None = None, skip: int = 0,
                     start: int = 0) -> None:
        """Fill `partial` up to `size` bytes, continuing what is there. Its byte `skip` is the remote byte
        `start` (a range of a bigger file after a header of our own); a whole file has both at 0."""
        from huggingface_hub.utils import build_hf_headers

        have = partial.stat().st_size if partial.exists() else 0
        if have > size:
            partial.unlink()
            have = 0
        if not partial.exists():
            partial.touch()
        ranged = bool(skip or start)
        attempt = 0
        while have < size:
            if cancel.is_set():
                raise DownloadCancelled
            headers = build_hf_headers()
            if ranged:
                headers["Range"] = f"bytes={start + have - skip}-{start + size - skip - 1}"
            elif have:
                headers["Range"] = f"bytes={have}-"
            before = have
            try:
                with _stream(url, headers) as response, self._reading(cancel, response):
                    if response.status_code == 416 and have >= size:
                        break
                    if response.status_code >= 400:
                        raise HTTPStatusError(response.status_code, name)
                    if ranged and response.status_code != 206:
                        raise RuntimeError(f"the server sent all of {name} instead of a part of it")
                    restart = have and response.status_code != 206  # the server ignored the Range
                    with partial.open("wb" if restart else "ab") as out:
                        if restart:
                            have = 0
                        for chunk in response.iter_bytes(CHUNK):
                            if cancel.is_set():
                                raise DownloadCancelled
                            chunk = chunk[:size - have]
                            if not chunk:
                                break
                            out.write(chunk)
                            have += len(chunk)
                            if current():
                                state.done = completed + min(have, size)
                if have < size and have == before:
                    raise RuntimeError(f"the connection closed before {name} was complete")
            except DownloadCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - a dropped connection resumes a few times
                if cancel.is_set():  # the cancel shut the connection
                    raise DownloadCancelled from exc
                attempt = 1 if have > before else attempt + 1
                if attempt > RETRIES or not _retryable(exc):
                    raise
                log.warning("download of %s interrupted (%s); resuming at %.2f GB", name, exc, have / 1e9)
                if cancel.wait(min(2 ** attempt, 10)):
                    raise DownloadCancelled from exc
        # The file on disk is what counts, not the bytes this thread received (fix round 2 of task 19).
        on_disk = partial.stat().st_size if partial.exists() else 0
        if have != size or on_disk != size:
            if on_disk > size:
                partial.unlink(missing_ok=True)
            raise RuntimeError(f"{name}: got {on_disk} bytes, expected {size}")
        if sha256 and _file_sha256(partial, cancel) != sha256:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"{name} was damaged in transfer; download it again")


def _replace_link(link: Path, target: str) -> None:
    """Point `link` at `target` in one step: two downloads of a shared repo may write the same link."""
    tmp = link.with_name(f".{link.name}.{threading.get_ident()}.link")
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(target)
    os.replace(tmp, link)


def _replace_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _head(path: Path, n: int) -> bytes:
    with path.open("rb") as f:
        return f.read(n)


def _is_commit(revision: str | None) -> bool:
    return bool(revision) and len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)
