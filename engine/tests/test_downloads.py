import os
from pathlib import Path

from conftest import fake_spec
from dmlx import downloads


def test_local_lookup_uses_pinned_revision(monkeypatch, tmp_path):
    spec = fake_spec("pinned")
    spec.revision = "commit-123"
    for component in spec.components:
        folder = tmp_path / component
        folder.mkdir()
        (folder / "model.safetensors").touch()
    calls = []

    def snapshot(repo, **kwargs):
        calls.append((repo, kwargs))
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    assert downloads.local_path(spec) == tmp_path
    assert calls == [(spec.repo, {"revision": "commit-123", "local_files_only": True})]


def test_listed_size_before_and_after_download():
    """Before a model is downloaded its size comes from the spec (repo files plus external parts), so the
    first-run setup can show the total; once downloaded, the size on disk is used. Both in decimal GB
    (bytes / 1e9), as the app shows every disk size (task 19: 4.3 GB on the card, 4.6 GB in the progress)."""
    from dmlx.registry import GB, Registry

    flux = Registry.load_builtin().get("flux2-klein-4b")
    assert flux.local_path is None
    assert flux.as_dict()["size_gb"] == 4.62  # its 4,619,704,206 bytes on Hugging Face

    qwen = Registry.load_builtin().get("qwen-image-2.1-q4")
    assert qwen.as_dict()["size_gb"] == round((5.13 + 2.4) * GB / 1e9, 2)  # repo files plus the shared encoder

    flux.local_path = Path("/downloaded")
    flux.components["transformer"].size = 3_000_000_000
    assert flux.as_dict()["size_gb"] == 3.0


def _cached_repo(hub, repo, files, shared=()):
    """A repo in Hugging Face cache layout. Names in `shared` are blobs linked into the cache-wide store."""
    import hashlib

    folder = hub / ("models--" + repo.replace("/", "--"))
    commit = hashlib.sha1(repo.encode()).hexdigest()  # real repos never share a commit
    (folder / "refs").mkdir(parents=True)
    (folder / "refs" / "main").write_text(commit)
    (folder / "blobs").mkdir()
    snap = folder / "snapshots" / commit
    snap.mkdir(parents=True)
    for name, data in files.items():
        etag = hashlib.sha1(data).hexdigest()
        blob = folder / "blobs" / etag
        if name in shared:
            store = hub / "blobs" / etag[:2]
            store.mkdir(parents=True, exist_ok=True)
            (store / etag).write_bytes(data)
            blob.symlink_to(f"../../blobs/{etag[:2]}/{etag}")
        else:
            blob.write_bytes(data)
        (snap / name).parent.mkdir(parents=True, exist_ok=True)
        (snap / name).symlink_to(os.path.relpath(blob, (snap / name).parent))
    return folder


def test_remove_keeps_a_duplicated_repo_with_the_same_commits(tmp_path, monkeypatch):
    from dmlx import downloads

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    ours = _cached_repo(tmp_path, "org/model", {"a.bin": b"weights"})
    # A Hugging Face "duplicate" keeps the git history, so both repos have the same commit.
    theirs = _cached_repo(tmp_path, "someone/copy", {"a.bin": b"weights"})
    commit = next((ours / "snapshots").iterdir()).name
    (theirs / "snapshots" / next((theirs / "snapshots").iterdir()).name).rename(theirs / "snapshots" / commit)
    (theirs / "refs" / "main").write_text(commit)

    spec = fake_spec("model")
    spec.repo = "org/model"
    downloads.delete_local(spec)

    assert not ours.exists()
    assert (theirs / "snapshots" / commit / "a.bin").read_bytes() == b"weights"


def test_remove_reports_the_bytes_freed(tmp_path, monkeypatch):
    from dmlx import downloads

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    _cached_repo(tmp_path, "org/model", {"a.bin": b"w" * 100_000})
    spec = fake_spec("model")
    spec.repo = "org/model"
    assert downloads.delete_local(spec) >= 100_000
    assert downloads.delete_local(spec) == 0  # already gone


def test_a_failed_delete_raises_and_keeps_the_repo_folder(tmp_path, monkeypatch):
    import huggingface_hub
    import pytest

    from dmlx import downloads

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    ours = _cached_repo(tmp_path, "org/model", {"a.bin": b"weights"}, shared={"a.bin"})

    def broken(_cache_dir):
        raise OSError("permission denied")

    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", broken)
    spec = fake_spec("model")
    spec.repo = "org/model"
    with pytest.raises(OSError):
        downloads.delete_local(spec)
    assert ours.exists()  # its link into the shared store is not lost


def test_a_failed_extra_delete_still_marks_the_model_removed(monkeypatch):
    import pytest

    from dmlx import downloads

    spec = fake_spec("model")
    spec.repo = "org/model"
    spec.local_path = "/somewhere"
    spec.disk_size = 100
    monkeypatch.setattr(downloads, "extra_repos", lambda _spec: [downloads.ZEN_IMAGE_EDIT])

    def delete(repo):
        if repo != "org/model":
            raise OSError("permission denied")
        return 100

    monkeypatch.setattr(downloads, "_delete_repo", delete)
    with pytest.raises(OSError):
        downloads.delete_local(spec)
    assert spec.local_path is None and spec.disk_size == 0  # its own weights are gone


def test_marks_only_a_data_folder_it_creates(tmp_path, monkeypatch):
    from dmlx import paths

    existing = tmp_path / "Documents"
    existing.mkdir()
    monkeypatch.setenv("DMLX_HOME", str(existing))
    paths.home()
    assert not (existing / paths.DATA_FOLDER_MARKER).exists()  # Uninstall must not take it

    created = tmp_path / "dmlx-data"
    monkeypatch.setenv("DMLX_HOME", str(created))
    paths.home()
    assert (created / paths.DATA_FOLDER_MARKER).exists()
