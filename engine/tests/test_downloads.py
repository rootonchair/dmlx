import os
from pathlib import Path
from types import SimpleNamespace

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


def test_adopt_moves_models_out_of_the_shared_cache(tmp_path, monkeypatch):
    from dmlx import downloads, paths

    old_hub, old_mflux = tmp_path / "shared" / "hub", tmp_path / "shared" / "mflux"
    new_hub, new_mflux = tmp_path / "dmlx" / "models-cache", tmp_path / "dmlx" / "mflux-cache"
    new_hub.mkdir(parents=True)
    new_mflux.mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(new_hub))
    monkeypatch.setenv("MFLUX_CACHE_DIR", str(new_mflux))
    monkeypatch.setattr(paths, "_shared_caches", (old_hub, old_mflux))
    files = {"config.json": b"{}", "transformer/model.safetensors": b"weights" * 100}
    _cached_repo(old_hub, "org/model", files, shared={"transformer/model.safetensors"})
    other = _cached_repo(old_hub, "someone/else", {"a.bin": b"other tool"})
    (old_mflux / "zen-image-edit").mkdir(parents=True)

    assert downloads.adopt(["org/model", "org/not-downloaded"]) == ["org/model"]

    snap = next((new_hub / "models--org--model" / "snapshots").iterdir())
    assert (snap / "config.json").read_bytes() == b"{}"
    assert (snap / "transformer/model.safetensors").read_bytes() == b"weights" * 100
    blobs = list((new_hub / "models--org--model" / "blobs").iterdir())
    assert blobs and not any(b.is_symlink() for b in blobs)  # nothing still points into the shared cache
    assert not (old_hub / "models--org--model").exists()
    assert other.exists()  # another tool's model stays
    assert (new_mflux / "zen-image-edit").is_dir() and not (old_mflux / "zen-image-edit").exists()
    # Already in dMLX's cache: nothing moves again.
    _cached_repo(old_hub, "org/model", files)
    assert downloads.adopt(["org/model"]) == []


def test_adopt_does_nothing_without_a_redirected_cache(monkeypatch):
    from dmlx import downloads, paths

    monkeypatch.setattr(paths, "_shared_caches", None)
    assert downloads.adopt(["org/model"]) == []


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
