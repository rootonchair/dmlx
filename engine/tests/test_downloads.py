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


def test_download_uses_same_pinned_revision_for_listing_and_files(monkeypatch, tmp_path):
    spec = fake_spec("pinned")
    spec.revision = "commit-123"
    info_calls = []
    file_calls = []

    class API:
        def model_info(self, repo, **kwargs):
            info_calls.append((repo, kwargs))
            return SimpleNamespace(siblings=[SimpleNamespace(rfilename="config.json", size=12)])

    monkeypatch.setattr(downloads, "_api", lambda: API())
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo, name, **kwargs: file_calls.append((repo, name, kwargs)),
    )
    monkeypatch.setattr(downloads, "local_path", lambda _spec: tmp_path)
    monkeypatch.setattr(downloads, "fill_component_sizes", lambda _spec, _path: None)
    monkeypatch.setattr(downloads.Downloader, "_watch_partial", lambda *_args: None)
    state = downloads.DownloadState(spec.id)

    downloads.Downloader()._run(spec, state, None)

    assert state.status == "done" and state.done == 12
    assert info_calls == [(spec.repo, {"revision": "commit-123", "files_metadata": True})]
    assert file_calls == [(spec.repo, "config.json", {"revision": "commit-123"})]


def test_download_skips_adapter_managed_external_component(monkeypatch, tmp_path):
    spec = fake_spec("external")
    spec.components["text_encoder"].external = True
    files = [
        SimpleNamespace(rfilename="text_encoder/model.safetensors", size=100),
        SimpleNamespace(rfilename="transformer/model.safetensors", size=200),
    ]

    class API:
        def model_info(self, *_args, **_kwargs):
            return SimpleNamespace(siblings=files)

    downloaded = []
    monkeypatch.setattr(downloads, "_api", lambda: API())
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda _repo, name, **_kwargs: downloaded.append(name),
    )
    monkeypatch.setattr(downloads, "local_path", lambda _spec: tmp_path)
    monkeypatch.setattr(downloads, "fill_component_sizes", lambda _spec, _path: None)
    monkeypatch.setattr(downloads.Downloader, "_watch_partial", lambda *_args: None)
    state = downloads.DownloadState(spec.id)

    downloads.Downloader()._run(spec, state, None)

    assert state.status == "done" and state.total == 200
    assert downloaded == ["transformer/model.safetensors"]
