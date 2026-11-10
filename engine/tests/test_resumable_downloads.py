"""Task 19: a download resumes from its partial files, can be cancelled, leaves no orphan .incomplete
files, and reports progress of its own files only. Hugging Face is faked: no network."""

import hashlib
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import fake_spec

from dmlx import downloads

COMMIT = "c" * 40
_REAL_STREAM = downloads._stream  # before any test fakes it


class FakeHub:
    """Serves a repo's files with Range support; can stall, cut a connection, ignore Range."""

    def __init__(self, files: dict[str, bytes], lfs=("transformer/model.safetensors",), commit=COMMIT):
        self.files = files
        self.commit = commit
        self.lfs = set(lfs)
        self.requests: list[tuple[str, str | None]] = []
        self.info_calls: list[tuple[str, dict]] = []
        self.gate: threading.Event | None = None  # when set, each chunk waits for it
        self.chunk_served = threading.Event()
        self.free_chunks = 0  # served before the gate applies
        self.ignore_range = False
        self.corrupt = False

    def model_info(self, repo, **kwargs):
        self.info_calls.append((repo, kwargs))
        siblings = []
        for name, data in self.files.items():
            lfs = SimpleNamespace(sha256=hashlib.sha256(data).hexdigest(), size=len(data)) if name in self.lfs else None
            siblings.append(SimpleNamespace(rfilename=name, size=len(data), blob_id=hashlib.sha1(data).hexdigest(),
                                            lfs=lfs))
        return SimpleNamespace(sha=self.commit, siblings=siblings)

    @contextmanager
    def stream(self, url, headers):
        name = url.split(f"/resolve/{self.commit}/", 1)[1]
        rng = headers.get("Range")
        self.requests.append((name, rng))
        data = self.files[name]
        if self.corrupt:
            data = data[:-1] + b"!"
        start, end = 0, len(data)
        if rng and not self.ignore_range:
            first, _, last = rng.removeprefix("bytes=").partition("-")
            start, end = int(first), (int(last) + 1 if last else len(data))
        data = data[:end]
        status = 206 if rng and not self.ignore_range and start < len(data) else 200
        if start >= len(data) and rng:
            status = 416

        def chunks(size):
            for i in range(start, len(data), 1000):
                if self.gate is not None and self.free_chunks > 0:
                    self.free_chunks -= 1
                elif self.gate is not None:
                    self.chunk_served.set()
                    assert self.gate.wait(5)
                yield data[i:i + 1000]

        yield SimpleNamespace(status_code=status, headers={}, iter_bytes=chunks)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    files = {
        "config.json": b'{"a": 1}',
        "text_encoder/model.safetensors": b"t" * 5000,
        "transformer/model.safetensors": bytes(range(256)) * 40,  # 10240 bytes, LFS (sha256 checked)
        "vae/model.safetensors": b"v" * 1500,
    }
    fake = FakeHub(files)
    monkeypatch.setattr(downloads, "_api", lambda: fake)
    monkeypatch.setattr(downloads, "_stream", fake.stream)
    monkeypatch.setattr(downloads, "fill_component_sizes", lambda _spec, _path: None)
    return fake


def _spec():
    spec = fake_spec("m")
    spec.local_path = None
    return spec


def _blobs(spec):
    return downloads._blobs(spec.repo)


def _etag(hub, name):
    data = hub.files[name]
    return hashlib.sha256(data).hexdigest() if name in hub.lfs else hashlib.sha1(data).hexdigest()


def _run(spec):
    state = downloads.DownloadState(spec.id)
    downloads.Downloader()._run(spec, state, None)
    return state


def test_a_download_lands_in_the_hugging_face_cache_layout(hub):
    spec = _spec()
    spec.revision = "commit-123"
    state = _run(spec)
    assert state.status == "done", state.error
    assert state.done == state.total == sum(len(d) for d in hub.files.values())
    assert hub.info_calls == [(spec.repo, {"revision": "commit-123", "files_metadata": True})]
    path = downloads.local_path(spec)  # huggingface_hub's own lookup finds it
    assert path is not None and (path / "transformer/model.safetensors").read_bytes() == hub.files[
        "transformer/model.safetensors"]
    assert not list(_blobs(spec).glob("*.incomplete"))


def test_external_components_are_not_downloaded(hub):
    spec = _spec()
    spec.components["text_encoder"].external = True
    state = _run(spec)
    assert state.status == "done"
    assert "text_encoder/model.safetensors" not in {name for name, _ in hub.requests}
    assert state.total == sum(len(d) for n, d in hub.files.items() if not n.startswith("text_encoder/"))


def test_an_interrupted_download_resumes_and_orphans_are_deleted(hub):
    spec = _spec()
    blobs = _blobs(spec)
    blobs.mkdir(parents=True)
    name = "transformer/model.safetensors"
    etag = _etag(hub, name)
    (blobs / f"{etag}.incomplete").write_bytes(hub.files[name][:4000])  # what a quit left
    (blobs / f"{etag}.0a1b2c3d.incomplete").write_bytes(b"x" * 9000)  # huggingface_hub's, never resumable
    (blobs / "deadbeef.incomplete").write_bytes(b"y" * 100)  # another revision's
    state = _run(spec)
    assert state.status == "done", state.error
    assert (name, "bytes=4000-") in hub.requests  # continued, not from 0
    assert (blobs / etag).read_bytes() == hub.files[name]  # and whole (its sha256 was checked)
    assert not list(blobs.glob("*.incomplete"))


def test_a_cancel_keeps_the_partial_files_and_a_new_pull_continues(hub):
    spec = _spec()
    hub.files["transformer/model.safetensors"] = bytes(range(256)) * 400  # 102400 bytes
    hub.gate = threading.Event()
    downloader = downloads.Downloader()
    state = downloader.start(spec)
    assert hub.chunk_served.wait(5)
    for _ in range(30):  # let some chunks through, then cancel mid-file
        hub.chunk_served.clear()
        hub.gate.set()
        hub.gate.clear()
        assert hub.chunk_served.wait(5)
    canceller = threading.Thread(target=lambda: downloader.cancel(spec.id))
    canceller.start()
    hub.gate.set()
    canceller.join(5)
    assert state.status == "idle" and state.error is None
    assert not downloader.cancel(spec.id)  # nothing runs any more
    partial = downloads.partial_bytes(spec)
    assert 0 < partial < sum(len(d) for d in hub.files.values())
    assert list(_blobs(spec).glob("*.incomplete"))

    hub.gate = None
    hub.requests.clear()
    state = downloader.start(spec)
    assert state.as_dict()["done_bytes"] >= partial  # the pull's own answer already counts what is there
    downloader._threads[spec.id].join(5)
    assert state.status == "done", state.error
    resumed = [rng for _, rng in hub.requests if rng]
    assert resumed, hub.requests  # the file cut by the cancel went on from where it was
    assert not list(_blobs(spec).glob("*.incomplete"))


def test_progress_counts_this_downloads_files_only(hub, monkeypatch):
    spec = _spec()
    blobs = _blobs(spec)
    blobs.mkdir(parents=True)
    name = "transformer/model.safetensors"
    (blobs / f"{_etag(hub, name)}.incomplete").write_bytes(hub.files[name][:4000])
    (blobs / "0123.4567abcd.incomplete").write_bytes(b"z" * 50_000)  # a big orphan
    starts = []
    original = hub.stream
    state = downloads.DownloadState(spec.id)

    def first_request(url, headers):
        starts.append(state.done)
        return original(url, headers)

    monkeypatch.setattr(downloads, "_stream", first_request)
    downloads.Downloader()._run(spec, state, None)
    assert starts[0] <= 4000  # never the orphan's 50000; the resumed 4000 count from the start
    assert state.done == state.total


def test_a_file_damaged_in_transfer_is_an_error_and_not_kept(hub):
    spec = _spec()
    hub.corrupt = True
    state = _run(spec)
    assert state.status == "error" and "damaged" in state.error
    assert not (_blobs(spec) / f"{_etag(hub, 'transformer/model.safetensors')}.incomplete").exists()


def test_a_server_that_ignores_the_range_starts_the_file_again(hub):
    spec = _spec()
    blobs = _blobs(spec)
    blobs.mkdir(parents=True)
    name = "transformer/model.safetensors"
    (blobs / f"{_etag(hub, name)}.incomplete").write_bytes(hub.files[name][:4000])
    hub.ignore_range = True
    state = _run(spec)
    assert state.status == "done", state.error
    assert (blobs / _etag(hub, name)).read_bytes() == hub.files[name]


def test_cancel_endpoint_partial_bytes_and_remove(hub, make_engine, client_for):
    spec = _spec()
    engine = make_engine([spec])
    spec.local_path = None
    client = client_for(engine)
    assert client.delete("/admin/models/m/pull").status_code == 404
    assert client.delete("/admin/models/nope/pull").status_code == 404

    hub.gate, hub.free_chunks = threading.Event(), 4
    assert client.post("/admin/models/m/pull").json()["status"] == "downloading"
    assert hub.chunk_served.wait(5)
    threading.Timer(0.2, hub.gate.set).start()
    res = client.delete("/admin/models/m/pull")
    assert res.status_code == 200 and res.json() == {"cancelled": True}
    entry = next(m for m in client.get("/v1/models").json()["data"] if m["id"] == "m")
    assert entry["downloaded"] is False and entry["download"]["status"] == "idle"
    assert entry["partial_bytes"] > 0

    partial = entry["partial_bytes"]
    res = client.delete("/admin/models/m/files")  # Remove on a partial download
    assert res.status_code == 200 and res.json()["freed_gb"] == round(partial / 1e9, 2)
    assert not (downloads._hub_cache() / "models--fake--m").exists()
    entry = next(m for m in client.get("/v1/models").json()["data"] if m["id"] == "m")
    assert entry["partial_bytes"] == 0


def test_the_engine_deletes_orphans_at_start_and_keeps_resumable_parts(hub, make_engine, monkeypatch):
    spec = _spec()
    blobs = _blobs(spec)
    blobs.mkdir(parents=True)
    (blobs / "aaaa.incomplete").write_bytes(b"resumable")
    (blobs / "aaaa.1234abcd.incomplete").write_bytes(b"orphan")
    from dmlx.engine import Engine

    monkeypatch.setattr(Engine, "_owned_repos", lambda self: [spec.repo])
    make_engine([spec])
    assert sorted(p.name for p in blobs.iterdir()) == ["aaaa.incomplete"]


def test_a_linked_repo_folder_is_never_cleaned(hub, tmp_path):
    real = tmp_path / "real" / "models--fake--m" / "blobs"
    real.mkdir(parents=True)
    (real / "x.1234abcd.incomplete").write_bytes(b"other cache")
    downloads._hub_cache().mkdir(parents=True)
    (downloads._hub_cache() / "models--fake--m").symlink_to(real.parent)
    assert downloads.clean_partials("fake/m", everything=True) == 0
    assert (real / "x.1234abcd.incomplete").exists()


def test_remove_counts_a_partial_downloads_bytes(hub):
    """t19-app: Remove on a partial download said "freed 0.00 GB" (the cache scan sees only finished files)."""
    spec = _spec()
    blobs = _blobs(spec)
    blobs.mkdir(parents=True)
    (blobs / "abcd.incomplete").write_bytes(b"p" * 300_000)
    (blobs / "ef01").write_bytes(b"f" * 200_000)
    assert downloads.delete_local(spec) == 500_000
    assert not blobs.parent.exists()


# -- test round 1 of task 19 ----------------------------------------------------------------------------------

import json  # noqa: E402
import struct  # noqa: E402
import time  # noqa: E402

REAL_FILL = downloads.fill_component_sizes
EXTRA_COMMIT = "e" * 40


def _shard(tensors: dict[str, bytes]) -> bytes:
    header, offset = {}, 0
    for name, data in tensors.items():
        header[name] = {"dtype": "U8", "shape": [len(data)], "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    encoded = json.dumps(header).encode()
    return struct.pack("<Q", len(encoded)) + encoded + b"".join(tensors.values())


@pytest.fixture
def extra(hub, tmp_path, monkeypatch):
    """A second repo the model needs (as Qwen Image 2.1 needs AiArtLab/zen-image-edit), with a tensor range
    kept as a file of its own under mflux's cache."""
    tensors = {"other.w": b"o" * 3000, "text_fusion.a": b"A" * 2500, "text_fusion.b": b"B" * 1500, "z.w": b"z" * 99}
    index = json.dumps({"weight_map": {name: "s1.safetensors" for name in tensors}}).encode()
    files = {"processor/tokenizer.json": b"{}" * 10, "enc/model.safetensors": b"e" * 4000, "enc/config.json": b"{}",
             "big/index.json": index, "big/s1.safetensors": _shard(tensors), "big/unused.bin": b"u" * 9999}
    xhub = FakeHub(files, lfs=("enc/model.safetensors",), commit=EXTRA_COMMIT)
    repo = downloads.ExtraRepo(
        repo="fake/extra", revision=EXTRA_COMMIT, allow=("processor/*", "enc/*", "big/index.json"),
        required=("processor/tokenizer.json", "enc/model.safetensors", "big/index.json"), size=12345,
        tensors=("big/index.json", "text_fusion.", f"extra/{EXTRA_COMMIT}/tf.safetensors"))
    monkeypatch.setitem(downloads.EXTRA_REPOS, "fake-ext", (repo,))
    monkeypatch.setenv("MFLUX_CACHE_DIR", str(tmp_path / "mflux"))
    monkeypatch.setattr(downloads, "_api", lambda: SimpleNamespace(
        model_info=lambda name, **kw: (xhub if name == "fake/extra" else hub).model_info(name, **kw)))
    monkeypatch.setattr(downloads, "_stream", lambda url, headers: (
        xhub if "/fake/extra/resolve/" in url else hub).stream(url, headers))
    return SimpleNamespace(hub=xhub, repo=repo, tensors=tensors, dest=tmp_path / "mflux" / repo.tensors[2])


def _ext_spec(model_id="q"):
    spec = fake_spec(model_id)
    spec.adapter = "fake-ext"
    spec.components["text_encoder"].external = True
    spec.local_path = None
    return spec


def _main_total(hub):
    return sum(len(d) for n, d in hub.files.items() if not n.startswith("text_encoder/"))


def test_a_model_downloads_its_extra_repos_with_it(hub, extra):
    spec = _ext_spec()
    state = _run(spec)
    assert state.status == "done", state.error
    allowed = ["processor/tokenizer.json", "enc/model.safetensors", "enc/config.json", "big/index.json"]
    tensor_bytes = 2500 + 1500
    tensor_file = extra.dest.read_bytes()
    assert state.done == state.total == _main_total(hub) + sum(len(extra.hub.files[n]) for n in allowed) + len(
        tensor_file)
    fetched = {name for name, _ in extra.hub.requests}
    assert "big/unused.bin" not in fetched
    assert all(rng for name, rng in extra.hub.requests if name == "big/s1.safetensors")  # ranges only
    assert downloads.extra_path(extra.repo) is not None and downloads.local_path(spec) is not None
    # The tensor file is what mflux's own extraction writes: its header, then the text_fusion bytes.
    size = struct.unpack("<Q", tensor_file[:8])[0]
    header = json.loads(tensor_file[8:8 + size])
    assert sorted(header) == ["text_fusion.a", "text_fusion.b"] and size % 8 == 0
    assert header["text_fusion.a"]["data_offsets"] == [0, 2500]
    assert tensor_file[8 + size:] == b"A" * 2500 + b"B" * 1500 and len(tensor_file) == 8 + size + tensor_bytes
    assert not list(downloads._blobs("fake/extra").glob("*.incomplete"))
    assert downloads.partial_bytes(spec) == state.total


def test_the_model_counts_as_downloaded_only_when_its_extra_repos_are(hub, extra):
    spec = _ext_spec()
    assert _run(spec).status == "done"
    extra.dest.unlink()
    assert downloads.local_path(spec) is None
    assert _run(spec).status == "done"  # a new pull fetches only what is missing
    assert downloads.local_path(spec) is not None
    snapshot = downloads.extra_path(extra.repo)
    (snapshot / "enc/model.safetensors").unlink()
    assert downloads.local_path(spec) is None


def test_a_cut_tensor_file_resumes_with_a_range(hub, extra):
    spec = _ext_spec()
    plan = downloads.Downloader()._plan_tensors(extra.repo, EXTRA_COMMIT)
    shard = extra.hub.files["big/s1.safetensors"]
    extra.dest.parent.mkdir(parents=True)
    partial = extra.dest.with_name(extra.dest.name + ".incomplete")
    partial.write_bytes(plan.header + shard[plan.start:plan.start + 1000])
    assert downloads.partial_bytes(spec) == len(plan.header) + 1000
    state = _run(spec)
    assert state.status == "done", state.error
    assert ("big/s1.safetensors", f"bytes={plan.start + 1000}-{plan.end - 1}") in extra.hub.requests
    assert extra.dest.read_bytes() == plan.header + shard[plan.start:plan.end] and not partial.exists()


def test_a_cancel_in_an_extra_repo_keeps_its_files_and_progress_counts_them(hub, extra):
    spec = _ext_spec()
    extra.hub.files["enc/model.safetensors"] = b"e" * 100_000
    extra.hub.gate, extra.hub.free_chunks = threading.Event(), 5
    downloader = downloads.Downloader()
    state = downloader.start(spec)
    assert extra.hub.chunk_served.wait(5)
    threading.Timer(0.2, extra.hub.gate.set).start()
    assert downloader.cancel(spec.id)
    assert state.status == "idle"
    on_disk = downloads.partial_bytes(spec)
    assert on_disk > _main_total(hub)  # the model's repo, and part of the extra one
    assert downloads.local_path(spec) is None
    extra.hub.gate = None
    state = downloader.start(spec)
    assert state.done >= on_disk
    downloader._threads[spec.id].join(5)
    assert state.status == "done", state.error
    assert any(rng for name, rng in extra.hub.requests if name == "enc/model.safetensors")


def test_remove_keeps_an_extra_repo_another_downloaded_model_uses(hub, extra, make_engine):
    first, second = _ext_spec("q1"), _ext_spec("q2")
    engine = make_engine([first, second])  # both count as downloaded
    first.local_path = None
    assert _run(first).status == "done"
    extra_folder = downloads._hub_cache() / "models--fake--extra"
    engine.delete_model_files("q1")
    assert not (downloads._hub_cache() / "models--fake--q1").exists()
    assert extra_folder.exists() and extra.dest.exists()  # q2 still uses them
    engine.delete_model_files("q2")  # nothing else uses them now
    assert not extra_folder.exists() and not extra.dest.exists()


def test_a_cancel_answers_at_once_and_a_resume_right_after_is_not_lost(hub):
    """Test round 1: the cancel said {"cancelled": true} while the download still ran (in model_info), so
    the Resume right after it saw "downloading" and was silently dropped."""
    spec = _spec()
    entered, release = threading.Event(), threading.Event()
    original = hub.model_info

    def slow(repo, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(repo, **kwargs)

    hub.model_info = slow
    downloader = downloads.Downloader()
    downloader.start(spec)
    assert entered.wait(5)
    began = time.monotonic()
    assert downloader.cancel(spec.id, wait=0.2)
    assert time.monotonic() - began < 1 and downloader.state(spec.id).status == "idle"
    assert not downloader.cancel(spec.id)  # it counts as stopped
    hub.model_info = original
    state = downloader.start(spec)  # Resume, while the old thread is still in model_info
    assert state.status == "downloading"
    release.set()
    downloader._threads[spec.id].join(5)
    assert state.status == "done" and state.error is None


def test_download_errors_read_as_plain_sentences(hub, monkeypatch):
    import errno

    import httpx

    spec = _spec()
    real_stream = hub.stream

    @contextmanager
    def refused(url, headers):
        hub.requests.append((url, None))
        yield SimpleNamespace(status_code=401, headers={}, iter_bytes=lambda size: iter(()))

    monkeypatch.setattr(downloads, "_stream", refused)
    state = _run(spec)
    assert state.status == "error"
    assert state.error.startswith("Hugging Face refused the download") and "HTTP" not in state.error
    assert len(hub.requests) == 1  # a refusal is not tried again

    @contextmanager
    def disk_full(url, headers):
        with real_stream(url, headers) as response:
            def chunks(size):
                yield b"x" * 10
                raise OSError(errno.ENOSPC, "No space left on device")
            yield SimpleNamespace(status_code=response.status_code, headers={}, iter_bytes=chunks)

    hub.requests.clear()
    monkeypatch.setattr(downloads, "_stream", disk_full)
    began = time.monotonic()
    state = _run(spec)
    assert state.error.startswith("The disk is full.") and "Errno" not in state.error
    assert len(hub.requests) == 1 and time.monotonic() - began < 1  # no retries on a full disk
    assert "internet connection" in downloads.plain_error(httpx.ConnectError("boom"))


def test_the_size_is_the_same_before_and_after_the_download(hub, monkeypatch):
    """Test round 1: Klein 4B read 4.62 GB, then 4.61 GB once downloaded (the weights alone)."""
    monkeypatch.setattr(downloads, "fill_component_sizes", REAL_FILL)
    spec = _spec()
    for component in spec.components.values():
        component.size = 0
    spec.download_size = sum(len(d) for d in hub.files.values())
    before = spec.listed_size
    assert _run(spec).status == "done"
    assert spec.local_path is not None and spec.listed_size == before == spec.download_size


def test_zen_image_edit_matches_what_mflux_reads():
    from mflux.cli.defaults.defaults import MFLUX_CACHE_DIR
    from mflux.models.qwen21.model.qwen21_small_text_encoder.zen_image_edit_weights import ZenImageEditWeights

    from dmlx.registry import Registry

    zen = downloads.ZEN_IMAGE_EDIT
    assert zen.repo == ZenImageEditWeights.REPO and zen.revision == ZenImageEditWeights.revision()
    assert list(zen.allow) == ZenImageEditWeights.SNAPSHOT_PATTERNS
    assert zen.tensors[1] == ZenImageEditWeights.FUSION_PREFIX
    assert zen.tensors[2] == f"zen-image-edit/{ZenImageEditWeights.revision()}/text_fusion.safetensors"
    assert downloads._mflux_cache() == Path(MFLUX_CACHE_DIR)
    qwen = Registry.load_builtin().get("qwen-image-2.1-q4")
    assert downloads.extra_repos(qwen) == (zen,)
    qwen.extra_size = downloads.extras_download_size(qwen)
    assert qwen.as_dict()["size_gb"] == round((qwen.download_size + zen.size) / 1e9, 2)


def test_the_qwen_image_21_encoder_never_downloads_in_a_job(hub, tmp_path, monkeypatch):
    from mflux.models.qwen21.model.qwen21_small_text_encoder.zen_image_edit_weights import ZenImageEditWeights

    from dmlx.adapters.mflux_family import _local_zen_image_edit

    monkeypatch.setenv("MFLUX_CACHE_DIR", str(tmp_path / "mflux"))
    original = ZenImageEditWeights.__dict__["snapshot"]
    with pytest.raises(RuntimeError, match="not downloaded"), _local_zen_image_edit():
        pass
    zen = downloads.ZEN_IMAGE_EDIT
    snapshot = downloads._hub_cache() / "models--AiArtLab--zen-image-edit" / "snapshots" / zen.revision
    for name in zen.required:
        (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
        (snapshot / name).write_bytes(b"x")
    (tmp_path / "mflux" / zen.tensors[2]).parent.mkdir(parents=True)
    (tmp_path / "mflux" / zen.tensors[2]).write_bytes(b"x")
    with _local_zen_image_edit():
        assert ZenImageEditWeights.snapshot() == snapshot
        with pytest.raises(RuntimeError, match="not downloaded"):
            ZenImageEditWeights._extract("transformer/x.safetensors", tmp_path / "x")
    assert ZenImageEditWeights.__dict__["snapshot"] is original


def test_progress_never_goes_back_with_several_partial_files(hub, extra):
    spec = _ext_spec()
    first = "transformer/model.safetensors"
    blobs = _blobs(spec)
    blobs.mkdir(parents=True)
    (blobs / f"{_etag(hub, first)}.incomplete").write_bytes(hub.files[first][:4000])
    xblobs = downloads._blobs("fake/extra")
    xblobs.mkdir(parents=True)
    later = "enc/model.safetensors"
    (xblobs / f"{hashlib.sha256(extra.hub.files[later]).hexdigest()}.incomplete").write_bytes(b"e" * 3000)
    state = downloads.DownloadState(spec.id)
    state.done = downloads.partial_bytes(spec)  # as Downloader.start() sets it
    seen = []
    real = downloads._stream

    def watch(url, headers):
        seen.append((url.rsplit("/", 1)[-1], state.done))
        return real(url, headers)

    downloads._stream = watch
    try:
        downloads.Downloader()._run(spec, state, None)
    finally:
        downloads._stream = real
    assert state.status == "done", state.error
    done = [d for _, d in seen]
    assert done == sorted(done) and done[0] >= 4000 + 3000


# -- fix round 2 of task 19 -----------------------------------------------------------------------------------

def _slow(fake):
    """Serve each chunk a little late, so two downloads overlap."""
    serve = fake.stream

    @contextmanager
    def stream(url, headers):
        with serve(url, headers) as response:
            inner = response.iter_bytes

            def chunks(size):
                for chunk in inner(size):
                    time.sleep(0.0005)
                    yield chunk

            yield SimpleNamespace(status_code=response.status_code, headers=response.headers, iter_bytes=chunks)

    fake.stream = stream


@pytest.mark.parametrize("delay", [0, 0.01, 0.05, 0.1])
def test_two_models_sharing_an_extra_repo_download_at_once(hub, extra, delay):
    """Test round 2: both wrote the same .incomplete; one accepted a damaged blob that passed the sha256
    check (it covered the bytes received, not the file), the other failed with a raw Errno."""
    big = bytes(range(256)) * 800
    extra.hub.files["enc/model.safetensors"] = big
    tensors = {"other.w": b"o" * 3000, "text_fusion.a": bytes(range(256)) * 400, "text_fusion.b": b"B" * 50000,
               "z.w": b"z" * 99}
    extra.hub.files["big/s1.safetensors"] = _shard(tensors)
    _slow(extra.hub)
    downloader = downloads.Downloader()
    first, second = _ext_spec("q1"), _ext_spec("q2")
    states = [downloader.start(first)]
    time.sleep(delay)
    states.append(downloader.start(second))
    downloader._threads["q1"].join(30)
    downloader._threads["q2"].join(30)
    assert [s.status for s in states] == ["done", "done"], [s.error for s in states]
    blob = downloads._blobs("fake/extra") / hashlib.sha256(big).hexdigest()
    assert blob.read_bytes() == big
    tensor_file = extra.dest.read_bytes()
    size = struct.unpack("<Q", tensor_file[:8])[0]
    assert tensor_file[8 + size:] == tensors["text_fusion.a"] + tensors["text_fusion.b"]
    assert downloads.local_path(first) is not None and downloads.local_path(second) is not None
    assert not list(downloads._blobs("fake/extra").glob("*.incomplete"))
    # Each file of the shared repo was fetched once: the second download reused the first one's.
    whole = [name for name, rng in extra.hub.requests if name == "enc/model.safetensors" and not rng]
    assert len(whole) == 1


def test_a_file_changed_on_disk_while_downloading_is_not_accepted(hub, monkeypatch):
    """The check reads the file on disk: bytes another writer added are caught even though the bytes this
    download received were right."""
    spec = _spec()
    name = "transformer/model.safetensors"
    partial = _blobs(spec) / f"{_etag(hub, name)}.incomplete"
    serve = hub.stream

    @contextmanager
    def tamper(url, headers):
        with serve(url, headers) as response:
            inner = response.iter_bytes

            def chunks(size):
                done = False
                for chunk in inner(size):
                    if name in url and not done and partial.exists() and partial.stat().st_size >= 10:
                        with partial.open("r+b") as f:  # same length, different bytes
                            f.write(b"!" * 10)
                        done = True
                    yield chunk
                assert done or name not in url

            yield SimpleNamespace(status_code=response.status_code, headers=response.headers, iter_bytes=chunks)

    monkeypatch.setattr(downloads, "_stream", tamper)
    state = _run(spec)
    assert state.status == "error" and "damaged" in state.error
    assert not (_blobs(spec) / _etag(hub, name)).exists() and not partial.exists()
    monkeypatch.setattr(downloads, "_stream", serve)
    assert _run(spec).status == "done"
    assert (_blobs(spec) / _etag(hub, name)).read_bytes() == hub.files[name]


def test_a_cancel_on_a_stalled_connection_answers_at_once_and_the_thread_ends(hub, monkeypatch):
    """Test round 3: the cancel joined the download thread for up to 10 s, and on a stalled connection
    (no bytes until the 30 s read timeout) the thread was still reading then. A real HTTP server on
    this Mac sends one chunk and then nothing; the download reads it through the real httpx client."""
    import socketserver

    name = "transformer/model.safetensors"
    hub.files[name] = b"s" * 200_000
    stall = threading.Event()

    class Stalling(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.recv(65536)
            body = hub.files[name]
            self.request.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % len(body) + body[:1000])
            stall.wait(25)  # sends nothing more

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Stalling)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    real_stream, fake_stream = _REAL_STREAM, hub.stream
    reading = threading.Event()

    def stream(url, headers):
        if url.endswith(name):
            reading.set()
            return real_stream(f"http://127.0.0.1:{server.server_address[1]}/{name}", {})
        return fake_stream(url, headers)

    monkeypatch.setattr(downloads, "_stream", stream)
    spec = _spec()
    downloader = downloads.Downloader()
    try:
        downloader.start(spec)
        assert reading.wait(5)
        time.sleep(0.3)  # the first chunk is in; the read now waits on the stalled socket
        began = time.monotonic()
        assert downloader.cancel(spec.id)
        assert time.monotonic() - began < 0.5  # answers at once
        assert downloader.state(spec.id).status == "idle"
        thread = downloader._threads[spec.id]
        thread.join(2)
        assert not thread.is_alive()  # the shut connection ended the stalled read
        assert downloader.state(spec.id).status == "idle" and downloader.state(spec.id).error is None
        assert downloads.partial_bytes(spec) > 0  # what came is kept

        monkeypatch.setattr(downloads, "_stream", fake_stream)
        state = downloader.start(spec)  # Resume
        downloader._threads[spec.id].join(5)
        assert state.status == "done", state.error
    finally:
        stall.set()
        server.shutdown()
        server.server_close()


# -- the agent model counts as installed only when every file it needs is there (fix round 4) -------------

AGENT = "qwen3-vl-8b-instruct-4bit"


def _agent_files() -> dict[str, bytes]:
    from dmlx.prompting import local_vlm

    model = local_vlm.MODELS[AGENT]
    files = {name: b'{"file": "%s"}' % name.encode() for name in model.files}
    files["model-00001-of-00002.safetensors"] = _shard({"a": b"1" * 6000})
    files["model-00002-of-00002.safetensors"] = _shard({"b": b"2" * 9000})
    files["README.md"] = b"readme"
    return dict(sorted(files.items()))


def _agent_hub(hub):
    from dmlx.prompting import local_vlm

    hub.files = _agent_files()
    hub.commit = local_vlm.MODELS[AGENT].revision
    hub.lfs = {name for name in hub.files if name.endswith(".safetensors")}


def test_a_half_downloaded_agent_is_not_installed(hub):
    """What a quit inside shard 2 left: config.json and shard 1 linked, shard 2 and the tokenizer not."""
    from dmlx.prompting import local_vlm

    _agent_hub(hub)
    model = local_vlm.MODELS[AGENT]
    blobs = downloads._blobs(model.repo)
    snapshot = blobs.parent / "snapshots" / model.revision
    snapshot.mkdir(parents=True)
    blobs.mkdir(parents=True)

    def link(name: str, data: bytes | None = None) -> None:
        blob = blobs / _etag(hub, name)
        blob.write_bytes(hub.files[name] if data is None else data)
        (snapshot / name).symlink_to(blob)

    link("config.json")
    link("model-00001-of-00002.safetensors")
    assert local_vlm.locate(AGENT) is None
    for name in hub.files:
        if not (snapshot / name).exists() and name != "README.md":
            link(name)
    assert local_vlm.locate(AGENT) == snapshot  # every file there: installed (README.md is not needed)
    (blobs / f"{_etag(hub, 'tokenizer.json')}x.incomplete").write_bytes(b"x")
    assert local_vlm.locate(AGENT) is None  # a download of it is unfinished
    next(blobs.glob("*.incomplete")).unlink()
    (snapshot / "model-00002-of-00002.safetensors").unlink()
    link("model-00002-of-00002.safetensors", hub.files["model-00002-of-00002.safetensors"][:-10])
    assert local_vlm.locate(AGENT) is None  # a cut shard


def test_an_agent_download_stopped_in_shard_2_is_not_installed_and_the_pull_resumes_it(hub, make_engine):
    from dmlx import engine as engine_mod
    from dmlx.prompting import local_vlm

    _agent_hub(hub)
    spec = engine_mod.prompt_assistant_spec(AGENT)
    shard2 = "model-00002-of-00002.safetensors"
    before = list(hub.files).index(shard2)
    hub.free_chunks = sum(-(-len(d) // 1000) for d in list(hub.files.values())[:before]) + 3
    hub.gate = threading.Event()
    downloader = downloads.Downloader()
    state = downloader.start(spec)
    assert hub.chunk_served.wait(5)
    canceller = threading.Thread(target=lambda: downloader.cancel(spec.id))
    canceller.start()
    hub.gate.set()
    canceller.join(5)
    assert downloader.wait_stopped(spec.id, 5)
    snapshot = downloads._blobs(spec.repo).parent / "snapshots" / spec.revision
    assert {"config.json", "model-00001-of-00002.safetensors"} <= {p.name for p in snapshot.iterdir()}
    assert not (snapshot / shard2).exists() and not (snapshot / "tokenizer.json").exists()
    assert local_vlm.locate(AGENT) is None
    engine = make_engine()  # a restart: the agent shows as not installed, its partial is kept
    entry = next(m for m in engine.prompt_assistant_status()["models"] if m["id"] == AGENT)
    assert not entry["installed"]
    assert downloads.partial_bytes(spec) > 0

    hub.gate = None
    hub.requests.clear()
    state = downloader.start(spec)
    downloader._threads[spec.id].join(5)
    assert state.status == "done", state.error
    ranges = [rng for name, rng in hub.requests if name == shard2]
    assert ranges and ranges[0] and ranges[0] != "bytes=0-", ranges  # shard 2 went on from where it stopped
    assert "model-00001-of-00002.safetensors" not in {name for name, _ in hub.requests}
    assert "tokenizer.json" in {name for name, _ in hub.requests}
    assert local_vlm.locate(AGENT) == snapshot
