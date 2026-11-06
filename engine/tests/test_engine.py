import base64
import io
import json
import threading
import time

from PIL import Image

from conftest import fake_spec
from dmlx.adapters.fake import FakeAdapter
from dmlx.registry import GB
from dmlx.memory import JobGuard, Plan, PressureMonitor


def _gen(client, **body):
    return client.post("/v1/images/generations", json={"model": "fake-a", "prompt": "a boat", "size": "512x512", **body})


def test_generate_returns_openai_shape(make_engine, client_for):
    client = client_for(make_engine())
    res = _gen(client, seed=1)
    assert res.status_code == 200, res.text
    body = res.json()
    image = Image.open(io.BytesIO(base64.b64decode(body["data"][0]["b64_json"])))
    assert image.size == (512, 512)
    assert body["dmlx"]["strategy"] == "eager"
    assert set(body["dmlx"]["loads"]) == {"text_encoder", "transformer", "vae"}
    rid = body["data"][0]["result_id"]
    assert client.get(f"/v1/results/{rid}").status_code == 200
    assert client.get(f"/v1/results/{rid}/meta").json()["seed"] == 1


def test_warm_second_request_has_no_loads(make_engine, client_for):
    client = client_for(make_engine())
    _gen(client)
    second = _gen(client, prompt="a different boat").json()
    assert second["dmlx"]["loads"] == []


def test_edit_with_source_id_and_errors(make_engine, client_for):
    client = client_for(make_engine())
    rid = _gen(client).json()["data"][0]["result_id"]
    res = client.post("/v1/images/edits", json={"model": "fake-a", "prompt": "make it night", "source_id": rid})
    assert res.status_code == 200, res.text
    meta = client.get(f"/v1/results/{res.json()['data'][0]['result_id']}/meta").json()
    assert meta["sources"] == [rid] and meta["task"] == "edit"
    assert client.post("/v1/images/edits", json={"model": "fake-a", "prompt": "x"}).status_code == 400
    assert client.post("/v1/images/edits", json={"model": "fake-a", "prompt": "x", "source_id": "img_nope"}).status_code == 404
    # fake-b cannot edit
    res = client.post("/v1/images/edits", json={"model": "fake-b", "prompt": "x", "source_id": rid})
    assert res.status_code == 400


def test_multipart_edit_upload(make_engine, client_for):
    client = client_for(make_engine())
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), (10, 20, 30)).save(buf, format="PNG")
    res = client.post(
        "/v1/images/edits",
        data={"model": "fake-a", "prompt": "make it red"},
        files={"image": ("in.png", buf.getvalue(), "image/png")},
    )
    assert res.status_code == 200, res.text
    out = Image.open(io.BytesIO(base64.b64decode(res.json()["data"][0]["b64_json"])))
    assert out.size == (640, 480)  # auto size follows the source


def test_upload_normalizes_exif_orientation(make_engine, client_for):
    client = client_for(make_engine())
    buf = io.BytesIO()
    image = Image.new("RGB", (40, 20), (10, 20, 30))
    exif = image.getexif()
    exif[274] = 6  # displayed orientation is 90 degrees clockwise
    image.save(buf, format="JPEG", exif=exif)
    response = client.post(
        "/v1/uploads", files={"file": ("rotated.jpg", buf.getvalue(), "image/jpeg")}
    )
    assert response.status_code == 200
    with Image.open(client.app.state.engine.results.path(response.json()["id"])) as stored:
        assert stored.size == (20, 40)
        assert stored.getexif().get(274) is None


def test_sse_event_sequence(make_engine, client_for):
    client = client_for(make_engine())
    with client.stream("POST", "/v1/images/generations",
                       json={"model": "fake-a", "prompt": "boat", "size": "512x512", "stream": True}) as res:
        events = [line.split(": ", 1)[1] for line in res.iter_lines() if line.startswith("event:")]
    assert events[0] == "queued"
    assert "plan" in events and "loading" in events and "encoding" in events and "decoding" in events
    assert events.count("step") == 4
    assert events[-1] == "done"


def test_cancel_running_job(make_engine, client_for):
    engine = make_engine()
    FakeAdapter.step_delay = 0.05
    try:
        job = engine.submit(task="t2i", prompt="slow", model="fake-a", size="512x512", steps=50)
        while job.status != "running":
            time.sleep(0.01)
        time.sleep(0.1)
        engine.cancel(job.id)
        job.finished.wait(5)
        assert job.status == "cancelled"
        # the engine keeps working afterwards
        job2 = engine.submit(task="t2i", prompt="next", model="fake-a", size="512x512")
        job2.finished.wait(5)
        assert job2.status == "done"
    finally:
        FakeAdapter.step_delay = 0.0


def test_sequential_when_tight(make_engine, client_for):
    # 2 GB encoder + 2 GB transformer + 0.2 vae; headroom 2.5 GB -> only sequential fits
    engine = make_engine(headroom_gb=2.5)
    client = client_for(engine)
    body = _gen(client).json()
    assert body["dmlx"]["strategy"] == "sequential"
    resident = {r["component"] for r in engine.memory.resident_components()}
    assert resident == {"vae"}  # big components freed after their phase


def test_phase_free_keeps_transformer(make_engine, client_for):
    engine = make_engine(headroom_gb=3.5)
    client = client_for(engine)
    body = _gen(client).json()
    assert body["dmlx"]["strategy"] == "phase_free"
    resident = {r["component"] for r in engine.memory.resident_components()}
    assert resident == {"vae", "transformer"}


def test_multimodal_layout_loads_transformer_before_releasing_preprocess_vae(make_engine):
    engine = make_engine(headroom_gb=64)
    spec = engine.registry.get("fake-a")
    engine.adapter(spec)
    guard = JobGuard(
        engine.memory, spec, Plan(spec.id, "sequential", 0, 64 * GB, []),
        threading.Event(), lambda *_args: None,
    )

    guard.start()
    guard.before_phase("encode")
    guard.after_phase("encode")
    guard.before_phase("preprocess")
    guard.before_phase("layout")
    assert {r["component"] for r in engine.memory.resident_components()} == {"vae", "transformer"}

    guard.before_phase("denoise")
    assert {r["component"] for r in engine.memory.resident_components()} == {"transformer"}


def test_insufficient_memory_is_readable(make_engine, client_for):
    client = client_for(make_engine(headroom_gb=1.0))
    res = _gen(client)
    assert res.status_code == 507
    err = res.json()["error"]
    assert err["code"] == "insufficient_memory" and "GB" in err["message"]


def test_an_image_job_waits_for_memory_that_is_coming_back(make_engine, client_for, monkeypatch):
    import dmlx.engine as engine_module

    monkeypatch.setattr(engine_module, "RECLAIM_POLL_S", 0.01)
    engine = make_engine()
    rising = iter([1.0, 2.0, 64.0])  # just after the agent was unloaded, then its memory comes back
    engine.memory.headroom = lambda: int(next(rising, 64.0) * GB)
    assert _gen(client_for(engine)).status_code == 200


def test_force_runs_anyway(make_engine, client_for):
    client = client_for(make_engine(headroom_gb=1.0))
    res = _gen(client, dmlx={"force": True})
    assert res.status_code == 200
    assert res.json()["dmlx"]["forced"] is True


def test_eviction_of_other_model(make_engine, client_for):
    specs = [fake_spec("fake-a"), fake_spec("fake-b", tasks=("t2i",))]
    engine = make_engine(specs=specs, headroom_gb=None)
    state = {"room": 6.0 * GB}
    engine.memory.headroom = lambda: int(state["room"] - engine.memory.resident_bytes())
    client = client_for(engine)
    assert _gen(client).json()["dmlx"]["strategy"] == "eager"
    body = _gen(client, model="fake-b").json()
    assert any(e.startswith("fake-a/") for e in body["dmlx"]["evicted"])
    assert "fake-b" in engine.memory.resident_models()


def test_pinned_model_is_not_evicted(make_engine, client_for):
    specs = [fake_spec("fake-a"), fake_spec("fake-b", tasks=("t2i",))]
    engine = make_engine(specs=specs, headroom_gb=None)
    engine.memory.headroom = lambda: int(6.0 * GB - engine.memory.resident_bytes())
    client = client_for(engine)
    assert client.post("/admin/models/fake-a/load", json={"pin": True}).status_code == 200
    res = _gen(client, model="fake-b")
    assert res.status_code == 507  # would need fake-a's memory, but it is pinned
    assert {r["component"] for r in engine.memory.resident_components() if r["model"] == "fake-a"} == {
        "text_encoder", "transformer", "vae"}


def test_auth_token(make_engine, client_for):
    client = client_for(make_engine(), token="secret")
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code == 200
    assert client.get("/health").status_code == 200


def test_models_listing(make_engine, client_for):
    client = client_for(make_engine())
    data = client.get("/v1/models").json()["data"]
    assert {m["id"] for m in data} == {"fake-a", "fake-b"}
    assert all(m["downloaded"] for m in data)


def test_validation(make_engine, client_for):
    client = client_for(make_engine())
    assert _gen(client, prompt=" ").status_code == 400
    assert _gen(client, size="10x10").status_code == 400
    assert _gen(client, size="banana").status_code == 400
    assert _gen(client, model="nope").status_code == 404


def test_model_specific_minimum_steps_is_validated_before_queueing(make_engine, client_for):
    spec = fake_spec("fake-a")
    spec.min_steps = 2
    response = _gen(client_for(make_engine(specs=[spec])), steps=1)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_steps"
    assert "between 2 and 100" in response.json()["error"]["message"]


def test_builtin_registry_parses():
    from dmlx.registry import Registry

    reg = Registry.load_builtin()
    assert reg.get("flux2-klein-4b").supports("edit")
    assert reg.get("z-image-turbo").license == "apache-2.0"
    assert reg.default_for("edit", 16).id == "flux2-klein-4b"
    assert reg.default_for("t2i", 16).id == "flux2-klein-4b"
    qwen = reg.get("qwen-image-2.1-q4")
    assert qwen.revision == "235fd28de40d745382e9d7a33f008c7735ec21c2"
    assert qwen.auto_select is False
    assert qwen.experimental is True
    assert qwen.license == "qwen-research" and qwen.license_acceptance_id
    assert qwen.max_references == 10
    assert qwen.min_steps == 2
    assert qwen.reference_activation_factor == 2.2
    assert qwen.min_ram_gb == 16
    assert qwen.components["text_encoder"].external is True
    assert 2.3 * GB < qwen.components["text_encoder"].size < 2.5 * GB


def test_sysmem_snapshot():
    from dmlx import sysmem

    snap = sysmem.snapshot()
    assert snap.total > 4 * GB and snap.available >= 0
    assert snap.pressure in (1, 2, 4)
    json.dumps(snap.as_dict())


def test_new_pressure_window_drops_previous_job_swap_samples():
    from dmlx import sysmem

    old = sysmem.MemorySnapshot(1, 1, 0, 0, 0, sysmem.PRESSURE_NORMAL, swapins=0, taken=1.0)
    current = sysmem.MemorySnapshot(
        1, 1, 0, 0, 0, sysmem.PRESSURE_NORMAL,
        swapins=500 * 1024**2, taken=4.0,
    )
    monitor = object.__new__(PressureMonitor)
    monitor._lock = threading.Lock()
    monitor._last = current
    monitor._history = [old, current]
    monitor._window_max_pressure = sysmem.PRESSURE_CRITICAL

    assert monitor.swapin_rate() > 64 * 1024**2
    monitor.start_window()
    assert monitor.swapin_rate() == 0
    assert monitor.window_max_pressure() == sysmem.PRESSURE_NORMAL


def test_cached_prompt_skips_encoder_in_planning(make_engine, client_for):
    engine = make_engine(headroom_gb=2.5)  # only sequential fits when the encoder must load
    client = client_for(engine)
    first = _gen(client, prompt="same prompt").json()["dmlx"]
    assert first["strategy"] == "sequential" and "text_encoder" in first["loads"]
    second = _gen(client, prompt="same prompt").json()["dmlx"]
    assert "text_encoder" not in second["loads"]
    assert second["embed_cache"]["hits"] >= 1


def test_cached_prompt_starts_low_memory_strategy_without_eager_components(make_engine):
    engine = make_engine(headroom_gb=64)
    first = engine.submit(task="t2i", prompt="same prompt", model="fake-a", size="512x512")
    first.finished.wait(5)
    assert first.status == "done" and first.report["strategy"] == "eager"
    assert {r["component"] for r in engine.memory.resident_components()} == {
        "text_encoder", "transformer", "vae",
    }

    second = engine.submit(
        task="t2i", prompt="same prompt", model="fake-a", size="512x512",
        strategy="sequential", force=True,
    )
    second.finished.wait(5)
    assert second.status == "done"
    assert "text_encoder" not in second.report["loads"]
    assert {r["component"] for r in engine.memory.resident_components()} == {"vae"}


def test_low_memory_decode_only_when_tight(make_engine, client_for):
    roomy_engine = make_engine(headroom_gb=64)
    roomy_engine.memory.total = 64 * GB
    roomy = client_for(roomy_engine)
    assert _gen(roomy).json()["dmlx"]["decode_mode"] == "full"
    def large_spec():
        spec = fake_spec("fake-a")
        spec.activation_gb_per_mp, spec.decode_gb_per_mp, spec.decode_tiled_gb_per_mp = 0.4, 0.8, 0.4
        return spec

    tight = client_for(make_engine(specs=[large_spec()], headroom_gb=4.0))
    body = _gen(tight, size="2048x2048").json()["dmlx"]  # full decode would need ~3.4 GB * margin
    assert body["strategy"] == "sequential" and body["decode_mode"] == "tiled"
    banded_spec = large_spec()
    banded_spec.decode_banded_gb_per_mp = 0.2
    body = _gen(client_for(make_engine(specs=[banded_spec], headroom_gb=4.0)), size="2048x2048").json()["dmlx"]
    assert body["decode_mode"] == "banded"


def test_swapin_thrash_aborts_and_degrades(make_engine):
    engine = make_engine()
    engine.memory.monitor.swapin_rate = lambda window=3.0: 500 * 1024**2  # 500 MB/s read back from swap
    job = engine.submit(task="t2i", prompt="boat", model="fake-a", size="512x512")
    job.finished.wait(5)
    assert job.status == "error" and job.error["code"] == "memory_pressure"
    assert engine.memory.resident_models() == set()  # everything freed after the abort
    engine.memory.monitor.swapin_rate = lambda window=3.0: 0.0
    job2 = engine.submit(task="t2i", prompt="boat", model="fake-a", size="512x512")
    job2.finished.wait(5)
    assert job2.status == "done" and job2.report["strategy"] == "sequential"  # degraded mode afterwards


def test_streamed_strategy_when_only_block_streaming_fits(make_engine):
    spec = fake_spec("fake-a")
    for name in ("text_encoder", "transformer"):
        spec.components[name].stream = ["blocks"]
        spec.components[name].stream_size = int(0.3 * GB)
    engine = make_engine(specs=[spec], headroom_gb=1.0)  # sequential needs ~2.3 GB, streamed ~0.6 GB
    job = engine.submit(task="t2i", prompt="boat", model="fake-a", size="512x512")
    job.finished.wait(5)
    assert job.status == "done", job.error
    assert job.report["strategy"] == "streamed"
    adapter = engine.adapter(spec)
    assert set(adapter.streamed_loads) == {"text_encoder", "transformer"}
    # The streamed big components are freed after their phase, like sequential.
    assert {r["component"] for r in engine.memory.resident_components()} == {"vae"}


def test_streamed_is_the_cheapest_refusal(make_engine):
    spec = fake_spec("fake-a")
    spec.components["transformer"].stream = ["blocks"]
    spec.components["transformer"].stream_size = int(0.3 * GB)
    engine = make_engine(specs=[spec], headroom_gb=0.2)
    job = engine.submit(task="t2i", prompt="boat", model="fake-a", size="512x512")
    job.finished.wait(5)
    assert job.error["code"] == "insufficient_memory"
    # The refusal quotes the cheapest strategy: the whole 2 GB text encoder (not streamable here) + VAE + work.
    assert 2.0 < job.error["need_gb"] < 2.5
    spec.components["text_encoder"].stream = ["blocks"]
    spec.components["text_encoder"].stream_size = int(0.3 * GB)
    job = engine.submit(task="t2i", prompt="boat", model="fake-a", size="512x512")
    job.finished.wait(5)
    assert job.error["code"] == "insufficient_memory" and job.error["need_gb"] < 1.0


def _inspection(repo="someone/flux2-klein-9b-mflux-q4"):
    from dmlx import catalog

    return catalog.Inspection(repo=repo, id=catalog._slug(repo), name="FLUX.2 Klein 9B (4-bit)", family="flux2-klein",
                              mflux_config="flux2_klein_9b", tasks=["t2i", "edit"], license="other",
                              permissive_license=False, quantization=4, size_bytes=9 * GB,
                              defaults={"steps": 4}, activation_gb_per_mp=2.0)


def test_add_and_remove_a_model(make_engine, monkeypatch, tmp_path):
    from dmlx import catalog, downloads

    engine = make_engine()
    monkeypatch.setattr(catalog, "inspect", lambda text: _inspection())
    monkeypatch.setattr(downloads, "local_path", lambda spec: None)
    added = engine.add_model("someone/flux2-klein-9b-mflux-q4", download=False)
    assert added["user_added"] and not added["downloaded"]
    assert (tmp_path / "models" / f"{added['id']}.yaml").exists()
    assert engine.add_model("someone/flux2-klein-9b-mflux-q4", download=False)["id"] == added["id"]  # no duplicate
    assert engine.inspect_model("x")["already_added"] == added["id"]
    monkeypatch.setattr(downloads, "delete_local", lambda spec: 0)
    assert engine.remove_model(added["id"])["removed"]
    assert added["id"] not in engine.registry
    assert not (tmp_path / "models" / f"{added['id']}.yaml").exists()


def test_builtin_models_keep_their_entry_and_busy_models_are_protected(make_engine, monkeypatch):
    from dmlx import downloads
    from dmlx.engine import RequestError

    engine = make_engine()
    try:
        engine.remove_model("fake-a")
        raise AssertionError("expected a refusal")
    except RequestError as exc:
        assert exc.code == "builtin_model"
    FakeAdapter.step_delay = 0.2
    try:
        job = engine.submit(task="t2i", prompt="boat", model="fake-a", size="512x512")
        time.sleep(0.1)
        try:
            engine.delete_model_files("fake-a")
            raise AssertionError("expected a refusal")
        except RequestError as exc:
            assert exc.code == "model_busy"
        job.finished.wait(5)
    finally:
        FakeAdapter.step_delay = 0.0
    deleted = []
    monkeypatch.setattr(downloads, "delete_local", lambda spec: deleted.append(spec.id) or 2 * GB)
    assert engine.delete_model_files("fake-a")["freed_gb"] == 2.0 and deleted == ["fake-a"]
    assert "fake-a" in engine.registry and engine.memory.resident_models() == set()
