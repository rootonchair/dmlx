import base64
import io
import json
import time

from PIL import Image

from conftest import fake_spec
from dmlx.adapters.fake import FakeAdapter
from dmlx.registry import GB


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


def test_insufficient_memory_is_readable(make_engine, client_for):
    client = client_for(make_engine(headroom_gb=1.0))
    res = _gen(client)
    assert res.status_code == 507
    err = res.json()["error"]
    assert err["code"] == "insufficient_memory" and "GB" in err["message"]


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


def test_builtin_registry_parses():
    from dmlx.registry import Registry

    reg = Registry.load_builtin()
    assert reg.get("flux2-klein-4b").supports("edit")
    assert reg.get("z-image-turbo").license == "apache-2.0"
    assert reg.default_for("edit", 16).id == "flux2-klein-4b"
    assert reg.default_for("t2i", 16).id == "flux2-klein-4b"


def test_sysmem_snapshot():
    from dmlx import sysmem

    snap = sysmem.snapshot()
    assert snap.total > 4 * GB and snap.available >= 0
    assert snap.pressure in (1, 2, 4)
    json.dumps(snap.as_dict())


def test_cached_prompt_skips_encoder_in_planning(make_engine, client_for):
    engine = make_engine(headroom_gb=2.5)  # only sequential fits when the encoder must load
    client = client_for(engine)
    first = _gen(client, prompt="same prompt").json()["dmlx"]
    assert first["strategy"] == "sequential" and "text_encoder" in first["loads"]
    second = _gen(client, prompt="same prompt").json()["dmlx"]
    assert "text_encoder" not in second["loads"]
    assert second["embed_cache"]["hits"] >= 1


def test_low_memory_decode_only_when_tight(make_engine, client_for):
    roomy_engine = make_engine(headroom_gb=64)
    roomy_engine.memory.total = 64 * GB
    roomy = client_for(roomy_engine)
    assert _gen(roomy).json()["dmlx"]["decode_mode"] == "full"
    tight = client_for(make_engine(headroom_gb=4.0))
    body = _gen(tight, size="2048x2048").json()["dmlx"]  # full decode would need ~3.4 GB * margin
    assert body["strategy"] == "sequential" and body["decode_mode"] == "tiled"
    banded_spec = fake_spec("fake-a")
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
