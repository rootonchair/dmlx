"""A job stopped for memory pressure frees its weights at once (task 17: the log said "released 0.00 GB of
4.86 GB; something still holds its weights" after a stop).

The free ran inside the `except JobAborted` block, where the exception's traceback still holds the
stopped job's frames, and a frame holds the weights. The memory came back once the block ended, so the
weights were not kept until a restart, but the guard's check (and anything planned right then) saw them.
"""

import logging
import weakref

from conftest import fake_spec

from dmlx.adapters.base import JobAborted
from dmlx.adapters.fake import FakeAdapter


class Weights:
    """Stands for a part's arrays: counted as active memory while any reference is alive. Simulated, as
    in test_memory_checks: CI runners have no Metal GPU."""

    def __init__(self, counter: dict, nbytes: int):
        counter["active"] += nbytes
        weakref.finalize(self, counter.__setitem__, "active", 0)  # one part at a time in this test
        self.nbytes = nbytes


class HoldingAdapter(FakeAdapter):
    """Keeps each part as a Weights object, and the denoise loop holds the transformer in a local, as
    mflux's loop holds its arrays while a step callback raises."""

    counter = {"active": 0}

    def load_component(self, name):
        nbytes = super().load_component(name)
        if name == "transformer":
            self.transformer = Weights(self.counter, nbytes)
        return nbytes

    def free_component(self, name):
        super().free_component(name)
        if name == "transformer":
            self.transformer = None

    def generate(self, req, hooks):
        hooks.before_phase("denoise")
        transformer = self.transformer  # noqa: F841 - a frame of the running job holds the weights
        raise JobAborted("memory_pressure", "Stopped to keep your Mac responsive.")


def test_a_job_stopped_for_memory_pressure_frees_its_weights(make_engine, monkeypatch, caplog):
    from dmlx import memory

    engine = make_engine([fake_spec("m")])
    engine._adapter_factory = HoldingAdapter
    monkeypatch.setattr(memory.mx, "get_active_memory", lambda: HoldingAdapter.counter["active"])
    with caplog.at_level(logging.INFO, logger="dmlx.memory"):
        job = engine.submit(task="t2i", prompt="a jazz poster", model="m", strategy="phase_free")
        assert job.finished.wait(10)
    assert job.error["code"] == "memory_pressure"
    assert engine.memory.resident_components() == []
    assert HoldingAdapter.counter["active"] == 0
    freed = [r.getMessage() for r in caplog.records if "freed m/transformer" in r.getMessage()]
    assert freed == ["freed m/transformer (2.00 GB)"]
    assert not [r for r in caplog.records if r.levelno == logging.WARNING and "still holds" in r.getMessage()]


# -- task 19: a stopped job frees what its plan frees at the end; the planner evicts idle models first ----

class StoppingAdapter(FakeAdapter):
    """Stops in the middle of the denoise loop, as a Stop does at step 2."""

    error: Exception = JobAborted("cancelled", "Generation cancelled")

    def generate(self, req, hooks):
        hooks.before_phase("encode")
        hooks.after_phase("encode")
        hooks.before_phase("denoise")
        hooks.on_step(1, req.steps)
        raise self.error


def _stopped_job(make_engine, strategy, error=None):
    spec = fake_spec("m")
    spec.components["transformer"].stream_size = int(0.5 * 1024**3)  # so a streamed plan exists
    engine = make_engine([spec])
    engine._adapter_factory = StoppingAdapter
    StoppingAdapter.error = error or JobAborted("cancelled", "Generation cancelled")
    job = engine.submit(task="t2i", prompt="a red bicycle", model="m", strategy=strategy)
    assert job.finished.wait(10)
    return engine, job, {r["component"] for r in engine.memory.resident_components()}


def test_a_stop_in_a_sequential_plan_frees_the_denoise_parts(make_engine):
    for strategy in ("sequential", "streamed"):
        engine, job, resident = _stopped_job(make_engine, strategy)
        assert job.status == "cancelled"
        assert resident == set()  # encoder and transformer gone (the VAE was freed before denoise)
        assert not engine.adapter(engine.registry.get("m")).is_loaded("transformer")


def test_a_stop_in_an_eager_plan_keeps_the_model_warm(make_engine):
    _, job, resident = _stopped_job(make_engine, "eager")
    assert job.status == "cancelled" and resident == {"text_encoder", "transformer", "vae"}
    _, job, resident = _stopped_job(make_engine, "phase_free")
    assert resident == {"transformer", "vae"}


def test_invalid_numbers_and_errors_free_like_a_stop(make_engine, caplog):
    _, job, resident = _stopped_job(make_engine, "sequential", JobAborted("invalid_output", "bad numbers"))
    assert job.error["code"] == "invalid_output" and resident == set()
    _, job, resident = _stopped_job(make_engine, "sequential", RuntimeError("Metal said no"))
    assert job.status == "error" and job.error == {"code": "internal_error", "message": "Metal said no"}
    assert resident == set()


def _two_models(make_engine, total_gb):
    """Room is `total_gb` less what dMLX keeps resident, as on a real Mac."""
    from dmlx.registry import GB

    engine = make_engine([fake_spec("a"), fake_spec("b", tasks=("t2i",))], headroom_gb=None)
    engine.memory.headroom = lambda: int(total_gb * GB - engine.memory.resident_bytes())
    return engine


def _run(engine, model, **kwargs):
    job = engine.submit(task="t2i", prompt=f"a {model} picture", model=model, **kwargs)
    assert job.finished.wait(10) and job.status == "done", job.error
    return job.report


def test_the_planner_evicts_an_idle_model_rather_than_run_slower(make_engine):
    engine = _two_models(make_engine, 7.0)
    assert _run(engine, "a", strategy="eager")["strategy"] == "eager"  # 4.2 GB warm, 2.8 GB left
    report = _run(engine, "b")  # sequential fits beside it, eager only once "a" is gone
    assert report["strategy"] == "eager"
    assert {e.split("/")[0] for e in report["evicted"]} == {"a"}
    left = {(r["model"], r["component"]) for r in engine.memory.resident_components() if r["model"] == "a"}
    assert left == {("a", "vae")}  # its big parts went; that was enough


def test_the_planner_never_evicts_a_pinned_model(make_engine):
    engine = _two_models(make_engine, 7.0)
    engine.load("a", pin=True)
    report = _run(engine, "b")
    assert report["strategy"] == "sequential" and report["evicted"] == []
    assert "a" in engine.memory.resident_models()


def test_the_planner_does_not_evict_when_it_would_not_help(make_engine):
    engine = _two_models(make_engine, 3.0)
    _run(engine, "a", strategy="sequential")  # leaves only its 0.2 GB VAE resident
    report = _run(engine, "b")  # eager or phase_free would not fit even with "a" gone
    assert report["strategy"] == "sequential" and report["evicted"] == []
    assert "a" in engine.memory.resident_models()


def test_the_planner_never_plans_past_the_room(make_engine, monkeypatch):
    """The 16 GB rule: an eviction that does not give back what it promised falls back to a slower plan."""
    from dmlx import memory
    from dmlx.registry import GB

    monkeypatch.setattr(memory, "EVICT_RETURN_S", 0.2)  # it waits for memory that never comes
    engine = _two_models(make_engine, 7.0)
    _run(engine, "a", strategy="eager")
    engine.memory.headroom = lambda: int(2.8 * GB)  # freeing "a" gives nothing back (something holds it)
    report = _run(engine, "b")
    assert report["strategy"] == "sequential"
    assert report["need_gb"] <= report["headroom_gb"]


def _memory_comes_back_late(engine, monkeypatch, total_gb, late):
    """As on macOS: a freed part's memory shows up in the room only `late` seconds after the free (test
    round 1 of task 19: the room was measured 50 ms after the free, 3.91 -> 3.76 GB instead of about 5.9)."""
    import time

    from dmlx.registry import GB

    freed = []  # (when, bytes)
    original = engine.memory.free

    def free(model_id, name):
        resident = engine.memory._resident.get((model_id, name))
        freed.append((time.monotonic(), resident.bytes if resident else 0))
        original(model_id, name)

    monkeypatch.setattr(engine.memory, "free", free)
    engine.memory.headroom = lambda: int(total_gb * GB - engine.memory.resident_bytes() - sum(
        nbytes for when, nbytes in freed if time.monotonic() - when < late))
    return freed


def test_the_planner_waits_for_evicted_memory_to_come_back(make_engine, monkeypatch):
    engine = _two_models(make_engine, 7.0)
    _run(engine, "a", strategy="eager")  # 4.2 GB warm
    _memory_comes_back_late(engine, monkeypatch, 7.0, late=0.4)
    report = _run(engine, "b")
    assert report["strategy"] == "eager"  # not streamed or sequential with "a" gone anyway
    assert sorted(report["evicted"]) == ["a/text_encoder", "a/transformer"]  # as when memory comes back at once


def test_the_planner_evicts_no_more_than_needed_when_memory_comes_back_late(make_engine, monkeypatch):
    engine = _two_models(make_engine, 8.5)
    _run(engine, "a", strategy="eager")  # 4.3 GB left; one 2 GB part of "a" is enough for "b" eager
    freed = _memory_comes_back_late(engine, monkeypatch, 8.5, late=0.4)
    report = _run(engine, "b")
    assert report["strategy"] == "eager"
    assert len(report["evicted"]) == 1 and len(freed) == 1
    assert ("a", "transformer") in {(r["model"], r["component"]) for r in engine.memory.resident_components()} or \
        ("a", "text_encoder") in {(r["model"], r["component"]) for r in engine.memory.resident_components()}


def test_the_wait_for_evicted_memory_is_bounded(make_engine, monkeypatch):
    import time

    from dmlx import memory

    monkeypatch.setattr(memory, "EVICT_RETURN_S", 0.3)
    engine = _two_models(make_engine, 7.0)
    _run(engine, "a", strategy="eager")
    _memory_comes_back_late(engine, monkeypatch, 7.0, late=60)  # not within any wait
    began = time.monotonic()
    report = _run(engine, "b")
    assert report["strategy"] == "sequential" and report["need_gb"] <= report["headroom_gb"]
    assert time.monotonic() - began < 3 * 0.3 + 2  # at most one bounded wait per evicted part


# -- fix round 2 of task 19 ----------------------------------------------------------------------------------

def _peak_resident(engine, monkeypatch):
    """The most dMLX held at once during the run, as the guard's loads go."""
    peak = [0]
    original = engine.memory.ensure

    def ensure(model_id, name):
        loaded = original(model_id, name)
        peak[0] = max(peak[0], engine.memory.resident_bytes())
        return loaded

    monkeypatch.setattr(engine.memory, "ensure", ensure)
    return peak


def test_an_eager_plan_for_a_cached_prompt_does_not_load_the_encoder(make_engine, monkeypatch):
    """Test round 2: the plan left the encoder out (the prompt's embedding was cached), then JobGuard.start
    loaded every part of the eager plan, encoder included: on 16 GB swap grew 8.26 -> 11.0 GB."""
    from dmlx.registry import GB

    engine = _two_models(make_engine, 4.0)
    first = _run(engine, "b")  # encodes "a b picture"; the encoder is freed after the encode
    assert "text_encoder" in first["loads"]
    engine.unload("b")
    peak = _peak_resident(engine, monkeypatch)
    report = _run(engine, "b")  # the same prompt: cached
    assert report["strategy"] == "eager" and report["skipped"] == ["text_encoder"]
    assert "text_encoder" not in report["loads"] and sorted(report["loads"]) == ["transformer", "vae"]
    assert "text_encoder" not in {r["component"] for r in engine.memory.resident_components()}
    assert peak[0] <= report["need_gb"] * GB + 0.01 * GB  # what was loaded is what the plan counted


def test_a_cached_prompt_skips_the_encoder_in_every_strategy(make_engine):
    engine = _two_models(make_engine, 64.0)
    _run(engine, "b")
    for strategy in ("eager", "phase_free", "sequential"):
        engine.unload("b")
        report = _run(engine, "b", strategy=strategy)
        assert "text_encoder" not in report["loads"], strategy
        assert report["skipped"] == ["text_encoder"]


class MissingCacheAdapter(FakeAdapter):
    """Says the prompt is cached, then encodes anyway (its cache had the prompt with another negative
    prompt): the encoder the plan skipped is needed after all."""

    def has_embedding(self, prompt):
        return True

    def generate(self, req, hooks):
        self._cache.pop(req.prompt, None)
        return super().generate(req, hooks)


def test_an_encode_the_plan_skipped_makes_room_first(make_engine, monkeypatch):
    from dmlx.registry import GB

    engine = _two_models(make_engine, 4.0)
    engine._adapter_factory = MissingCacheAdapter
    peak = _peak_resident(engine, monkeypatch)
    report = _run(engine, "b")
    assert report["strategy"] == "eager" and report["unplanned_loads"] == ["text_encoder"]
    # The transformer was freed for the encode and loaded again for the denoise: never both resident.
    assert report["loads"] == ["vae", "transformer", "text_encoder", "transformer"]
    assert peak[0] <= 2.2 * GB + 1
    assert "text_encoder" not in {r["component"] for r in engine.memory.resident_components()}


def test_requests_answer_during_an_eviction_wait_and_a_stop_ends_the_evictions(make_engine, monkeypatch):
    """Test round 2: the wait for evicted memory (up to 2 s per part) held the memory lock, and the async
    GET /v1/models took it on the event loop: /v1/models, /health and the Stop all took 3.7 s."""
    import socket
    import threading
    import time

    import httpx
    import uvicorn

    from dmlx.api import create_app

    engine = _two_models(make_engine, 7.0)
    _run(engine, "a", strategy="eager")
    _memory_comes_back_late(engine, monkeypatch, 7.0, late=60)  # each eviction waits its full 2 s
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(create_app(engine, token=""), port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        base = f"http://127.0.0.1:{port}"
        job = engine.submit(task="t2i", prompt="a b picture", model="b")
        time.sleep(0.3)  # planning: the first eviction is waiting
        timings = {}
        for name, method, path in (("models", "GET", "/v1/models"), ("health", "GET", "/health"),
                                   ("memory", "GET", "/admin/memory"), ("cancel", "DELETE", f"/v1/jobs/{job.id}")):
            began = time.monotonic()
            assert httpx.request(method, base + path, timeout=10).status_code == 200
            timings[name] = time.monotonic() - began
        assert max(timings.values()) < 0.5, timings
        assert job.finished.wait(10)
        assert job.status == "cancelled"
        assert "a" in engine.memory.resident_models()  # the Stop ended the evictions: not every part went
    finally:
        server.should_exit = True


def test_models_report_the_memory_a_job_needs(make_engine):
    from dmlx.registry import GB

    engine = make_engine([fake_spec("m")])
    entry = next(m for m in engine.models() if m["id"] == "m")
    activation = fake_spec("m").activation_bytes(512, 512)
    assert entry["memory_gb"] == round((4.2 * GB + activation) / GB, 2)  # every part, at the default size
    assert entry["memory_min_gb"] == round((2.2 * GB + activation) / GB, 2)  # one big part at a time
    assert entry["memory_estimated"] is False
    # Before the download the parts' sizes are not known: the download's size stands in.
    spec = engine.registry.get("m")
    for comp in spec.components.values():
        comp.size = 0
    spec.download_size = int(4 * GB)
    entry = next(m for m in engine.models() if m["id"] == "m")
    assert entry["memory_gb"] == round((4 * GB + activation) / GB, 2) and entry["memory_min_gb"] is None
    assert entry["memory_estimated"] is True
    spec.download_size = 0
    entry = next(m for m in engine.models() if m["id"] == "m")
    assert entry["memory_gb"] is None and entry["memory_min_gb"] is None


# -- fix round 3 of task 19 ----------------------------------------------------------------------------------

def test_a_stop_during_an_eviction_wait_ends_the_job_at_once(make_engine, monkeypatch):
    """Test round 3: the Stop was checked only before each eviction, so the job ended after the eviction's
    wait for memory (up to 2 s, measured 1.92 s)."""
    import time

    engine = _two_models(make_engine, 7.0)
    _run(engine, "a", strategy="eager")
    _memory_comes_back_late(engine, monkeypatch, 7.0, late=60)  # each eviction waits its full 2 s
    job = engine.submit(task="t2i", prompt="a b picture", model="b")
    time.sleep(0.3)  # planning: the first eviction is waiting
    began = time.monotonic()
    engine.cancel(job.id)
    assert job.finished.wait(5) and job.status == "cancelled"
    assert time.monotonic() - began < 0.5
    assert "b" not in engine.memory.resident_models()


class SlowLoadAdapter(FakeAdapter):
    """Each part takes a while to load, as real weights do (2.1 s for Z-Image's three parts)."""

    seconds = 0.3
    loading = None  # set when the first load begins

    def load_component(self, name):
        if self.loading is not None:
            self.loading.set()
        import time

        time.sleep(self.seconds)
        return super().load_component(name)


def test_a_stop_while_an_eager_plan_loads_ends_the_job_and_frees_what_it_loaded(make_engine):
    """Test round 3: an eager plan loaded every part before the first Stop check (2.1 s after the Stop),
    and the stopped model stayed resident, as an eager plan keeps its parts warm after a job."""
    import threading
    import time

    engine = make_engine([fake_spec("m")])
    engine._adapter_factory = SlowLoadAdapter
    SlowLoadAdapter.loading = threading.Event()
    job = engine.submit(task="t2i", prompt="a red bicycle", model="m", strategy="eager")
    assert SlowLoadAdapter.loading.wait(5)
    began = time.monotonic()
    engine.cancel(job.id)
    assert job.finished.wait(5) and job.status == "cancelled", job.error
    assert time.monotonic() - began < 0.5  # at most the part loading when the Stop came
    assert len(job.report["loads"]) < len(fake_spec("m").components)  # it did not load every part
    assert engine.memory.resident_components() == []  # nothing of the stopped job stays
    assert not any(engine.adapter(engine.registry.get("m")).is_loaded(n) for n in fake_spec("m").components)


def test_a_job_stopped_before_it_ran_keeps_parts_that_were_warm_before(make_engine):
    """It frees only what it loaded itself: parts a finished job left warm stay."""
    import threading

    engine = make_engine([fake_spec("m")])
    engine._adapter_factory = SlowLoadAdapter
    SlowLoadAdapter.loading = None
    _run(engine, "m", strategy="phase_free")  # leaves the VAE and transformer warm
    warm = {r["component"] for r in engine.memory.resident_components()}
    assert warm and "text_encoder" not in warm
    SlowLoadAdapter.loading = threading.Event()
    job = engine.submit(task="t2i", prompt="a green bicycle", model="m", strategy="eager")
    assert SlowLoadAdapter.loading.wait(5)  # loading the text encoder
    engine.cancel(job.id)
    assert job.finished.wait(5) and job.status == "cancelled"
    assert {r["component"] for r in engine.memory.resident_components()} == warm
