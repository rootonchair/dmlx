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
