"""Memory checks: the guard warns when freeing a part does not release its memory (task 15: a lazy cached
encoding kept FLUX.3's text encoder alive after the guard had freed it)."""

import logging

from dmlx.memory import MemoryManager


class Adapter:
    """A 300 MB part. The memory readings are simulated: CI runners have no Metal GPU, so MLX's own
    counters do not show allocations the way a Mac does."""

    PART = 300 * 1024**2

    def __init__(self, keep_alive: bool):
        self.keep_alive = keep_alive
        self.loaded = False
        self.active = 1024**3

    def load_component(self, name):
        self.loaded = True
        self.active += self.PART
        return self.PART

    def is_loaded(self, name):
        return self.loaded

    def free_component(self, name):
        self.loaded = False
        if not self.keep_alive:  # kept alive: like a lazy result that still references the weights
            self.active -= self.PART


def _free(keep_alive, caplog, monkeypatch):
    from dmlx import memory

    adapter = Adapter(keep_alive)
    monkeypatch.setattr(memory.mx, "get_active_memory", lambda: adapter.active)
    manager = MemoryManager()
    manager.register_adapter("m", adapter)
    manager.ensure("m", "text_encoder")
    with caplog.at_level(logging.INFO, logger="dmlx.memory"):
        manager.free("m", "text_encoder")
    return [r for r in caplog.records if r.levelno == logging.WARNING]


def test_a_clean_free_does_not_warn(caplog, monkeypatch):
    assert _free(False, caplog, monkeypatch) == []


def test_a_free_that_releases_nothing_warns(caplog, monkeypatch):
    warnings = _free(True, caplog, monkeypatch)
    assert len(warnings) == 1 and "still holds" in warnings[0].getMessage()


def test_small_sizes_are_estimated_as_one_megapixel():
    # Working memory stops shrinking below 1 MP (Krea 2's tiled decode: 6.1 GB at 512 px and at 1024 px).
    from conftest import fake_spec

    from dmlx.registry import GB

    spec = fake_spec("m")
    spec.activation_gb_per_mp, spec.decode_tiled_gb_per_mp = 1.5, 6.8
    assert spec.decode_bytes(512, 512, "tiled") == spec.decode_bytes(1000, 1000, "tiled") == int(6.8 * GB)
    assert spec.activation_bytes(512, 512) == spec.activation_bytes(1000, 1000)
    assert spec.decode_bytes(2000, 1000, "tiled") == int(6.8 * 2 * GB)


def test_the_refusal_never_reads_as_needing_what_it_has():
    """Test round 3 of task 17: "needs about 3.3 GB but only 3.3 GB can be used"."""
    from dmlx.memory import GB, InsufficientMemory

    close = str(InsufficientMemory(int(3.34 * GB), int(3.26 * GB), "klein-9b", "FLUX.2 Klein 9B"))
    assert "needs about 3.34 GB but only 3.26 GB can be used" in close
    closer = str(InsufficientMemory(int(3.3449 * GB), int(3.3401 * GB), "klein-9b"))
    assert "needs about 3.345 GB but only 3.340 GB" in closer
    apart = str(InsufficientMemory(int(9.5 * GB), int(7.6 * GB), "qwen-image", "Qwen Image 2.1"))
    assert "needs about 9.5 GB but only 7.6 GB can be used" in apart
    assert "only 0.0 GB" in str(InsufficientMemory(int(2 * GB), -5, "x"))


def test_the_refusal_suggests_a_smaller_size_only_when_there_is_one(make_engine):
    """Task 19: "pick a smaller size" at 1024x1024, the app's smallest size."""
    from dmlx.memory import GB, InsufficientMemory

    at_1mp = str(InsufficientMemory(int(9.5 * GB), int(7.3 * GB), "q", "Qwen", pixels=1216 * 832))
    assert at_1mp.endswith("Close some apps, or choose a lighter model.")
    big = str(InsufficientMemory(int(9.5 * GB), int(7.3 * GB), "q", "Qwen", pixels=1536 * 1536))
    assert big.endswith("Close some apps, pick a smaller size, or choose a lighter model.")

    engine = make_engine(headroom_gb=1.0)
    for size, smaller in (("1024x1024", False), ("2048x2048", True)):
        job = engine.submit(task="t2i", prompt="a boat", model="fake-a", size=size)
        assert job.finished.wait(10)
        assert job.error["code"] == "insufficient_memory"
        assert ("pick a smaller size" in job.error["message"]) is smaller
