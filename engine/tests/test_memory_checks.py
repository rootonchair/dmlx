"""Memory checks: the guard warns when freeing a part does not release its memory (task 15: a lazy cached
encoding kept FLUX.3's text encoder alive after the guard had freed it)."""

import logging

import mlx.core as mx

from dmlx.memory import MemoryManager


class Adapter:
    def __init__(self, keep_alive: bool):
        self.keep_alive = keep_alive
        self.part = None
        self.held = None

    def load_component(self, name):
        self.part = mx.zeros((300 * 1024**2 // 4,), dtype=mx.float32)  # 300 MB
        mx.eval(self.part)
        return self.part.nbytes

    def is_loaded(self, name):
        return self.part is not None

    def free_component(self, name):
        if self.keep_alive:
            self.held = self.part  # like a lazy result that still references the weights
        self.part = None
        mx.clear_cache()


def _free(keep_alive, caplog):
    manager = MemoryManager()
    manager.register_adapter("m", Adapter(keep_alive))
    manager.ensure("m", "text_encoder")
    with caplog.at_level(logging.INFO, logger="dmlx.memory"):
        manager.free("m", "text_encoder")
    return [r for r in caplog.records if r.levelno == logging.WARNING]


def test_a_clean_free_does_not_warn(caplog):
    assert _free(False, caplog) == []


def test_a_free_that_releases_nothing_warns(caplog):
    warnings = _free(True, caplog)
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
