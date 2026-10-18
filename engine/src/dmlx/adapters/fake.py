"""Fake adapter for tests: no GPU, deterministic images, honours the phase-hook contract."""

from __future__ import annotations

import time
from pathlib import Path

from PIL import Image, ImageDraw

from dmlx.adapters.base import GenRequest, PhaseHooks
from dmlx.registry import ModelSpec


class FakeAdapter:
    step_delay = 0.0

    def __init__(self, spec: ModelSpec, path: Path):
        self.spec = spec
        self.path = path
        self.loaded: set[str] = set()
        self.load_calls: list[str] = []
        self._cache: dict[str, bool] = {}
        self.hits = self.misses = 0

    def component_names(self) -> list[str]:
        return list(self.spec.components)

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def load_component(self, name: str) -> int:
        self.loaded.add(name)
        self.load_calls.append(name)
        return self.spec.components[name].size

    def free_component(self, name: str) -> None:
        self.loaded.discard(name)

    def embed_cache_stats(self) -> dict:
        return {"entries": len(self._cache), "hits": self.hits, "misses": self.misses}

    def generate(self, req: GenRequest, hooks: PhaseHooks) -> Image.Image:
        if req.prompt in self._cache:
            self.hits += 1
        else:
            self.misses += 1
            hooks.before_phase("encode")
            assert "text_encoder" in self.loaded, "encoder must be loaded during encode"
            self._cache[req.prompt] = True
            hooks.after_phase("encode")
        hooks.before_phase("denoise")
        assert "transformer" in self.loaded, "transformer must be loaded during denoise"
        for step in range(1, req.steps + 1):
            if self.step_delay:
                time.sleep(self.step_delay)
            hooks.on_step(step, req.steps)
            hooks.check()
        hooks.after_phase("denoise")
        hooks.before_phase("decode")
        assert "vae" in self.loaded
        image = Image.new("RGB", (req.width, req.height), ((req.seed * 37) % 256, 120, 180))
        ImageDraw.Draw(image).text((10, 10), req.prompt[:40], fill=(255, 255, 255))
        hooks.after_phase("decode")
        return image
