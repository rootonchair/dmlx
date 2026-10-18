"""Engine adapter contract.

The server owns component lifecycle; adapters only know how to load/free a single
component and how to run one generation, calling `PhaseHooks` at phase boundaries
so the memory manager can load and free components in between.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from PIL import Image

from dmlx.registry import ModelSpec


class JobAborted(Exception):
    """Raised from inside a generation to stop it (cancel, memory pressure)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class GenRequest:
    task: str  # t2i | edit
    prompt: str
    width: int
    height: int
    steps: int
    seed: int
    guidance: float | None = None
    image_paths: list[Path] = field(default_factory=list)  # edit sources/references, primary first


class PhaseHooks(Protocol):
    def before_phase(self, phase: str) -> None: ...
    def after_phase(self, phase: str) -> None: ...
    def on_step(self, step: int, total: int) -> None: ...
    def check(self) -> None: ...  # raises JobAborted


class Adapter(Protocol):
    spec: ModelSpec

    def component_names(self) -> list[str]: ...
    def is_loaded(self, name: str) -> bool: ...
    def load_component(self, name: str) -> int: ...  # returns resident bytes
    def free_component(self, name: str) -> None: ...
    def generate(self, req: GenRequest, hooks: PhaseHooks) -> Image.Image: ...
    def embed_cache_stats(self) -> dict: ...


AdapterFactory = Callable[[ModelSpec, Path], Adapter]
