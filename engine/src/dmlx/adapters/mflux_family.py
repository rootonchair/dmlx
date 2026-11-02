"""One adapter for any mflux family that follows mflux's common layout (task 15).

Every mflux family has an initializer (`_init_config`, `_init_models`), a weight definition
(`get_components`, `get_tokenizers`) and a pipeline class with `generate_image`. This adapter
uses those directly, so a family is a table entry, not a class.

Parts load when the pipeline first uses them. Each part is a property on the pipeline; the
first read asks the job guard for that part's phase (the spec's `phase`: encode, denoise or
decode). A pipeline that reads the transformer before its denoise loop therefore still gets it
at the right time, and a prompt found in the pipeline's cache never loads the text encoder.
"""

from __future__ import annotations

import importlib
import inspect
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

from mlx import nn
from PIL import Image

from dmlx.adapters.base import GenRequest
from dmlx.adapters.mflux_adapter import _EMBED_CACHE_ENTRIES, _MfluxAdapterBase
from dmlx.registry import ModelSpec

# adapter id -> where mflux keeps the family. "config" is the default ModelConfig factory.
FAMILIES: dict[str, dict] = {
    "mflux.ernie_image": {
        "initializer": "mflux.models.ernie_image.ernie_image_initializer:ErnieImageInitializer",
        "weights": "mflux.models.ernie_image.weights.ernie_weight_definition:ErnieWeightDefinition",
        "pipelines": {"t2i": "mflux.models.ernie_image.variants.txt2img.ernie_image:ErnieImage"},
        "config": "ernie_image_turbo",
    },
}


def _load(path):
    if not isinstance(path, str):
        return path  # already a class (tests)
    module, _, name = path.partition(":")
    return getattr(importlib.import_module(module), name)


def _call(fn, **available):
    """Call an mflux helper with the arguments its signature names (they differ per family)."""
    names = list(inspect.signature(fn).parameters)
    return fn(**{name: available[name] for name in names if name in available})


class _PromptCache(OrderedDict):
    """The pipeline's own prompt cache, bounded (mflux's is a plain dict that only grows)."""

    def __init__(self, capacity: int = _EMBED_CACHE_ENTRIES):
        super().__init__()
        self.capacity = capacity
        self.hits = 0
        self.misses = 0

    def __contains__(self, key) -> bool:
        found = super().__contains__(key)
        if found:
            self.hits += 1
            self.move_to_end(key)
        else:
            self.misses += 1
        return found

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self.capacity:
            self.popitem(last=False)

    def has_prompt(self, prompt: str) -> bool:
        return any(key == prompt or (isinstance(key, tuple) and key and key[0] == prompt) for key in self.keys())


class MfluxFamilyAdapter(_MfluxAdapterBase):
    family: dict = {}

    def __init__(self, spec: ModelSpec, path: Path):
        from mflux.models.common.config.model_config import ModelConfig

        self._modules: dict[str, nn.Module] = {}
        self.initializer = _load(self.family["initializer"])
        self.weight_definition = _load(self.family["weights"])
        self.model_config = getattr(ModelConfig, spec.mflux_config or self.family["config"])()
        self._init_config = getattr(self.initializer, "_init_config", None) or self.initializer.init_config
        super().__init__(spec, path)
        # `stream: [auto]` in the spec: stream every block stack the part has.
        from dmlx.adapters.streaming import find_stacks

        for name, lists in self.stream_lists.items():
            if lists == ["auto"]:
                self.stream_lists[name] = find_stacks(self._construct(name))

    # -- parts -----------------------------------------------------------------------
    def _module(self, name: str):
        return self._modules.get(name)

    def _install(self, name: str, module, bits=None) -> None:
        if module is None:
            self._modules.pop(name, None)
        else:
            self._modules[name] = module
        if bits is not None:
            for pipe in self.pipes.values():
                pipe.bits = bits

    def _construct(self, name: str) -> nn.Module:
        # Build every part on a scratch object and keep one. MLX is lazy: the unused parts are
        # never evaluated (measured: 15 ms, no memory for ERNIE).
        scratch = SimpleNamespace(model_config=self.model_config)
        _call(self._init_config, model=scratch, model_config=self.model_config)
        _call(self.initializer._init_models, model=scratch, model_config=self.model_config, model_path=self.path)
        return getattr(scratch, name)

    def _touch(self, name: str) -> nn.Module:
        """The pipeline reads a part: enter its phase, so the guard loads it and frees what it replaces."""
        hooks = self._current_hooks
        if hooks is not None:
            phase = self.spec.components[name].phase
            if phase == "denoise":
                if not self._loop_hooks.denoise_started:
                    hooks.after_phase("encode")
                    hooks.before_phase("denoise")
                    self._loop_hooks.denoise_started = True
            elif not self.is_loaded(name):
                if phase == "encode":
                    hooks.before_phase("encode")
                else:
                    hooks.before_phase("decode" if self._loop_hooks.finished else "preprocess")
        if not self.is_loaded(name):
            self.load_component(name)  # no guard (tests, tools) or a phase the guard does not manage
        return self._modules[name]

    # -- pipelines -------------------------------------------------------------------
    def _build_empty_pipelines(self) -> dict:
        return {task: self._empty(_load(path)) for task, path in self.family["pipelines"].items()}

    def _empty(self, base: type):
        from mflux.models.common.compute_precision import ComputePrecision
        from mflux.models.common.tokenizer import TokenizerLoader

        adapter = self

        def __init__(pipe):
            nn.Module.__init__(pipe)
            _call(adapter._init_config, model=pipe, model_config=adapter.model_config)
            pipe.prompt_cache = _PromptCache()
            pipe.tokenizers = TokenizerLoader.load_all(
                definitions=adapter.weight_definition.get_tokenizers(), model_path=str(adapter.path)
            )
            pipe.bits = None
            pipe.lora_paths = None
            pipe.lora_scales = None
            pipe.compute_precision = ComputePrecision(None)

        attrs = {"__init__": __init__}
        for name in self.spec.components:
            attrs[name] = property(lambda pipe, n=name: adapter._touch(n))
        return type(f"Empty{base.__name__}", (base,), attrs)()

    def embed_cache_stats(self) -> dict:
        caches = [pipe.prompt_cache for pipe in self.pipes.values()]
        return {
            "entries": sum(len(c) for c in caches),
            "hits": sum(c.hits for c in caches),
            "misses": sum(c.misses for c in caches),
        }

    def has_embedding(self, prompt: str) -> bool:
        return any(pipe.prompt_cache.has_prompt(prompt) for pipe in self.pipes.values())

    def _run(self, req: GenRequest) -> Image.Image:
        pipe = self.pipes["edit"] if req.image_paths and "edit" in self.pipes else self.pipes["t2i"]
        accepted = inspect.signature(pipe.generate_image).parameters
        kwargs = {
            "seed": req.seed,
            "prompt": req.prompt,
            "num_inference_steps": req.steps,
            "width": req.width,
            "height": req.height,
        }
        if req.guidance is not None and "guidance" in accepted:
            kwargs["guidance"] = req.guidance
        if req.negative_prompt and "negative_prompt" in accepted:
            kwargs["negative_prompt"] = req.negative_prompt
        if req.image_paths:
            if "image_paths" in accepted:
                kwargs["image_paths"] = [str(p) for p in req.image_paths]
            elif "image_path" in accepted:
                kwargs["image_path"] = str(req.image_paths[0])
        return pipe.generate_image(**kwargs).image


def adapter_for(adapter_id: str) -> type[MfluxFamilyAdapter]:
    return type(f"Mflux_{adapter_id.split('.')[-1]}", (MfluxFamilyAdapter,), {"family": FAMILIES[adapter_id]})
