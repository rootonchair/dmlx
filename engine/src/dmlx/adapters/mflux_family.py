"""One adapter for any mflux family that follows mflux's common layout (task 15).

Every mflux family has an initializer (`_init_config`, `_init_models`), a weight definition
(`get_components`, `get_tokenizers`) and a pipeline class with `generate_image`. This adapter
uses those directly, so a family is a table entry (`FAMILIES`), not a class.

Parts load when the pipeline first uses them. Each part is a property on the pipeline; the
first read asks the job guard for that part's phase (the spec's `phase`: encode, denoise or
decode):

- the text side enters `encode`;
- the VAE before the denoise loop enters `preprocess` (an edit source), and ends `encode` first,
  so the text encoder is gone before the VAE runs; after the loop it is `decode`;
- the transformer before the loop enters `layout` (some pipelines read its settings early),
  and ends `encode`; the loop callback then enters `denoise` as before.

A prompt found in the prompt cache never loads the text encoder.
"""

from __future__ import annotations

import importlib
import inspect
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
from PIL import Image

from dmlx.adapters.base import GenRequest, PhaseHooks
from dmlx.adapters.mflux_adapter import (
    _EMBED_CACHE_ENTRIES,
    _MfluxAdapterBase,
    _module_bytes,
)
from dmlx.registry import ModelSpec

_M = "mflux.models"

# adapter id -> the family's mflux classes and the few settings that differ.
#   initializer, pipelines {task: class}, config (ModelConfig factory used when the spec names none)
#   weights        the weight definition; None: ask the initializer's weight_definition(model_config)
#   encode         an encode method to cache (pipelines that do not use their own prompt_cache)
#   attrs          extra attributes set on the pipelines and on the scratch object parts are built on
#   after_load     {part: fn(module)} run after a part loads
#   banded_decode  the VAE supports dMLX's banded decode (FLUX.2 VAE)
#   skip_from      {part: {stack: first unused layer}} never read while streaming
FAMILIES: dict[str, dict] = {
    "mflux.flux2_klein": {
        "initializer": f"{_M}.flux2.flux2_initializer:Flux2Initializer",
        "weights": f"{_M}.flux2.weights.flux2_weight_definition:Flux2KleinWeightDefinition",
        "pipelines": {
            "t2i": f"{_M}.flux2.variants:Flux2Klein",
            "edit": f"{_M}.flux2.variants:Flux2KleinEdit",
        },
        "config": "flux2_klein_4b",
        "encode": "_encode_prompt_pair",
        "attrs": {"small_text_encoder": False},
        "banded_decode": True,
        # Prompt embeddings stack hidden states 9, 18, 27 (the output of layers 8, 17, 26).
        "skip_from": {"text_encoder": {"layers": 27}},
    },
    "mflux.z_image": {
        "initializer": f"{_M}.z_image.z_image_initializer:ZImageInitializer",
        "weights": f"{_M}.z_image.weights.z_image_weight_definition:ZImageWeightDefinition",
        "pipelines": {"t2i": f"{_M}.z_image.variants.z_image:ZImage"},
        "config": "z_image_turbo",
        "encode": "_encode_prompts",
        "attrs": {"float32": False},
        "after_load": {"transformer": lambda module: module.set_float32(False)},
    },
    "mflux.ernie_image": {
        "initializer": f"{_M}.ernie_image.ernie_image_initializer:ErnieImageInitializer",
        "weights": f"{_M}.ernie_image.weights.ernie_weight_definition:ErnieWeightDefinition",
        "pipelines": {"t2i": f"{_M}.ernie_image.variants.txt2img.ernie_image:ErnieImage"},
        "config": "ernie_image_turbo",
    },
    "mflux.moss_image": {
        "initializer": f"{_M}.moss_image.moss_image_initializer:MOSSImageInitializer",
        "weights": f"{_M}.moss_image.weights.moss_image_weight_definition:MOSSImageWeightDefinition",
        "pipelines": {"t2i": f"{_M}.moss_image.variants.txt2img.moss_image:MOSSImage"},
        "config": "moss_image_sft",
    },
    "mflux.flux3_klein": {
        "initializer": f"{_M}.flux3.flux3_initializer:Flux3Initializer",
        "weights": None,
        "pipelines": {
            "t2i": f"{_M}.flux3.variants.txt2img.flux3_klein:Flux3Klein",
            "edit": f"{_M}.flux3.variants.edit.flux3_klein_edit:Flux3KleinEdit",
        },
        "config": "flux3_image_klein",
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
    """The pipelines' prompt cache: shared by a family's pipelines and bounded (mflux's only grows)."""

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

    def stats(self) -> dict:
        return {"entries": len(self), "hits": self.hits, "misses": self.misses}


class _SettlingHooks:
    """The job's hooks, with encoded prompts evaluated before the encode phase ends."""

    def __init__(self, hooks: PhaseHooks, settle):
        self._hooks = hooks
        self._settle = settle

    def after_phase(self, phase: str) -> None:
        if phase == "encode":
            self._settle()
        self._hooks.after_phase(phase)

    def __getattr__(self, name):
        return getattr(self._hooks, name)


class MfluxFamilyAdapter(_MfluxAdapterBase):
    family: dict = {}

    def __init__(self, spec: ModelSpec, path: Path):
        from mflux.models.common.config.model_config import ModelConfig

        self._modules: dict[str, nn.Module] = {}
        self.prompt_cache = _PromptCache()
        self.initializer = _load(self.family["initializer"])
        self.model_config = self._model_config(ModelConfig, spec)
        self._init_config = getattr(self.initializer, "_init_config", None) or self.initializer.init_config
        if getattr(self, "weight_definition", None) is None:
            weights = self.family.get("weights")
            self.weight_definition = _load(weights) if weights else self.initializer.weight_definition(self.model_config)
        self.supports_banded_decode = bool(self.family.get("banded_decode"))
        self.stream_skip_from = self.family.get("skip_from", {})
        super().__init__(spec, path)
        # `stream: [auto]` in the spec: stream every block stack the part has.
        from dmlx.adapters.streaming import find_stacks

        for name, lists in self.stream_lists.items():
            if lists == ["auto"]:
                self.stream_lists[name] = find_stacks(self._construct(name))

    def _model_config(self, model_config_cls, spec: ModelSpec):
        return getattr(model_config_cls, spec.mflux_config or self.family["config"])()

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
        # never evaluated (measured: 11-27 ms and under 1 MB across eight families).
        scratch = SimpleNamespace(model_config=self.model_config)
        _call(self._init_config, model=scratch, model_config=self.model_config)
        for key, value in self.family.get("attrs", {}).items():
            setattr(scratch, key, value)
        _call(self.initializer._init_models, model=scratch, model_config=self.model_config, model_path=self.path)
        return getattr(scratch, name)

    def _after_transformer_load(self, module: nn.Module) -> None:
        hook = self.family.get("after_load", {}).get("transformer")
        if hook:
            hook(module)

    def _touch(self, name: str) -> nn.Module:
        """The pipeline reads a part: enter its phase, so the guard loads it and frees what it replaces."""
        hooks = self._current_hooks
        if hooks is not None:
            phase = self.spec.components[name].phase
            loop = self._loop_hooks
            if phase == "encode":
                if not self.is_loaded(name):
                    hooks.before_phase("encode")
            elif phase == "denoise":
                # Read before the loop (settings such as FLUX.3's flow_map, or Qwen 2.1's axes). If the
                # text encoder runs after this read, it frees the transformer again in the low-memory
                # strategies, so the transformer loads twice (about 1 s); every load goes through the guard.
                if not loop.denoise_started and not loop.finished and not self.is_loaded(name):
                    if self._encoder_loaded():
                        hooks.after_phase("encode")
                    hooks.before_phase("layout")
            elif loop.finished:
                if not self.is_loaded(name):
                    hooks.before_phase("decode")
            elif not loop.denoise_started:
                if self._encoder_loaded():
                    hooks.after_phase("encode")
                if not self.is_loaded(name):
                    hooks.before_phase("preprocess")
        if not self.is_loaded(name):
            self.load_component(name)  # no guard (tests, tools) or a phase the guard does not manage
        return self._modules[name]

    def _encoder_loaded(self) -> bool:
        return any(self.is_loaded(n) for n, c in self.spec.components.items() if c.phase == "encode")

    # -- pipelines -------------------------------------------------------------------
    def _build_empty_pipelines(self) -> dict:
        return {task: self._empty(_load(path)) for task, path in self.family["pipelines"].items()}

    def _pipe_init(self, pipe) -> None:
        from mflux.models.common.compute_precision import ComputePrecision
        from mflux.models.common.tokenizer import TokenizerLoader

        nn.Module.__init__(pipe)
        _call(self._init_config, model=pipe, model_config=self.model_config)
        pipe.prompt_cache = self.prompt_cache
        pipe.tokenizers = TokenizerLoader.load_all(
            definitions=self.weight_definition.get_tokenizers(), model_path=str(self.path)
        )
        pipe.bits = None
        pipe.lora_paths = None
        pipe.lora_scales = None
        pipe.compute_precision = ComputePrecision(None)
        for key, value in self.family.get("attrs", {}).items():
            setattr(pipe, key, value)

    def _empty(self, base: type):
        adapter = self
        attrs = {"__init__": lambda pipe: adapter._pipe_init(pipe)}
        for name in self.spec.components:
            attrs[name] = property(lambda pipe, n=name: adapter._touch(n))
        encode = self.family.get("encode")
        if encode:
            attrs[encode] = self._cached_encode(getattr(base, encode))
        return type(f"Empty{base.__name__}", (base,), attrs)()

    def _cached_encode(self, parent):
        """Wrap an encode method with the prompt cache; free the text side as soon as it is done."""
        adapter = self

        def encode(pipe, *, prompt, negative_prompt=None, guidance=None, **kwargs):
            key = (prompt, negative_prompt, guidance is not None and guidance > 1.0, parent.__name__)
            cache = adapter.prompt_cache
            if key in cache:
                return cache[key]
            value = parent(pipe, prompt=prompt, negative_prompt=negative_prompt, guidance=guidance, **kwargs)
            mx.eval([v for _, v in tree_flatten(value) if isinstance(v, mx.array)])
            cache[key] = value
            if adapter._current_hooks is not None:
                adapter._current_hooks.after_phase("encode")
            return value

        return encode

    def generate(self, req: GenRequest, hooks: PhaseHooks) -> Image.Image:
        return super().generate(req, _SettlingHooks(hooks, self._settle_encodings))

    def _settle_encodings(self) -> None:
        """Evaluate cached prompt encodings. Some pipelines cache them unevaluated, and a lazy result
        keeps the text encoder's weights alive after the guard frees it (measured on FLUX.3: the
        encoder stayed resident through the whole denoise loop)."""
        mx.eval([v for _, v in tree_flatten(list(self.prompt_cache.values())) if isinstance(v, mx.array)])

    def embed_cache_stats(self) -> dict:
        return self.prompt_cache.stats()

    def has_embedding(self, prompt: str) -> bool:
        return self.prompt_cache.has_prompt(prompt)

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


class QwenImage21Adapter(MfluxFamilyAdapter):
    """Qwen Image 2.1: one unified pipeline for new images and ordered reference edits.

    It differs from the common layout in three ways: its parts are built from the checkpoint's
    config files, its text side is mflux's small multimodal encoder (a separate download, which
    also brings the processor), and the pipeline needs the processor before it encodes.
    """

    family = {
        "initializer": f"{_M}.qwen21.qwen21_initializer:Qwen21Initializer",
        "weights": f"{_M}.qwen21.weights.qwen_image21_weight_definition:QwenImage21NoTextEncoderWeightDefinition",
        "pipelines": {"unified": f"{_M}.qwen21.variants.edit.qwen_image_21_edit:QwenImage21Edit"},
        "config": "qwen_image_21",
    }

    def __init__(self, spec: ModelSpec, path: Path):
        import json

        self._component_configs = {
            name: json.loads((Path(path) / name / "config.json").read_text()) for name in ("vae", "transformer")
        }
        super().__init__(spec, path)

    def _pipe_init(self, pipe) -> None:
        from mflux.models.common.compute_precision import ComputePrecision

        nn.Module.__init__(pipe)
        self.initializer.init_config(pipe, self.model_config)
        pipe.prompt_cache = self.prompt_cache
        pipe._checkpoint_path = str(self.path)
        pipe._component_configs = self._component_configs
        pipe.small_text_encoder = True
        pipe.tokenizers = {}
        pipe.processor = None
        pipe.bits = None
        pipe.lora_paths = None
        pipe.lora_scales = None
        pipe.compute_precision = ComputePrecision(None)

    def load_component(self, name: str) -> int:
        if name != "text_encoder":
            return super().load_component(name)
        from mflux.models.qwen21.model.qwen21_small_text_encoder.qwen21_small_text_encoder import (
            Qwen21SmallTextEncoder,
        )

        with self._lock:
            if self.is_loaded(name):
                return _module_bytes(self._modules[name])
            self.free_component(name)
            encoder, processor = Qwen21SmallTextEncoder.load()
            mx.eval(encoder.parameters())
            self._install(name, encoder)
            for pipe in self.pipes.values():
                pipe.processor = processor
            self._streamed[name] = False
            return _module_bytes(encoder)

    def _construct(self, name: str) -> nn.Module:
        from mflux.models.qwen21.model.qwen21_transformer.qwen_image21_transformer import (
            QwenImage21Transformer,
        )
        from mflux.models.qwen21.model.qwen21_vae.vae import QwenImage21VAE

        config = self._component_configs[name]
        if name == "vae":
            return QwenImage21VAE(config)
        if name == "transformer":
            return QwenImage21Transformer(config)
        raise KeyError(name)

    def generate(self, req: GenRequest, hooks: PhaseHooks) -> Image.Image:
        # The pipeline reads the processor (which comes with the text encoder) before it encodes.
        hooks.before_phase("encode")
        return super().generate(req, hooks)

    def _run(self, req: GenRequest) -> Image.Image:
        negative = req.negative_prompt or None
        # The base checkpoint is guidance-free by default. A negative prompt opts
        # into true CFG, which evaluates both branches and therefore costs more.
        guidance = float(req.guidance or 4.0) if negative else 1.0
        generated = self.pipe.generate_image(
            seed=req.seed,
            prompt=req.prompt,
            negative_prompt=negative,
            num_inference_steps=req.steps,
            width=req.width,
            height=req.height,
            guidance=guidance,
            image_paths=[str(path) for path in req.image_paths],
            output_resolution=max(req.width, req.height),
            use_kv_cache=True,
        )
        return generated.image


def weight_definition(adapter_id: str, config: str):
    """The weight definition a family uses for an mflux config (the catalog checks repos against it)."""
    from mflux.models.common.config.model_config import ModelConfig

    family = FAMILIES[adapter_id]
    if family.get("weights"):
        return _load(family["weights"])
    return _load(family["initializer"]).weight_definition(getattr(ModelConfig, config)())


def adapter_for(adapter_id: str) -> type[MfluxFamilyAdapter]:
    if adapter_id == "mflux.qwen_image_21":
        return QwenImage21Adapter
    return type(f"Mflux_{adapter_id.split('.')[-1]}", (MfluxFamilyAdapter,), {"family": FAMILIES[adapter_id]})
