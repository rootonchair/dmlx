"""Adapters over mflux pipelines with per-component load/free.

mflux pipelines normally load every component at construction. Here the pipeline object
is built empty and each component is loaded on demand with mflux's single-component
loader, so the memory manager can run phases sequentially when memory is tight.
"""

from __future__ import annotations

import gc
import hashlib
import threading
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
from PIL import Image

from dmlx.adapters.base import GenRequest, JobAborted, PhaseHooks
from dmlx.registry import ModelSpec

_EMBED_CACHE_ENTRIES = 32


def _module_bytes(module: nn.Module) -> int:
    return sum(v.nbytes for _, v in tree_flatten(module.parameters()))


class _EmbedCache:
    """Small LRU of prompt embeddings, keyed by (model, prompt, extra)."""

    def __init__(self, capacity: int = _EMBED_CACHE_ENTRIES):
        self.capacity = capacity
        self._items: OrderedDict[str, object] = OrderedDict()
        self._prompts: dict[str, str] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(*parts: object) -> str:
        return hashlib.sha256(repr(parts).encode()).hexdigest()

    def has_prompt(self, prompt: str) -> bool:
        return prompt in self._prompts.values()

    def get(self, key: str):
        if key in self._items:
            self._items.move_to_end(key)
            self.hits += 1
            return self._items[key]
        self.misses += 1
        return None

    def put(self, key: str, value: object, prompt: str = "") -> None:
        self._items[key] = value
        self._prompts[key] = prompt
        self._items.move_to_end(key)
        while len(self._items) > self.capacity:
            old, _ = self._items.popitem(last=False)
            self._prompts.pop(old, None)

    def stats(self) -> dict:
        return {"entries": len(self._items), "hits": self.hits, "misses": self.misses}


class _LoopHooks:
    """mflux callback object bridging the denoise loop to PhaseHooks."""

    def __init__(self):
        self.hooks: PhaseHooks | None = None
        self.total = 0

    def call_before_loop(self, seed, prompt, latents, config, **_kwargs):
        if self.hooks:
            self.total = config.num_inference_steps
            self.hooks.before_phase("denoise")

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps, **_kwargs):
        if self.hooks:
            mx.eval(latents)
            self.hooks.on_step(t + 1, config.num_inference_steps)
            self.hooks.check()

    def call_after_loop(self, seed, prompt, latents, config, **_kwargs):
        if self.hooks:
            self.hooks.after_phase("denoise")
            self.hooks.before_phase("decode")

    def call_interrupt(self, *args, **kwargs):
        pass


class _MfluxAdapterBase:
    """Shared lifecycle; subclasses provide the pipeline class, weight definition, constructors."""

    weight_definition = None

    def __init__(self, spec: ModelSpec, path: Path):
        self.spec = spec
        self.path = Path(path)
        self._lock = threading.RLock()
        self.embeds = _EmbedCache()
        self._loop_hooks = _LoopHooks()
        self.pipes = self._build_empty_pipelines()  # task -> pipeline; all share the same component modules
        self.pipe = next(iter(self.pipes.values()))
        for pipe in self.pipes.values():
            pipe.callbacks.register(self._loop_hooks)
        self._components = {c.name: c for c in self.weight_definition.get_components()}
        self._quant_bits = None

    # -- lifecycle -------------------------------------------------------------------
    def component_names(self) -> list[str]:
        return list(self.spec.components)

    def is_loaded(self, name: str) -> bool:
        return getattr(self.pipe, name, None) is not None

    def load_component(self, name: str) -> int:
        from mflux.models.common.weights.loading.weight_applier import WeightApplier
        from mflux.models.common.weights.loading.weight_loader import WeightLoader

        with self._lock:
            if self.is_loaded(name):
                return _module_bytes(getattr(self.pipe, name))
            component = self._components[name]
            weights = WeightLoader.load_single_local(component=component, root_path=self.path)
            module = self._construct(name)
            bits = WeightApplier.apply_and_quantize_single(
                weights=weights, model=module, component=component, quantize_arg=None
            )
            del weights
            mx.eval(module.parameters())
            if name == "transformer":
                self._after_transformer_load(module)
            for pipe in self.pipes.values():
                setattr(pipe, name, module)
                if bits is not None:
                    pipe.bits = bits
            if bits is not None:
                self._quant_bits = bits
            gc.collect()
            return _module_bytes(module)

    def free_component(self, name: str) -> None:
        with self._lock:
            if getattr(self.pipe, name, None) is None:
                return
            for pipe in self.pipes.values():
                setattr(pipe, name, None)
            gc.collect()
            mx.clear_cache()

    def embed_cache_stats(self) -> dict:
        return self.embeds.stats()

    def has_embedding(self, prompt: str) -> bool:
        return self.embeds.has_prompt(prompt)

    def set_tiled_decode(self, tiled: bool) -> None:
        from mflux.models.common.vae.tiling_config import TilingConfig

        config = TilingConfig() if tiled else None
        for pipe in self.pipes.values():
            pipe.tiling_config = config

    # -- generation ------------------------------------------------------------------
    def generate(self, req: GenRequest, hooks: PhaseHooks) -> Image.Image:
        with self._lock:
            self._loop_hooks.hooks = hooks
            self._current_hooks = hooks
            try:
                image = self._run(req)
            except KeyboardInterrupt as exc:  # mflux converts this; keep ours intact
                raise JobAborted("cancelled", "Generation cancelled") from exc
            finally:
                self._loop_hooks.hooks = None
                self._current_hooks = None
            hooks.after_phase("decode")
            return image

    def _encode_with_cache(self, key_parts: tuple, encode_fn):
        key = self.embeds.key(self.spec.id, *key_parts)
        cached = self.embeds.get(key)
        if cached is not None:
            return cached, True
        hooks = self._current_hooks
        if hooks:
            hooks.before_phase("encode")
        value = encode_fn()
        mx.eval([v for v in (value if isinstance(value, tuple) else (value,)) if isinstance(v, mx.array)])
        self.embeds.put(key, value, prompt=str(key_parts[0]))
        if hooks:
            hooks.after_phase("encode")
        return value, False

    # -- subclass API ----------------------------------------------------------------
    def _build_empty_pipelines(self) -> dict:
        raise NotImplementedError

    def _construct(self, name: str) -> nn.Module:
        raise NotImplementedError

    def _after_transformer_load(self, module: nn.Module) -> None:
        pass

    def _run(self, req: GenRequest) -> Image.Image:
        raise NotImplementedError


def _empty_pipeline(adapter, base_cls, init_config, model_config, encode_method: str, extra: dict | None = None):
    """Subclass an mflux pipeline so it starts with no weights and caches prompt embeddings."""
    from mflux.models.common.compute_precision import ComputePrecision
    from mflux.models.common.tokenizer import TokenizerLoader

    def __init__(self):
        nn.Module.__init__(self)
        init_config(self, model_config)
        self.tokenizers = TokenizerLoader.load_all(
            definitions=adapter.weight_definition.get_tokenizers(), model_path=str(adapter.path)
        )
        self.vae = None
        self.transformer = None
        self.text_encoder = None
        self.bits = None
        self.lora_paths = None
        self.lora_scales = None
        self.compute_precision = ComputePrecision(None)
        for key, value in (extra or {}).items():
            setattr(self, key, value)

    def cached_encode(self, *, prompt, negative_prompt, guidance):
        parent = getattr(base_cls, encode_method)

        def encode():
            return parent(self, prompt=prompt, negative_prompt=negative_prompt, guidance=guidance)

        value, _hit = adapter._encode_with_cache((prompt, negative_prompt, guidance is not None and guidance > 1.0), encode)
        return value

    cls = type(f"Empty{base_cls.__name__}", (base_cls,), {"__init__": __init__, encode_method: cached_encode})
    return cls()


class Flux2KleinAdapter(_MfluxAdapterBase):
    def __init__(self, spec: ModelSpec, path: Path):
        from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition

        self.weight_definition = Flux2KleinWeightDefinition
        super().__init__(spec, path)

    def _build_empty_pipelines(self) -> dict:
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.flux2.flux2_initializer import Flux2Initializer
        from mflux.models.flux2.variants import Flux2Klein, Flux2KleinEdit

        cfg = ModelConfig.flux2_klein_4b()
        init = Flux2Initializer._init_config
        return {
            "t2i": _empty_pipeline(self, Flux2Klein, init, cfg, "_encode_prompt_pair"),
            "edit": _empty_pipeline(self, Flux2KleinEdit, init, cfg, "_encode_prompt_pair"),
        }

    def _construct(self, name: str) -> nn.Module:
        from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
        from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
        from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE

        cfg = self.pipe.model_config
        if name == "vae":
            return Flux2VAE()
        if name == "transformer":
            return Flux2Transformer(**cfg.transformer_overrides)
        if name == "text_encoder":
            return Qwen3TextEncoder(**cfg.text_encoder_overrides)
        raise KeyError(name)

    def _run(self, req: GenRequest) -> Image.Image:
        guidance = req.guidance if req.guidance is not None else 1.0
        if req.image_paths:
            generated = self.pipes["edit"].generate_image(
                seed=req.seed,
                prompt=req.prompt,
                num_inference_steps=req.steps,
                width=req.width,
                height=req.height,
                guidance=guidance,
                image_paths=[str(p) for p in req.image_paths],
            )
        else:
            generated = self.pipes["t2i"].generate_image(
                seed=req.seed,
                prompt=req.prompt,
                num_inference_steps=req.steps,
                width=req.width,
                height=req.height,
                guidance=guidance,
            )
        return generated.image


class ZImageAdapter(_MfluxAdapterBase):
    def __init__(self, spec: ModelSpec, path: Path):
        from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition

        self.weight_definition = ZImageWeightDefinition
        super().__init__(spec, path)

    def _build_empty_pipelines(self) -> dict:
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.z_image.variants.z_image import ZImage
        from mflux.models.z_image.z_image_initializer import ZImageInitializer

        cfg = ModelConfig.z_image_turbo()
        return {"t2i": _empty_pipeline(self, ZImage, ZImageInitializer._init_config, cfg, "_encode_prompts", {"float32": False})}

    def _construct(self, name: str) -> nn.Module:
        from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder
        from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
        from mflux.models.z_image.model.z_image_vae.vae import VAE

        if name == "vae":
            return VAE()
        if name == "transformer":
            return ZImageTransformer()
        if name == "text_encoder":
            return TextEncoder()
        raise KeyError(name)

    def _after_transformer_load(self, module: nn.Module) -> None:
        module.set_float32(False)

    def _run(self, req: GenRequest) -> Image.Image:
        generated = self.pipe.generate_image(
            seed=req.seed,
            prompt=req.prompt,
            num_inference_steps=req.steps,
            width=req.width,
            height=req.height,
            image_path=str(req.image_paths[0]) if req.image_paths else None,
        )
        return generated.image


ADAPTERS = {
    "mflux.flux2_klein": Flux2KleinAdapter,
    "mflux.z_image": ZImageAdapter,
}
