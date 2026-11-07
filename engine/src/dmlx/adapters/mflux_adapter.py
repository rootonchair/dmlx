"""Adapters over mflux pipelines with per-component load/free.

mflux pipelines normally load every component at construction. Here the pipeline object
is built empty and each component is loaded on demand with mflux's single-component
loader, so the memory manager can run phases sequentially when memory is tight.
"""

from __future__ import annotations

import functools
import gc
import os
import threading
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
from PIL import Image

from dmlx.adapters.base import GenRequest, JobAborted, NonFinite, PhaseHooks
from dmlx.registry import ModelSpec

_EMBED_CACHE_ENTRIES = 32


def _module_bytes(module: nn.Module) -> int:
    return sum(v.nbytes for _, v in tree_flatten(module.parameters()))


class _LoopHooks:
    """mflux callback object bridging the denoise loop to PhaseHooks."""

    def __init__(self):
        self.hooks: PhaseHooks | None = None
        self.total = 0
        self.denoise_started = False
        self.finished = False  # the denoise loop has ended (decode comes next)

    def call_before_loop(self, seed, prompt, latents, config, **_kwargs):
        if self.hooks:
            # Most adapters close this phase immediately after producing embeddings.
            # Unified multimodal pipelines also encode reference images before entering
            # their denoise callback, so this is their safe boundary for releasing the encoder.
            self.hooks.after_phase("encode")
            self.total = config.num_inference_steps
            if not self.denoise_started:
                self.hooks.before_phase("denoise")
                self.denoise_started = True

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps, **_kwargs):
        if self.hooks:
            mx.eval(latents)
            # Stop at the first bad step, not after the whole run: the decode would be a black square.
            if not bool(mx.isfinite(latents).all()):
                raise NonFinite(t + 1)
            self.hooks.on_step(t + 1, config.num_inference_steps)
            self.hooks.check()

    def call_after_loop(self, seed, prompt, latents, config, **_kwargs):
        if self.hooks:
            self.hooks.after_phase("denoise")
            self.hooks.before_phase("decode")
        self.denoise_started = False
        self.finished = True

    def call_interrupt(self, *args, **kwargs):
        pass


_FIXES_INSTALLED = False


def _install_mflux_fixes() -> None:
    """mflux's tiled VAE encode assembles tiles in float32 numpy and returns float32 latents.

    For an edit that promotes the whole transformer to float32 (about 2x slower and more
    memory). The untiled encode returns the model precision, so cast the tiled result to it.
    """
    global _FIXES_INSTALLED
    if _FIXES_INSTALLED:
        return
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.common.vae.vae_util import VAEUtil

    original = VAEUtil.encode

    def encode(*args, **kwargs):
        latents = original(*args, **kwargs)
        return latents.astype(ModelConfig.precision) if latents.dtype == mx.float32 else latents

    VAEUtil.encode = staticmethod(encode)
    from dmlx.adapters.mflux_fixes import install_gated_delta_fix

    install_gated_delta_fix()  # MOSS Image's text encoder: wrong past one chunk, NaN past ~250 tokens
    _FIXES_INSTALLED = True


class _MfluxAdapterBase:
    """Shared lifecycle; subclasses provide the pipeline class, weight definition, constructors."""

    weight_definition = None
    supports_banded_decode = False
    # Per component, stack name -> first layer whose output the pipeline never reads.
    stream_skip_from: dict[str, dict[str, int]] = {}

    def __init__(self, spec: ModelSpec, path: Path):
        _install_mflux_fixes()
        self.spec = spec
        self.path = Path(path)
        self._lock = threading.RLock()
        self._loop_hooks = _LoopHooks()
        self.pipes = self._build_empty_pipelines()  # task -> pipeline; all share the same component modules
        self.pipe = next(iter(self.pipes.values()))
        for pipe in self.pipes.values():
            pipe.callbacks.register(self._loop_hooks)
        self._components = {c.name: c for c in self.weight_definition.get_components()}
        self._quant_bits = None
        self.stream_mode = False  # set per job by the memory guard
        self.stream_lists = {n: c.stream for n, c in spec.components.items() if c.stream}
        self._streamed: dict[str, bool] = {}

    # -- lifecycle -------------------------------------------------------------------
    def component_names(self) -> list[str]:
        return list(self.spec.components)

    def supports_streaming(self, name: str) -> bool:
        return name in self.stream_lists

    def _wants_stream(self, name: str) -> bool:
        return self.stream_mode and self.supports_streaming(name)

    def is_loaded(self, name: str) -> bool:
        """Loaded in the form the current job wants (streamed or whole)."""
        return self._module(name) is not None and self._streamed.get(name, False) == self._wants_stream(name)

    def load_component(self, name: str) -> int:
        from mflux.models.common.weights.loading.weight_applier import WeightApplier
        from mflux.models.common.weights.loading.weight_loader import WeightLoader

        with self._lock:
            if self.is_loaded(name):
                return _module_bytes(self._module(name))
            self.free_component(name)  # loaded in the other form
            if self._wants_stream(name):
                return self._load_streamed(name)
            component = self._components[name]
            weights = WeightLoader.load_single_local(component=component, root_path=self.path)
            module = self._construct(name)
            self._before_apply(name, module, weights)
            bits = WeightApplier.apply_and_quantize_single(
                weights=weights, model=module, component=component, quantize_arg=None
            )
            del weights
            mx.eval(module.parameters())
            if name == "transformer":
                self._after_transformer_load(module)
            self._install(name, module, bits)
            if bits is not None:
                self._quant_bits = bits
            self._streamed[name] = False
            gc.collect()
            return _module_bytes(module)

    def _before_apply(self, name: str, module, weights) -> None:
        """A family's own step before its weights go in (Ideogram 4 rebuilds its quantized layers)."""

    def _load_streamed(self, name: str) -> int:
        from dmlx.adapters.streaming import make_streamed

        module = self._construct(name)
        component = self._components[name]
        nbytes = make_streamed(
            module,
            self.path / component.hf_subdir,
            self.stream_lists[name],
            self.stream_skip_from.get(name),
            check=lambda: self._current_hooks.check() if self._current_hooks else None,
        )
        if name == "transformer":
            self._after_transformer_load(module)
        self._install(name, module)
        self._streamed[name] = True
        gc.collect()
        return nbytes

    def free_component(self, name: str) -> None:
        with self._lock:
            if self._module(name) is None:
                return
            self._install(name, None)
            self._streamed.pop(name, None)
            gc.collect()
            mx.clear_cache()

    def embed_cache_stats(self) -> dict:
        return {"entries": 0, "hits": 0, "misses": 0}

    def has_embedding(self, prompt: str) -> bool:
        return False

    decode_tile_size: int | None = None  # pixels per tile in the tiled decode; None: mflux's 512

    def set_decode_mode(self, mode: str) -> None:
        """full: one piece. tiled: mflux tiles. banded: exact low-memory decode (FLUX.2 VAE only)."""
        from mflux.models.common.vae.tiling_config import TilingConfig

        if mode == "banded" and not self.supports_banded_decode:
            mode = "tiled"
        # The same config also tiles the VAE encode of edit sources; keep that whenever memory is tight.
        tiled = TilingConfig(vae_decode_tile_size=self.decode_tile_size) if self.decode_tile_size else TilingConfig()
        config = {"full": None, "tiled": tiled, "banded": TilingConfig(vae_decode_tiles_per_dim=None)}[mode]
        for pipe in self.pipes.values():
            pipe.tiling_config = config
        vae = self._module("vae")
        if vae is not None:
            if mode == "banded":
                from dmlx.adapters.banded_vae import decode

                object.__setattr__(vae, "decode", functools.partial(decode, vae))
            else:
                vae.__dict__.pop("decode", None)

    # -- generation ------------------------------------------------------------------
    def generate(self, req: GenRequest, hooks: PhaseHooks) -> Image.Image:
        with self._lock:
            self._loop_hooks.hooks = hooks
            self._loop_hooks.denoise_started = False  # a cancelled job can leave it set
            self._loop_hooks.finished = False
            self._current_hooks = hooks
            if self.stream_mode:
                # A compiled forward pass cannot load weights part-way through.
                mx.disable_compile()
            try:
                image = self._run(req)
            except KeyboardInterrupt as exc:  # mflux converts this; keep ours intact
                raise JobAborted("cancelled", "Generation cancelled") from exc
            finally:
                if self.stream_mode and not os.environ.get("MLX_DISABLE_COMPILE"):
                    mx.enable_compile()
                self._loop_hooks.hooks = None
                self._current_hooks = None
            hooks.after_phase("decode")
            return image

    # -- subclass API ----------------------------------------------------------------
    def _module(self, name: str):
        """The loaded part, or None."""
        return getattr(self.pipe, name, None)

    def _install(self, name: str, module, bits=None) -> None:
        """Give every pipeline the part (None removes it)."""
        for pipe in self.pipes.values():
            setattr(pipe, name, module)
            if bits is not None:
                pipe.bits = bits

    def _build_empty_pipelines(self) -> dict:
        raise NotImplementedError

    def _construct(self, name: str) -> nn.Module:
        raise NotImplementedError

    def _after_transformer_load(self, module: nn.Module) -> None:
        pass

    def _run(self, req: GenRequest) -> Image.Image:
        raise NotImplementedError


class SeedVR2Adapter(_MfluxAdapterBase):
    """Lifecycle integration for the gated SeedVR2 upscaler.

    The adapter is registered for benchmark/prototype use, but no built-in model
    spec exposes it until its memory gate has passed.
    """

    def __init__(self, spec: ModelSpec, path: Path):
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.seedvr2.weights.seedvr2_weight_definition import (
            SeedVR2WeightDefinition,
        )

        self.model_config = getattr(ModelConfig, spec.mflux_config or "seedvr2_3b")()
        self.weight_definition = SeedVR2WeightDefinition.resolve(self.model_config)
        super().__init__(spec, path)

    def _build_empty_pipelines(self) -> dict:
        from mflux.models.seedvr2.seedvr2_initializer import SeedVR2Initializer
        from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2

        model_config = self.model_config

        class EmptySeedVR2(SeedVR2):
            def __init__(self):
                nn.Module.__init__(self)
                SeedVR2Initializer._init_config(self, model_config)
                self.vae = None
                self.transformer = None
                self.bits = None

        return {"upscale": EmptySeedVR2()}

    def _construct(self, name: str) -> nn.Module:
        from mflux.models.seedvr2.model.seedvr2_transformer.transformer import (
            SeedVR2Transformer,
        )
        from mflux.models.seedvr2.model.seedvr2_vae.vae import SeedVR2VAE

        if name == "vae":
            return SeedVR2VAE()
        if name == "transformer":
            return SeedVR2Transformer(**(self.model_config.transformer_overrides or {}))
        raise KeyError(name)

    def _run(self, req: GenRequest) -> Image.Image:
        if len(req.image_paths) != 1:
            raise ValueError("SeedVR2 requires exactly one source image.")
        from mflux.utils.scale_factor import ScaleFactor

        scale = int(req.extras.get("scale", 2))
        generated = self.pipe.generate_image(
            seed=req.seed,
            image_path=req.image_paths[0],
            resolution=ScaleFactor.parse(f"{scale}x"),
            softness=float(req.extras.get("softness", 0.0)),
        )
        return generated.image

    def generate(self, req: GenRequest, hooks: PhaseHooks) -> Image.Image:
        # SeedVR2 encodes the source with its VAE before callbacks enter the
        # denoise loop. Announce that phase so sequential mode can load the VAE,
        # release it before the transformer, and reload it for final decode.
        hooks.before_phase("preprocess")
        return super().generate(req, hooks)


# The image families use the generic adapter (dmlx.adapters.mflux_family).
ADAPTERS = {
    "mflux.seedvr2": SeedVR2Adapter,
}
