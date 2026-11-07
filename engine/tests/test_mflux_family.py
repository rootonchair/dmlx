"""The generic mflux family adapter: parts load on first use, in the right phase (task 15)."""

from types import SimpleNamespace

import mlx.core as mx
import yaml
from mlx import nn

from dmlx.adapters.base import GenRequest
from dmlx.adapters.mflux_family import MfluxFamilyAdapter, _PromptCache
from dmlx.registry import _parse

SPEC = yaml.safe_load("""
id: fake-family
name: Fake family
adapter: mflux.fake_family
repo: x/fake
license: apache-2.0
tasks: [t2i]
components:
  text_encoder: { phase: encode }
  transformer:  { phase: denoise }
  vae:          { phase: decode, keep: true }
defaults: { steps: 2, guidance: 1.0, width: 64, height: 64 }
""")


class Part(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.marker = 1  # read before the loop, like MOSS's time_signal or FLUX.3's flow_map

    def __call__(self, x):
        return x


class Init:
    @staticmethod
    def _init_config(model, model_config):
        from mflux.callbacks.callback_registry import CallbackRegistry

        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None

    @staticmethod
    def _init_models(model, model_config):
        model.vae = Part("vae")
        model.transformer = Part("transformer")
        model.text_encoder = Part("text_encoder")


class Weights:
    @staticmethod
    def get_components():
        return [SimpleNamespace(name=n, hf_subdir=n) for n in ("vae", "transformer", "text_encoder")]

    @staticmethod
    def get_tokenizers():
        return []


class Pipe(nn.Module):
    def generate_image(self, seed, prompt, num_inference_steps, width, height, guidance=1.0):
        config = SimpleNamespace(num_inference_steps=num_inference_steps)
        if prompt not in self.prompt_cache:
            self.prompt_cache[prompt] = self.text_encoder(mx.array([1.0]))
        _ = self.transformer.marker  # before the loop starts
        for cb in self.callbacks.before_loop_callbacks():
            cb.call_before_loop(seed=seed, prompt=prompt, latents=None, config=config)
        for t in range(num_inference_steps):
            self.transformer(mx.array([0.0]))
        for cb in self.callbacks.after_loop_callbacks():
            cb.call_after_loop(seed=seed, prompt=prompt, latents=None, config=config)
        self.vae(mx.array([0.0]))
        from PIL import Image

        return SimpleNamespace(image=Image.new("RGB", (width, height)))


class Adapter(MfluxFamilyAdapter):
    family = {"initializer": Init, "weights": Weights, "pipelines": {"t2i": Pipe}, "config": "ernie_image_turbo"}

    def load_component(self, name):  # no weights: build the part only
        self.loaded.append(name)
        self._install(name, self._construct(name))
        self._streamed[name] = False
        return 0


class Hooks:
    def __init__(self, adapter):
        self.adapter, self.events = adapter, []

    def before_phase(self, phase):
        self.events.append(f"+{phase}")
        part = {"encode": "text_encoder", "denoise": "transformer", "decode": "vae", "preprocess": "vae",
                "layout": "transformer"}[phase]
        if not self.adapter.is_loaded(part):
            self.adapter.load_component(part)

    def after_phase(self, phase):
        self.events.append(f"-{phase}")
        if phase == "encode":
            self.adapter.free_component("text_encoder")

    def on_step(self, step, total):
        pass

    def check(self):
        pass


def _adapter(tmp_path):
    Adapter.loaded = []
    adapter = Adapter(_parse(SPEC), tmp_path)
    adapter.loaded = []
    return adapter


def _request(prompt="a fox"):
    return GenRequest(task="t2i", prompt=prompt, width=64, height=64, steps=2, seed=1)


def test_parts_load_in_their_phase_even_when_read_before_the_loop(tmp_path):
    adapter = _adapter(tmp_path)
    hooks = Hooks(adapter)

    adapter.generate(_request(), hooks)

    # The early transformer read loads it (layout) and frees the text encoder first; the loop enters denoise.
    assert hooks.events[:3] == ["+encode", "-encode", "+layout"]
    assert hooks.events.count("+denoise") == 1
    assert "+decode" in hooks.events
    assert adapter.loaded == ["text_encoder", "transformer", "vae"]
    assert not adapter.is_loaded("text_encoder")


def test_a_cached_prompt_never_loads_the_text_encoder(tmp_path):
    adapter = _adapter(tmp_path)
    adapter.generate(_request(), Hooks(adapter))
    adapter.loaded.clear()

    hooks = Hooks(adapter)
    adapter.generate(_request(), hooks)

    assert "+encode" not in hooks.events
    assert "text_encoder" not in adapter.loaded
    assert adapter.has_embedding("a fox") and not adapter.has_embedding("a cat")
    assert adapter.embed_cache_stats()["hits"] >= 1


def test_without_a_guard_parts_load_on_use(tmp_path):
    adapter = _adapter(tmp_path)
    adapter.generate(_request(), SimpleNamespace(
        before_phase=lambda p: None, after_phase=lambda p: None, on_step=lambda *a: None, check=lambda: None))
    assert set(adapter.loaded) == {"text_encoder", "transformer", "vae"}


def test_the_prompt_cache_is_bounded():
    cache = _PromptCache(capacity=2)
    for key in ("a", "b", "c"):
        cache[key] = key
    assert list(cache) == ["b", "c"]


def test_find_stacks_finds_nested_block_lists():
    from dmlx.adapters.streaming import find_stacks

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Part(str(i)) for i in range(5)]
            self.norm = Part("norm")

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.blocks = [Part(str(i)) for i in range(4)]
            self.pair = [Part("a"), Part("b")]  # too short to be a stack

    assert sorted(find_stacks(Outer())) == ["blocks", "model.layers"]


def test_stacks_from_the_weight_index():
    from dmlx.adapters.streaming import stacks_from_index

    class Index:
        names = [f"language_model.model.layers.{i}.mlp.weight" for i in range(6)]
        names += [f"layers.{i}.attn.weight" for i in range(5)]
        names += ["pair.0.weight", "pair.1.weight", "norm.weight"]

        def keys(self, prefix=""):
            return [n for n in self.names if n.startswith(prefix)]

    assert stacks_from_index(Index()) == ["language_model.model.layers", "layers"]


def test_cached_encodings_are_evaluated_when_encoding_ends(tmp_path):
    adapter = _adapter(tmp_path)
    settled = []
    original = adapter._settle_encodings
    adapter._settle_encodings = lambda: (settled.append(len(adapter.prompt_cache)), original())
    hooks = Hooks(adapter)

    adapter.generate(_request(), hooks)

    # Settled before the guard frees the text encoder, with the new encoding already in the cache.
    assert settled and settled[0] == 1


class PositionalEncodePipe(Pipe):
    """A pipeline with its own encode method that takes the prompt positionally (Boogu)."""

    def _encode_instruction(self, prompt):
        return self.text_encoder(mx.array([float(len(prompt))]))

    def generate_image(self, seed, prompt, num_inference_steps, width, height):
        self.encoded = self._encode_instruction(prompt)
        return super().generate_image(seed, prompt, num_inference_steps, width, height)


class PositionalEncodeAdapter(Adapter):
    family = {**Adapter.family, "pipelines": {"t2i": PositionalEncodePipe}, "encode": "_encode_instruction"}


def test_an_encode_method_with_a_positional_prompt_is_cached(tmp_path):
    PositionalEncodeAdapter.loaded = []
    adapter = PositionalEncodeAdapter(_parse(SPEC), tmp_path)
    adapter.generate(_request(), Hooks(adapter))
    adapter.loaded.clear()

    hooks = Hooks(adapter)
    adapter.generate(_request(), hooks)

    assert "+encode" not in hooks.events
    assert "text_encoder" not in adapter.loaded
    assert adapter.embed_cache_stats()["hits"] >= 1


def test_a_streamed_block_keeps_its_block_type():
    from dmlx.adapters.streaming import StreamedBlock, stand_in

    class DualBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.width = 3

    block = DualBlock()
    proxy = stand_in(block, SimpleNamespace(keys=lambda prefix: []), "blocks.0.", None, None)
    # Iris picks each block's call by type.
    assert isinstance(proxy, DualBlock) and isinstance(proxy, StreamedBlock)
    assert proxy.width == 3 and "width" not in proxy.parameters()
    assert type(stand_in(DualBlock(), SimpleNamespace(keys=lambda p: []), "b.1.", None, None)) is type(proxy)


def test_a_family_minimum_size_generates_larger_and_scales_down(tmp_path):
    from dmlx.adapters.mflux_family import _at_least

    assert _at_least(512, 512, 640) == (640, 640)
    assert _at_least(512, 768, 640) == (640, 960)
    assert _at_least(1024, 1024, 640) == (1024, 1024)
    assert _at_least(512, 512, None) == (512, 512)

    class MinSideAdapter(Adapter):
        family = {**Adapter.family, "min_side": 128}

    MinSideAdapter.loaded = []
    adapter = MinSideAdapter(_parse(SPEC), tmp_path)
    generated = []
    inner = adapter._generate
    adapter._generate = lambda req, w, h: generated.append((w, h)) or inner(req, w, h)
    image = adapter.generate(_request(), Hooks(adapter))  # asks for 64 x 64
    assert generated == [(128, 128)] and image.size == (64, 64)


def test_a_family_can_set_the_tiled_decode_tile_size(tmp_path):
    # Krea 2's VAE needs 6.1 GB per 512 px tile; 256 px tiles need 3.3 GB and look the same.
    class SmallTiles(Adapter):
        family = {**Adapter.family, "decode_tile_size": 256}

    SmallTiles.loaded = []
    adapter = SmallTiles(_parse(SPEC), tmp_path)
    adapter.set_decode_mode("tiled")
    assert adapter.pipe.tiling_config.vae_decode_tile_size == 256
    adapter.set_decode_mode("full")
    assert adapter.pipe.tiling_config is None
    plain = _adapter(tmp_path)
    plain.set_decode_mode("tiled")
    assert plain.pipe.tiling_config.vae_decode_tile_size == 512


TWO_TRANSFORMERS = yaml.safe_load("""
id: two
name: Two transformers
adapter: mflux.fake_family
repo: x/two
license: apache-2.0
tasks: [t2i]
components:
  text_encoder:              { phase: encode, size_gb: 4 }
  conditional_transformer:   { phase: denoise, size_gb: 3 }
  unconditional_transformer: { phase: denoise, size_gb: 3 }
  vae:                       { phase: decode, keep: true, size_gb: 0.2 }
defaults: { steps: 20, guidance: 7.0, width: 64, height: 64 }
""")


def test_both_transformers_count_together_in_the_low_memory_plans():
    # Ideogram 4 runs its conditional and unconditional transformers in the same loop.
    from dmlx.memory import GB, MemoryManager

    spec = _parse(TWO_TRANSFORMERS)
    assert spec.parts("denoise") == ["conditional_transformer", "unconditional_transformer"]
    manager = MemoryManager()
    try:
        needs = manager._needs(spec, 0)
    finally:
        manager.monitor.stop()
    assert round(needs["phase_free"] / GB, 1) == 6.2  # both transformers (6) and the VAE, not the 4 GB encoder
    assert round(needs["sequential"] / GB, 1) == 6.2
    assert round(needs["eager"] / GB, 1) == 10.2


def test_a_preset_family_passes_the_nearest_preset_instead_of_steps(tmp_path):
    seen = {}

    class PresetPipe(Pipe):
        def generate_image(self, seed, prompt, num_inference_steps, width, height, guidance=None, preset=None):
            seen.update(steps=num_inference_steps, guidance=guidance, preset=preset)
            return super().generate_image(seed, prompt, 2, width, height)

    class PresetAdapter(Adapter):
        family = {**Adapter.family, "pipelines": {"t2i": PresetPipe},
                  "presets": {12: "TURBO_12", 20: "DEFAULT_20", 48: "QUALITY_48"}}

    PresetAdapter.loaded = []
    adapter = PresetAdapter(_parse(SPEC), tmp_path)
    request = _request()
    request.steps, request.guidance = 30, 7.0
    adapter.generate(request, Hooks(adapter))
    assert seen == {"steps": None, "guidance": None, "preset": "DEFAULT_20"}


def test_a_family_hook_runs_before_a_parts_weights_go_in(tmp_path):
    from dmlx.adapters.mflux_adapter import _MfluxAdapterBase

    calls = []

    class HookAdapter(Adapter):
        family = {**Adapter.family, "before_apply": {"transformer": lambda module, weights: calls.append(module.name)}}

    HookAdapter.loaded = []
    adapter = HookAdapter(_parse(SPEC), tmp_path)
    adapter._before_apply("transformer", adapter._construct("transformer"), None)
    adapter._before_apply("vae", adapter._construct("vae"), None)
    assert calls == ["transformer"]
    assert _MfluxAdapterBase._before_apply(adapter, "vae", None, None) is None  # the base does nothing
