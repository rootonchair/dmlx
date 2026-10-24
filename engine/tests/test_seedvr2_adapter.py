from dmlx.capabilities import CapabilitySpec, InputConstraint
from dmlx.adapters.base import GenRequest
from dmlx.adapters import mflux_adapter
from dmlx.registry import ComponentSpec, ModelSpec


def test_seedvr2_adapter_builds_empty_without_loading_weights(tmp_path):
    from dmlx.adapters.mflux_adapter import SeedVR2Adapter

    spec = ModelSpec(
        id="seedvr2-test", name="SeedVR2 Test", adapter="mflux.seedvr2", repo="local/seedvr2",
        license="apache-2.0", tasks=["upscale"],
        components={
            "transformer": ComponentSpec("transformer", "denoise"),
            "vae": ComponentSpec("vae", "decode"),
        },
        defaults={}, mflux_config="seedvr2_3b",
        capabilities={
            "image.upscale": CapabilitySpec(
                "image.upscale", {"image": InputConstraint(1, 1, True)}, positive_prompt=False)
        },
    )
    adapter = SeedVR2Adapter(spec, tmp_path)
    assert adapter.component_names() == ["transformer", "vae"]
    assert not adapter.is_loaded("transformer")
    assert not adapter.is_loaded("vae")


def test_seedvr2_announces_source_preprocess_before_generation(tmp_path, monkeypatch):
    from dmlx.adapters.mflux_adapter import SeedVR2Adapter

    spec = ModelSpec(
        id="seedvr2-test", name="SeedVR2 Test", adapter="mflux.seedvr2", repo="local/seedvr2",
        license="apache-2.0", tasks=["upscale"],
        components={
            "transformer": ComponentSpec("transformer", "denoise"),
            "vae": ComponentSpec("vae", "decode"),
        }, defaults={}, mflux_config="seedvr2_3b",
        capabilities={"image.upscale": CapabilitySpec(
            "image.upscale", {"image": InputConstraint(1, 1, True)}, positive_prompt=False)},
    )
    adapter = SeedVR2Adapter(spec, tmp_path)
    phases = []

    class Hooks:
        def before_phase(self, phase): phases.append(phase)

    sentinel = object()
    monkeypatch.setattr(mflux_adapter._MfluxAdapterBase, "generate", lambda *_args: sentinel)
    result = adapter.generate(GenRequest("upscale", "upscale", 512, 512, 1, 1), Hooks())
    assert result is sentinel
    assert phases == ["preprocess"]
