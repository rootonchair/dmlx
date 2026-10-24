import pytest

from dmlx import catalog
from dmlx.registry import Registry

FILES = {
    "text_encoder/0.safetensors": 2_000_000_000,
    "transformer/0.safetensors": 2_000_000_000,
    "transformer/model.safetensors.index.json": 1000,
    "vae/0.safetensors": 160_000_000,
    "tokenizer/tokenizer.json": 1000,
}


def _names(double=0, single=0, refiners=0, layers=0):
    names = [f"transformer_blocks.{i}.attn.to_q.weight" for i in range(double)]
    names += [f"single_transformer_blocks.{i}.attn.to_qkv_mlp_proj.weight" for i in range(single)]
    names += [f"noise_refiner.{i}.w" for i in range(refiners)] + [f"context_refiner.{i}.w" for i in range(refiners)]
    names += [f"layers.{i}.w" for i in range(layers)]
    return names


@pytest.fixture
def remote(monkeypatch):
    def set_(names, meta=None):
        meta = {"mflux_version": "0.19.0", "quantization_level": "4"} if meta is None else meta
        monkeypatch.setattr(catalog, "_tensor_names", lambda repo, files, component: (names, meta))
    return set_


@pytest.mark.parametrize("repo,double,config,name,steps", [
    ("mflux-community/flux2-klein-4b-mflux-q4", 5, "flux2_klein_4b", "FLUX.2 Klein 4B (4-bit)", 4),
    ("mflux-community/flux2-klein-9b-mflux-q4", 8, "flux2_klein_9b", "FLUX.2 Klein 9B (4-bit)", 4),
    ("mflux-community/flux2-klein-9b-kv-mflux-q4", 8, "flux2_klein_9b_kv", "FLUX.2 Klein 9B KV (4-bit)", 4),
    ("mflux-community/flux-2-klein-base-4b-mflux-q4", 5, "flux2_klein_base_4b", "FLUX.2 Klein Base 4B (4-bit)", 50),
])
def test_detects_flux2_klein_variants(remote, repo, double, config, name, steps):
    remote(_names(double=double, single=20))
    found = catalog.detect(repo, FILES, "apache-2.0")
    assert (found.family, found.mflux_config, found.name) == ("flux2-klein", config, name)
    assert found.tasks == ["t2i", "edit"] and found.defaults["steps"] == steps
    assert found.permissive_license and found.size_bytes == 4_160_000_000


def test_detects_z_image_and_flags_restrictive_license(remote):
    remote(_names(refiners=2, layers=30))
    turbo = catalog.detect("mflux-community/z-image-turbo-mflux-q6", FILES, "apache-2.0")
    assert (turbo.mflux_config, turbo.tasks, turbo.defaults["guidance"]) == ("z_image_turbo", ["t2i"], 0.0)
    base = catalog.detect("someone/z-image-base-mflux-q4", FILES, "flux-non-commercial-license")
    assert base.mflux_config == "z_image" and base.name == "Z-Image Base (4-bit)" and not base.permissive_license


def test_detects_qwen_image_21_base_as_explicit_research_model(remote):
    files = {
        **{key: value for key, value in FILES.items() if not key.startswith("tokenizer/")},
        "processor/preprocessor_config.json": 1000,
    }
    remote(_names(double=60), {"mflux_version": "0.23.0", "quantization_level": "4"})
    found = catalog.detect("someone/Qwen-Image-2.1-mflux-4bit", files, "qwen-research")
    spec = found.to_spec_data()

    assert (found.family, found.mflux_config) == ("qwen-image-2.1", "qwen_image_21")
    assert found.tasks == ["t2i", "edit"] and found.defaults["steps"] == 40
    assert spec["auto_select"] is False and spec["min_ram_gb"] == 16
    assert spec["experimental"] is True
    assert spec["min_steps"] == 2
    assert spec["max_references"] == 10
    assert spec["components"]["text_encoder"]["stream"] == []
    assert spec["components"]["text_encoder"]["external"] is True
    assert spec["components"]["transformer"]["stream"] == []
    assert spec["reference_activation_factor"] == 2.2
    assert spec["capabilities"]["image.edit"]["inputs"]["image"]["ordered"] is True
    assert spec["capabilities"]["image.edit"]["prompt"]["negative"] is True


def test_refuses_qwen_image_21_turbo(remote):
    remote(_names(double=60), {"mflux_version": "0.23.0", "quantization_level": "4"})
    with pytest.raises(catalog.InspectError, match="Turbo is not supported"):
        catalog.detect("someone/Qwen-Image-2.1-Turbo-mflux-4bit", FILES, "qwen-research")


@pytest.mark.parametrize("repo,files,names,meta,message", [
    ("x/z-image-turbo-controlnet-q4", FILES, [], None, "add-ons"),
    ("x/model", {"model.safetensors": 1}, [], None, "different model"),
    ("x/model", {k: v for k, v in FILES.items() if not k.startswith("vae/")}, [], None, "missing its vae"),
    ("x/flux2-klein-4b", FILES, _names(double=5, single=20), {}, "not in mflux's format"),
    ("x/model", FILES, _names(layers=10), None, "different model"),
    ("x/flux2-klein-7b", FILES, _names(double=6, single=20), None, "4B or 9B"),
])
def test_refuses_what_it_cannot_run(remote, repo, files, names, meta, message):
    remote(names, meta)
    with pytest.raises(catalog.InspectError, match=message):
        catalog.detect(repo, files, None)


@pytest.mark.parametrize("text,repo", [
    ("mflux-community/flux2-klein-9b-mflux-q4", "mflux-community/flux2-klein-9b-mflux-q4"),
    ("https://huggingface.co/mflux-community/flux2-klein-9b-mflux-q4/tree/main", "mflux-community/flux2-klein-9b-mflux-q4"),
    ("  owner/name  ", "owner/name"),
])
def test_normalize_repo(text, repo):
    assert catalog.normalize_repo(text) == repo


def test_normalize_rejects_garbage():
    with pytest.raises(catalog.InspectError):
        catalog.normalize_repo("just words")


def test_saved_specs_load_and_bad_files_are_skipped(tmp_path, monkeypatch, remote):
    monkeypatch.setenv("DMLX_HOME", str(tmp_path))
    remote(_names(double=8, single=24))
    spec = catalog.save_spec(catalog.detect("mflux-community/flux2-klein-9b-mflux-q4", FILES, "other"))
    (tmp_path / "models" / "broken.yaml").write_text("id: [unclosed")
    registry = Registry.load_builtin(extra_dir=tmp_path / "models")
    loaded = registry.get(spec.id)
    assert loaded.user_added and loaded.mflux_config == "flux2_klein_9b"
    assert loaded.components["transformer"].stream == ["transformer_blocks", "single_transformer_blocks"]
    assert "flux2-klein-4b" in registry  # built-ins still load alongside


@pytest.mark.parametrize("repo,expected", [
    ("mflux-community/flux-2-klein-base-9b-mflux-q4", "FLUX.2 Klein Base 9B · 4-bit"),
    ("mflux-community/flux2-klein-9b-kv-mflux-q8", "FLUX.2 Klein 9B KV · 8-bit"),
    ("mflux-community/z-image-turbo-mflux-bf16", "Z-Image Turbo · full precision"),
    ("mflux-community/z-image-base-mflux-q3", "Z-Image Base · 3-bit"),
])
def test_suggestion_labels(repo, expected):
    assert catalog.label(repo) == expected
