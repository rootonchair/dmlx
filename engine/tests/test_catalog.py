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


@pytest.mark.parametrize("repo", ["someone/Qwen-Image-2.1-mflux-4bit", "mflux-community/qwen-image-2-1-mflux-q4"])
def test_detects_qwen_image_21_under_both_names(remote, repo):
    remote(_names(double=60), {"mflux_version": "0.23.0", "quantization_level": "4"})
    assert catalog.detect(repo, FILES, "qwen-research").family == "qwen-image-2.1"


def test_suggestions_keep_supported_families_only(monkeypatch):
    from types import SimpleNamespace

    import huggingface_hub

    repos = [
        "mflux-community/flux2-klein-4b-mflux-q4",
        "mflux-community/qwen-image-2-1-mflux-q4",
        "mflux-community/qwen-image-2-1-turbo-mflux-q4",
        "mflux-community/flux-3-image-klein-mflux-q4",
        "mflux-community/qwen-image-2-1-controlnet-union-mflux-q4",
        "mflux-community/krea-2-turbo-mflux-q4",
        "mflux-community/ideogram-4-mflux-q4",
    ]

    class Api:
        def list_models(self, **_):
            return [SimpleNamespace(id=r, downloads=1) for r in repos]

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(catalog, "_suggestions", None)
    assert [s["repo"] for s in catalog.suggestions()] == [repos[0], repos[1], repos[3], repos[5]]
    monkeypatch.setattr(catalog, "_suggestions", None)


GENERIC_FILES = {
    "transformer/0.safetensors": 3_000_000_000,
    "text_encoder/0.safetensors": 2_000_000_000,
    "text_encoder/tokenizer.json": 1000,
    "vae/0.safetensors": 160_000_000,
}


def test_detects_a_generic_family_from_its_weight_definition(remote):
    remote([], {"mflux_version": "0.23.0", "quantization_level": "4"})
    found = catalog.detect("mflux-community/flux-3-image-klein-mflux-q4", GENERIC_FILES, "flux-3-image-klein-license")
    spec = found.to_spec_data()

    assert (found.family, found.mflux_config, found.name) == ("flux3-klein", "flux3_image_klein", "FLUX.3 Klein (4-bit)")
    assert found.tasks == ["t2i", "edit"] and not found.permissive_license
    assert spec["adapter"] == "mflux.flux3_klein" and spec["agent_note"]
    assert spec["components"]["transformer"]["stream"] == ["auto"]


def test_a_part_in_another_folder_keeps_its_folder_in_the_spec(remote, tmp_path):
    from dmlx import downloads, registry

    remote([], {"mflux_version": "0.23.0", "quantization_level": "4"})
    files = {"transformer/0.safetensors": 6_000_000_000, "vae/0.safetensors": 160_000_000,
             "mllm/0.safetensors": 4_700_000_000, "mllm/tokenizer.json": 1000}
    found = catalog.detect("mflux-community/boogu-image-turbo-mflux-q4", files, "apache-2.0")
    data = found.to_spec_data()

    assert found.name == "Boogu Image Turbo (4-bit)"
    assert data["components"]["text_encoder"]["subdir"] == "mllm"
    assert "subdir" not in data["components"]["transformer"]
    spec = registry._parse(data)
    assert spec.components["text_encoder"].folder == "mllm"
    import mlx.core as mx

    for name in files:
        (tmp_path / name).parent.mkdir(exist_ok=True)
        if name.endswith(".safetensors"):
            mx.save_safetensors(str(tmp_path / name), {"layers.0.w": mx.zeros((4,))})
    downloads.fill_component_sizes(spec, tmp_path)
    assert spec.components["text_encoder"].size == (tmp_path / "mllm/0.safetensors").stat().st_size


def test_a_transformer_at_the_repo_root_is_found(monkeypatch):
    # Krea 2 keeps its transformer's shards at the repo root.
    read = []
    monkeypatch.setattr(catalog, "_remote_header", lambda repo, name: read.append(name) or {
        "__metadata__": {"mflux_version": "0.24.0", "quantization_level": "4"}})
    monkeypatch.setattr(catalog, "_get", lambda repo, name, byte_range=None: b'{"weight_map": {}}')
    files = {"0.safetensors": 4_000_000_000, "model.safetensors.index.json": 100,
             "text_encoder/0.safetensors": 8_000_000_000, "tokenizer/tokenizer.json": 1000,
             "vae/0.safetensors": 250_000_000}
    found = catalog.detect("mflux-community/krea-2-turbo-mflux-q4", files, "krea-2-community-license")

    assert found.name == "Krea 2 Turbo (4-bit)" and not found.permissive_license
    assert read == ["0.safetensors"]
    assert found.to_spec_data()["components"]["transformer"]["subdir"] == ""


def test_a_generic_family_needs_every_part_and_its_tokenizer(remote):
    remote([], {"mflux_version": "0.23.0", "quantization_level": "4"})
    files = {k: v for k, v in GENERIC_FILES.items() if k != "vae/0.safetensors"}
    with pytest.raises(catalog.InspectError, match="missing its vae"):
        catalog.detect("mflux-community/flux-3-image-klein-mflux-q4", files, None)
    # ERNIE keeps its tokenizer in tokenizer/, FLUX.3 in text_encoder/.
    with pytest.raises(catalog.InspectError, match="missing its tokenizer"):
        catalog.detect("mflux-community/ernie-image-turbo-mflux-q4", GENERIC_FILES, None)


def test_unsupported_versions_of_a_generic_family_are_named():
    with pytest.raises(catalog.InspectError, match="ERNIE Image Turbo"):
        catalog.detect("mflux-community/ernie-image-base-mflux-q4", GENERIC_FILES, None)
    assert catalog.label("mflux-community/flux-3-image-klein-mflux-q8") == "FLUX.3 Klein · 8-bit"
