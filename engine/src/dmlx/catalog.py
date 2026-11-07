"""Adding models from Hugging Face: inspect a repo, build a model spec, suggest compatible repos.

dMLX can run any repo of a model family it has an adapter for, saved in
mflux's format (pre-quantized packs such as those from mflux-community). Inspection reads only
the repo's file list, each component's weight index and one shard header (a ranged request of a
few kilobytes), so nothing is downloaded before the user confirms.
"""

from __future__ import annotations

import json
import re
import struct
import time
from dataclasses import dataclass, field

import yaml

from dmlx import paths
from dmlx.registry import ModelSpec, _parse

COMPONENTS = ("text_encoder", "transformer", "vae")
_PERMISSIVE = {"apache-2.0", "mit", "bsd-3-clause", "openrail", "creativeml-openrail-m"}

# Per family: what the adapter needs, and measured working memory (docs/benchmarks/b1.md, b2.md).
FAMILIES = {
    "flux2-klein": {
        "adapter": "mflux.flux2_klein",
        "tasks": ["t2i", "edit"],
        "max_references": 3,
        "stream": {"text_encoder": ["layers"], "transformer": ["transformer_blocks", "single_transformer_blocks"]},
        "decode_gb_per_mp": 8.6,
        "decode_tiled_gb_per_mp": 2.4,
        "decode_banded_gb_per_mp": 1.2,
    },
    "z-image": {
        "adapter": "mflux.z_image",
        "tasks": ["t2i"],
        "max_references": 0,
        "stream": {"text_encoder": ["layers"], "transformer": ["noise_refiner", "context_refiner", "layers"]},
        "decode_gb_per_mp": 6.7,
        "decode_tiled_gb_per_mp": 2.0,
    },
    "qwen-image-2.1": {
        "adapter": "mflux.qwen_image_21",
        "tasks": ["t2i", "edit"],
        "max_references": 10,
        "stream": {
            "text_encoder": [],
            "transformer": [],
        },
        "decode_gb_per_mp": 8.0,
        "decode_tiled_gb_per_mp": 1.3,
        "reference_activation_factor": 2.2,
        "min_ram_gb": 16,
        "min_steps": 2,
        "auto_select": False,
        "experimental": True,
        "negative_prompt": True,
        "small_text_encoder": True,
    },
    # Families the generic adapter runs from mflux's weight definitions (task 15). They are found by
    # repo name ("match"), then checked against the weight definition's parts and tokenizers.
    # Memory numbers: `dmlx bench-family` on an M5 (sequential, 10% margin), docs/benchmarks/families.md.
    "ernie-image": {
        "adapter": "mflux.ernie_image",
        "tasks": ["t2i"],
        "max_references": 0,
        "match": ("ernie-image",),
        "variants": [{"when": "turbo", "config": "ernie_image_turbo", "name": "ERNIE Image Turbo",
                      "defaults": {"steps": 8, "guidance": 1.0}}],
        "stream": {"text_encoder": ["auto"], "transformer": ["auto"]},
        # Measured 2026-11-02: denoise +1.35 GB, decode +1.31 GB at 1024 px (mflux's block decoder
        # is always low-memory, so full and tiled are the same).
        "activation_gb_per_mp": 1.5,
        "decode_gb_per_mp": 1.4,
        "decode_tiled_gb_per_mp": 1.4,
        "agent_note": "New images only; slow (about 2.5 minutes at 1024 px); "
        "an 8B model by Baidu, good at following long prompts.",
    },
    "moss-image": {
        "adapter": "mflux.moss_image",
        "tasks": ["t2i"],
        "max_references": 0,
        "match": ("moss-image",),
        "variants": [{"when": "sft", "config": "moss_image_sft", "name": "MOSS Image",
                      "defaults": {"steps": 50, "guidance": 3.0}}],
        "stream": {"text_encoder": ["auto"], "transformer": ["auto"]},
        # Measured 2026-11-02 at 1024 px: denoise +0.86 GB, decode +7.04 GB full, +2.02 GB tiled.
        "activation_gb_per_mp": 1.0,
        "decode_gb_per_mp": 7.4,
        "decode_tiled_gb_per_mp": 2.3,
        "agent_note": "New images only; very slow (about 13 minutes at 1024 px, 3 minutes at 512 px); "
        "use only when the user asks for it.",
    },
    "flux3-klein": {
        "adapter": "mflux.flux3_klein",
        "tasks": ["t2i", "edit"],
        "max_references": 3,
        "match": ("flux-3-image-klein", "flux3-image-klein", "flux3-klein"),
        "variants": [{"unless": "base", "config": "flux3_image_klein", "name": "FLUX.3 Klein",
                      "defaults": {"steps": 16, "guidance": 1.0}}],
        "stream": {"text_encoder": ["auto"], "transformer": ["auto"]},
        # Measured 2026-11-02 at 1024 px: denoise +1.06 GB (+1.52 GB with one reference), decode +1.31 GB.
        "activation_gb_per_mp": 1.2,
        "reference_activation_factor": 0.5,
        "decode_gb_per_mp": 1.4,
        "decode_tiled_gb_per_mp": 1.4,
        "agent_note": "New images and edits with up to 3 images; about 2 minutes per 1024 px image "
        "(4 for an edit), slower than FLUX.2 Klein.",
    },    "boogu-image": {
        "adapter": "mflux.boogu",
        "tasks": ["t2i"],
        "max_references": 0,
        "match": ("boogu-image",),
        "variants": [{"when": "turbo", "config": "boogu_image_turbo", "name": "Boogu Image Turbo",
                      "defaults": {"steps": 8, "guidance": 1.0}}],
        "stream": {"text_encoder": ["auto"], "transformer": ["auto"]},
        # Measured 2026-11-06 at 1024 px: denoise +3.05 GB, decode +8.60 GB full, +2.98 GB tiled
        # (the FLUX.1 VAE's tiles are a fixed size, so tiled decode is the same at 512 px).
        "activation_gb_per_mp": 3.3,
        "decode_gb_per_mp": 9.1,
        "decode_tiled_gb_per_mp": 3.3,
        "agent_note": "New images only; slow (about 3.5 minutes at 1024 px, 45 seconds at 512 px); "
        "photographic, good at text in English and Chinese.",
    },
    "krea2": {
        "adapter": "mflux.krea2",
        "tasks": ["t2i"],
        "max_references": 0,
        "match": ("krea-2-",),
        "variants": [{"when": "turbo", "config": "krea2", "name": "Krea 2 Turbo",
                      "defaults": {"steps": 8, "guidance": 1.0}}],
        "stream": {"text_encoder": ["auto"], "transformer": ["auto"]},
        # Measured 2026-11-06 at 1024 px: denoise +1.67 GB, decode +9.73 GB full. Its VAE (Qwen Image's)
        # needs 6.1 GB per 512 px tile, so the adapter decodes in 256 px tiles: +3.32 GB.
        "activation_gb_per_mp": 1.8,
        "decode_gb_per_mp": 10.3,
        "decode_tiled_gb_per_mp": 3.5,
        "agent_note": "New images only; about 2.5 minutes at 1024 px; strong photographic realism.",
    },
    "iris": {
        "adapter": "mflux.iris",
        "tasks": ["t2i"],
        "max_references": 0,
        "match": ("iris-3b",),
        "variants": [{"when": "turbo", "config": "iris_3b", "name": "Iris 3B Turbo",
                      "defaults": {"steps": 4, "guidance": 1.0}}],
        "stream": {"text_encoder": ["auto"], "transformer": ["auto"]},
        # Measured 2026-11-06 at 1024 px: denoise +1.01 to +1.25 GB. No VAE: the transformer outputs
        # pixels, so the decode phase only unpacks them (+0.02 GB).
        "activation_gb_per_mp": 1.4,
        "decode_gb_per_mp": 0.1,
        "decode_tiled_gb_per_mp": 0.1,
        "agent_note": "New images only; fast (about 20 seconds at 1024 px); a small 3B model, "
        "simpler detail than the larger models.",
    },
}


_OTHER_MODEL = (
    "dMLX can run FLUX.2 Klein, FLUX.3 Klein, Z-Image, Qwen Image 2.1 Base, ERNIE Image Turbo and MOSS Image "
    "models; this repo is a different model."
)


class InspectError(Exception):
    """A repo dMLX cannot run; the message is shown to the user as is."""


@dataclass
class Inspection:
    repo: str
    id: str
    name: str
    family: str
    mflux_config: str
    tasks: list[str]
    license: str
    permissive_license: bool
    quantization: int | None
    size_bytes: int
    defaults: dict
    activation_gb_per_mp: float
    subdirs: dict[str, str] = field(default_factory=dict)  # part -> folder, where they differ
    parts: tuple[str, ...] = COMPONENTS  # the parts the weight definition has (Iris: no VAE)

    def as_dict(self) -> dict:
        return {
            "repo": self.repo,
            "id": self.id,
            "name": self.name,
            "family": self.family,
            "tasks": self.tasks,
            "license": self.license,
            "permissive_license": self.permissive_license,
            "quantization": self.quantization,
            "size_gb": round(self.size_bytes / 1024**3, 2),
            "url": f"https://huggingface.co/{self.repo}",
        }

    def to_spec_data(self) -> dict:
        family = FAMILIES[self.family]
        data = {
            "id": self.id,
            "name": self.name,
            "adapter": family["adapter"],
            "mflux_config": self.mflux_config,
            "repo": self.repo,
            "quantization": self.quantization,
            "license": self.license,
            "tasks": self.tasks,
            "max_references": family["max_references"],
            "min_ram_gb": family.get("min_ram_gb", 16),
            "min_steps": family.get("min_steps", 1),
            "auto_select": family.get("auto_select", True),
            "experimental": family.get("experimental", False),
            "user_added": True,
            # Shown as the size before the model is downloaded.
            "download_gb": round(self.size_bytes / 1024**3, 2),
            "components": {
                "text_encoder": {
                    "phase": "encode",
                    "stream": family["stream"]["text_encoder"],
                    **({"external": True, "size_gb": 2.4} if family.get("small_text_encoder") else {}),
                },
                "transformer": {"phase": "denoise", "stream": family["stream"]["transformer"]},
                "vae": {"phase": "decode", "keep": True},
            },
            "defaults": self.defaults,
            "activation_gb_per_mp": self.activation_gb_per_mp,
            "reference_activation_factor": family.get("reference_activation_factor", 0.75),
            **{k: family[k] for k in ("decode_gb_per_mp", "decode_tiled_gb_per_mp", "decode_banded_gb_per_mp") if k in family},
        }
        data["components"] = {n: c for n, c in data["components"].items() if n in self.parts}
        for name, subdir in self.subdirs.items():
            data["components"][name]["subdir"] = subdir
        if family.get("agent_note"):
            data["agent_note"] = family["agent_note"]
        if family.get("negative_prompt"):
            data["capabilities"] = {
                "image.generate": {"inputs": {}, "prompt": {"positive": True, "negative": True}},
                "image.edit": {
                    "inputs": {"image": {"min": 1, "max": family["max_references"], "ordered": True}},
                    "prompt": {"positive": True, "negative": True},
                },
            }
        return data


# -- remote reads ---------------------------------------------------------------------------
def _get(repo: str, filename: str, byte_range: tuple[int, int] | None = None) -> bytes:
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers, get_session

    headers = build_hf_headers()
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    response = get_session().get(hf_hub_url(repo, filename), headers=headers, timeout=30)
    response.raise_for_status()
    return response.content


def _remote_header(repo: str, filename: str) -> dict:
    """The JSON header of a remote safetensors file, fetched with two small ranged requests."""
    length = struct.unpack("<Q", _get(repo, filename, (0, 7))[:8])[0]
    if length > 64 * 1024**2:
        raise InspectError(f"{filename} does not look like a safetensors file.")
    return json.loads(_get(repo, filename, (8, 8 + length - 1)))


def _tensor_names(repo: str, files: dict[str, int], component: str) -> tuple[list[str], dict]:
    """Tensor names of a component (from its index, or a shard header) and one shard's metadata."""
    # `component` is the part's folder; "" is the repo root (Krea 2 keeps its transformer there).
    shards = sorted(f for f in _subdir_files(files, component) if f.endswith(".safetensors"))
    if not shards:
        name = component.replace("_", " ") or "transformer"
        raise InspectError(f"This repo has no {name} weights, so it is not a full model.")
    index = f"{component}/model.safetensors.index.json" if component else "model.safetensors.index.json"
    header = _remote_header(repo, shards[0])
    if index in files:
        names = list(json.loads(_get(repo, index)).get("weight_map", {}))
    else:
        names = [k for k in header if k != "__metadata__"]
        for shard in shards[1:]:
            names += [k for k in _remote_header(repo, shard) if k != "__metadata__"]
    return names, header.get("__metadata__") or {}


def _count(names: list[str], stack: str) -> int:
    return len({n.split(".")[1] for n in names if n.startswith(f"{stack}.") and n.split(".")[1].isdigit()})


# -- detection ------------------------------------------------------------------------------
def _slug(repo: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", repo.split("/")[-1].lower()).strip("-")


def _bits_label(quantization: int | None) -> str:
    return f"{quantization}-bit" if quantization else "full precision"


def _is_qwen_21(repo: str) -> bool:
    """Qwen Image 2.1, named either "Qwen-Image-2.1" or "qwen-image-2-1" (mflux-community)."""
    return "qwen-image-2-1" in _slug(repo)


def detect(repo: str, files: dict[str, int], card_license: str | None) -> Inspection:
    lowered = repo.lower()
    if "controlnet" in lowered or "lora" in lowered:
        raise InspectError("ControlNet and LoRA repos are add-ons, not full models; dMLX does not support them yet.")
    if _is_qwen_21(repo) and "turbo" in lowered:
        raise InspectError("Qwen Image 2.1 Turbo is not supported yet; use a Base MFLUX checkpoint.")
    family_id = _generic_family(repo)
    if family_id:
        return _detect_generic(repo, files, card_license, family_id)
    folders = {f.split("/")[0] for f in files if "/" in f and f.endswith(".safetensors")}
    if "transformer" not in folders:
        raise InspectError(_OTHER_MODEL)
    missing = [c.replace("_", " ") for c in ("text_encoder", "vae") if c not in folders]
    if not any(f.startswith(("tokenizer/", "processor/")) for f in files):
        missing.append("tokenizer or processor")
    if missing:
        raise InspectError(f"This repo is missing its {' and '.join(missing)}, so dMLX cannot run it on its own.")
    names, meta = _tensor_names(repo, files, "transformer")
    if not meta.get("mflux_version"):
        raise InspectError(
            "This repo is not in mflux's format. Use an mflux version of the model "
            "(for example one from huggingface.co/mflux-community)."
        )
    level = meta.get("quantization_level")
    quantization = int(level) if level not in (None, "", "None", "null") else None
    double, single = _count(names, "transformer_blocks"), _count(names, "single_transformer_blocks")
    if _is_qwen_21(repo) and double and not single:
        config, family = "qwen_image_21", "qwen-image-2.1"
        name = "Qwen Image 2.1 Base"
        defaults = {"steps": 40, "guidance": 4.0, "width": 1024, "height": 1024}
        activation = 1.3
    elif double and single:
        size_label = {5: "4B", 8: "9B"}.get(double)
        if size_label is None:
            raise InspectError("This looks like a FLUX.2 model, but not a Klein size dMLX knows (4B or 9B).")
        base = "base" in lowered
        kv = "kv" in re.split(r"[^a-z0-9]+", lowered) and not base and size_label == "9B"
        config = f"flux2_klein_{'base_' if base else ''}{size_label.lower()}{'_kv' if kv else ''}"
        family = "flux2-klein"
        name = f"FLUX.2 Klein {'Base ' if base else ''}{size_label}{' KV' if kv else ''}"
        defaults = {"steps": 50 if base else 4, "guidance": 3.5 if base else 1.0, "width": 1024, "height": 1024}
        activation = 1.4 if size_label == "4B" else 2.0
    elif _count(names, "noise_refiner") and _count(names, "context_refiner"):
        turbo = "turbo" in lowered
        config, family = ("z_image_turbo" if turbo else "z_image"), "z-image"
        name = "Z-Image Turbo" if turbo else "Z-Image Base"
        defaults = {"steps": 9 if turbo else 50, "guidance": 0.0 if turbo else 3.5, "width": 1024, "height": 1024}
        activation = 1.0
    else:
        raise InspectError(_OTHER_MODEL)
    size = sum(sz for f, sz in files.items() if f.split("/")[0] in COMPONENTS and f.endswith(".safetensors"))
    license_id = (card_license or "unknown").lower()
    return Inspection(
        repo=repo,
        id=_slug(repo),
        name=f"{name} ({_bits_label(quantization)})",
        family=family,
        mflux_config=config,
        tasks=list(FAMILIES[family]["tasks"]),
        license=license_id,
        permissive_license=license_id in _PERMISSIVE,
        quantization=quantization,
        size_bytes=size,
        defaults=defaults,
        activation_gb_per_mp=activation,
    )


def _generic_family(repo: str) -> str | None:
    slug = _slug(repo)
    return next((fid for fid, fam in FAMILIES.items() if any(m in slug for m in fam.get("match", ()))), None)


def _variant(repo: str, family: dict) -> dict | None:
    slug = _slug(repo)
    for variant in family["variants"]:
        if variant.get("when") and variant["when"] not in slug:
            continue
        if variant.get("unless") and variant["unless"] in slug:
            continue
        return variant
    return None


def _subdir_files(files: dict[str, int], subdir: str) -> list[str]:
    subdir = subdir.strip("/")
    return [f for f in files if (f.startswith(f"{subdir}/") if subdir else "/" not in f)]


def _detect_generic(repo: str, files: dict[str, int], card_license: str | None, family_id: str) -> Inspection:
    """Check a repo against the weight definition of its family: every part and tokenizer must be there."""
    from dmlx.adapters.mflux_family import weight_definition

    family = FAMILIES[family_id]
    variant = _variant(repo, family)
    if variant is None:
        names = ", ".join(v["name"] for v in family["variants"])
        raise InspectError(f"dMLX runs {names} from this family; this version is not supported yet.")
    definition = weight_definition(family["adapter"], variant["config"])
    missing, size, transformer_dir, subdirs = [], 0, None, {}
    for component in definition.get_components():
        if component.hf_subdir.strip("/") != component.name:
            subdirs[component.name] = component.hf_subdir.strip("/")
        shards = [f for f in _subdir_files(files, component.hf_subdir) if f.endswith(".safetensors")]
        if not shards:
            missing.append(component.name.replace("_", " "))
        size += sum(files[f] for f in shards)
        if component.name == "transformer":
            transformer_dir = component.hf_subdir.strip("/")
    for tokenizer in definition.get_tokenizers():
        if not _subdir_files(files, tokenizer.hf_subdir):
            missing.append("tokenizer")
    if missing:
        raise InspectError(f"This repo is missing its {' and '.join(missing)}, so dMLX cannot run it on its own.")
    _names, meta = _tensor_names(repo, files, "transformer" if transformer_dir is None else transformer_dir)
    if not meta.get("mflux_version"):
        raise InspectError(
            "This repo is not in mflux's format. Use an mflux version of the model "
            "(for example one from huggingface.co/mflux-community)."
        )
    level = meta.get("quantization_level")
    quantization = int(level) if level not in (None, "", "None", "null") else None
    license_id = (card_license or "unknown").lower()
    return Inspection(
        repo=repo,
        id=_slug(repo),
        name=f"{variant['name']} ({_bits_label(quantization)})",
        family=family_id,
        mflux_config=variant["config"],
        tasks=list(family["tasks"]),
        license=license_id,
        permissive_license=license_id in _PERMISSIVE,
        quantization=quantization,
        size_bytes=size,
        defaults={**variant["defaults"], "width": 1024, "height": 1024},
        activation_gb_per_mp=family["activation_gb_per_mp"],
        subdirs=subdirs,
        parts=tuple(c.name for c in definition.get_components()),
    )


def normalize_repo(text: str) -> str:
    """Accept `owner/name`, or a huggingface.co URL to the repo or one of its pages."""
    text = text.strip().rstrip("/")
    match = re.search(r"huggingface\.co/([^/\s]+/[^/\s?#]+)", text)
    repo = match.group(1) if match else text
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise InspectError("Enter a Hugging Face repo like mflux-community/flux2-klein-9b-mflux-q4, or its URL.")
    return repo


def inspect(text: str) -> Inspection:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    repo = normalize_repo(text)
    try:
        info = HfApi().model_info(repo, files_metadata=True)
    except GatedRepoError as exc:
        raise InspectError("This repo needs you to accept its terms on huggingface.co first.") from exc
    except RepositoryNotFoundError as exc:
        raise InspectError(f"No public Hugging Face model named {repo}.") from exc
    except Exception as exc:
        raise InspectError(f"Could not reach Hugging Face: {exc}") from exc
    files = {s.rfilename: s.size or 0 for s in info.siblings or []}
    card = info.card_data.to_dict() if info.card_data else {}
    return detect(info.id, files, card.get("license_name") or card.get("license"))


# -- user models folder ----------------------------------------------------------------------
def spec_path(model_id: str):
    return paths.models_dir() / f"{model_id}.yaml"


def save_spec(inspection: Inspection) -> ModelSpec:
    data = inspection.to_spec_data()
    spec_path(inspection.id).write_text(yaml.safe_dump(data, sort_keys=False))
    return _parse(data)


def delete_spec(model_id: str) -> None:
    spec_path(model_id).unlink(missing_ok=True)


# -- suggestions -----------------------------------------------------------------------------
def label(repo: str) -> str:
    """A readable name from a repo name alone (inspection gives the authoritative one)."""
    name = repo.split("/")[-1].lower()
    tokens = re.split(r"[^a-z0-9]+", name)
    precision = next((f"{t[1:]}-bit" for t in tokens if re.fullmatch(r"q\d", t)), "full precision" if "bf16" in tokens else "")
    family_id = _generic_family(repo)
    variant = _variant(repo, FAMILIES[family_id]) if family_id else None
    if variant:
        title = variant["name"]
    elif "klein" in tokens:
        size = "9B" if "9b" in tokens else "4B"
        title = f"FLUX.2 Klein {'Base ' if 'base' in tokens else ''}{size}{' KV' if 'kv' in tokens else ''}"
    elif "z" in tokens and "image" in tokens:
        title = "Z-Image Turbo" if "turbo" in tokens else "Z-Image Base"
    elif "qwen" in tokens and "2" in tokens and "1" in tokens:
        title = "Qwen Image 2.1 Base"
    else:
        title = repo.split("/")[-1]
    return f"{title} · {precision}" if precision else title


def quantization_of(repo: str) -> int | None:
    """The bits of a build from its repo name (`-q4`, `-4bit`); None for full precision or unknown."""
    tokens = re.split(r"[^a-z0-9]+", repo.split("/")[-1].lower())
    for token in tokens:
        match = re.fullmatch(r"q(\d)|(\d)bit", token)
        if match:
            return int(match.group(1) or match.group(2))
    return None


def group(repo: str, quantization: int | None = None) -> dict:
    """The model a repo is one build of, so the app can show one entry per model (family and size, e.g.
    "FLUX.2 Klein 4B") with its builds (4-bit, 8-bit, full size) as versions of it. Base (undistilled)
    and KV checkpoints are separate models: they run differently (much slower, or faster edits), while
    the builds of one model differ only in download size and quality."""
    name = label(repo).split(" · ")[0]
    bits = quantization_of(repo)
    return {
        "group": _slug(name),
        "group_name": name,
        "quantization": bits if bits is not None else quantization,
    }


def family_of(repo: str) -> str | None:
    """The supported family of a repo, from its name alone."""
    family_id = _generic_family(repo)
    if family_id:
        return family_id
    lowered = repo.lower()
    if _is_qwen_21(repo):
        return "qwen-image-2.1"
    if "klein" in lowered:
        return "flux2-klein"
    if "z-image" in lowered:
        return "z-image"
    return None



_SUGGESTION_TTL = 3600.0
_suggestions: tuple[float, list[dict]] | None = None


def suggestions() -> list[dict]:
    """Repos from mflux-community whose names match a family dMLX supports (checked fully on select)."""
    global _suggestions
    if _suggestions and time.monotonic() - _suggestions[0] < _SUGGESTION_TTL:
        return _suggestions[1]
    from huggingface_hub import HfApi

    out = []
    models = HfApi().list_models(author="mflux-community", limit=500, sort="downloads", expand=["downloads", "cardData"])
    for model in models:
        lowered = model.id.lower()
        if "controlnet" in lowered or "lora" in lowered:
            continue
        family_id = _generic_family(model.id)
        if family_id:
            if _variant(model.id, FAMILIES[family_id]) is None:
                continue
        elif not ("klein" in lowered or "z-image" in lowered or (_is_qwen_21(model.id) and "turbo" not in lowered)):
            continue
        family = FAMILIES[family_of(model.id) or "flux2-klein"]
        card = model.card_data.to_dict() if getattr(model, "card_data", None) else {}
        license_id = str(card.get("license_name") or card.get("license") or "unknown").lower()
        out.append({
            "repo": model.id,
            "label": label(model.id),
            "downloads": getattr(model, "downloads", None) or 0,
            **group(model.id),
            "tasks": list(family["tasks"]),
            "license": license_id,
            "permissive_license": license_id in _PERMISSIVE,
            "min_ram_gb": family.get("min_ram_gb", 16),
        })
    _suggestions = (time.monotonic(), out)
    return out


_sizes: dict[str, float] = {}


def repo_sizes(repos: list[str]) -> dict[str, float]:
    """Download size (GiB) of each repo's weights, read from its file list (nothing is downloaded).
    Repos that cannot be read are left out; sizes are kept for the engine's lifetime."""
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi

    def size(repo: str) -> tuple[str, float | None]:
        try:
            info = HfApi().model_info(repo, files_metadata=True)
        except Exception:
            return repo, None
        total = sum(s.size or 0 for s in info.siblings or [] if s.rfilename.endswith(".safetensors"))
        return repo, round(total / 1024**3, 2) if total else None

    wanted = [r for r in dict.fromkeys(repos) if r not in _sizes][:16]
    if wanted:
        with ThreadPoolExecutor(max_workers=6) as pool:
            for repo, gb in pool.map(size, wanted):
                if gb is not None:
                    _sizes[repo] = gb
    return {r: _sizes[r] for r in repos if r in _sizes}
