"""Declarative model catalog loaded from the YAML specs in dmlx/models/."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import yaml

from dmlx.capabilities import CapabilitySpec, parse_capabilities

GB = 1024**3


@dataclass
class ComponentSpec:
    name: str
    phase: str  # encode | denoise | decode
    share_key: str | None = None
    keep: bool = False  # small components that are never freed between phases
    size: int = 0  # bytes on disk; filled in once the model is downloaded
    stream: list[str] = field(default_factory=list)  # block stacks that can be streamed from disk
    stream_size: int = 0  # weights resident while streaming (non-block part + largest block)
    external: bool = False  # loaded by the adapter from a smaller auxiliary checkpoint
    subdir: str | None = None  # the part's folder in the repo when it is not the part's name (Boogu: mllm)

    @property
    def folder(self) -> str:
        return self.name if self.subdir is None else self.subdir


def _megapixels(width: int, height: int) -> float:
    """Megapixels for memory estimates. Below 1 MP the working memory stops shrinking (fixed costs,
    and VAE tiles of a fixed size: Krea 2's decode needs 6.1 GB at 512 px and at 1024 px), and the
    per-MP numbers are measured that way (`dmlx bench-family`), so small sizes count as 1 MP."""
    return max(width * height / 1_000_000, 1.0)


@dataclass
class ModelSpec:
    id: str
    name: str
    adapter: str
    repo: str
    license: str
    tasks: list[str]
    components: dict[str, ComponentSpec]
    defaults: dict
    quantization: int | None = None
    max_references: int = 0
    min_steps: int = 1
    min_ram_gb: int = 16
    activation_gb_per_mp: float = 1.5  # denoise working memory per megapixel
    reference_activation_factor: float = 0.75  # additional activation multiplier per reference image
    decode_gb_per_mp: float = 9.0  # untiled VAE decode spike per megapixel
    decode_tiled_gb_per_mp: float = 2.5  # tiled VAE decode spike per megapixel
    decode_banded_gb_per_mp: float | None = None  # banded (exact, low-memory) decode, where the VAE supports it
    mflux_config: str | None = None  # mflux ModelConfig factory (e.g. flux2_klein_9b); adapter default if None
    revision: str | None = None  # optional pinned Hugging Face revision for reproducible downloads
    auto_select: bool = True  # false for restricted/experimental models that require explicit selection
    experimental: bool = False  # integration is available but has not passed every release gate
    license_url: str | None = None
    license_acceptance_id: str | None = None
    agent_note: str = ""  # what the planning agent should know when it picks a model for a message
    user_added: bool = False  # added from the app; stored in the user's models folder and removable
    download_size: int = 0  # the repo's files on Hugging Face, for showing the size before the model is downloaded
    local_path: Path | None = field(default=None, repr=False)
    capabilities: dict[str, CapabilitySpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Tests and third-party callers may still construct ModelSpec directly
        # with the v0.1 task fields.
        if not self.capabilities:
            self.capabilities = parse_capabilities(None, self.tasks, self.max_references)

    def supports(self, task: str) -> bool:
        return task in self.tasks

    @property
    def low_memory_decode(self) -> str:
        """Decode mode used whenever the full decode does not clearly fit."""
        return "banded" if self.decode_banded_gb_per_mp else "tiled"

    def activation_bytes(self, width: int, height: int, references: int = 0) -> int:
        """Peak working memory of the cheapest safe path: denoise, or a low-memory decode, whichever is larger."""
        megapixels = _megapixels(width, height)
        # Reference images add their tokens to the sequence. Measured with one reference at 1024x1024
        # (bf16 throughout): 1.45x the working memory of a plain generation; 0.75 per image keeps margin.
        denoise = self.activation_gb_per_mp * megapixels * (
            1 + self.reference_activation_factor * references)
        return int(max(denoise, self.decode_bytes(width, height, self.low_memory_decode) / GB) * GB)

    def decode_bytes(self, width: int, height: int, mode: str) -> int:
        per_mp = {
            "full": self.decode_gb_per_mp,
            "tiled": self.decode_tiled_gb_per_mp,
            "banded": self.decode_banded_gb_per_mp or self.decode_tiled_gb_per_mp,
        }[mode]
        return int(per_mp * _megapixels(width, height) * GB)

    @property
    def size(self) -> int:
        return sum(c.size for c in self.components.values())

    @property
    def listed_size(self) -> int:
        """On disk once downloaded; before that, the repo's download size plus the parts it shares with other
        models (external), so the first-run setup can show the total."""
        if self.local_path is not None or not self.download_size:
            return self.size
        external = sum(c.size for c in self.components.values() if c.external)
        return max(self.size, self.download_size + external)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "license": self.license,
            "license_url": self.license_url,
            "license_acceptance_id": self.license_acceptance_id,
            "auto_select": self.auto_select,
            "experimental": self.experimental,
            "tasks": self.tasks,
            "max_references": self.max_references,
            "min_steps": self.min_steps,
            "min_ram_gb": self.min_ram_gb,
            "reference_activation_factor": self.reference_activation_factor,
            "repo": self.repo,
            "user_added": self.user_added,
            "defaults": self.defaults,
            "size_gb": round(self.listed_size / GB, 2) if self.listed_size else None,
            "capabilities": {name: capability.as_dict() for name, capability in self.capabilities.items()},
        }


def _parse(data: dict) -> ModelSpec:
    comps = {
        name: ComponentSpec(
            name=name,
            phase=c["phase"],
            share_key=c.get("share_key"),
            keep=bool(c.get("keep", False)),
            stream=list(c.get("stream", [])),
            size=int(float(c.get("size_gb", 0)) * GB),
            external=bool(c.get("external", False)),
            subdir=c.get("subdir"),
        )
        for name, c in data["components"].items()
    }
    tasks = list(data["tasks"])
    max_references = int(data.get("max_references", 0))
    return ModelSpec(
        id=data["id"],
        name=data["name"],
        adapter=data["adapter"],
        repo=data["repo"],
        license=data["license"],
        tasks=tasks,
        components=comps,
        defaults=dict(data.get("defaults", {})),
        quantization=data.get("quantization"),
        max_references=max_references,
        min_steps=int(data.get("min_steps", 1)),
        min_ram_gb=int(data.get("min_ram_gb", 16)),
        activation_gb_per_mp=float(data.get("activation_gb_per_mp", 1.5)),
        reference_activation_factor=float(data.get("reference_activation_factor", 0.75)),
        decode_gb_per_mp=float(data.get("decode_gb_per_mp", 9.0)),
        decode_tiled_gb_per_mp=float(data.get("decode_tiled_gb_per_mp", 2.5)),
        decode_banded_gb_per_mp=float(data["decode_banded_gb_per_mp"]) if "decode_banded_gb_per_mp" in data else None,
        mflux_config=data.get("mflux_config"),
        revision=data.get("revision"),
        auto_select=bool(data.get("auto_select", True)),
        experimental=bool(data.get("experimental", False)),
        license_url=data.get("license_url"),
        license_acceptance_id=data.get("license_acceptance_id"),
        agent_note=str(data.get("agent_note", "")),
        user_added=bool(data.get("user_added", False)),
        download_size=int(float(data.get("download_gb", 0)) * GB),
        capabilities=parse_capabilities(data.get("capabilities"), tasks, max_references),
    )


class Registry:
    def __init__(self, specs: list[ModelSpec]):
        self._specs = {s.id: s for s in specs}

    @classmethod
    def load_builtin(cls, extra_dir: Path | None = None) -> "Registry":
        specs = []
        for entry in sorted(resources.files("dmlx.models").iterdir(), key=lambda p: p.name):
            if entry.name.endswith(".yaml"):
                specs.append(_parse(yaml.safe_load(entry.read_text())))
        builtin = {s.id for s in specs}
        if extra_dir and extra_dir.is_dir():
            for path in sorted(extra_dir.glob("*.yaml")):
                try:
                    spec = _parse(yaml.safe_load(path.read_text()))
                except Exception:  # a hand-edited or truncated file must not stop the engine
                    continue
                if spec.id not in builtin:
                    specs.append(spec)
        return cls(specs)

    def add(self, spec: ModelSpec) -> None:
        self._specs[spec.id] = spec

    def remove(self, model_id: str) -> None:
        self._specs.pop(model_id, None)

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._specs[model_id]
        except KeyError:
            raise KeyError(f"Unknown model '{model_id}'. Known: {', '.join(self._specs)}") from None

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._specs

    def all(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def default_for(self, task: str, total_ram_gb: float) -> ModelSpec | None:
        # One model for both tasks by default: alternating generate/edit turns never swap models.
        preference = {"t2i": ["flux2-klein-4b", "z-image-turbo"], "edit": ["flux2-klein-4b"]}
        for model_id in preference.get(task, []):
            spec = self._specs.get(model_id)
            if spec and spec.auto_select and spec.supports(task) and spec.min_ram_gb <= total_ram_gb + 0.5:
                return spec
        for spec in self._specs.values():
            if spec.auto_select and spec.supports(task) and spec.min_ram_gb <= total_ram_gb + 0.5:
                return spec
        return None

    def default_for_operation(self, operation: str, total_ram_gb: float) -> ModelSpec | None:
        legacy_task = {"image.generate": "t2i", "image.edit": "edit"}.get(operation)
        if legacy_task:
            preferred = self.default_for(legacy_task, total_ram_gb)
            if preferred and operation in preferred.capabilities:
                return preferred
        return next((
            spec for spec in self._specs.values()
            if spec.auto_select and operation in spec.capabilities and spec.min_ram_gb <= total_ram_gb + 0.5
        ), None)
