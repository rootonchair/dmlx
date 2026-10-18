"""Declarative model catalog loaded from the YAML specs in dmlx/models/."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import yaml

GB = 1024**3


@dataclass
class ComponentSpec:
    name: str
    phase: str  # encode | denoise | decode
    share_key: str | None = None
    keep: bool = False  # small components that are never freed between phases
    size: int = 0  # bytes on disk; filled in once the model is downloaded


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
    min_ram_gb: int = 16
    activation_gb_per_mp: float = 1.5
    local_path: Path | None = field(default=None, repr=False)

    def supports(self, task: str) -> bool:
        return task in self.tasks

    def activation_bytes(self, width: int, height: int, references: int = 0) -> int:
        megapixels = (width * height) / 1_000_000
        # Reference images add their tokens to the sequence; roughly one more image each.
        return int(self.activation_gb_per_mp * megapixels * (1 + 0.6 * references) * GB)

    @property
    def size(self) -> int:
        return sum(c.size for c in self.components.values())

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "license": self.license,
            "tasks": self.tasks,
            "max_references": self.max_references,
            "min_ram_gb": self.min_ram_gb,
            "repo": self.repo,
            "defaults": self.defaults,
            "size_gb": round(self.size / GB, 2) if self.size else None,
        }


def _parse(data: dict) -> ModelSpec:
    comps = {
        name: ComponentSpec(name=name, phase=c["phase"], share_key=c.get("share_key"), keep=bool(c.get("keep", False)))
        for name, c in data["components"].items()
    }
    return ModelSpec(
        id=data["id"],
        name=data["name"],
        adapter=data["adapter"],
        repo=data["repo"],
        license=data["license"],
        tasks=list(data["tasks"]),
        components=comps,
        defaults=dict(data.get("defaults", {})),
        quantization=data.get("quantization"),
        max_references=int(data.get("max_references", 0)),
        min_ram_gb=int(data.get("min_ram_gb", 16)),
        activation_gb_per_mp=float(data.get("activation_gb_per_mp", 1.5)),
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
        if extra_dir and extra_dir.is_dir():
            for path in sorted(extra_dir.glob("*.yaml")):
                specs.append(_parse(yaml.safe_load(path.read_text())))
        return cls(specs)

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
        preference = {"t2i": ["z-image-turbo", "flux2-klein-4b"], "edit": ["flux2-klein-4b"]}
        for model_id in preference.get(task, []):
            spec = self._specs.get(model_id)
            if spec and spec.supports(task) and spec.min_ram_gb <= total_ram_gb + 0.5:
                return spec
        for spec in self._specs.values():
            if spec.supports(task) and spec.min_ram_gb <= total_ram_gb + 0.5:
                return spec
        return None
