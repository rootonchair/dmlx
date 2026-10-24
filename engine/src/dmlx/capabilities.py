"""Declarative workflow capabilities exposed by model specifications."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InputConstraint:
    minimum: int = 0
    maximum: int = 0
    ordered: bool = True

    def as_dict(self) -> dict:
        return {"min": self.minimum, "max": self.maximum, "ordered": self.ordered}


@dataclass(frozen=True)
class CapabilitySpec:
    operation: str
    inputs: dict[str, InputConstraint] = field(default_factory=dict)
    positive_prompt: bool = True
    negative_prompt: bool = False

    def as_dict(self) -> dict:
        return {
            "operation": self.operation,
            "inputs": {kind: constraint.as_dict() for kind, constraint in self.inputs.items()},
            "prompt": {"positive": self.positive_prompt, "negative": self.negative_prompt},
        }


def parse_capabilities(data: dict | None, tasks: list[str], max_references: int) -> dict[str, CapabilitySpec]:
    """Parse the new schema, or derive it from v0.1 task declarations."""
    if data:
        out: dict[str, CapabilitySpec] = {}
        for operation, raw in data.items():
            constraints = {
                kind: InputConstraint(
                    minimum=int(value.get("min", 0)),
                    maximum=int(value.get("max", 0)),
                    ordered=bool(value.get("ordered", True)),
                )
                for kind, value in (raw.get("inputs") or {}).items()
            }
            prompt = raw.get("prompt") or {}
            out[operation] = CapabilitySpec(
                operation=operation,
                inputs=constraints,
                positive_prompt=bool(prompt.get("positive", True)),
                negative_prompt=bool(prompt.get("negative", False)),
            )
        return out

    out = {}
    if "t2i" in tasks:
        out["image.generate"] = CapabilitySpec("image.generate")
    if "edit" in tasks:
        out["image.edit"] = CapabilitySpec(
            "image.edit", {"image": InputConstraint(1, max(max_references, 1), True)}
        )
    return out
