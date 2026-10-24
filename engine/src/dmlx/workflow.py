"""Versioned workflow wire types and validation.

Plans are deliberately data-only. The engine resolves executors and validates
capabilities before converting a supported step into an adapter request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dmlx.prompting import PassthroughPromptCompiler


class WorkflowError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class MediaInput:
    id: str
    media_type: str = "image"
    role: str | None = None
    order: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict, fallback_order: int = 0) -> "MediaInput":
        result_id = str(value.get("id", "")).strip()
        if not result_id:
            raise WorkflowError("invalid_input", "Every workflow input needs an id.")
        media_type = str(value.get("media_type", "image"))
        if media_type not in ("image", "video", "mask"):
            raise WorkflowError("invalid_input", f"Unsupported media type '{media_type}'.")
        role = value.get("role")
        if role is not None and role not in ("source", "reference", "composite", "mask", "sketch"):
            raise WorkflowError("invalid_input", f"Unsupported input role '{role}'.")
        return cls(result_id, media_type, role, int(value.get("order", fallback_order)), dict(value.get("metadata") or {}))

    def as_dict(self) -> dict:
        value = {"id": self.id, "media_type": self.media_type, "order": self.order}
        if self.role is not None:
            value["role"] = self.role
        if self.metadata:
            value["metadata"] = self.metadata
        return value


@dataclass(frozen=True)
class PromptSpec:
    user: str
    positive: str
    negative: str | None = None
    enhancement_mode: str = "off"
    compiler: str = "passthrough@1"

    @classmethod
    def from_dict(cls, value: dict | str) -> "PromptSpec":
        if isinstance(value, str):
            value = {"user": value, "positive": value}
        user = str(value.get("user", value.get("positive", ""))).strip()
        positive = str(value.get("positive", user)).strip()
        if not user or not positive:
            raise WorkflowError("invalid_prompt", "Workflow prompt is empty.")
        mode = str(value.get("enhancement_mode", "off"))
        if mode not in ("off", "ask", "auto"):
            raise WorkflowError("invalid_prompt", f"Unknown enhancement mode '{mode}'.")
        negative = value.get("negative")
        compiler_id = str(value.get("compiler", "passthrough@1"))
        if compiler_id == "passthrough@1" and not negative and positive == user:
            compiled = PassthroughPromptCompiler().compile(user)
            positive, negative = compiled.positive, compiled.negative
        return cls(user, positive, str(negative).strip() if negative else None, mode, compiler_id)

    def as_dict(self) -> dict:
        return {
            "user": self.user,
            "positive": self.positive,
            "negative": self.negative,
            "enhancement_mode": self.enhancement_mode,
            "compiler": self.compiler,
        }


@dataclass(frozen=True)
class WorkflowStep:
    id: str
    operation: str
    executor: str | None
    inputs: tuple[MediaInput, ...]
    prompt: PromptSpec
    parameters: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict, index: int = 0) -> "WorkflowStep":
        operation = str(value.get("operation", "")).strip()
        if not operation:
            raise WorkflowError("invalid_workflow", "Every workflow step needs an operation.")
        inputs = tuple(
            sorted(
                (MediaInput.from_dict(item, i) for i, item in enumerate(value.get("inputs") or [])),
                key=lambda item: item.order,
            )
        )
        return cls(
            id=str(value.get("id") or f"step_{index + 1}"),
            operation=operation,
            executor=str(value["executor"]) if value.get("executor") else None,
            inputs=inputs,
            prompt=PromptSpec.from_dict(value.get("prompt") or ""),
            parameters=dict(value.get("parameters") or {}),
        )

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "operation": self.operation,
            "executor": self.executor,
            "inputs": [item.as_dict() for item in self.inputs],
            "prompt": self.prompt.as_dict(),
            "parameters": self.parameters,
        }


@dataclass(frozen=True)
class WorkflowPlan:
    steps: tuple[WorkflowStep, ...]
    schema_version: int = 1
    route: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, value: dict) -> "WorkflowPlan":
        version = int(value.get("schema_version", 1))
        if version != 1:
            raise WorkflowError("unsupported_schema", f"Unsupported workflow schema version {version}.")
        raw_steps = value.get("steps") or []
        if not raw_steps:
            raise WorkflowError("invalid_workflow", "A workflow needs at least one step.")
        return cls(tuple(WorkflowStep.from_dict(step, i) for i, step in enumerate(raw_steps)), version, value.get("route"))

    def as_dict(self) -> dict:
        value = {"schema_version": self.schema_version, "steps": [step.as_dict() for step in self.steps]}
        if self.route is not None:
            value["route"] = self.route
        return value

    @classmethod
    def legacy(
        cls, *, task: str, prompt: str, model: str, input_ids: list[str], parameters: dict[str, Any],
        negative_prompt: str | None = None
    ) -> "WorkflowPlan":
        operation = "image.generate" if task == "t2i" else "image.edit"
        inputs = tuple(
            MediaInput(result_id, role="source" if i == 0 else "reference", order=i)
            for i, result_id in enumerate(input_ids)
        )
        step = WorkflowStep("step_1", operation, model, inputs, PromptSpec(prompt, prompt, negative_prompt), parameters)
        return cls((step,))
