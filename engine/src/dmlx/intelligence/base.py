from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class IntelligenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Decision:
    intent: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    provider: str = "unknown"
    input_result_ids: tuple[str, ...] = ()
    model_version: str | None = None
    latency_ms: float | None = None
    memory_mb: float | None = None

    def as_dict(self) -> dict:
        value = {
            "intent": self.intent,
            "confidence": round(max(0.0, min(self.confidence, 1.0)), 4),
            "probabilities": {key: round(value, 4) for key, value in self.probabilities.items()},
            "provider": self.provider,
            "input_result_ids": list(self.input_result_ids),
        }
        if self.model_version is not None:
            value["model_version"] = self.model_version
        if self.latency_ms is not None:
            value["latency_ms"] = round(self.latency_ms, 2)
        if self.memory_mb is not None:
            value["memory_mb"] = round(self.memory_mb, 2)
        return value


class DecisionProvider(Protocol):
    name: str

    def decide(self, state: dict, candidates: dict[str, str]) -> Decision: ...
