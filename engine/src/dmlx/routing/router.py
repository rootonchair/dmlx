"""Capability-bounded workflow routing with deterministic safe cases."""

from __future__ import annotations

from dmlx.intelligence.base import Decision, DecisionProvider, IntelligenceError


DEFAULT_DESCRIPTIONS = {
    "image.generate": (
        "Create a completely new and independent image from scratch. Use when the request describes "
        "a new picture, scene, or logo and does not ask to preserve or change the current image."
    ),
    "image.edit": (
        "Modify the current or attached image while preserving everything not requested to change. "
        "Includes follow-ups that say keep, same, make, change, add, remove, or refer to existing content."
    ),
    "image.variation": (
        "Create another alternative take based on the current image or the same idea, without requesting "
        "one specific change."
    ),
    "image.upscale": (
        "Increase pixel dimensions or resolution while preserving the same image and content; includes "
        "2x, 4x, 4K, larger, or higher resolution."
    ),
    "image.restore": (
        "Repair image defects such as blur, noise, JPEG artifacts, scratches, or damage without changing "
        "its content."
    ),
    "image.enhance": (
        "Generatively reconstruct or add realistic fine detail and improve perceived image quality, rather "
        "than making a specific content edit or only increasing pixel dimensions."
    ),
    "video.generate": "Create, animate, or convert into a moving video or clip.",
}


class WorkflowRouter:
    def __init__(self, provider: DecisionProvider | None = None):
        self.provider = provider

    def deterministic_decision(self, state: dict, candidates: list[str]) -> Decision | None:
        allowed = list(dict.fromkeys(candidates))
        if not allowed:
            return Decision("ambiguous", 0.0, {}, "deterministic")
        has_current = bool(state.get("current_result_id") or state.get("attachments"))
        if not has_current and "image.generate" in allowed:
            return Decision(
                "image.generate", 1.0, {name: float(name == "image.generate") for name in allowed}, "deterministic"
            )
        if len(allowed) == 1:
            return Decision(allowed[0], 1.0, {allowed[0]: 1.0}, "deterministic")
        return None

    def decide_with_provider(self, state: dict, candidates: list[str]) -> Decision:
        allowed = list(dict.fromkeys(candidates))
        if self.provider is None:
            return Decision("ambiguous", 0.0, {name: 0.0 for name in allowed}, "manual")
        try:
            descriptions = {name: DEFAULT_DESCRIPTIONS.get(name, name) for name in allowed}
            return self.provider.decide(state, descriptions)
        except IntelligenceError:
            return Decision("ambiguous", 0.0, {name: 0.0 for name in allowed}, self.provider.name)

    def decide(self, state: dict, candidates: list[str]) -> Decision:
        deterministic = self.deterministic_decision(state, candidates)
        return deterministic or self.decide_with_provider(state, candidates)
