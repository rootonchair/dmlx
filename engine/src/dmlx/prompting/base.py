from __future__ import annotations

from typing import Protocol

from dmlx.prompting.compiler import CompiledPrompt


class PromptProvider(Protocol):
    name: str

    def enhance(self, text: str, *, supports_negative: bool) -> CompiledPrompt: ...
