"""Deterministic prompt syntax understood without loading an LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CompiledPrompt:
    positive: str
    negative: str | None = None
    compiler: str = "passthrough@1"


class PassthroughPromptCompiler:
    """Preserve normal text and split only explicit negative-prompt syntax."""

    id = "passthrough@1"
    _negative_line = re.compile(r"(?im)^\s*negative\s*:\s*(.+?)\s*$")
    _negative_flag = re.compile(r"(?is)\s+--negative(?:\s+|=)(.+?)\s*$")

    def compile(self, text: str) -> CompiledPrompt:
        source = text.strip()
        match = self._negative_flag.search(source)
        if match:
            positive = source[: match.start()].strip()
            negative = match.group(1).strip()
            return CompiledPrompt(positive or source, negative or None, self.id)

        matches = list(self._negative_line.finditer(source))
        if matches:
            negative = ", ".join(match.group(1).strip() for match in matches if match.group(1).strip())
            positive = self._negative_line.sub("", source).strip()
            return CompiledPrompt(positive or source, negative or None, self.id)
        return CompiledPrompt(source, None, self.id)
