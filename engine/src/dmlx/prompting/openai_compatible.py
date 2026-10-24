"""OpenAI-compatible prompt enhancement with locally validated output."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from dmlx.intelligence.base import IntelligenceError
from dmlx.intelligence.openai_compatible import (
    ProviderCapabilityError,
    ProviderPayloadError,
    ProviderRefusalError,
    ProviderTransportError,
)
from dmlx.prompting.compiler import CompiledPrompt


class OpenAICompatiblePromptProvider:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 30.0):
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("Provider URL must be an absolute HTTP(S) URL.")
        if parsed.scheme != "https" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("Remote intelligence providers must use HTTPS.")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.name = f"openai-compatible:{model}:prompt-v1"

    @property
    def endpoint(self) -> str:
        suffix = "/chat/completions"
        return self.base_url + suffix if self.base_url.endswith("/v1") else self.base_url + "/v1" + suffix

    def enhance(self, text: str, *, supports_negative: bool) -> CompiledPrompt:
        schema = {
            "type": "object",
            "properties": {
                "positive": {"type": "string"},
                "negative": {"type": ["string", "null"]},
            },
            "required": ["positive", "negative"],
            "additionalProperties": False,
        }
        negative_rule = (
            "A separate negative prompt is supported; include one only when it materially helps."
            if supports_negative
            else "The executor cannot use a separate negative prompt; negative must be null."
        )
        instructions = (
            "Improve the image prompt while preserving the user's intent, subjects, relationships, and requested text. "
            "Do not invent a different composition. Return only the schema. " + negative_rule
        )
        common = {
            "model": self.model,
            "messages": [{"role": "system", "content": instructions}, {"role": "user", "content": text}],
            "temperature": 0.2,
        }
        tool = {
            "type": "function",
            "function": {
                "name": "compile_prompt", "description": "Return the enhanced prompt fields.",
                "strict": True, "parameters": schema,
            },
        }
        attempts = [
            {**common, "response_format": {"type": "json_schema", "json_schema": {
                "name": "compiled_prompt", "strict": True, "schema": schema}}},
            {**common, "tools": [tool], "tool_choice": {"type": "function", "function": {"name": "compile_prompt"}}},
            {**common, "response_format": {"type": "json_object"}},
            common,
        ]
        last_error: Exception | None = None
        repaired_payload = False
        for payload in attempts:
            try:
                return self._request(payload, supports_negative=supports_negative)
            except ProviderCapabilityError as exc:
                last_error = exc
                continue
            except ProviderPayloadError as exc:
                if repaired_payload:
                    raise IntelligenceError(str(exc)) from exc
                repaired_payload = True
                last_error = exc
                continue
            except (ProviderRefusalError, ProviderTransportError) as exc:
                raise IntelligenceError(str(exc)) from exc
        raise IntelligenceError(str(last_error or "Prompt provider did not return a compiled prompt."))

    def _request(self, payload: dict, *, supports_negative: bool) -> CompiledPrompt:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint, data=json.dumps(payload).encode(), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read())
            message = body["choices"][0]["message"]
            if message.get("refusal"):
                raise ProviderRefusalError("Prompt provider refused the enhancement request.")
            calls = message.get("tool_calls") or []
            content = calls[0]["function"]["arguments"] if calls else message.get("content")
            if isinstance(content, str):
                content = content.strip()
                if content.startswith("```") and content.endswith("```"):
                    content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                value = json.loads(content)
            else:
                value = content
            positive = str(value["positive"]).strip()
            raw_negative = value.get("negative")
            negative = str(raw_negative).strip() if raw_negative else None
        except urllib.error.HTTPError as exc:
            if exc.code in {400, 404, 415, 422}:
                raise ProviderCapabilityError(
                    f"Prompt provider does not support this response format (HTTP {exc.code})."
                ) from exc
            raise ProviderTransportError(f"Prompt provider request failed with HTTP {exc.code}.") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ProviderTransportError(f"Prompt provider request failed: {exc}") from exc
        except ProviderRefusalError:
            raise
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderPayloadError(f"Prompt provider returned an invalid payload: {exc}") from exc
        if not positive:
            raise ProviderPayloadError("Prompt provider returned an empty positive prompt.")
        if negative and not supports_negative:
            raise ProviderPayloadError("Prompt provider returned unsupported negative conditioning.")
        return CompiledPrompt(positive, negative, self.name)
