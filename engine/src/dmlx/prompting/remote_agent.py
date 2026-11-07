"""The planning agent on a service the user chose instead of a model on this Mac.

Two kinds of service:
  - Claude (Anthropic), through the official `anthropic` SDK and the Messages API;
  - ChatGPT (OpenAI), Ollama, and any other service with an OpenAI-compatible
    `/v1/chat/completions` endpoint.

It plays the same part as the local agent (local_vlm.LocalVLMPromptProvider): `chat` gets the planner's
messages and the chat images and returns the model's text. The chat and the images it is shown leave
this Mac, which the app says before the user picks a service. The API key is held in memory only; the app
keeps it in the Keychain and passes it in when the engine starts.
"""

from __future__ import annotations

import base64
import io
import json
import re
import threading
import urllib.error
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass

from dmlx.intelligence.base import IntelligenceError
from dmlx.prompting.local_vlm import ChatPromptProvider, LocalVLMPromptProvider

KINDS = ("anthropic", "openai", "ollama", "openai-compatible")
DEFAULT_BASE_URL = {
    "openai": "https://api.openai.com/v1",
    "ollama": "http://localhost:11434/v1",
}
# Models that take the server-side refusal fallback (Claude API): a declined plan is re-run on a fallback
# model inside the same request instead of failing the message.
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
_MAX_TOKENS = 16_000  # the plan is a short JSON reply; this also leaves room for the model's thinking
_IMAGE_SIDE = 1024
_TIMEOUT = 90.0


@dataclass(frozen=True)
class RemoteConfig:
    kind: str
    model: str
    base_url: str = ""
    api_key: str = ""

    def public(self) -> dict:
        """What the app may show and the engine may save: never the key."""
        return {"kind": self.kind, "model": self.model, "base_url": self.endpoint_base}

    @property
    def endpoint_base(self) -> str:
        if self.kind == "anthropic":
            return ""
        return (self.base_url or DEFAULT_BASE_URL.get(self.kind, "")).rstrip("/")


def check(config: RemoteConfig) -> None:
    """Refuse a configuration that cannot work, before anything is sent."""
    if config.kind not in KINDS:
        raise ValueError(f"Unknown agent service '{config.kind}'.")
    if not config.model.strip():
        raise ValueError("Choose a model.")
    if config.kind in ("anthropic", "openai") and not config.api_key.strip():
        raise ValueError("Enter your API key.")
    if config.kind in ("ollama", "openai-compatible"):
        from urllib.parse import urlparse

        url = urlparse(config.endpoint_base)
        if url.scheme not in ("http", "https") or not url.hostname:
            raise ValueError("Enter the service's address, for example http://localhost:11434/v1.")
        if url.scheme == "http" and url.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("A service on another computer must use https://.")


class RemoteAgent(ChatPromptProvider):
    """Plans each message on the chosen service. Nothing is loaded on this Mac, so it needs no memory."""

    uses_accelerator = False
    resident = False
    need_bytes = 0

    def __init__(self, config: RemoteConfig, *, client=None, opener=None):
        check(config)
        self.config = config
        self.name = f"remote:{config.kind}:{config.model}"
        self._lock = threading.RLock()
        self._cache: OrderedDict = OrderedDict()
        self._client = client  # an anthropic.Anthropic, injected by tests
        self._opener = opener or urllib.request.urlopen

    def unload(self) -> None:
        """Nothing to free: the model runs on the service."""

    def status(self) -> dict:
        return {"kind": "remote", "name": self.name, "resident": False, **self.config.public()}

    def chat(self, messages: list[dict], images: list) -> str:
        pictures = [_jpeg(item) for item in images]
        if self.config.kind == "anthropic":
            return self._claude(messages, pictures)
        return self._openai(messages, pictures)

    # -- Claude ---------------------------------------------------------------------------------------
    def _claude(self, messages: list[dict], pictures: list[str]) -> str:
        import anthropic

        client = self._client or anthropic.Anthropic(api_key=self.config.api_key, timeout=_TIMEOUT, max_retries=2)
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
        if pictures and turns:
            # The images belong to the message being planned: they go before its text, as on this Mac.
            last = turns[-1]
            last["content"] = [
                *({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}
                  for data in pictures),
                {"type": "text", "text": last["content"]},
            ]
        request = {
            "model": self.config.model,
            "max_tokens": _MAX_TOKENS,
            "system": system,
            "messages": turns,
            # A plan is a short routing decision that the user waits for: low effort keeps it quick.
            "output_config": {"effort": "low"},
        }
        try:
            if self.config.model in _FALLBACK_MODELS:
                response = client.beta.messages.create(
                    **request, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            else:
                response = client.messages.create(**request)
        except anthropic.AuthenticationError as exc:
            raise IntelligenceError("Claude did not accept the API key. Check it in Settings → Intelligence.") from exc
        except anthropic.PermissionDeniedError as exc:
            raise IntelligenceError("This Claude API key cannot use that model.") from exc
        except anthropic.NotFoundError as exc:
            raise IntelligenceError(f"Claude has no model '{self.config.model}'.") from exc
        except anthropic.RateLimitError as exc:
            raise IntelligenceError("Claude is rate limiting this key. Try again in a minute.") from exc
        except anthropic.APIStatusError as exc:
            raise IntelligenceError(f"Claude returned an error ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise IntelligenceError("Could not reach Claude. Check the internet connection.") from exc
        if response.stop_reason == "refusal":
            raise IntelligenceError("Claude declined to plan this message.")
        text = "".join(block.text for block in response.content if block.type == "text")
        if not text.strip():
            raise IntelligenceError("Claude returned no answer.")
        return text

    # -- OpenAI-compatible (ChatGPT, Ollama, others) --------------------------------------------------
    def _openai(self, messages: list[dict], pictures: list[str]) -> str:
        turns = [{"role": m["role"], "content": m["content"]} for m in messages]
        if pictures:
            last = next(m for m in reversed(turns) if m["role"] == "user")
            last["content"] = [
                *({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}} for data in pictures),
                {"type": "text", "text": last["content"]},
            ]
        body = self._post("/chat/completions", {"model": self.config.model, "messages": turns})
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise IntelligenceError("The agent service returned an answer dMLX cannot read.") from exc
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if not isinstance(content, str) or not content.strip():
            raise IntelligenceError("The agent service returned no answer.")
        return content

    def _post(self, path: str, payload: dict) -> dict:
        return self._request(path, json.dumps(payload).encode())

    def _request(self, path: str, data: bytes | None = None) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib.request.Request(self.config.endpoint_base + path, data=data, headers=headers,
                                         method="POST" if data is not None else "GET")
        try:
            with self._opener(request, timeout=_TIMEOUT) as reply:
                return json.loads(reply.read())
        except urllib.error.HTTPError as exc:
            detail = _error_text(exc)
            if exc.code in (401, 403):
                raise IntelligenceError("The service did not accept the API key.") from exc
            if exc.code == 404:
                raise IntelligenceError(f"The service has no model '{self.config.model}', or the address is wrong.") from exc
            raise IntelligenceError(f"The agent service returned an error ({exc.code}){': ' + detail if detail else ''}.") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            where = "Ollama. Is it running?" if self.config.kind == "ollama" else "the agent service."
            raise IntelligenceError(f"Could not reach {where}") from exc
        except json.JSONDecodeError as exc:
            raise IntelligenceError("The agent service returned an answer dMLX cannot read.") from exc

    # -- setup helpers ------------------------------------------------------------------------------
    def models(self) -> list[dict]:
        """The service's models for the setup's model menu, recommended ones first.

        Each: {"id", "name", "line", "recommended", "images"}. `images` is True or None (unknown); models
        known to take no images are left out, because the agent must see the chat's pictures.
        """
        if self.config.kind == "anthropic":
            return self._claude_models()
        body = self._request("/models")
        items = [item for item in body.get("data", []) if isinstance(item, dict) and item.get("id")]
        if self.config.kind == "openai":
            return _openai_choices(items)
        return [_choice(item["id"]) for item in sorted(items, key=lambda item: item["id"])]

    def _claude_models(self) -> list[dict]:
        import anthropic

        client = self._client or anthropic.Anthropic(api_key=self.config.api_key, timeout=_TIMEOUT, max_retries=2)
        try:
            listed = list(client.models.list(limit=100))  # newest first; the page iterator follows further pages
        except anthropic.AuthenticationError as exc:
            raise IntelligenceError("Claude did not accept the API key.") from exc
        except anthropic.APIStatusError as exc:
            raise IntelligenceError(f"Claude returned an error ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise IntelligenceError("Could not reach Claude. Check the internet connection.") from exc
        usable = []
        for info in listed:
            if getattr(info, "lifecycle", "active") != "active":
                continue
            capabilities = getattr(info, "capabilities", None)
            images = capabilities.image_input.supported if capabilities is not None else None
            if images is False:
                continue
            usable.append((info.id, info.display_name or info.id, getattr(info, "line", None), images))
        # The newest model of each line is recommended: Opus, Sonnet, Haiku, then any other line.
        newest: dict[str, str] = {}
        for model_id, _, line, _ in usable:
            if line and line not in newest:
                newest[line] = model_id
        lines = sorted(newest, key=lambda line: (_CLAUDE_LINES.index(line) if line in _CLAUDE_LINES else 99, line))
        picks = [newest[line] for line in lines]
        choices = {model_id: _choice(model_id, name, line, model_id in picks, images)
                   for model_id, name, line, images in usable}
        return [choices[model_id] for model_id in picks] + [c for c in choices.values() if not c["recommended"]]

    def test(self) -> None:
        """One tiny request, so a wrong key or address shows up during setup, not on the first message."""
        reply = self.chat([{"role": "system", "content": "Answer with the single word OK."},
                           {"role": "user", "content": "Say OK."}], [])
        if not reply.strip():
            raise IntelligenceError("The agent service returned no answer.")


_CLAUDE_LINES = ("opus", "sonnet", "haiku")
# OpenAI lists every model the key can call, including ones that cannot chat or cannot see images.
_NOT_CHAT = ("embedding", "tts", "whisper", "dall-e", "moderation", "audio", "realtime", "transcribe", "image",
             "search", "davinci", "babbage", "sora", "codex", "computer-use", "-instruct", "-pro")
_NO_IMAGES = ("gpt-3.5", "gpt-4-0", "o1-mini", "o1-preview", "o3-mini")
_SNAPSHOT = re.compile(r"-\d{4}-\d{2}-\d{2}$|-\d{4}$|preview|latest")
_OPENAI_RECOMMENDED = 3


def _choice(model_id: str, name: str | None = None, line: str | None = None, recommended: bool = False,
            images: bool | None = None) -> dict:
    return {"id": model_id, "name": name or model_id, "line": line, "recommended": recommended, "images": images}


def _openai_choices(items: list[dict]) -> list[dict]:
    """Chat models that can see images; the newest main models (not dated snapshots or previews) first."""
    chat = [item for item in items
            if not any(word in item["id"] for word in _NOT_CHAT)
            and not item["id"].startswith(_NO_IMAGES) and item["id"] != "gpt-4"]
    chat.sort(key=lambda item: (-(item.get("created") or 0), item["id"]))
    main = [item["id"] for item in chat if not _SNAPSHOT.search(item["id"])][:_OPENAI_RECOMMENDED]
    return ([_choice(model_id, recommended=True, images=True) for model_id in main]
            + [_choice(item["id"]) for item in chat if item["id"] not in main])


def _jpeg(item) -> str:
    """A chat image as base64 JPEG, no larger than the service needs to see it."""
    if isinstance(item, tuple):
        path, side = item
        image = LocalVLMPromptProvider._feed((path, min(side, _IMAGE_SIDE)))
    else:
        image = LocalVLMPromptProvider._feed((item, _IMAGE_SIDE))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    return base64.standard_b64encode(buffer.getvalue()).decode("ascii")


def _error_text(exc: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(exc.read() or b"{}")
    except (json.JSONDecodeError, OSError):
        return ""
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        return str(error.get("message", ""))[:200]
    return str(error or "")[:200]
