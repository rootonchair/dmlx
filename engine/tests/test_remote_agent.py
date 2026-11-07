"""The planning agent on a service the user chose: Claude, or an OpenAI-compatible endpoint."""

import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from dmlx.intelligence.base import IntelligenceError
from dmlx.prompting import remote_agent
from dmlx.prompting.remote_agent import RemoteAgent, RemoteConfig

MESSAGES = [{"role": "system", "content": "You plan."}, {"role": "user", "content": "make the boat red"}]


def picture(tmp_path):
    path = tmp_path / "boat.png"
    Image.new("RGB", (2048, 1024), "red").save(path)
    return path


class FakeClaude:
    def __init__(self, stop_reason="end_turn", text='{"intent": "image.edit"}'):
        self.calls = []
        reply = SimpleNamespace(stop_reason=stop_reason, content=[
            SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)])

        def create(**kwargs):
            self.calls.append(kwargs)
            return reply

        self.messages = SimpleNamespace(create=create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


def test_claude_gets_the_system_prompt_and_the_images_before_the_text(tmp_path):
    client = FakeClaude()
    agent = RemoteAgent(RemoteConfig("anthropic", "claude-opus-5-5", api_key="k"), client=client)
    assert agent.chat(MESSAGES, [picture(tmp_path)]) == '{"intent": "image.edit"}'
    call = client.calls[0]
    assert call["system"] == "You plan."
    assert call["output_config"] == {"effort": "low"}
    assert call["fallbacks"] == "default" and call["betas"] == ["server-side-fallback-2026-07-01"]
    content = call["messages"][0]["content"]
    assert [part["type"] for part in content] == ["image", "text"]
    shrunk = Image.open(io.BytesIO(__import__("base64").b64decode(content[0]["source"]["data"])))
    assert max(shrunk.size) == 1024 and content[1]["text"] == "make the boat red"


def test_haiku_has_no_fallback_and_a_refusal_is_an_error():
    client = FakeClaude(stop_reason="refusal")
    agent = RemoteAgent(RemoteConfig("anthropic", "claude-haiku-5-5", api_key="k"), client=client)
    with pytest.raises(IntelligenceError, match="declined"):
        agent.chat(MESSAGES, [])
    assert "fallbacks" not in client.calls[0]


class FakeOpener:
    def __init__(self, body):
        self.body, self.requests = body, []

    def __call__(self, request, timeout):
        self.requests.append(request)
        return io.BytesIO(json.dumps(self.body).encode())


def test_openai_compatible_sends_data_url_images_and_the_key(tmp_path):
    opener = FakeOpener({"choices": [{"message": {"content": "OK"}}]})
    agent = RemoteAgent(RemoteConfig("openai", "some-model", api_key="sk-1"), opener=opener)
    assert agent.chat(MESSAGES, [picture(tmp_path)]) == "OK"
    request = opener.requests[0]
    assert request.full_url == "https://api.openai.com/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer sk-1"
    user = json.loads(request.data)["messages"][1]["content"]
    assert user[0]["image_url"]["url"].startswith("data:image/jpeg;base64,") and user[1]["text"] == "make the boat red"


def test_ollama_needs_no_key_and_lists_models():
    opener = FakeOpener({"data": [{"id": "qwen2.5vl:7b"}, {"id": "llava"}]})
    agent = RemoteAgent(RemoteConfig("ollama", "llava"), opener=opener)
    assert [m["id"] for m in agent.models()] == ["llava", "qwen2.5vl:7b"]
    assert opener.requests[0].full_url == "http://localhost:11434/v1/models"
    assert opener.requests[0].get_header("Authorization") is None


@pytest.mark.parametrize("config, message", [
    (RemoteConfig("anthropic", "claude-opus-5-5"), "API key"),
    (RemoteConfig("openai-compatible", "m", base_url="http://example.com/v1"), "https"),
    (RemoteConfig("ollama", ""), "model"),
    (RemoteConfig("gemini", "m"), "Unknown"),
])
def test_impossible_settings_are_refused_before_sending(config, message):
    with pytest.raises(ValueError, match=message):
        RemoteAgent(config)


def test_the_key_is_never_in_the_status():
    agent = RemoteAgent(RemoteConfig("openai", "m", api_key="sk-secret"), opener=FakeOpener({}))
    assert "sk-secret" not in json.dumps(agent.status())
    assert agent.need_bytes == 0 and agent.uses_accelerator is False


def test_engine_switches_to_a_service_and_back(make_engine, client_for, monkeypatch):
    monkeypatch.setattr(remote_agent.RemoteAgent, "test", lambda self: None)
    client = client_for(make_engine(), token="t")
    headers = {"Authorization": "Bearer t"}
    status = client.post("/admin/prompt-assistant/provider", headers=headers,
                         json={"kind": "openai", "model": "m", "api_key": "sk-secret", "test": True}).json()
    assert status["provider"] == {"kind": "openai", "model": "m", "base_url": "https://api.openai.com/v1"}
    assert status["installed"] is True and "sk-secret" not in json.dumps(status)
    status = client.post("/admin/prompt-assistant/provider", headers=headers, json={"kind": "local"}).json()
    assert status["provider"] == {"kind": "local"}
    bad = client.post("/admin/prompt-assistant/provider", headers=headers, json={"kind": "anthropic", "model": "x"})
    assert bad.status_code == 422


def test_chatgpt_lists_newest_image_models_first_and_leaves_out_the_rest():
    ids = [("gpt-9", 300), ("gpt-9-mini", 290), ("gpt-9-2026-10-01", 299), ("gpt-8", 200), ("gpt-7", 100),
           ("text-embedding-3-large", 400), ("gpt-9-realtime", 305), ("gpt-3.5-turbo", 50), ("gpt-4", 40),
           ("o3-mini", 60), ("gpt-9-codex", 310), ("gpt-image-2", 320), ("gpt-9-chat-latest", 301)]
    opener = FakeOpener({"data": [{"id": i, "created": c} for i, c in ids]})
    models = RemoteAgent(RemoteConfig("openai", "-", api_key="sk"), opener=opener).models()

    assert [m["id"] for m in models if m["recommended"]] == ["gpt-9", "gpt-9-mini", "gpt-8"]
    assert [m["id"] for m in models if not m["recommended"]] == ["gpt-9-chat-latest", "gpt-9-2026-10-01", "gpt-7"]


def test_claude_lists_live_models_newest_of_each_line_first():
    from types import SimpleNamespace as NS

    def info(model_id, name, line, images=True, lifecycle="active"):
        caps = NS(image_input=NS(supported=images))
        return NS(id=model_id, display_name=name, line=line, capabilities=caps, lifecycle=lifecycle)

    listed = [info("claude-haiku-6", "Claude Haiku 6", "haiku"), info("claude-opus-6", "Claude Opus 6", "opus"),
              info("claude-opus-5-5", "Claude Opus 5.5", "opus"), info("claude-sonnet-5-5", "Claude Sonnet 5.5", "sonnet"),
              info("claude-text-1", "Text only", None, images=False),
              info("claude-old", "Old", "sonnet", lifecycle="deprecated")]
    client = NS(models=NS(list=lambda limit: iter(listed)))
    models = RemoteAgent(RemoteConfig("anthropic", "-", api_key="sk"), client=client).models()

    assert [(m["id"], m["recommended"]) for m in models] == [
        ("claude-opus-6", True), ("claude-sonnet-5-5", True), ("claude-haiku-6", True), ("claude-opus-5-5", False)]
    assert models[0]["name"] == "Claude Opus 6" and models[0]["line"] == "opus"
