import json
import urllib.error

import pytest

from dmlx.prompting.compiler import CompiledPrompt
from dmlx.prompting import PassthroughPromptCompiler
from dmlx.prompting.openai_compatible import OpenAICompatiblePromptProvider
from dmlx.workflow import PromptSpec


def test_passthrough_preserves_an_ordinary_prompt():
    result = PassthroughPromptCompiler().compile("A red fox in falling snow")
    assert result.positive == "A red fox in falling snow"
    assert result.negative is None


def test_passthrough_splits_negative_line():
    result = PromptSpec.from_dict(
        {"user": "A red fox\nNegative: text, blur", "positive": "A red fox\nNegative: text, blur"}
    )
    assert result.positive == "A red fox"
    assert result.negative == "text, blur"


def test_passthrough_splits_negative_flag():
    result = PassthroughPromptCompiler().compile("A red fox --negative text, blur")
    assert result.positive == "A red fox"
    assert result.negative == "text, blur"


def test_openai_prompt_provider_validates_output(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            content = {"positive": "A detailed red fox in snow", "negative": None}
            return json.dumps({"choices": [{"message": {"content": json.dumps(content)}}]}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    provider = OpenAICompatiblePromptProvider("http://localhost:8000/v1", "prompt-model")
    result = provider.enhance("a fox", supports_negative=False)
    assert result.positive == "A detailed red fox in snow"
    assert result.compiler == "openai-compatible:prompt-model:prompt-v1"


@pytest.mark.parametrize("failure", ["timeout", "refusal"])
def test_prompt_provider_does_not_repeat_transport_failures_or_refusals(monkeypatch, failure):
    calls = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"refusal": "no", "content": None}}]}).encode()

    def open_url(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if failure == "timeout":
            raise TimeoutError("timed out")
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", open_url)
    with pytest.raises(Exception):
        OpenAICompatiblePromptProvider("http://localhost:8000/v1", "prompt-model").enhance(
            "a fox", supports_negative=False)
    assert calls == 1


class FakePromptProvider:
    name = "fake-prompt"

    def enhance(self, text, *, supports_negative):
        assert not supports_negative
        return CompiledPrompt(f"Enhanced: {text}", None, self.name)


def test_prompt_compile_endpoint_is_model_aware(make_engine, client_for):
    engine = make_engine()
    engine.prompt_provider = FakePromptProvider()
    response = client_for(engine).post(
        "/v1/prompts/compile",
        json={"prompt": "a fox", "operation": "image.generate", "model": "fake-a"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "user": "a fox",
        "positive": "Enhanced: a fox",
        "negative": None,
        "compiler": "fake-prompt",
        "negative_supported": False,
    }
