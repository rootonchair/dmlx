import pytest

from dmlx.intelligence.base import IntelligenceError
from dmlx.prompting.local_vlm import LocalVLMPromptProvider, parse_reply


class _Stub:
    """Counts loads and replies with a fixed rewrite; records what the VLM was shown."""

    def __init__(self, reply='{"positive": "A detailed description of the edited card.", "negative": null}'):
        self.reply = reply
        self.loads = 0
        self.calls = []

    def loader(self, _path):
        self.loads += 1
        return object(), object()

    def generator(self, _model, _processor, messages, images):
        self.calls.append((messages, images))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _install(engine, tmp_path, stub):
    (tmp_path / "vlm").mkdir(exist_ok=True)
    (tmp_path / "vlm" / "model.safetensors").write_bytes(b"x" * 1024)
    engine.prompt_assistant = LocalVLMPromptProvider(tmp_path / "vlm", loader=stub.loader, generator=stub.generator)
    return engine.prompt_assistant


def _generate(client, prompt="a card"):
    body = client.post("/v1/images/generations", json={"prompt": prompt, "model": "fake-a", "size": "512x512"}).json()
    return body["data"][0]["result_id"]


def _edit(client, source, prompt="remove all text", **prompt_fields):
    return client.post("/v1/workflows", json={"steps": [{
        "operation": "image.edit", "executor": "fake-a",
        "inputs": [{"id": source, "media_type": "image", "role": "source", "order": 0}],
        "prompt": {"user": prompt, "positive": prompt, **prompt_fields},
    }]})


def test_every_edit_is_rewritten_with_the_image_and_the_assistant_is_released(make_engine, client_for, tmp_path):
    engine = make_engine()
    client = client_for(engine)
    source = _generate(client)
    stub = _Stub()
    assistant = _install(engine, tmp_path, stub)

    response = _edit(client, source)

    assert response.status_code == 200, response.text
    meta = client.get(f"/v1/results/{response.json()['data'][0]['result_id']}/meta").json()
    assert meta["prompt"] == "remove all text"  # the user's words stay the job's prompt
    assert meta["parameters"]["rewritten_prompt"] == "A detailed description of the edited card."
    assert meta["parameters"]["prompt_assistant"] == assistant.name
    messages, images = stub.calls[0]
    assert messages[1]["content"] == "remove all text" and len(images) == 1
    assert not assistant.resident
    assert "prompt_assistant" not in {r["component"] for r in engine.memory.resident_components()}


def test_repeating_an_edit_reuses_the_rewrite_without_loading(make_engine, client_for, tmp_path):
    engine = make_engine()
    client = client_for(engine)
    source = _generate(client)
    stub = _Stub()
    _install(engine, tmp_path, stub)

    assert _edit(client, source).status_code == 200
    assert _edit(client, source).status_code == 200

    assert stub.loads == 1 and len(stub.calls) == 1


def test_an_enhanced_prompt_and_plain_generations_are_left_alone(make_engine, client_for, tmp_path):
    engine = make_engine()
    client = client_for(engine)
    source = _generate(client)
    stub = _Stub()
    _install(engine, tmp_path, stub)

    accepted = _edit(client, source, "remove all text", positive="Already rewritten.", compiler="openai-compatible:x")
    assert accepted.status_code == 200
    _generate(client, "a boat")

    assert stub.calls == []


def test_auto_mode_rewrites_new_images_too(make_engine, client_for, tmp_path):
    engine = make_engine()
    client = client_for(engine)
    stub = _Stub('{"positive": "A small wooden rowboat at golden hour.", "negative": null}')
    _install(engine, tmp_path, stub)

    response = client.post("/v1/workflows", json={"steps": [{
        "operation": "image.generate", "executor": "fake-a",
        "prompt": {"user": "a boat", "positive": "a boat", "enhancement_mode": "auto"},
        "parameters": {"width": 512, "height": 512},
    }]})

    assert response.status_code == 200, response.text
    meta = client.get(f"/v1/results/{response.json()['data'][0]['result_id']}/meta").json()
    assert meta["parameters"]["rewritten_prompt"] == "A small wooden rowboat at golden hour."
    assert stub.calls[0][1] == []  # nothing to look at for a new image


def test_a_failed_rewrite_runs_the_users_words(make_engine, client_for, tmp_path):
    engine = make_engine()
    client = client_for(engine)
    source = _generate(client)
    assistant = _install(engine, tmp_path, _Stub(RuntimeError("out of tokens")))

    response = _edit(client, source)

    assert response.status_code == 200, response.text
    meta = client.get(f"/v1/results/{response.json()['data'][0]['result_id']}/meta").json()
    assert "rewritten_prompt" not in meta["parameters"]
    assert response.json()["dmlx"]["rewrite"]["skipped"] == "prompt_provider_unavailable"
    assert not assistant.resident


def test_enhance_button_shows_the_assistant_the_edited_image(make_engine, client_for, tmp_path):
    engine = make_engine()
    client = client_for(engine)
    source = _generate(client)
    stub = _Stub()
    _install(engine, tmp_path, stub)

    response = client.post("/v1/prompts/compile", json={
        "prompt": "remove all text", "operation": "image.edit", "model": "fake-a", "image_ids": [source]})

    assert response.status_code == 200, response.text
    assert response.json()["positive"] == "A detailed description of the edited card."
    assert len(stub.calls[0][1]) == 1


def test_without_the_assistant_edits_run_as_written(make_engine, client_for):
    engine = make_engine()
    client = client_for(engine)
    assert engine.prompt_assistant is None
    response = _edit(client, _generate(client))
    meta = client.get(f"/v1/results/{response.json()['data'][0]['result_id']}/meta").json()
    assert meta["prompt"] == "remove all text" and "rewritten_prompt" not in meta["parameters"]


@pytest.mark.parametrize(("reply", "expected"), [
    ('thinking... {"positive": "first"} then {"positive": "final", "negative": "blur"}', ("final", "blur")),
    ('{"positive": "only positive", "negative": ""}', ("only positive", None)),
    ("A plain paragraph without JSON.", ("A plain paragraph without JSON.", None)),
])
def test_parse_reply(reply, expected):
    assert parse_reply(reply, supports_negative=True) == expected


def test_parse_reply_drops_unsupported_negative_and_rejects_empty():
    assert parse_reply('{"positive": "p", "negative": "n"}', supports_negative=False) == ("p", None)
    with pytest.raises(IntelligenceError):
        parse_reply('{"positive": ""}', supports_negative=True)
