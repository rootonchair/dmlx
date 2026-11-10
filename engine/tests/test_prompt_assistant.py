import pytest

from dmlx.intelligence.base import IntelligenceError
from dmlx.prompting.local_vlm import MODELS as AGENT_MODELS
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


# -- the agent models' file lists (test round 5 of task 19) -------------------------------------------------
# An agent counts as installed only when every name in `AgentModel.files` is in its snapshot. A new agent
# without the list never reads as installed; a revision bump that renames or adds a file reads wrong. Each
# entry here is the revision and its file list as checked against Hugging Face's listing at that commit
# (all files but README.md and .gitattributes). Changing a revision or a list without checking it again
# fails here: check the new listing (huggingface_hub `model_info(repo, revision=...).siblings`), fix `files`,
# then write the new fingerprint below.
_CHECKED_AGENT_FILES = {
    "qwen3-vl-8b-instruct-4bit": ("defcdea7cc7a4b0858fea563cbbce171d328e457", "4842269d5cbd0e69"),
    "qwen3.5-9b-mlx-4bit": ("938d8919941c6e7efd3c7150eff7fe9d12afa631", "0ccf60776682b624"),
    "ministral-3-3b-instruct-2512-4bit": ("a962dcb09eee4169c890e544c9eb938f1113fdee", "59c3d2f24e382a7a"),
    "qwen3-vl-4b-instruct-4bit": ("2fd8dacbdb8f1e54b8c005f081ec5bf79c56376b", "8fbe990d66cf093e"),
}


def _files_fingerprint(files) -> str:
    import hashlib

    return hashlib.sha256("\n".join(sorted(files)).encode()).hexdigest()[:16]


def _weight_shards_complete(files) -> bool:
    """One `model.safetensors`, or every `model-0000k-of-0000N.safetensors` for k = 1..N with its index."""
    import re

    weights = [f for f in files if f.endswith(".safetensors")]
    if not weights:
        return False
    shards = [re.fullmatch(r"model-(\d+)-of-(\d+)\.safetensors", f) for f in weights if f != "model.safetensors"]
    if not shards:
        return True
    if not all(shards) or "model.safetensors.index.json" not in files:
        return False
    totals = {int(m.group(2)) for m in shards}
    if len(totals) != 1:
        return False
    (total,) = totals
    return sorted(int(m.group(1)) for m in shards) == list(range(1, total + 1))


@pytest.mark.parametrize("model_id", sorted(AGENT_MODELS))
def test_every_agent_has_a_pinned_revision_and_a_complete_file_list(model_id):
    import re

    from dmlx.prompting.local_vlm import MODELS

    model = MODELS[model_id]
    assert re.fullmatch(r"[0-9a-f]{40}", model.revision), f"{model_id}: pin a full commit, not {model.revision!r}"
    files = set(model.files)
    assert files, f"{model_id} has no `files`: it would download but never read as installed"
    assert len(files) == len(model.files), f"{model_id} lists a file twice"
    assert "config.json" in files
    assert "tokenizer_config.json" in files and ({"tokenizer.json", "tekken.json"} & files)
    assert _weight_shards_complete(files), f"{model_id}: the weight files are not a complete set: {sorted(files)}"
    assert not files & {"README.md", ".gitattributes"}  # never needed, and not counted


@pytest.mark.parametrize("model_id", sorted(AGENT_MODELS))
def test_an_agents_revision_and_files_were_checked_together(model_id):
    from dmlx.prompting.local_vlm import MODELS

    model = MODELS[model_id]
    assert model_id in _CHECKED_AGENT_FILES, (
        f"New agent {model_id}: check its `files` against Hugging Face's listing at {model.revision}, then add "
        f"({model.revision!r}, {_files_fingerprint(model.files)!r}) to _CHECKED_AGENT_FILES.")
    revision, fingerprint = _CHECKED_AGENT_FILES[model_id]
    assert model.revision == revision, (
        f"{model_id} moved to revision {model.revision} (checked: {revision}). Its files may have changed: check "
        f"`files` against Hugging Face's listing at the new commit, then update _CHECKED_AGENT_FILES.")
    assert _files_fingerprint(model.files) == fingerprint, (
        f"{model_id}'s `files` changed but its revision did not: check the list against Hugging Face's listing at "
        f"{revision}, then write {_files_fingerprint(model.files)!r} into _CHECKED_AGENT_FILES.")


def test_the_file_list_checks_catch_a_missing_shard_or_an_unchecked_revision(monkeypatch):
    import dataclasses

    from dmlx.prompting import local_vlm

    assert not _weight_shards_complete(("config.json", "model-00001-of-00002.safetensors",
                                        "model.safetensors.index.json"))
    assert not _weight_shards_complete(("config.json", "model-00001-of-00002.safetensors",
                                        "model-00002-of-00002.safetensors"))  # no index
    assert not _weight_shards_complete(("config.json",))
    model_id = "ministral-3-3b-instruct-2512-4bit"
    bumped = dataclasses.replace(local_vlm.MODELS[model_id], revision="0" * 40)
    monkeypatch.setitem(local_vlm.MODELS, model_id, bumped)
    with pytest.raises(AssertionError, match="moved to revision"):
        test_an_agents_revision_and_files_were_checked_together(model_id)
    monkeypatch.setitem(local_vlm.MODELS, model_id, dataclasses.replace(bumped, files=()))
    with pytest.raises(AssertionError, match="no `files`"):
        test_every_agent_has_a_pinned_revision_and_a_complete_file_list(model_id)
