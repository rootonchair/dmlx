import json

from dmlx.prompting import agent
from dmlx.prompting.local_vlm import LocalVLMPromptProvider

CANDIDATES = ["image.edit", "image.generate"]
STATE = {"message": "more", "current_result_id": "img_2", "attachments": [], "history": [
    {"kind": "user", "text": "a teapot", "input_ids": []},
    {"kind": "image", "text": "a teapot", "image_number": 1, "operation": "image.generate",
     "result_id": "img_1", "input_ids": []},
    {"kind": "user", "text": "make it blue", "input_ids": []},
    {"kind": "image", "text": "make it blue", "image_number": 2, "operation": "image.edit",
     "result_id": "img_2", "input_ids": ["img_1"]},
]}
ALLOWED = {"img_1", "img_2"}


def check(reply, state=STATE):
    return agent.validate(json.dumps(reply) if isinstance(reply, dict) else reply,
                          state=state, candidates=CANDIDATES, allowed_ids=ALLOWED, provider="agent:test")


def test_short_intent_labels_are_normalized():
    plan = check({"intent": "edit", "inputs": ["img_2"], "prompt": "The teapot is navy.", "ask": None})
    assert plan.intent == "image.edit" and plan.inputs == ("img_2",) and plan.prompt == "The teapot is navy."


def test_invented_ids_and_unknown_workflows_are_rejected():
    assert check({"intent": "image.edit", "inputs": ["img_9"], "prompt": "x"}) is None
    assert check({"intent": "video.generate", "inputs": [], "prompt": "x"}) is None
    assert check("no json here") is None


def test_a_new_image_takes_no_inputs():
    assert check({"intent": "image.generate", "inputs": ["img_2"], "prompt": "x"}) is None
    plan = check({"intent": "image.generate", "inputs": [], "prompt": "A cat in space."})
    assert plan.intent == "image.generate" and plan.inputs == ()


def test_an_edit_without_inputs_edits_the_current_image():
    assert check({"intent": "image.edit", "inputs": [], "prompt": "x"}).inputs == ("img_2",)


def test_explicit_attachments_are_always_used_first():
    state = {**STATE, "attachments": ["up_1"]}
    allowed = ALLOWED | {"up_1"}
    plan = agent.validate(json.dumps({"intent": "image.edit", "inputs": ["img_1"], "prompt": "x"}),
                          state=state, candidates=CANDIDATES, allowed_ids=allowed, provider="agent:test")
    assert plan.inputs == ("up_1", "img_1")
    # A new image with attachments uses them: "a dragon in the style of this".
    styled = agent.validate(json.dumps({"intent": "image.generate", "inputs": [], "prompt": "x"}),
                            state=state, candidates=CANDIDATES, allowed_ids=allowed, provider="agent:test")
    assert styled.intent == "image.edit" and styled.inputs == ("up_1",)


def test_edit_and_references_fields_give_the_edited_image_first():
    plan = check({"intent": "image.edit", "edit": "img_2", "references": ["img_1"], "prompt": "x"})
    assert plan.inputs == ("img_2", "img_1")
    assert check({"intent": "image.edit", "edit": "img_2", "references": "img_1", "prompt": "x"}) is None


def test_a_clarifying_question_is_passed_on():
    plan = check({"intent": "ambiguous", "inputs": [], "prompt": None, "ask": "Edit image 2 or make a new one?"})
    assert plan.as_decision()["ask"] == "Edit image 2 or make a new one?"
    assert plan.as_decision()["intent"] == "ambiguous"


def test_image_labels_are_resolved_to_ids_and_inventions_rejected():
    assert check({"intent": "image.edit", "edit": "<image1>", "prompt": "x"}).inputs == ("img_2",)
    assert check({"intent": "image.edit", "edit": "image 1", "prompt": "x"}).inputs == ("img_1",)
    assert check({"intent": "image.edit", "edit": "2", "references": ["1"], "prompt": "x"}).inputs == ("img_2", "img_1")
    assert check({"intent": "image.edit", "edit": "image 7", "prompt": "x"}) is None


def test_attachments_replace_the_current_image_in_view():
    assert agent.shown_images({**STATE, "attachments": ["up_1"]}) == ["up_1"]
    assert agent.shown_images(STATE) == ["img_2"]


def test_the_model_reads_the_chat_and_sees_attachments_then_the_current_image():
    shown = agent.shown_images(STATE)
    body = agent.messages(STATE, {"image.edit": "edit"}, shown)[1]["content"]
    assert "image 1 id=img_1 (generated) from prompt: a teapot" in body
    assert "image 2 id=img_2 (image.edit of img_1)" in body
    assert body.endswith("NEW message: more")


# -- through the engine's routing ---------------------------------------------------------
class _Model:
    """Replies in turn (the last one repeats): the plan, then the description of an edit."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def install(self, engine, tmp_path):
        (tmp_path / "vlm").mkdir(exist_ok=True)
        (tmp_path / "vlm" / "model.safetensors").write_bytes(b"x")
        engine.prompt_assistant = LocalVLMPromptProvider(
            tmp_path / "vlm", loader=lambda _p: (object(), object()), generator=self.generate)

    def generate(self, _model, _processor, messages, images):
        self.calls.append((messages, images))
        reply = self.replies[min(len(self.calls), len(self.replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply


def _route(client, source, plan=True, **extra):
    body = {"message": "more", "current_result_id": source, "attachments": [], "history": [
        {"sequence": 0, "kind": "image", "text": "a card", "image_number": 1, "operation": "image.generate",
         "result_id": source, "input_ids": []}], **extra}
    if plan:
        body["plan"] = True
    return client.post("/v1/workflows/route", json=body).json()


def _source(client):
    body = client.post("/v1/images/generations", json={"prompt": "a card", "model": "fake-a", "size": "512x512"}).json()
    return body["data"][0]["result_id"]


def test_a_sent_message_is_planned_by_the_agent(make_engine, client_for, tmp_path):
    engine = make_engine()
    client = client_for(engine)
    source = _source(client)
    model = _Model(json.dumps({"intent": "image.edit", "edit": source, "prompt": "Make the card bluer.", "ask": None}),
                   json.dumps({"positive": "The same card, its colors shifted to deep blue.", "negative": None}))
    model.install(engine, tmp_path)

    decision = _route(client, source)

    assert decision["intent"] == "image.edit" and decision["input_result_ids"] == [source]
    assert decision["instruction"] == "Make the card bluer."
    assert decision["prompt"] == "The same card, its colors shifted to deep blue."
    assert decision["provider"].startswith("agent:")
    assert len(model.calls) == 2 and len(model.calls[0][1]) == 1  # looked at the image, then described the edit


def test_a_failed_description_keeps_the_agents_instruction(make_engine, client_for, tmp_path):
    engine = make_engine()
    client = client_for(engine)
    source = _source(client)
    _Model(json.dumps({"intent": "image.edit", "edit": source, "prompt": "Make the card bluer."}),
           RuntimeError("description failed")).install(engine, tmp_path)

    decision = _route(client, source)

    assert decision["provider"].startswith("agent:") and decision["prompt"] == "Make the card bluer."


def test_the_typing_preview_never_runs_the_agent(make_engine, client_for, tmp_path):
    engine = make_engine()
    client = client_for(engine)
    source = _source(client)
    model = _Model("{}")
    model.install(engine, tmp_path)

    decision = _route(client, source, plan=False)

    assert model.calls == [] and "prompt" not in decision


def test_a_bad_or_failed_answer_falls_back_to_the_router(make_engine, client_for, tmp_path):
    engine = make_engine()
    client = client_for(engine)
    source = _source(client)
    for reply in ('{"intent": "image.edit", "inputs": ["img_invented"], "prompt": "x"}', RuntimeError("boom")):
        _Model(reply).install(engine, tmp_path)
        decision = _route(client, source)
        assert not decision["provider"].startswith("agent:") and "prompt" not in decision
        assert not engine.prompt_assistant.resident or engine.memory.total >= 24 * 1024**3


def test_an_image_job_always_unloads_the_assistant_first(make_engine, client_for, tmp_path):
    engine = make_engine()
    client = client_for(engine)
    source = _source(client)
    model = _Model(json.dumps({"intent": "image.edit", "inputs": [source], "prompt": "The card, bluer."}))
    model.install(engine, tmp_path)
    engine.memory.total = 32 * 1024**3  # stays loaded between messages here
    _route(client, source)
    assert engine.prompt_assistant.resident

    _source(client)

    assert not engine.prompt_assistant.resident


def test_a_planned_prompt_runs_once_and_keeps_the_users_words(make_engine, client_for, tmp_path):
    engine = make_engine()
    client = client_for(engine)
    source = _source(client)
    model = _Model("{}")
    model.install(engine, tmp_path)

    response = client.post("/v1/workflows", json={"steps": [{
        "operation": "image.edit", "executor": "fake-a",
        "inputs": [{"id": source, "media_type": "image", "role": "source", "order": 0}],
        "prompt": {"user": "more", "positive": "The card, much bluer.", "compiler": "agent:local-vlm"},
    }]})

    assert response.status_code == 200, response.text
    meta = client.get(f"/v1/results/{response.json()['data'][0]['result_id']}/meta").json()
    assert meta["prompt"] == "more"
    assert meta["parameters"]["rewritten_prompt"] == "The card, much bluer."
    assert model.calls == []  # not rewritten a second time


def test_agent_eval_dataset_is_well_formed():
    from pathlib import Path

    from dmlx.prompting.agent_eval import load, state_for

    cases = load(Path(__file__).parents[1] / "evals" / "agent-routing.jsonl")
    assert len(cases) >= 30 and len({c["id"] for c in cases}) == len(cases)
    for case in cases:
        state, ids = state_for(case)
        assert set(case["labels"]) <= {"image.generate", "image.edit", "ambiguous"}, case["id"]
        for option in case["inputs"]:
            assert all(name in ids for name in option), case["id"]
        assert all(rid in ids.values() for rid in state["attachments"]), case["id"]


def test_a_loaded_assistant_only_needs_room_for_its_working_memory(tmp_path):
    from dmlx.prompting.local_vlm import ACTIVATION_BYTES

    (tmp_path / "model.safetensors").write_bytes(b"x" * 4096)
    assistant = LocalVLMPromptProvider(tmp_path, loader=lambda _p: (object(), object()),
                                       generator=lambda *_a: '{"positive": "p"}')
    assert assistant.need_bytes == ACTIVATION_BYTES + 4096
    assistant.chat([], [])
    assert assistant.need_bytes == ACTIVATION_BYTES
