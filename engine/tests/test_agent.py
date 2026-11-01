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


def test_an_edit_that_names_no_image_is_sent_back_with_the_reason():
    assert check({"intent": "image.edit", "inputs": [], "prompt": "x"}) is None
    problem = agent.check(json.dumps({"intent": "image.edit", "edit": None, "prompt": "x"}), state=STATE,
                          candidates=CANDIDATES, allowed_ids=ALLOWED, provider="agent:test")
    assert problem == 'image.edit needs the image to change in "edit"'
    model = _Replies('{"intent": "image.edit", "edit": null, "references": [], "prompt": "Bluer."}',
                     '{"intent": "image.edit", "edit": "image 2", "references": [], "prompt": "Bluer."}')
    result = agent.plan(model, STATE, {"image.edit": "e", "image.generate": "g"}, image_path=lambda rid: f"/{rid}.png")
    assert result.inputs == ("img_2",) and len(model.calls) == 2
    assert "Your last answer could not be used: image.edit needs the image" in model.calls[1][0][1]["content"]


def test_the_agents_choice_of_images_is_kept():
    state = {**STATE, "attachments": ["up_1"]}
    allowed = ALLOWED | {"up_1"}
    plan = agent.validate(json.dumps({"intent": "image.edit", "edit": "img_1", "references": ["up_1"], "prompt": "x"}),
                          state=state, candidates=CANDIDATES, allowed_ids=allowed, provider="agent:test")
    assert plan.inputs == ("img_1", "up_1")  # not reordered: it chose what to edit
    new = agent.validate(json.dumps({"intent": "image.generate", "inputs": [], "prompt": "x"}),
                         state=state, candidates=CANDIDATES, allowed_ids=allowed, provider="agent:test")
    assert new.intent == "image.generate"  # not turned into an edit because something is attached


def test_edit_and_references_fields_give_the_edited_image_first():
    plan = check({"intent": "image.edit", "edit": "img_2", "references": ["img_1"], "prompt": "x"})
    assert plan.inputs == ("img_2", "img_1")
    assert check({"intent": "image.edit", "edit": "img_2", "references": "img_1", "prompt": "x"}) is None


def test_a_clarifying_question_is_passed_on():
    plan = check({"intent": "ambiguous", "inputs": [], "prompt": None, "ask": "Edit image 2 or make a new one?"})
    assert plan.as_decision()["ask"] == "Edit image 2 or make a new one?"
    assert plan.as_decision()["intent"] == "ambiguous"


def test_every_label_form_means_the_same_chat_image():
    # One image shown (image 2, the current one): no form may read "2" as "the second image shown".
    for label in ("image2", "<image2>", "image 2", "2", "Image 2", "img_2", "<img_2>"):
        assert check({"intent": "image.edit", "edit": label, "prompt": "x"}).inputs == ("img_2",), label
    for label in ("image1", "<image1>", "image 1", "1", "#1"):
        assert check({"intent": "image.edit", "edit": label, "prompt": "x"}).inputs == ("img_1",), label
    assert check({"intent": "image.edit", "edit": "2", "references": ["1"], "prompt": "x"}).inputs == ("img_2", "img_1")
    assert check({"intent": "image.edit", "edit": "image 7", "prompt": "x"}) is None


def test_labels_with_several_images_shown_follow_the_chat_numbers():
    state = {**STATE, "attachments": ["up_1", "up_2"]}
    allowed = ALLOWED | {"up_1", "up_2"}

    def inputs(reply):
        plan = agent.validate(json.dumps(reply), state=state, candidates=CANDIDATES, allowed_ids=allowed,
                              provider="agent:test")
        return plan.inputs if plan else None

    # Two attachments shown; "image 1" is still the chat's image 1, never "the first image shown".
    assert inputs({"intent": "image.edit", "edit": "<image1>", "prompt": "x"}) == ("img_1",)
    assert inputs({"intent": "image.edit", "edit": "attachment 2", "prompt": "x"}) == ("up_2",)
    assert inputs({"intent": "image.edit", "edit": "attached1", "prompt": "x"}) == ("up_1",)
    assert inputs({"intent": "image.edit", "edit": "attachment 3", "prompt": "x"}) is None


def test_a_number_the_chat_lacks_falls_back_to_the_images_shown():
    state = {"message": "make it red", "current_result_id": None, "history": [], "attachments": ["up_1", "up_2"]}
    plan = agent.validate(json.dumps({"intent": "image.edit", "edit": "<image2>", "prompt": "x"}), state=state,
                          candidates=CANDIDATES, allowed_ids={"up_1", "up_2"}, provider="agent:test")
    assert plan.inputs == ("up_2",)


def test_the_model_sees_one_numbering_for_the_images_in_view():
    body = agent.messages(STATE, {"image.edit": "edit"}, agent.shown_images(STATE))[1]["content"]
    assert "Images shown to you, in order: image 2 id=img_2" in body and "<image" not in body
    state = {**STATE, "attachments": ["up_1"]}
    body = agent.messages(state, {"image.edit": "edit"}, agent.shown_images(state))[1]["content"]
    assert "Images shown to you, in order: attachment 1 id=up_1" in body


def test_the_agent_starts_with_the_image_this_and_it_mean():
    assert agent.shown_images(STATE) == ["img_2"]
    assert agent.shown_images({**STATE, "attachments": ["up_1"]}) == ["up_1"]


class _Replies:
    """A stand-in model: replies in turn, recording what it was shown."""

    name = "test"

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, images):
        self.calls.append((messages, images))
        return self.replies[min(len(self.calls), len(self.replies)) - 1]

    def enhance(self, text, **_):
        from dmlx.prompting.compiler import CompiledPrompt
        return CompiledPrompt(f"described: {text}", None, "test")


def test_the_agent_looks_at_an_older_image_before_deciding():
    model = _Replies('{"look": ["image 1"]}',
                     '{"intent": "image.edit", "edit": "image 1", "references": [], "prompt": "Make the teapot red."}')
    result = agent.plan(model, {**STATE, "message": "make the first one red"}, {"image.edit": "e", "image.generate": "g"},
                        image_path=lambda rid: f"/{rid}.png")

    assert result.inputs == ("img_1",) and result.looked == ("img_1",)
    assert result.as_decision()["looked"] == ["img_1"]
    assert model.calls[0][1] == ["/img_2.png"]  # first only the latest image
    assert model.calls[1][1] == ["/img_2.png", ("/img_1.png", 384)]  # then the one it asked for, smaller
    assert "You asked to look at image 1 id=img_1" in model.calls[1][0][1]["content"]


def test_the_agent_answers_without_looking_when_the_latest_image_is_enough():
    model = _Replies('{"intent": "image.edit", "edit": "image 2", "references": [], "prompt": "More blue."}')
    result = agent.plan(model, STATE, {"image.edit": "e", "image.generate": "g"}, image_path=lambda rid: f"/{rid}.png")
    assert result.inputs == ("img_2",) and result.looked == () and len(model.calls) == 1


def test_looking_is_limited_and_then_the_agent_must_answer():
    model = _Replies('{"look": ["image 1"]}', '{"look": ["image 1"]}',
                     '{"intent": "image.edit", "edit": "image 1", "references": [], "prompt": "Red."}')
    result = agent.plan(model, STATE, {"image.edit": "e", "image.generate": "g"}, image_path=lambda rid: f"/{rid}.png")
    # Asking again for an image already shown ends the looking: the next turn must answer.
    assert len(model.calls) == 3 and result.inputs == ("img_1",)
    assert "You cannot look at more images now" in model.calls[2][0][1]["content"]
    model = _Replies('{"look": ["image 1"]}', '{"look": ["image 3"]}', '{"look": ["image 4"]}')
    state = {**STATE, "history": STATE["history"] + [
        {"kind": "image", "text": f"p{n}", "image_number": n, "operation": "image.generate", "result_id": f"img_{n}",
         "input_ids": []} for n in (3, 4)]}
    assert agent.plan(model, state, {"image.edit": "e", "image.generate": "g"}, image_path=lambda rid: f"/{rid}.png") is None
    assert len(model.calls) == 4  # two looks, then it must answer; a look then gets one retry, then gives up
    assert "You cannot look at more images now" in model.calls[2][0][1]["content"]


def test_a_long_chat_still_lists_every_image():
    history = []
    for n in range(1, 13):  # 24 chat lines, cut to the last 16
        history += [{"kind": "user", "text": f"picture {n}", "input_ids": []},
                    {"kind": "image", "text": f"picture {n}", "image_number": n, "operation": "image.generate",
                     "result_id": f"img_{n}", "input_ids": []}]
    state = {"message": "make image 1 red", "current_result_id": "img_12", "attachments": [], "history": history}
    body = agent.messages(state, {"image.edit": "edit"}, agent.shown_images(state))[1]["content"]
    assert "image 1 id=img_1: picture 1" in body and "from prompt: picture 1\n" not in body
    assert "Images shown to you, in order: image 12 id=img_12 (latest)\n" in body


def test_the_model_reads_the_chat_and_sees_attachments_then_the_current_image():
    shown = agent.shown_images(STATE)
    body = agent.messages(STATE, {"image.edit": "edit"}, shown)[1]["content"]
    assert "image 1 id=img_1 (generated) from prompt: a teapot" in body
    assert "image 2 id=img_2 (image.edit of img_1)" in body
    assert body.endswith("NEW message: more")


def test_the_question_the_app_asked_is_part_of_the_chat():
    state = {**STATE, "message": "the first one", "history": [*STATE["history"],
             {"kind": "user", "text": "make it red", "input_ids": []},
             {"kind": "notice", "text": "Which image do you mean?"}]}
    lines = agent.chat_lines(state)
    assert lines[-2:] == ["user: make it red", "app: Which image do you mean?"]
    body = agent.messages(state, {"image.edit": "edit"}, ["img_2"])[1]["content"]
    assert body.endswith('NEW message: make it red (asked "Which image do you mean?", the user answered: the first one)')


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


def test_tight_memory_never_skips_the_agent(make_engine, client_for, tmp_path):
    engine = make_engine()
    client = client_for(engine)
    source = _source(client)
    model = _Model(json.dumps({"intent": "image.edit", "edit": source, "prompt": "Make the card bluer.", "ask": None}),
                   json.dumps({"positive": "The same card, deep blue.", "negative": None}))
    model.install(engine, tmp_path)
    engine.memory.make_room = lambda _need: False

    decision = _route(client, source)

    assert decision["provider"].startswith("agent:") and decision["prompt"] == "The same card, deep blue."


def test_the_typing_preview_never_runs_the_agent(make_engine, client_for, tmp_path):
    engine = make_engine()
    client = client_for(engine)
    source = _source(client)
    model = _Model("{}")
    model.install(engine, tmp_path)

    decision = _route(client, source, plan=False)

    assert model.calls == [] and "prompt" not in decision


def test_the_first_message_is_planned_too(make_engine, client_for, tmp_path):
    engine = make_engine()
    client = client_for(engine)
    model = _Model(json.dumps({"intent": "image.generate", "edit": None, "references": [],
                               "prompt": "A glazed ceramic teapot on a wooden table, soft window light."}))
    model.install(engine, tmp_path)

    for body in ({"message": "a ceramic teapot", "attachments": [], "history": [], "plan": True},
                 {"message": "a ceramic teapot", "current_result_id": None, "current_prompt": None, "plan": True}):
        decision = client.post("/v1/workflows/route", json=body).json()

        assert decision["intent"] == "image.generate" and decision["provider"].startswith("agent:")
        assert decision["prompt"].startswith("A glazed ceramic teapot")
    assert len(model.calls) == 2 and model.calls[0][1] == []  # no image to show yet


def test_a_bad_or_failed_answer_asks_the_user_instead_of_guessing(make_engine, client_for, tmp_path, caplog):
    engine = make_engine()
    client = client_for(engine)
    source = _source(client)
    for reply, note in (('{"intent": "image.edit", "inputs": ["img_invented"], "prompt": "x"}', "rejected"),
                        (RuntimeError("boom"), "failed")):
        _Model(reply).install(engine, tmp_path)
        decision = _route(client, source)
        assert decision["intent"] == "ambiguous" and decision["planning"] == note and "current one" in decision["ask"]
        assert "prompt" not in decision and "input_result_ids" not in decision
        assert not engine.prompt_assistant.resident or engine.memory.total >= 24 * 1024**3
    assert "img_invented" in caplog.text  # the rejected reply is logged


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


def test_the_user_picks_the_agent_model(make_engine, client_for, tmp_path, monkeypatch):
    from dmlx.prompting import local_vlm

    engine = make_engine()
    client = client_for(engine)
    status = client.get("/admin/prompt-assistant").json()
    from dmlx import sysmem

    assert status["selected"] == local_vlm.default_model(sysmem.total_memory())
    assert {m["id"] for m in status["models"]} == set(local_vlm.MODELS)
    assert client.post("/admin/prompt-assistant/select", json={"model": "no-such-model"}).status_code == 404

    small = "qwen3-vl-4b-instruct-4bit"
    (tmp_path / "vlm4").mkdir()
    (tmp_path / "vlm4" / "model.safetensors").write_bytes(b"x")
    monkeypatch.setattr(local_vlm, "locate", lambda model_id=local_vlm.DEFAULT_MODEL:
                        tmp_path / "vlm4" if model_id == small else None)
    monkeypatch.setenv("DMLX_PROMPT_ASSISTANT", "on")
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a: object())
    status = client.post("/admin/prompt-assistant/select", json={"model": small}).json()

    assert status["selected"] == small and status["installed"]
    assert engine.prompt_assistant.name.startswith(f"local-vlm:{small}@")
    assert json.loads((tmp_path / "prompt-assistant.json").read_text()) == {"model": small}
    from dmlx.engine import selected_assistant_model
    assert selected_assistant_model() == small  # kept for the next launch


def test_smaller_macs_default_to_a_smaller_agent():
    from dmlx.prompting import local_vlm

    assert local_vlm.default_model(16 * 1024**3) == local_vlm.SMALL_MAC_MODEL
    assert local_vlm.default_model(24 * 1024**3) == local_vlm.DEFAULT_MODEL
    assert all(model.note for model in local_vlm.MODELS.values())


def _edit_model(source):
    return _Model(json.dumps({"intent": "image.edit", "edit": source, "prompt": "Make the card bluer.", "ask": None}),
                  json.dumps({"positive": "The same card, deep blue.", "negative": None}))


def test_a_message_sent_during_an_image_job_says_it_was_not_planned(make_engine, client_for, tmp_path):
    engine = make_engine()
    client = client_for(engine)
    source = _source(client)
    model = _edit_model(source)
    model.install(engine, tmp_path)
    engine.memory.total = 16 * 1024**3

    with engine._accelerator_lock:  # an image is being made
        decision = _route(client, source)

    assert decision["planning"] == "skipped_busy" and not decision["provider"].startswith("agent:")
    assert model.calls == []


def test_a_message_from_another_chat_waits_for_the_image_job(make_engine, client_for, tmp_path):
    import threading

    engine = make_engine()
    client = client_for(engine)
    source = _source(client)
    _edit_model(source).install(engine, tmp_path)
    engine.memory.total = 16 * 1024**3

    engine._accelerator_lock.acquire()
    threading.Timer(0.3, engine._accelerator_lock.release).start()  # the job finishes
    decision = _route(client, source, wait=True)

    assert decision["provider"].startswith("agent:") and decision["planning"] == "waited"
    assert decision["prompt"] == "The same card, deep blue."


def test_a_big_mac_plans_beside_the_image_job_without_freeing_its_memory(make_engine, client_for, tmp_path):
    engine = make_engine()
    client = client_for(engine)
    source = _source(client)
    _edit_model(source).install(engine, tmp_path)
    engine.memory.total = 64 * 1024**3
    freed = []
    engine.memory.make_room = lambda need: freed.append(need) or True

    with engine._accelerator_lock:  # still running when the plan is done
        decision = _route(client, source)

    assert decision["provider"].startswith("agent:") and decision["planning"] == "beside_job"
    assert freed == []


def test_the_agent_picks_an_offered_image_model_and_nothing_else():
    models = {"image.edit": [("fast-model", "Fast"), ("text-model", "Best at words")]}
    pick = lambda name: agent.validate(  # noqa: E731
        json.dumps({"intent": "image.edit", "edit": "img_2", "prompt": "x", "model": name}), state=STATE,
        candidates=CANDIDATES, allowed_ids=ALLOWED, provider="agent:test", models=models)
    assert pick("text-model").model == "text-model" and pick("text-model").as_decision()["model"] == "text-model"
    assert pick("made-up").model is None and pick(None).model is None  # the app's choice stands
    body = agent.messages(STATE, {"image.edit": "e"}, ["img_2"], models=models, free_gb=7.5)[1]["content"]
    assert "- image.edit: text-model (Best at words)" in body and "about 7.5 GB" in body


def test_auto_offers_the_agent_the_models_that_can_run_the_message(make_engine, client_for, tmp_path):
    engine = make_engine()
    client = client_for(engine)
    _source(client)
    model = _Model(json.dumps({"intent": "image.generate", "edit": None, "references": [],
                               "prompt": "A red fox in the snow.", "model": "fake-b"}))
    model.install(engine, tmp_path)

    # fake-a and fake-b can both make a new image; there is an image in the chat, so the agent plans.
    body = {"message": "a fox", "attachments": [], "history": [], "plan": True, "candidates": ["image.generate"]}
    decision = client.post("/v1/workflows/route", json={**body, "current_result_id": _source(client)}).json()
    assert decision.get("model") == "fake-b"
    assert "fake-a (" in model.calls[-1][0][1]["content"] and "fake-b (" in model.calls[-1][0][1]["content"]
    chosen = client.post("/v1/workflows/route", json={**body, "current_result_id": _source(client),
                                                       "selected_model": "fake-a"}).json()
    assert "model" not in chosen  # a model the user picked is never overridden


def test_the_agent_decides_whether_an_edit_is_described():
    reply = '{"intent": "image.edit", "edit": "image 2", "references": [], "prompt": "Make it black and white.", '
    plain = agent.plan(_Replies(reply + '"describe": false}'), STATE, {"image.edit": "e", "image.generate": "g"},
                       image_path=lambda rid: f"/{rid}.png")
    described = agent.plan(_Replies(reply + '"describe": true}'), STATE, {"image.edit": "e", "image.generate": "g"},
                           image_path=lambda rid: f"/{rid}.png")
    assert plain.prompt == "Make it black and white." and plain.instruction is None
    assert described.prompt == "described: Make it black and white."


def test_the_agent_can_read_the_whole_chat():
    history = []
    for n in range(1, 13):
        history += [{"kind": "user", "text": f"note {n}", "input_ids": []},
                    {"kind": "image", "text": f"picture {n}", "image_number": n, "operation": "image.generate",
                     "result_id": f"img_{n}", "input_ids": []}]
    state = {"message": "like my first note said", "current_result_id": "img_12", "attachments": [], "history": history}
    model = _Replies('{"read_chat": true}',
                     '{"intent": "image.edit", "edit": "image 12", "references": [], "prompt": "x", "describe": false}')
    result = agent.plan(model, state, {"image.edit": "e", "image.generate": "g"}, image_path=lambda rid: f"/{rid}.png")
    assert "user: note 1\n" not in model.calls[0][0][1]["content"]  # cut at first
    assert "Whole chat:" in model.calls[1][0][1]["content"] and "user: note 1\n" in model.calls[1][0][1]["content"]
    assert result.inputs == ("img_12",)
