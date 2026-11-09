"""Auto never picks, or keeps, an image model the memory plan will refuse when another one fits (task 17).

Seen in the app: after an Ideogram 4 poster, "make it warmer" on Auto planned image.edit with Qwen Image 2.1
(needs about 9.5 GB, 7.6 GB usable) while FLUX.2 Klein 4B and 9B, also installed, would fit.
"""
import json

from conftest import GB, fake_spec
from dmlx.prompting.local_vlm import LocalVLMPromptProvider


class _Agent:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def install(self, engine, tmp_path):
        (tmp_path / "vlm").mkdir(exist_ok=True)
        (tmp_path / "vlm" / "model.safetensors").write_bytes(b"x")
        engine.prompt_assistant = LocalVLMPromptProvider(
            tmp_path / "vlm", loader=lambda _p: (object(), object()), generator=self.generate)

    def generate(self, _model, _processor, messages, images):
        self.calls.append(messages)
        return self.reply


def _specs():
    # Klein-like models (small) and a Qwen-Image-like one (8 GB transformer, not auto-selected).
    klein4 = fake_spec("klein-4b", te=2.0, tr=2.0)
    klein9 = fake_spec("klein-9b", te=2.0, tr=4.0)
    qwen = fake_spec("qwen-image", te=4.0, tr=8.0)
    qwen.auto_select = False
    return [klein4, klein9, qwen]


def _source(client):
    body = client.post("/v1/images/generations", json={"prompt": "a poster", "model": "klein-4b",
                                                        "size": "512x512"}).json()
    return body["data"][0]["result_id"]


def _warmer(client, source, **extra):
    return client.post("/v1/workflows/route", json={
        "message": "make it warmer", "current_result_id": source, "attachments": [], "plan": True,
        "candidates": ["image.edit"], **extra}).json()


def _edit_reply(source, model):
    return json.dumps({"intent": "image.edit", "edit": source, "references": [], "prompt": "Warmer colours.",
                       "describe": False, "model": model})


def test_auto_does_not_offer_or_keep_a_model_that_will_not_fit(make_engine, client_for, tmp_path):
    engine = make_engine(_specs(), headroom_gb=7.6)
    client = client_for(engine)
    source = _source(client)
    engine.memory.unload_model("klein-4b")  # nothing of dMLX's resident, as in the app
    agent = _Agent(_edit_reply(source, "qwen-image"))
    agent.install(engine, tmp_path)

    decision = _warmer(client, source)

    assert decision["intent"] == "image.edit" and decision.get("model") != "qwen-image"
    offered = agent.calls[-1][1]["content"]
    assert "klein-4b (" in offered and "klein-9b (" in offered and "qwen-image (" not in offered
    # The job the app then sends (its usual Auto model, Klein 4B) runs.
    job = engine.submit(task="edit", prompt="Warmer colours.", model=decision.get("model") or "klein-4b",
                        image_ids=[source], size="512x512")
    job.finished.wait(10)
    assert job.status == "done"


def test_a_model_that_fits_is_still_offered_and_kept(make_engine, client_for, tmp_path):
    engine = make_engine(_specs(), headroom_gb=64)
    client = client_for(engine)
    source = _source(client)
    agent = _Agent(_edit_reply(source, "qwen-image"))
    agent.install(engine, tmp_path)

    # Qwen Image is not auto-selected, but it fits: the agent may still pick it (the user accepted its
    # license on download), and the pick is kept.
    assert _warmer(client, source)["model"] == "qwen-image"
    assert "qwen-image (" in agent.calls[-1][1]["content"]


def test_with_no_pick_a_default_that_will_not_fit_gives_way(make_engine, client_for, tmp_path):
    heavy = fake_spec("heavy", te=6.0, tr=6.0)  # first in the registry: dMLX's default here
    engine = make_engine([heavy, fake_spec("light")], headroom_gb=5.0)
    client = client_for(engine)
    source = client.post("/v1/images/generations", json={"prompt": "a card", "model": "light",
                                                          "size": "512x512"}).json()["data"][0]["result_id"]
    engine.memory.unload_model("light")

    decision = _warmer(client, source, plan=False)  # the rules and the router decide, no agent

    assert decision["intent"] == "image.edit" and decision["model"] == "light"


def test_a_model_the_user_picked_is_never_swapped_and_still_refused(make_engine, client_for, tmp_path):
    engine = make_engine(_specs(), headroom_gb=7.6)
    client = client_for(engine)
    source = _source(client)
    engine.memory.unload_model("klein-4b")
    _Agent(_edit_reply(source, None)).install(engine, tmp_path)

    decision = _warmer(client, source, selected_models={"image.edit": "qwen-image"})
    assert "model" not in decision

    job = engine.submit(task="edit", prompt="Warmer colours.", model="qwen-image", image_ids=[source],
                        size="512x512")
    job.finished.wait(10)
    assert job.status == "error" and job.error["code"] == "insufficient_memory"


def test_when_nothing_fits_the_choice_is_left_alone(make_engine, client_for, tmp_path):
    engine = make_engine(_specs())
    client = client_for(engine)
    source = _source(client)
    engine.memory.unload_model("klein-4b")
    engine.memory.headroom = lambda: 1 * 1024**3
    agent = _Agent(_edit_reply(source, "klein-9b"))
    agent.install(engine, tmp_path)

    assert _warmer(client, source)["model"] == "klein-9b"  # the job's refusal then says why


def _heavy_and_light():
    # dMLX's default (first in the registry) is the light one; "heavy" can edit too but needs 12 GB.
    light = fake_spec("light")
    heavy = fake_spec("heavy", te=6.0, tr=6.0)
    return [light, heavy]


def test_the_apps_own_auto_model_gives_way_when_it_will_not_fit(make_engine, client_for, tmp_path):
    # New is set to a heavy model that can edit, Edit on Auto: the app's Auto would run that one, not
    # dMLX's default, so the engine must check the app's model, sent as auto_models.
    engine = make_engine(_heavy_and_light(), headroom_gb=5.0)
    client = client_for(engine)
    source = client.post("/v1/images/generations", json={"prompt": "a card", "model": "light",
                                                          "size": "512x512"}).json()["data"][0]["result_id"]
    engine.memory.unload_model("light")

    decision = _warmer(client, source, plan=False, auto_models={"image.edit": "heavy"})

    assert decision["intent"] == "image.edit" and decision["model"] == "light"


def test_a_single_fitting_model_is_selected_though_the_agent_is_not_offered_it(make_engine, client_for, tmp_path):
    engine = make_engine(_heavy_and_light(), headroom_gb=5.0)
    client = client_for(engine)
    source = client.post("/v1/images/generations", json={"prompt": "a card", "model": "light",
                                                          "size": "512x512"}).json()["data"][0]["result_id"]
    engine.memory.unload_model("light")
    agent = _Agent(_edit_reply(source, None))
    agent.install(engine, tmp_path)

    decision = _warmer(client, source, auto_models={"image.edit": "heavy"})

    assert "light (" not in agent.calls[-1][1]["content"]  # one model fits: nothing to offer
    assert decision["intent"] == "image.edit" and decision["model"] == "light"


def test_the_apps_auto_model_that_fits_is_left_alone(make_engine, client_for, tmp_path):
    engine = make_engine(_specs(), headroom_gb=7.6)
    client = client_for(engine)
    source = _source(client)
    engine.memory.unload_model("klein-4b")

    decision = _warmer(client, source, plan=False, auto_models={"image.edit": "klein-9b"})

    assert "model" not in decision  # the app runs its klein-9b, which fits


def test_the_fit_check_uses_the_jobs_size(make_engine, client_for, tmp_path):
    big = fake_spec("big")  # dMLX's default here; working memory grows with the picture above 1 MP
    big.activation_gb_per_mp = 1.0
    engine = make_engine([big, fake_spec("light")], headroom_gb=5.0)
    client = client_for(engine)

    def new_image(**extra):
        return client.post("/v1/workflows/route", json={
            "message": "a poster for a jazz night", "attachments": [], "candidates": ["image.generate"],
            **extra}).json()

    assert "model" not in new_image()  # no size: 1024x1024, 1 GB working, fits
    assert "model" not in new_image(size="1216x832")
    assert new_image(size="2048x2048")["model"] == "light"  # 4.2 GB working: does not fit
    bad = client.post("/v1/workflows/route", json={"message": "a poster", "attachments": [], "size": "big"})
    assert bad.status_code == 422
    # An edit's source is scaled to 1 MP, whatever size a new image would have.
    source = client.post("/v1/images/generations", json={"prompt": "a card", "model": "light",
                                                          "size": "512x512"}).json()["data"][0]["result_id"]
    engine.memory.unload_model("light")
    assert "model" not in _warmer(client, source, plan=False, size="2048x2048")


def _new_image(client, **extra):
    return client.post("/v1/workflows/route", json={
        "message": "a poster for a jazz night", "attachments": [], "candidates": ["image.generate"],
        **extra}).json()


def _slow_format_model(model_id="ideo", prompt_format="ideogram-json", tr=1.0):
    # Like Ideogram 4, FIBO 2 or MOSS: auto-selectable, a prompt format, lighter than the default here.
    spec = fake_spec(model_id, tasks=("t2i",), te=1.0, tr=tr)
    spec.prompt_format = prompt_format
    return spec


def test_a_swap_does_not_pick_a_prompt_format_model_without_a_layout(make_engine, client_for, tmp_path):
    # The default won't fit; a plain model and a lighter prompt-format one both fit: the plain one runs.
    heavy = fake_spec("heavy", te=6.0, tr=6.0)
    engine = make_engine([heavy, fake_spec("light"), _slow_format_model()], headroom_gb=5.0)
    client = client_for(engine)

    assert _new_image(client)["model"] == "light"


def test_a_swap_keeps_the_refusal_when_only_a_prompt_format_model_fits(make_engine, client_for, tmp_path):
    heavy = fake_spec("heavy", te=6.0, tr=6.0)
    engine = make_engine([heavy, _slow_format_model("moss", "moss-layout")], headroom_gb=5.0)
    client = client_for(engine)

    assert "model" not in _new_image(client)  # the job's refusal then says why


def test_a_swap_does_not_pick_a_model_that_is_not_auto_selected(make_engine, client_for, tmp_path):
    heavy = fake_spec("heavy", te=6.0, tr=6.0)
    manual = fake_spec("manual")
    manual.auto_select = False
    engine = make_engine([heavy, manual], headroom_gb=5.0)
    client = client_for(engine)

    assert "model" not in _new_image(client)


def test_with_a_layout_a_swap_may_pick_a_prompt_format_model(make_engine, client_for, tmp_path):
    big = _slow_format_model("big-format", tr=8.0)
    engine = make_engine([big, _slow_format_model()], headroom_gb=5.0)
    client = client_for(engine)
    layout = [{"kind": "text", "label": "FRIDAY 9 PM", "box": [800, 600, 900, 900]}]

    # The layout step names the model that places best (both are ideogram-json, so the first); the fit
    # check then swaps it for the format model that fits.
    assert _new_image(client, layout=layout)["model"] == "ideo"


def test_a_one_workflow_send_without_the_agent_gets_the_swap(make_engine, client_for, tmp_path):
    # No agent, empty chat, a photo attached, "make it warmer": only image.edit is possible, so the app
    # asks for that workflow alone (candidates) and runs the model the engine names (test round 2).
    engine = make_engine(_heavy_and_light(), headroom_gb=5.0)
    client = client_for(engine)
    photo = client.post("/v1/images/generations", json={"prompt": "a photo", "model": "light",
                                                         "size": "512x512"}).json()["data"][0]["result_id"]
    engine.memory.unload_model("light")

    decision = client.post("/v1/workflows/route", json={
        "message": "make it warmer", "attachments": [photo], "candidates": ["image.edit"],
        "auto_models": {"image.edit": "heavy"}}).json()

    assert decision["intent"] == "image.edit" and decision["model"] == "light"


def test_the_hint_route_with_the_agent_names_the_model_that_fits(make_engine, client_for, tmp_path):
    # Agent installed, one workflow on Auto: the composer's hint asks the route (no plan) for that workflow.
    # With one fitting model the agent picks nothing, so the hint can name the swap (test round 3).
    engine = make_engine(_heavy_and_light(), headroom_gb=5.0)
    client = client_for(engine)
    photo = client.post("/v1/images/generations", json={"prompt": "a photo", "model": "light",
                                                         "size": "512x512"}).json()["data"][0]["result_id"]
    engine.memory.unload_model("light")
    agent = _Agent(_edit_reply(photo, None))
    agent.install(engine, tmp_path)

    decision = client.post("/v1/workflows/route", json={
        "message": "make it warmer", "attachments": [photo], "candidates": ["image.edit"],
        "auto_models": {"image.edit": "heavy"}}).json()

    assert decision["model"] == "light" and "agent_choices" not in decision
    assert agent.calls == []  # a hint never runs the agent


def test_the_hint_route_lists_the_models_the_agent_will_pick_from(make_engine, client_for, tmp_path):
    engine = make_engine(_specs(), headroom_gb=7.6)
    client = client_for(engine)
    source = _source(client)
    engine.memory.unload_model("klein-4b")

    without_agent = _warmer(client, source, plan=False, auto_models={"image.edit": "klein-9b"})
    _Agent(_edit_reply(source, None)).install(engine, tmp_path)
    with_agent = _warmer(client, source, plan=False, auto_models={"image.edit": "klein-9b"})
    picked = _warmer(client, source, plan=False, auto_models={"image.edit": "klein-9b"},
                     selected_models={"image.edit": "klein-9b"})

    assert "agent_choices" not in without_agent
    # Qwen Image would not fit: not offered. The app's Auto model comes first, as the hint names two.
    assert with_agent["agent_choices"] == ["klein-9b", "klein-4b"]
    assert "agent_choices" not in picked  # the user's model is not the agent's to change


def test_the_fit_check_counts_a_running_jobs_working_memory_as_coming_back(make_engine):
    # Test round 3: while a job runs, its working memory is not free, but a job routed now runs after it.
    engine = make_engine(_specs(), headroom_gb=None)
    memory = engine.memory
    klein9 = engine.registry.get("klein-9b")
    need = min(memory._needs(klein9, 0).values())
    gb = 1024**3
    memory.headroom = lambda: need + gb  # room for Klein 9B before the job
    running = memory.plan(engine.registry.get("klein-4b"), 3 * gb, strategy="sequential")
    memory.job_started(running, 3 * gb)
    memory.headroom = lambda: need - 2 * gb  # the job holds 3 GB of working memory now

    assert memory.could_fit(klein9, 0)  # it runs after the job, when the 3 GB are back
    memory.job_ended()
    assert not memory.could_fit(klein9, 0)  # no job: what is free now is all there is

    # Never more than the job could give back: now plus its planned working memory.
    memory.job_started(running, 1 * gb)
    assert not memory.could_fit(klein9, 0)
    memory.job_ended()


def test_a_finished_job_no_longer_counts(make_engine, client_for):
    engine = make_engine(_specs(), headroom_gb=7.6)
    client = client_for(engine)
    _source(client)
    assert engine.memory._job is None


def _generate_hint(client, **extra):
    return client.post("/v1/workflows/route", json={
        "message": "a poster for a jazz night", "attachments": [], "candidates": ["image.generate"],
        **extra}).json()


def _with_experimental_models():
    # dMLX's default (first) is heavy; a plain light model, and lighter ones Auto never runs by itself:
    # one not auto-selected (like Qwen Image 2.1) and a prompt-format one (like MOSS).
    manual = fake_spec("manual", te=0.5, tr=0.5)
    manual.auto_select = False
    return [fake_spec("heavy", te=6.0, tr=6.0), fake_spec("light"), manual,
            _slow_format_model("moss", "moss-layout", tr=0.5)]


def test_the_agent_is_offered_fitting_experimental_models_without_a_layout(make_engine, client_for, tmp_path):
    # Fix round 6: the agent's list is every downloaded model that can run the workflow and fits, also
    # the ones not auto-selected and the prompt-format ones (it picks by their notes). Fix round 5 had
    # filtered them out, so the agent could never pick Klein 9B, Qwen Image 2.1 or MOSS.
    engine = make_engine(_with_experimental_models(), headroom_gb=64)
    client = client_for(engine)
    for pick in ("manual", "moss"):
        agent = _Agent(json.dumps({"intent": "image.generate", "edit": None, "references": [],
                                   "prompt": "A jazz poster.", "describe": False, "model": pick}))
        agent.install(engine, tmp_path)
        assert _generate_hint(client, plan=True)["model"] == pick
        offered = agent.calls[-1][1]["content"]
        assert all(f"{model_id} (" in offered for model_id in ("heavy", "light", "manual", "moss"))

    # The hint names them too: the Auto model first, then the ones suited to Auto, then the rest.
    assert _generate_hint(client)["agent_choices"] == ["heavy", "light", "manual", "moss"]
    assert _generate_hint(client, auto_models={"image.generate": "light"})["agent_choices"] == [
        "light", "heavy", "manual", "moss"]


def test_a_swap_never_picks_an_experimental_model_the_agent_did_not_choose(make_engine, client_for, tmp_path):
    # The agent picks the default, which won't fit; "manual" and "moss" are lighter and fit, but a swap
    # keeps to models suited to Auto, so "light" runs. They are still offered to the agent.
    engine = make_engine(_with_experimental_models(), headroom_gb=5.0)
    client = client_for(engine)
    agent = _Agent(json.dumps({"intent": "image.generate", "edit": None, "references": [],
                               "prompt": "A jazz poster.", "describe": False, "model": "heavy"}))
    agent.install(engine, tmp_path)

    assert _generate_hint(client, plan=True)["model"] == "light"
    offered = agent.calls[-1][1]["content"]
    assert "manual (" in offered and "moss (" in offered and "heavy (" not in offered
    assert _generate_hint(client)["model"] == "light"  # no pick (the hint): the same swap


def test_with_a_layout_the_agent_is_offered_the_prompt_format_models(make_engine, client_for, tmp_path):
    specs = [fake_spec("klein-4b"), _slow_format_model("moss", "moss-layout"), _slow_format_model("ideo")]
    engine = make_engine(specs, headroom_gb=64)
    client = client_for(engine)
    _Agent(json.dumps({"intent": "image.generate"})).install(engine, tmp_path)
    layout = [{"kind": "text", "label": "FRIDAY 9 PM", "box": [800, 600, 900, 900]}]

    assert _generate_hint(client, layout=layout)["agent_choices"] == ["moss", "ideo"]


def _agent_unloaded_after_planning(make_engine, client_for, tmp_path, monkeypatch, total_gb=16):
    """A Mac under 24 GB, where the agent is unloaded right after it plans, with an agent the size of
    Ministral 3B (MLX frees 3 GB when it unloads; measured 2.75). The headroom drops while it runs and
    macOS hands its memory back a moment later."""
    import dmlx.engine as engine_module

    engine = make_engine(_specs(), headroom_gb=None)
    monkeypatch.setattr(engine_module, "_mlx_held",
                        lambda: 3 * GB if engine.prompt_assistant is not None and engine.prompt_assistant.resident else 0)
    engine.memory.total = total_gb * GB
    room = {"gb": 7.6}
    engine.memory.headroom = lambda: int(room["gb"] * GB)
    client = client_for(engine)
    source = _source(client)
    engine.memory.unload_model("klein-4b")

    class Agent(_Agent):
        def generate(self, *args):
            if not self.calls:  # just after it planned: its memory not handed back yet (7.50 -> 4.44 GB)
                room["gb"] = 3.5
            return super().generate(*args)

    agent = Agent(_edit_reply(source, "klein-9b"))
    agent.install(engine, tmp_path)
    engine.prompt_assistant.weights_bytes = int(2.5 * GB)
    return engine, client, source, agent, room


def test_an_agent_pick_is_kept_while_the_unloaded_agents_memory_comes_back(make_engine, client_for, tmp_path,
                                                                           monkeypatch):
    # Test round 5: on a 16 GB Mac the fit check read the headroom about 3 GB low right after planning and
    # swapped the agent's fitting pick (Qwen Image 2.1 there, Klein 9B here) to Klein 4B.
    engine, client, source, agent, room = _agent_unloaded_after_planning(make_engine, client_for, tmp_path, monkeypatch)

    decision = _warmer(client, source)

    assert "klein-9b (" in agent.calls[-1][1]["content"]  # offered at 7.6 GB
    assert not engine.prompt_assistant.resident
    assert decision.get("model") == "klein-9b"
    # The job's own plan is still the gate: it runs once the memory is back.
    room["gb"] = 7.6
    job = engine.submit(task="edit", prompt="Warmer colours.", model="klein-9b", image_ids=[source],
                        size="512x512")
    job.finished.wait(10)
    assert job.status == "done"


def test_the_returning_agent_memory_is_bounded_and_expires(make_engine, client_for, tmp_path, monkeypatch):
    engine, client, source, agent, room = _agent_unloaded_after_planning(make_engine, client_for, tmp_path, monkeypatch)
    _warmer(client, source)
    before, after, freed, _ = engine._agent_returning
    assert (before, after, freed) == (int(7.6 * GB), int(3.5 * GB), 3 * GB)
    assert engine._agent_memory_returning() == 3 * GB  # nothing back yet: what MLX freed

    # Never more than MLX freed, when something else took memory too ...
    room["gb"] = 1.0
    assert engine._agent_memory_returning() == 3 * GB
    # What has come back no longer counts ...
    room["gb"] = 5.0
    assert engine._agent_memory_returning() == 3 * GB - (int(5.0 * GB) - int(3.5 * GB))
    # ... nor more than the headroom fell since just before the agent ran.
    engine._agent_returning = (int(4.0 * GB), int(3.5 * GB), 3 * GB, engine._agent_returning[3])
    room["gb"] = 3.5
    assert engine._agent_memory_returning() == int(4.0 * GB) - int(3.5 * GB)

    # Only for the few seconds a job's plan waits for it: still counted just inside the window, gone just past it.
    import dmlx.engine as engine_module
    until = engine._agent_returning[3]
    monkeypatch.setattr(engine_module.time, "monotonic", lambda: until - 0.1)
    assert engine._agent_memory_returning() == int(4.0 * GB) - int(3.5 * GB)
    monkeypatch.setattr(engine_module.time, "monotonic", lambda: until + 0.1)
    assert engine._agent_memory_returning() == 0 and engine._agent_returning is None


def test_the_credit_ends_once_the_agents_memory_is_back(make_engine, client_for, tmp_path, monkeypatch):
    # Test round 6: back at 7.2 GB, then something takes memory down again inside the window. That is
    # not the agent's memory still coming back, so Klein 9B (needs about 4.4 GB here) must not count as fitting.
    # The part of the fall that never comes back (measured up to 0.7 GB) ends it too: MLX freed 3 GB of 4.1.
    engine, client, source, agent, room = _agent_unloaded_after_planning(make_engine, client_for, tmp_path, monkeypatch)
    _warmer(client, source)
    assert engine._agent_memory_returning() == 3 * GB
    room["gb"] = 6.5
    assert engine._agent_memory_returning() == 0
    room["gb"] = 3.5  # Klein 4B fits here, Klein 9B only with the credit
    assert engine._agent_memory_returning() == 0
    decision = _warmer(client, source)
    assert "klein-9b (" not in agent.calls[-1][1]["content"]  # not offered to the agent ...
    assert decision.get("model") != "klein-9b"  # ... nor kept when it picks it anyway
    assert engine._agent_returning is None


def test_the_credit_ends_when_a_job_starts(make_engine, client_for, tmp_path, monkeypatch):
    # Test round 6 (real, 16 GB): a Klein 4B job straight after planning. While it loaded, the memory it took
    # still counted as the agent's coming back, so Qwen fitted in the hint at 6.6 GB against a real 6.31 GB.
    engine, client, source, agent, room = _agent_unloaded_after_planning(make_engine, client_for, tmp_path, monkeypatch)
    _warmer(client, source)
    assert engine._agent_returning is not None
    room["gb"] = 7.6
    job = engine.submit(task="edit", prompt="Warmer colours.", model="klein-4b", image_ids=[source],
                        size="512x512")
    job.finished.wait(10)
    assert job.status == "done"
    room["gb"] = 3.5  # the job's weights and whatever else, not the agent's memory
    assert engine._agent_memory_returning() == 0
    assert engine._agent_returning is None


def test_without_the_credit_a_low_reading_still_swaps(make_engine, client_for, tmp_path, monkeypatch):
    # The same moment with the memory still low after the window: nothing is coming back, so Klein 9B,
    # which does not fit in 3.5 GB, gives way to Klein 4B.
    engine, client, source, agent, room = _agent_unloaded_after_planning(make_engine, client_for, tmp_path, monkeypatch)
    monkeypatch.setattr(engine, "_agent_memory_returning", lambda: 0)
    assert _warmer(client, source).get("model") == "klein-4b"


def test_on_24_gb_the_resident_agent_is_counted_as_before(make_engine, client_for, tmp_path, monkeypatch):
    engine, client, source, agent, room = _agent_unloaded_after_planning(make_engine, client_for, tmp_path,
                                                                         monkeypatch, total_gb=24)
    decision = _warmer(client, source)
    assert engine.prompt_assistant.resident and engine._agent_returning is None
    assert decision.get("model") == "klein-9b"  # its 2.5 GB of weights count as freed before the job
