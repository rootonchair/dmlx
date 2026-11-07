"""Structured prompts (task 17): the checks on the agent's caption, the user's boxes, and the engine's
fallbacks. The user never sees JSON; these tests are about what reaches the model."""

import json

import pytest

from dmlx.prompting import formats
from dmlx.prompting.local_vlm import LocalVLMPromptProvider
from dmlx.registry import _parse
from conftest import fake_spec

JAZZ_IDEOGRAM = {
    "high_level_description": "Art-deco poster for a jazz night.",
    "style_description": {"medium": "poster", "aesthetics": "vintage art deco", "art_style": "flat graphic",
                          "lighting": "warm spotlight", "color_palette": ["#1a1f3c", "c9a227", "nope"]},
    "mood": "smoky",
    "compositional_deconstruction": {"background": "deep navy", "elements": [
        {"type": "text", "text": "JAZZ NIGHT", "bbox": [120, 60, 880, 220], "desc": "gold lettering"},
        {"type": "obj", "bbox": [300, 260, 700, 820], "desc": "saxophonist silhouette, brass"},
        {"type": "text", "text": "FRIDAY 9 PM", "bbox": [280, 860, 720, 940], "desc": "cream sans"}]},
}
JAZZ_MOSS = {
    "summary": {"meta": ["poster"], "canvas": "single", "caption": "A vintage art-deco jazz poster."},
    "items": [
        {"label": "text", "caption": "gold lettering", "text": "JAZZ NIGHT", "bbox": [120, 60, 880, 220]},
        {"label": "object", "caption": "saxophonist silhouette", "bbox": [300, 260, 700, 820]},
        {"label": "text", "caption": "cream sans", "text": "FRIDAY 9 PM", "id": "date", "bbox": [0, 0, 100, 100]},
    ],
}
USER_DATE = {"id": "date", "kind": "text", "label": "FRIDAY 9 PM", "box": [880, 520, 960, 960], "by_user": True}


def test_ideogram_caption_is_tidied_into_the_order_the_model_expects():
    result = formats.normalize(formats.IDEOGRAM, "Here it is:\n" + json.dumps(JAZZ_IDEOGRAM))
    caption = json.loads(result.prompt)
    assert list(caption) == ["high_level_description", "style_description", "compositional_deconstruction"]
    assert list(caption["style_description"]) == ["aesthetics", "lighting", "medium", "art_style", "color_palette"]
    assert caption["style_description"]["color_palette"] == ["#1A1F3C", "#C9A227"]
    assert [b.label for b in result.layout] == ["JAZZ NIGHT", "saxophonist silhouette", "FRIDAY 9 PM"]
    assert any("mood" in fix for fix in result.fixes)


def test_ideogram_captions_pass_ideograms_own_check():
    """mflux's Ideogram4CaptionVerifier: the key names and orders the model was trained on."""
    verifier = pytest.importorskip("mflux.models.ideogram4.model.ideogram4_text_encoder.caption")
    check = verifier.Ideogram4CaptionVerifier().verify_raw
    layout = formats.parse_layout([USER_DATE])
    assert check(formats.normalize(formats.IDEOGRAM, json.dumps(JAZZ_IDEOGRAM), layout=layout).prompt) == []
    photo = json.loads(json.dumps(JAZZ_IDEOGRAM))
    photo["style_description"].pop("art_style")
    photo["style_description"]["photo"] = "35mm, shallow depth of field"
    assert check(formats.normalize(formats.IDEOGRAM, json.dumps(photo)).prompt) == []
    built = formats.from_layout(formats.IDEOGRAM, "a jazz poster", layout, width=1024, height=1024)
    assert check(built.prompt) == []


def test_ideogram_needs_exactly_one_style_and_refuses_a_cut_off_reply():
    both = json.loads(json.dumps(JAZZ_IDEOGRAM))
    both["style_description"]["photo_style"] = "studio photo"  # the old name still reads as photo
    caption = json.loads(formats.normalize(formats.IDEOGRAM, json.dumps(both)).prompt)
    assert "photo" in caption["style_description"] and "art_style" not in caption["style_description"]
    with pytest.raises(formats.FormatError, match="cut off"):
        formats.normalize(formats.IDEOGRAM, json.dumps(JAZZ_IDEOGRAM)[:300])


def test_moss_boxes_become_pixels_from_the_centre_at_the_generation_size():
    result = formats.normalize(formats.MOSS, json.dumps(JAZZ_MOSS), width=1024, height=768)
    prompt = json.loads(result.prompt)
    assert prompt["bbox_coord_space"] == "abs_centralized" and prompt["bbox_format"] == "yxyx"
    assert prompt["items"][0] == {"label": "text", "caption": "gold lettering", "text": "JAZZ NIGHT",
                                  "bbox": [-338, -390, -215, 390]}
    assert prompt["items"][1]["label"] == "object" and "text" not in prompt["items"][1]
    for item in prompt["items"]:
        y0, x0, y1, x1 = item["bbox"]
        assert -384 <= y0 < y1 <= 384 and -512 <= x0 < x1 <= 512


def test_the_users_boxes_are_kept_exactly_and_their_description_word_for_word():
    mine = dict(USER_DATE, desc="hand-painted cream script", desc_by_user=True)
    layout = formats.parse_layout([mine])
    result = formats.normalize(formats.MOSS, json.dumps(JAZZ_MOSS), layout=layout, width=1024, height=768)
    date = next(b for b in result.layout if b.id == "date")
    assert date.box == (880, 520, 960, 960) and date.by_user and date.desc == "hand-painted cream script"
    item = json.loads(result.prompt)["items"][2]
    assert item["caption"] == "hand-painted cream script" and item["bbox"][0] > 0  # bottom half, not [0,0,..]


def test_a_user_box_the_agent_dropped_is_put_back():
    layout = formats.parse_layout([USER_DATE])
    reply = json.dumps({**JAZZ_MOSS, "items": JAZZ_MOSS["items"][:2]})
    result = formats.normalize(formats.MOSS, reply, layout=layout, width=1024, height=1024)
    assert [b.id for b in result.layout][-1] == "date"
    assert any("Put back" in fix for fix in result.fixes)


def test_fibo_gets_place_words_for_the_users_boxes_and_keeps_an_empty_text_list():
    layout = formats.parse_layout([{"id": "fox", "kind": "obj", "label": "red fox", "box": [600, 650, 950, 980]}])
    reply = json.dumps({"short_description": "A fox at dawn.", "objects": [{"description": "a fox"}],
                        "lighting": {"conditions": "dawn", "direction": ""}, "weather": "mist"})
    caption = json.loads(formats.normalize(formats.FIBO, reply, layout=layout).prompt)
    assert caption["objects"][0] == {"description": "red fox", "location": "bottom right", "relative_size": "medium-sized"}
    assert caption["lighting"] == {"conditions": "dawn"} and caption["text_render"] == []
    assert "weather" not in caption
    with pytest.raises(formats.FormatError, match="edit_instruction"):
        formats.normalize(formats.FIBO, reply, operation="image.edit")


def test_fibo_keeps_the_agents_font_and_colour_for_a_text_box_the_user_moved():
    layout = formats.parse_layout([{"id": "date", "kind": "text", "label": "FRIDAY 9 PM", "box": [880, 560, 960, 960],
                                    "by_user": True}])
    reply = json.dumps({"short_description": "A jazz poster.", "text_render": [
        {"text": "FRIDAY 9 PM", "location": "center", "size": "medium", "font": "condensed sans", "color": "cream"},
        {"text": "JAZZ NIGHT", "location": "top", "size": "large", "font": "art deco", "color": "gold"}]})
    texts = json.loads(formats.normalize(formats.FIBO, reply, layout=layout).prompt)["text_render"]
    assert texts[0] == {"text": "FRIDAY 9 PM", "location": "bottom right", "size": "small",
                        "font": "condensed sans", "color": "cream"}
    assert [t["text"] for t in texts] == ["FRIDAY 9 PM", "JAZZ NIGHT"]


def test_captions_over_the_token_limit_are_refused():
    long = json.loads(json.dumps(JAZZ_IDEOGRAM))
    long["compositional_deconstruction"]["elements"] = [
        {"type": "obj", "bbox": [0, i * 80, 900, i * 80 + 70], "desc": "a very detailed thing " * 30} for i in range(12)]
    with pytest.raises(formats.FormatError, match="at most 2048"):
        formats.normalize(formats.IDEOGRAM, json.dumps(long))


def test_fallbacks_and_plain_words_notes():
    assert formats.fallback(formats.IDEOGRAM, "a poster") == "a poster"
    assert json.loads(formats.fallback(formats.FIBO, "a fox")) == {"short_description": "a fox"}
    assert json.loads(formats.fallback(formats.FIBO, "make it night", "image.edit")) == {"edit_instruction": "make it night"}
    assert formats.fallback_note(formats.MOSS, "MOSS Image", had_layout=False) is None
    note = formats.fallback_note(formats.IDEOGRAM, "Ideogram 4", had_layout=False)
    assert "JSON" not in note and note.startswith("Ideogram 4 works best")
    assert "layout" in formats.fallback_note(formats.MOSS, "MOSS Image", had_layout=True)


def test_moss_gets_a_layout_only_when_placement_matters():
    assert not formats.wants_structure(formats.MOSS, "a red fox in a misty forest at dawn", [])
    assert formats.wants_structure(formats.MOSS, "a poster for a jazz night", [])
    assert formats.wants_structure(formats.MOSS, 'a shop sign reading "OPEN"', [])
    assert formats.wants_structure(formats.IDEOGRAM, "a red fox", [])


def test_bad_layouts_are_refused():
    for bad in ([{"kind": "text", "box": [0, 0, 100, 100]}], [{"box": [0, 0, 1, 1]}], [{"box": "top"}], "x"):
        with pytest.raises(formats.FormatError):
            formats.parse_layout(bad)


def test_specs_saved_before_task_17_get_their_familys_format():
    data = {"id": "moss", "name": "MOSS Image", "adapter": "mflux.moss_image", "repo": "r", "license": "apache-2.0",
            "tasks": ["t2i"], "components": {"transformer": {"phase": "denoise"}}}
    assert _parse(data).prompt_format == formats.MOSS
    assert _parse({**data, "adapter": "mflux.flux2_klein"}).prompt_format is None
    assert _parse({**data, "prompt_format": "made-up"}).prompt_format is None


# -- through the engine --------------------------------------------------------------------
class _Assistant:
    def __init__(self, *replies):
        self.replies, self.calls, self.loads = list(replies), [], 0

    def loader(self, _path):
        self.loads += 1
        return object(), object()

    def generator(self, _model, _processor, messages, images, max_tokens=None):
        self.calls.append((messages, max_tokens))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _engine(make_engine, tmp_path, assistant=None, fmt=formats.MOSS):
    spec = fake_spec("moss", tasks=("t2i",))
    spec.prompt_format, spec.name = fmt, "MOSS Image"
    engine = make_engine([spec, fake_spec("plain", tasks=("t2i",))])
    if assistant:
        (tmp_path / "vlm").mkdir(exist_ok=True)
        (tmp_path / "vlm" / "model.safetensors").write_bytes(b"x" * 1024)
        engine.prompt_assistant = LocalVLMPromptProvider(tmp_path / "vlm", loader=assistant.loader,
                                                         generator=assistant.generator)
    return engine


def _make(client, prompt, model="moss", **parameters):
    return client.post("/v1/workflows", json={"steps": [{
        "operation": "image.generate", "executor": model,
        "prompt": {"user": prompt, "positive": prompt, "compiler": "agent:local-vlm:test"},
        "parameters": {"width": 1024, "height": 768, **parameters},
    }]})


def _meta(client, response):
    assert response.status_code == 200, response.text
    return client.get(f"/v1/results/{response.json()['data'][0]['result_id']}/meta").json()


def test_a_poster_on_moss_gets_a_checked_layout_and_the_users_words_stay(make_engine, client_for, tmp_path):
    assistant = _Assistant(json.dumps(JAZZ_MOSS))
    engine = _engine(make_engine, tmp_path, assistant)
    client = client_for(engine)

    meta = _meta(client, _make(client, "A vintage jazz club poster reading \"JAZZ NIGHT\" and \"FRIDAY 9 PM\""))

    sent = json.loads(meta["parameters"]["rewritten_prompt"])
    assert sent["bbox_format"] == "yxyx" and len(sent["items"]) == 3
    assert meta["parameters"]["plain_prompt"].startswith("A vintage jazz club poster")
    assert meta["parameters"]["structure"]["structured"] is True
    assert [b["label"] for b in meta["parameters"]["layout"]][0] == "JAZZ NIGHT"
    assert assistant.calls[0][1] == 1200  # the longer reply allowed for a caption
    assert not engine.prompt_assistant.resident


def test_a_bad_caption_is_retried_then_falls_back_to_the_plain_sentence(make_engine, client_for, tmp_path):
    assistant = _Assistant('{"summary": {"caption": "cut', '{"items": []}')
    engine = _engine(make_engine, tmp_path, assistant)
    client = client_for(engine)

    response = _make(client, "a poster for a jazz night")
    meta = _meta(client, response)

    assert len(assistant.calls) == 2 and "could not be used" in assistant.calls[1][0][1]["content"]
    assert meta["parameters"]["rewritten_prompt"] == "a poster for a jazz night"
    status = response.json()["dmlx"]["structure"]
    assert status["fallback"] == "plain" and "note" not in status  # MOSS reads plain sentences well


def test_without_the_agent_the_users_boxes_still_reach_the_model(make_engine, client_for, tmp_path):
    engine = _engine(make_engine, tmp_path)
    client = client_for(engine)

    response = _make(client, "a flyer for a book fair", layout=[USER_DATE])
    meta = _meta(client, response)

    sent = json.loads(meta["parameters"]["rewritten_prompt"])
    assert sent["summary"]["caption"] == "a flyer for a book fair"
    assert sent["items"][0]["text"] == "FRIDAY 9 PM" and sent["items"][0]["bbox"][0] > 0
    assert response.json()["dmlx"]["structure"]["fallback"] == "layout"
    assert meta["parameters"]["layout"][0]["by_user"] is True


def test_ideogram_fallback_tells_the_user_in_plain_words(make_engine, client_for, tmp_path):
    engine = _engine(make_engine, tmp_path, _Assistant("no json here {", "still none {"), fmt=formats.IDEOGRAM)
    client = client_for(engine)

    status = _make(client, "a red fox").json()["dmlx"]["structure"]

    assert status["fallback"] == "plain"
    assert status["note"].startswith("MOSS Image works best") and "JSON" not in status["note"]


def test_a_scene_without_placement_runs_on_the_plain_sentence(make_engine, client_for, tmp_path):
    assistant = _Assistant()
    engine = _engine(make_engine, tmp_path, assistant)
    client = client_for(engine)

    meta = _meta(client, _make(client, "a red fox in a misty pine forest at dawn"))

    assert assistant.calls == [] and meta["parameters"]["structure"]["skipped"] == "no_placement"


def test_variations_reuse_the_caption(make_engine, client_for, tmp_path):
    assistant = _Assistant(json.dumps(JAZZ_MOSS))
    engine = _engine(make_engine, tmp_path, assistant)
    client = client_for(engine)

    _meta(client, _make(client, "a jazz poster", seed=1))
    _meta(client, _make(client, "a jazz poster", seed=2))

    assert len(assistant.calls) == 1


def test_a_layout_is_refused_for_models_without_a_format(make_engine, client_for, tmp_path):
    client = client_for(_engine(make_engine, tmp_path))

    response = _make(client, "a poster", model="plain", layout=[USER_DATE])

    assert response.status_code == 422 and response.json()["error"]["code"] == "layout_unsupported"
    assert _make(client, "a poster", layout=[{"box": [0, 0, 1, 1]}]).status_code == 422


def test_auto_with_a_layout_only_offers_models_that_can_use_it(make_engine, tmp_path):
    engine = _engine(make_engine, tmp_path)

    decision = engine.route({"message": "a flyer for a book fair", "layout": [USER_DATE]})

    assert decision["intent"] == "image.generate" and decision["model"] == "moss"
    assert engine._model_choices(["image.generate"], 0, False, layout=True) == {
        "image.generate": [("moss", "MOSS Image")]}


def test_agent_boxes_are_read_as_x_first_and_background_boxes_are_dropped():
    reply = json.dumps({"summary": {"meta": "poster", "caption": "A poster."}, "items": [
        {"label": "text", "text": "TITLE", "caption": "big", "bbox": [100, 40, 900, 200]},
        {"label": "object", "caption": "navy background", "bbox": [0, 0, 1000, 1000]}]})
    result = formats.normalize(formats.MOSS, reply, width=1000, height=1000)
    assert [b.box for b in result.layout] == [(40, 100, 200, 900)]  # a wide band across the top
    assert json.loads(result.prompt)["summary"]["meta"] == ["poster"]


def test_a_user_box_the_agent_copied_without_its_id_is_not_drawn_twice():
    layout = formats.parse_layout([USER_DATE, {"id": "sax", "kind": "obj", "label": "saxophonist",
                                               "box": [260, 300, 820, 700], "by_user": True}])
    reply = json.dumps({"summary": {"caption": "A poster."}, "items": [
        {"label": "text", "text": "FRIDAY 9 PM", "caption": "cream", "bbox": [520, 880, 960, 960]},
        {"label": "object", "caption": "a woman playing sax", "bbox": [310, 270, 690, 810]}]})
    result = formats.normalize(formats.MOSS, reply, layout=layout)
    assert [b.id for b in result.layout] == ["date", "sax"]
    assert result.layout[1].desc == "a woman playing sax" and result.layout[1].box == (260, 300, 820, 700)


def test_the_agent_is_told_the_users_boxes_in_the_order_it_writes_them():
    text = formats.request_text("a poster", formats.parse_layout([USER_DATE]))
    assert 'bbox [520, 880, 960, 960]' in text and 'id "date"' in text


def test_moss_drops_a_mood_backdrop_and_names_boxes_short():
    # The agent's reply from the first in-app run: a "moody" box wrapping the scene, long names.
    reply = json.dumps({"summary": {"meta": "poster", "caption": "A jazz night poster, smoky and moody."}, "items": [
        {"label": "text", "caption": "bold retro", "text": "Jazz Night Friday", "bbox": [200, 100, 800, 300]},
        {"label": "object", "caption": "Saxophone in the foreground, gleaming", "bbox": [150, 200, 450, 500]},
        {"label": "object", "name": "dancers", "caption": "Silhouettes of people dancing", "bbox": [400, 300, 850, 650]},
        {"label": "object", "caption": "Moody, smoky ambiance", "bbox": [100, 150, 900, 700]},
    ]})
    out = formats.normalize(formats.MOSS, reply)
    assert [b.label for b in out.layout] == ["Jazz Night Friday", "Saxophone", "dancers"]
    assert any("backdrop" in f for f in out.fixes)
    assert len(json.loads(out.prompt)["items"]) == 3


def test_a_user_box_is_never_dropped_as_a_backdrop():
    mine = formats.parse_layout([{"id": "room", "kind": "obj", "label": "club", "box": [100, 100, 900, 900],
                                  "by_user": True}])
    reply = json.dumps({"summary": {"caption": "A club."}, "items": [
        {"id": "room", "label": "object", "caption": "a club", "bbox": [100, 100, 900, 900]},
        {"label": "object", "name": "sax", "caption": "a sax", "bbox": [200, 200, 400, 400]},
        {"label": "object", "name": "drums", "caption": "drums", "bbox": [500, 500, 700, 700]},
    ]})
    out = formats.normalize(formats.MOSS, reply, layout=mine)
    assert "room" in [b.id for b in out.layout]


def test_instructions_forbid_invented_facts_and_mood_boxes():
    text = formats.instructions(formats.MOSS)
    assert "no dates" in text and "a mood" in text and '"name"' in text


def test_a_caption_that_gives_a_blank_image_is_drawn_again_from_the_plain_sentence(make_engine, client_for, tmp_path):
    # MOSS at 1024 px turned one of the agent's layouts into NaNs: a black square, every pixel 0.
    from PIL import Image

    engine = _engine(make_engine, tmp_path, _Assistant(json.dumps(JAZZ_MOSS)))
    adapter = engine.adapter(engine.registry.get("moss"))
    real, prompts = adapter.generate, []

    def generate(req, hooks):
        prompts.append(req.prompt)
        image = real(req, hooks)
        return Image.new("RGB", image.size) if req.prompt.startswith("{") else image

    adapter.generate = generate
    client = client_for(engine)
    response = _make(client, "A vintage jazz club poster reading \"JAZZ NIGHT\" and \"FRIDAY 9 PM\"")
    meta = _meta(client, response)

    assert len(prompts) == 2 and prompts[1].startswith("A vintage jazz club poster")
    assert meta["parameters"]["rewritten_prompt"] == prompts[1] and "layout" not in meta["parameters"]
    status = response.json()["dmlx"]["structure"]
    assert status["fallback"] == "plain" and "blank image" in status["note"]
    assert not engine._structure_cache


def test_a_caption_that_goes_to_nan_mid_run_is_drawn_from_the_plain_sentence(make_engine, client_for, tmp_path):
    from dmlx.adapters.base import NonFinite

    engine = _engine(make_engine, tmp_path, _Assistant(json.dumps(JAZZ_MOSS)))
    adapter = engine.adapter(engine.registry.get("moss"))
    real, prompts = adapter.generate, []

    def generate(req, hooks):
        prompts.append(req.prompt)
        if req.prompt.startswith("{"):
            raise NonFinite(3)
        return real(req, hooks)

    adapter.generate = generate
    client = client_for(engine)
    response = _make(client, "A vintage jazz club poster reading \"JAZZ NIGHT\" and \"FRIDAY 9 PM\"")
    meta = _meta(client, response)
    assert len(prompts) == 2 and meta["parameters"]["structure"]["reason"] == "blank image"


def test_moss_saved_at_50_steps_moves_to_30_but_a_users_choice_stays():
    from dmlx.catalog import MOSS_NOTE

    data = {"id": "moss", "name": "MOSS Image", "adapter": "mflux.moss_image", "repo": "r", "license": "apache-2.0",
            "tasks": ["t2i"], "components": {"transformer": {"phase": "denoise"}},
            "defaults": {"steps": 50, "guidance": 3.0},
            "agent_note": "New images only; very slow (about 13 minutes at 1024 px, 3 minutes at 512 px); "
                          "use only when the user asks for it."}
    spec = _parse(data)
    assert spec.defaults["steps"] == 30 and spec.agent_note == MOSS_NOTE
    assert _parse({**data, "defaults": {"steps": 40}}).defaults["steps"] == 40
    assert _parse({**data, "adapter": "mflux.flux2_klein"}).defaults["steps"] == 50


NO_WORDS_MOSS = {"summary": {"meta": ["poster"], "caption": "A moody jazz club."},
                 "items": [{"label": "object", "name": "piano", "caption": "a grand piano", "bbox": [300, 550, 600, 800]}]}


def test_quoted_words_that_the_caption_drops_get_a_retry(make_engine, client_for, tmp_path):
    # Seen in the app: the planner kept 'JAZZ NIGHT' and 'FRIDAY 9 PM', the caption had only objects.
    assistant = _Assistant(json.dumps(NO_WORDS_MOSS), json.dumps(JAZZ_MOSS))
    engine = _engine(make_engine, tmp_path, assistant)
    client = client_for(engine)
    meta = _meta(client, _make(client, 'a poster with the words "JAZZ NIGHT" and "FRIDAY 9 PM"'))
    retry = assistant.calls[1][0][1]["content"]
    assert '"JAZZ NIGHT", "FRIDAY 9 PM"' in retry and "text item" in retry
    assert [i.get("text") for i in json.loads(meta["parameters"]["rewritten_prompt"])["items"] if "text" in i] == \
        ["JAZZ NIGHT", "FRIDAY 9 PM"]


def test_a_caption_that_never_draws_the_quoted_words_falls_back_to_the_plain_sentence(make_engine, client_for, tmp_path):
    assistant = _Assistant(json.dumps(NO_WORDS_MOSS), json.dumps(NO_WORDS_MOSS))
    engine = _engine(make_engine, tmp_path, assistant)
    client = client_for(engine)
    response = _make(client, 'a poster with the words "JAZZ NIGHT"')
    meta = _meta(client, response)
    assert meta["parameters"]["rewritten_prompt"] == 'a poster with the words "JAZZ NIGHT"'
    assert response.json()["dmlx"]["structure"]["fallback"] == "plain"


def test_quoted_words_in_either_quote_style_and_not_apostrophes():
    assert formats.quoted_words('words "JAZZ NIGHT"', "and 'FRIDAY 9 PM', it's Friday's") == ["JAZZ NIGHT", "FRIDAY 9 PM"]
    assert formats.quoted_words("a sign that says “OPEN”") == ["OPEN"]


def test_moss_items_labelled_by_name_still_count_as_text_or_objects():
    # Ministral 3B's reply: "label" used as a name.
    reply = json.dumps({"summary": {"caption": "A jazz poster."}, "items": [
        {"label": "title", "caption": "serif", "text": "JAZZ NIGHT", "bbox": [100, 40, 900, 120]},
        {"label": "date", "caption": "script", "text": "FRIDAY 9 PM", "bbox": [100, 140, 900, 180]},
        {"label": "saxophone", "caption": "brass, warm glow", "bbox": [600, 500, 700, 600]},
    ]})
    out = formats.normalize(formats.MOSS, reply, words=["JAZZ NIGHT", "FRIDAY 9 PM"])
    assert [(b.kind, b.label) for b in out.layout] == [("text", "JAZZ NIGHT"), ("text", "FRIDAY 9 PM"),
                                                        ("obj", "saxophone")]


def test_agent_box_descriptions_lose_place_words_but_the_users_keep_theirs():
    # The redraw in the app: the moved date kept "below main title" and MOSS drew it there.
    mine = formats.parse_layout([
        {"id": "date", "kind": "text", "label": "FRIDAY 9 PM", "box": [880, 560, 960, 960], "by_user": True},
        {"id": "sax", "kind": "obj", "label": "sax", "desc": "gold sax at the bottom", "desc_by_user": True,
         "box": [300, 100, 700, 400], "by_user": True}])
    reply = json.dumps({"summary": {"caption": "A jazz poster."}, "items": [
        {"id": "date", "label": "text", "text": "FRIDAY 9 PM", "caption": "Complementary font, smaller, below main title",
         "bbox": [560, 880, 960, 960]},
        {"label": "object", "name": "piano", "caption": "In the corner, dimly lit, wooden finish", "bbox": [300, 500, 600, 750]},
    ]})
    out = {b.label: b.desc for b in formats.normalize(formats.MOSS, reply, layout=mine).layout}
    assert out["FRIDAY 9 PM"] == "Complementary font, smaller"
    assert out["piano"] == "dimly lit, wooden finish"
    assert out["sax"] == "gold sax at the bottom"
