from dmlx.prompting import PassthroughPromptCompiler
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


def test_rewriting_a_prompt_needs_the_agent(make_engine, client_for):
    response = client_for(make_engine()).post(
        "/v1/prompts/compile",
        json={"prompt": "a fox", "operation": "image.generate", "model": "fake-a"},
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "agent_not_available"
