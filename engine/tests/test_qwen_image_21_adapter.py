from pathlib import Path

from PIL import Image

from dmlx.adapters.base import GenRequest
from dmlx.adapters.mflux_family import QwenImage21Adapter, adapter_for
from dmlx.adapters.streaming import block_prefixes


class _Generated:
    image = Image.new("RGB", (8, 8))


class _Pipe:
    def __init__(self):
        self.calls = []

    def generate_image(self, **kwargs):
        self.calls.append(kwargs)
        return _Generated()


def _request(*, negative_prompt=None):
    return GenRequest(
        task="edit", prompt="combine in this order", negative_prompt=negative_prompt,
        width=768, height=512, steps=12, seed=7, guidance=4.5,
        image_paths=[Path("first.png"), Path("second.png")],
    )


def test_qwen_adapter_is_registered_and_preserves_reference_order():
    assert adapter_for("mflux.qwen_image_21") is QwenImage21Adapter
    adapter = QwenImage21Adapter.__new__(QwenImage21Adapter)
    adapter.pipe = _Pipe()

    result = adapter._run(_request())

    assert result.size == (8, 8)
    call = adapter.pipe.calls[0]
    assert call["image_paths"] == ["first.png", "second.png"]
    assert call["guidance"] == 1.0
    assert call["output_resolution"] == 768
    assert call["use_kv_cache"] is True


def test_qwen_adapter_enables_true_cfg_for_negative_prompt():
    adapter = QwenImage21Adapter.__new__(QwenImage21Adapter)
    adapter.pipe = _Pipe()

    adapter._run(_request(negative_prompt="text, watermark"))

    call = adapter.pipe.calls[0]
    assert call["negative_prompt"] == "text, watermark"
    assert call["guidance"] == 4.5


def test_streaming_finds_nested_text_encoder_block_stacks():
    class Index:
        values = [
            "language_model.layers.0.attn.weight",
            "language_model.layers.1.attn.weight",
            "visual.blocks.0.mlp.weight",
            "visual.merger.weight",
        ]

        def keys(self, prefix=""):
            return [key for key in self.values if key.startswith(prefix)]

    assert block_prefixes(Index(), ["language_model.layers", "visual.blocks"]) == [
        "language_model.layers.0.",
        "language_model.layers.1.",
        "visual.blocks.0.",
    ]
