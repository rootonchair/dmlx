"""Integration tests: network and/or real models. Run with `uv run pytest -m integration`."""

import pytest

pytestmark = pytest.mark.integration


def test_downloader_progress_and_completion(tmp_path):
    """Download a tiny public repo through the same code path the app uses (isolated HF cache)."""
    import os
    import subprocess
    import sys

    script = """
import time
from dmlx import downloads
from dmlx.registry import ComponentSpec, ModelSpec
spec = ModelSpec(id="tiny", name="Tiny", adapter="fake", repo="hf-internal-testing/tiny-sd-pipe",
                 license="apache-2.0", tasks=["t2i"], defaults={},
                 components={n: ComponentSpec(n, "encode") for n in ("text_encoder", "unet", "vae")})
assert downloads.local_path(spec) is None
state = downloads.Downloader().start(spec)
deadline = time.time() + 300
while state.status == "downloading" and time.time() < deadline:
    time.sleep(0.2)
assert state.status == "done", state.error
assert state.total > 1_000_000 and state.done == state.total
assert downloads.local_path(spec) is not None and spec.components["unet"].size > 0
print("ok")
"""
    env = {**os.environ, "HF_HUB_CACHE": str(tmp_path / "hub"), "HF_HOME": str(tmp_path / "hf")}
    out = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=400)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr[-2000:]


def _real_engine():
    from dmlx.engine import Engine

    engine = Engine()
    spec = engine.registry.get("flux2-klein-4b")
    if not engine.is_downloaded(spec):
        pytest.skip("flux2-klein-4b not downloaded")
    return engine


def test_fixed_seed_is_deterministic():
    """Same seed + params + strategy (no step caching) -> identical pixels (spec §6.2)."""
    from PIL import Image, ImageChops

    engine = _real_engine()
    images = []
    for _ in range(2):
        job = engine.submit(task="t2i", prompt="a red bicycle against a white wall", model="flux2-klein-4b",
                            size="512x512", seed=1234, strategy="sequential", force=True)
        job.finished.wait(600)
        assert job.status == "done", job.error
        images.append(Image.open(engine.results.path(job.result_ids[0])).convert("RGB"))
    assert ImageChops.difference(*images).getbbox() is None


def test_warm_edit_has_no_loads():
    engine = _real_engine()
    first = engine.submit(task="t2i", prompt="a ceramic mug", model="flux2-klein-4b", size="512x512", seed=1, force=True)
    first.finished.wait(600)
    assert first.status == "done", first.error
    edit = engine.submit(task="edit", prompt="make the mug blue", model="flux2-klein-4b", size="512x512",
                         image_ids=first.result_ids, force=True)
    edit.finished.wait(600)
    assert edit.status == "done", edit.error
    if first.report["strategy"] == "eager":
        assert edit.report["loads"] in ([], ["text_encoder"])
