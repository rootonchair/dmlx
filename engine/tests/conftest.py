import os

import pytest

from dmlx.registry import GB, ComponentSpec, ModelSpec, Registry


def fake_spec(model_id: str, tasks=("t2i", "edit"), te=2.0, tr=2.0, vae=0.2, refs=3) -> ModelSpec:
    comps = {
        "text_encoder": ComponentSpec("text_encoder", "encode", size=int(te * GB)),
        "transformer": ComponentSpec("transformer", "denoise", size=int(tr * GB)),
        "vae": ComponentSpec("vae", "decode", keep=True, size=int(vae * GB)),
    }
    spec = ModelSpec(
        id=model_id, name=model_id.title(), adapter="fake", repo=f"fake/{model_id}", license="apache-2.0",
        tasks=list(tasks), components=comps, defaults={"steps": 4, "width": 512, "height": 512},
        max_references=refs, min_ram_gb=8, activation_gb_per_mp=0.1, decode_gb_per_mp=0.2, decode_tiled_gb_per_mp=0.1,
    )
    return spec


@pytest.fixture
def make_engine(tmp_path, monkeypatch):
    monkeypatch.setenv("DMLX_HOME", str(tmp_path))
    # Never load a real prompt assistant from the Hugging Face cache; tests install a stub.
    monkeypatch.setenv("DMLX_PROMPT_ASSISTANT", "off")
    from dmlx.engine import Engine
    from dmlx.results import ResultStore

    engines = []

    def factory(specs=None, headroom_gb: float | None = 64.0):
        specs = specs or [fake_spec("fake-a"), fake_spec("fake-b", tasks=("t2i",))]
        for s in specs:
            s.local_path = tmp_path / "models" / s.id
            s.local_path.mkdir(parents=True, exist_ok=True)
        engine = Engine(registry=Registry(specs), results=ResultStore(tmp_path / "results"))
        if headroom_gb is not None:
            engine.memory.headroom = lambda: int(headroom_gb * GB)
        engines.append(engine)
        return engine

    yield factory
    for e in engines:
        e.memory.monitor.stop()


@pytest.fixture
def client_for():
    from fastapi.testclient import TestClient

    from dmlx.api import create_app

    def make(engine, token=None):
        return TestClient(create_app(engine, token=token or ""))

    return make


os.environ.setdefault("DMLX_HOME", "/tmp/dmlx-test-home")
