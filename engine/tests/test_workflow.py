import base64
import io
import time
import json
from pathlib import Path

from PIL import Image

from dmlx.capabilities import CapabilitySpec
from dmlx.capabilities import InputConstraint
from conftest import fake_spec
from dmlx.prompting.compiler import CompiledPrompt
from dmlx.adapters.fake import FakeAdapter
from dmlx.workflow import WorkflowPlan


def _generate(client, prompt="source") -> str:
    response = client.post(
        "/v1/images/generations", json={"model": "fake-a", "prompt": prompt, "size": "512x512"}
    )
    assert response.status_code == 200, response.text
    return response.json()["data"][0]["result_id"]


def _upscale_spec():
    spec = fake_spec("fake-upscaler", tasks=("upscale",), refs=1)
    spec.defaults["max_output_pixels"] = 16_777_216
    spec.capabilities = {
        "image.upscale": CapabilitySpec(
            "image.upscale", {"image": InputConstraint(1, 1, True)}, positive_prompt=False)
    }
    return spec


def test_legacy_translation_golden_fixtures():
    fixture_root = Path(__file__).parent / "fixtures" / "workflows"
    for path in sorted(fixture_root.glob("*.json")):
        fixture = json.loads(path.read_text())
        plan = WorkflowPlan.legacy(**fixture["legacy"])
        step = plan.steps[0]
        assert step.operation == fixture["expected"]["operation"], path.name
        assert [item.id for item in step.inputs] == fixture["expected"]["input_ids"], path.name


def test_capabilities_derive_from_legacy_specs(make_engine, client_for):
    client = client_for(make_engine())
    body = client.get("/v1/capabilities").json()
    assert body["schema_version"] == 1
    assert {item["model"] for item in body["operations"]["image.generate"]} == {"fake-a", "fake-b"}
    edit = body["operations"]["image.edit"][0]
    assert edit["inputs"]["image"] == {"min": 1, "max": 3, "ordered": True}


def test_workflow_generation_uses_existing_execution_path(make_engine, client_for):
    client = client_for(make_engine())
    response = client.post(
        "/v1/workflows",
        json={
            "schema_version": 1,
            "steps": [{
                "operation": "image.generate",
                "executor": "fake-a",
                "prompt": {"user": "a lighthouse", "positive": "a lighthouse"},
                "parameters": {"width": 512, "height": 512, "seed": 7},
            }],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    image = Image.open(io.BytesIO(base64.b64decode(body["data"][0]["b64_json"])))
    assert image.size == (512, 512)
    meta = client.get(f"/v1/results/{body['data'][0]['result_id']}/meta").json()
    assert meta["workflow"]["steps"][0]["operation"] == "image.generate"


def test_workflow_preserves_ordered_image_inputs(make_engine, client_for):
    client = client_for(make_engine())
    sources = [_generate(client, f"source {index}") for index in range(3)]
    response = client.post(
        "/v1/workflows",
        json={
            "steps": [{
                "id": "compose",
                "operation": "image.edit",
                "executor": "fake-a",
                "inputs": [
                    {"id": sources[2], "media_type": "image", "role": "reference", "order": 2},
                    {"id": sources[0], "media_type": "image", "role": "source", "order": 0},
                    {"id": sources[1], "media_type": "image", "role": "reference", "order": 1},
                ],
                "prompt": "combine them",
                "parameters": {},
            }]
        },
    )
    assert response.status_code == 200, response.text
    first_step = response.json()["dmlx"]["workflow"]["steps"][0]
    assert [item["id"] for item in first_step["inputs"]] == sources

    # A one-input request records the normalized order in metadata.
    response = client.post(
        "/v1/workflows",
        json={
            "steps": [{
                "operation": "image.edit",
                "executor": "fake-a",
                "inputs": [{"id": sources[0], "media_type": "image", "order": 9}],
                "prompt": "make it blue",
            }]
        },
    )
    assert response.status_code == 200, response.text
    step = response.json()["dmlx"]["workflow"]["steps"][0]
    assert [item["id"] for item in step["inputs"]] == [sources[0]]


def test_workflow_validation_is_explicit(make_engine, client_for):
    client = client_for(make_engine())
    assert client.post("/v1/workflows", json={"steps": []}).status_code == 422
    unsupported = client.post(
        "/v1/workflows", json={"steps": [{"operation": "image.upscale", "prompt": "upscale it"}]}
    )
    assert unsupported.status_code == 422
    assert unsupported.json()["error"]["code"] == "unsupported_operation"
    negative = client.post(
        "/v1/workflows",
        json={
            "steps": [{
                "operation": "image.generate",
                "executor": "fake-a",
                "prompt": {"user": "cat", "positive": "cat", "negative": "text"},
            }]
        },
    )
    assert negative.status_code == 422
    assert negative.json()["error"]["code"] == "negative_prompt_unsupported"

    parsed_negative = client.post(
        "/v1/workflows",
        json={
            "steps": [{
                "operation": "image.generate",
                "executor": "fake-a",
                "prompt": {"user": "cat --negative text", "positive": "cat --negative text"},
            }]
        },
    )
    assert parsed_negative.status_code == 422
    assert parsed_negative.json()["error"]["code"] == "negative_prompt_unsupported"


def test_workflow_auto_prompt_enhancement_is_persisted(make_engine, client_for):
    class Provider:
        name = "test-compiler@1"

        def enhance(self, text, *, supports_negative):
            return CompiledPrompt("Detailed " + text, None, self.name)

    engine = make_engine()
    engine.prompt_provider = Provider()
    response = client_for(engine).post(
        "/v1/workflows",
        json={
            "steps": [{
                "operation": "image.generate",
                "executor": "fake-a",
                "prompt": {"user": "a fox", "positive": "a fox", "enhancement_mode": "auto"},
                "parameters": {"size": "512x512"},
            }]
        },
    )
    assert response.status_code == 200, response.text
    prompt = response.json()["dmlx"]["workflow"]["steps"][0]["prompt"]
    assert prompt["positive"] == "Detailed a fox"
    assert prompt["compiler"] == "test-compiler@1"


def test_supported_negative_prompt_reaches_request_and_metadata(make_engine, client_for):
    engine = make_engine()
    engine.registry.get("fake-a").capabilities["image.generate"] = CapabilitySpec(
        "image.generate", negative_prompt=True)
    response = client_for(engine).post(
        "/v1/workflows",
        json={
            "steps": [{
                "operation": "image.generate", "executor": "fake-a",
                "prompt": {"user": "cat", "positive": "cat", "negative": "text, blur"},
                "parameters": {"size": "512x512"},
            }]
        },
    )
    assert response.status_code == 200, response.text
    result_id = response.json()["data"][0]["result_id"]
    metadata = client_for(engine).get(f"/v1/results/{result_id}/meta").json()
    assert metadata["negative_prompt"] == "text, blur"


def test_multi_step_workflow_hands_off_artifact_and_emits_step_events(make_engine, client_for):
    engine = make_engine()
    response = client_for(engine).post(
        "/v1/workflows",
        json={
            "steps": [
                {
                    "id": "base", "operation": "image.generate", "executor": "fake-a",
                    "prompt": "a white circle", "parameters": {"size": "512x512", "seed": 1},
                },
                {
                    "id": "finish", "operation": "image.edit", "executor": "fake-a",
                    "inputs": [{"id": "$base", "role": "source", "order": 0}],
                    "prompt": "make it blue", "parameters": {"seed": 2},
                },
            ]
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    completed = body["dmlx"]["completed_steps"]
    assert [item["id"] for item in completed] == ["base", "finish"]
    assert body["data"][0]["result_id"] == completed[-1]["result_id"]
    final_meta = engine.results.meta(completed[-1]["result_id"])
    assert final_meta["sources"] == [completed[0]["result_id"]]
    job = engine.job(body["dmlx"]["job_id"])
    step_events = [name for name, _ in job.events if name.startswith("workflow_step_")]
    assert step_events == ["workflow_step_started", "workflow_step_done"] * 2


def test_capability_installed_upscaler_executes_without_api_branch(make_engine, client_for):
    engine = make_engine(specs=[fake_spec("fake-a"), _upscale_spec()])
    client = client_for(engine)
    source = _generate(client)
    response = client.post("/v1/workflows", json={"steps": [{
        "id": "upscale", "operation": "image.upscale", "executor": "fake-upscaler",
        "inputs": [{"id": source, "role": "source"}], "prompt": "upscale",
        "parameters": {"scale": 2, "softness": 0.1, "seed": 4},
    }]})
    assert response.status_code == 200, response.text
    result_id = response.json()["data"][0]["result_id"]
    with Image.open(engine.results.path(result_id)) as output:
        assert output.size == (1024, 1024)
    metadata = engine.results.meta(result_id)
    assert metadata["task"] == "upscale"
    assert metadata["operation"] == "image.upscale"
    assert metadata["sources"] == [source]
    assert metadata["parameters"] == {"scale": 2, "softness": 0.1}
    assert metadata["workflow"]["steps"][0]["parameters"]["scale"] == 2


def test_edit_then_upscale_has_deterministic_handoff(make_engine, client_for):
    engine = make_engine(specs=[fake_spec("fake-a"), _upscale_spec()])
    response = client_for(engine).post("/v1/workflows", json={"steps": [
        {
            "id": "base", "operation": "image.generate", "executor": "fake-a",
            "prompt": "a circle", "parameters": {"size": "512x512", "seed": 1},
        },
        {
            "id": "edited", "operation": "image.edit", "executor": "fake-a",
            "inputs": [{"id": "$base", "role": "source"}],
            "prompt": "make it blue", "parameters": {"seed": 2},
        },
        {
            "id": "large", "operation": "image.upscale", "executor": "fake-upscaler",
            "inputs": [{"id": "$edited", "role": "source"}],
            "prompt": "upscale", "parameters": {"scale": 2, "seed": 3},
        },
    ]})
    assert response.status_code == 200, response.text
    completed = response.json()["dmlx"]["completed_steps"]
    assert [item["id"] for item in completed] == ["base", "edited", "large"]
    assert engine.results.meta(completed[2]["result_id"])["sources"] == [completed[1]["result_id"]]


def test_cancelled_upscale_frees_components_and_worker_recovers(make_engine, client_for):
    engine = make_engine(specs=[fake_spec("fake-a"), _upscale_spec()])
    source = _generate(client_for(engine))
    plan = WorkflowPlan.from_dict({"steps": [{
        "operation": "image.upscale", "executor": "fake-upscaler",
        "inputs": [{"id": source, "role": "source"}], "prompt": "upscale",
        "parameters": {"scale": 2, "steps": 50},
    }]})
    FakeAdapter.step_delay = 0.02
    try:
        job = engine.submit_workflow(plan)
        while job.status != "running":
            time.sleep(0.005)
        engine.cancel(job.id)
        assert job.finished.wait(5)
        assert job.status == "cancelled"
        assert not [
            item for item in engine.memory.resident_components()
            if item["model"] == "fake-upscaler"
        ]
        next_job = engine.submit(task="t2i", prompt="still works", model="fake-a", size="512x512")
        assert next_job.finished.wait(5)
        assert next_job.status == "done"
    finally:
        FakeAdapter.step_delay = 0


def test_multi_step_rejects_forward_artifact_reference(make_engine, client_for):
    response = client_for(make_engine()).post(
        "/v1/workflows",
        json={
            "steps": [
                {
                    "id": "first", "operation": "image.edit", "executor": "fake-a",
                    "inputs": [{"id": "$later", "role": "source"}], "prompt": "edit it",
                },
                {"id": "later", "operation": "image.generate", "executor": "fake-a", "prompt": "source"},
            ]
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_step_reference"


def _two_step_plan():
    return {
        "steps": [
            {
                "id": "base", "operation": "image.generate", "executor": "fake-a",
                "prompt": "a white circle", "parameters": {"size": "512x512", "seed": 1},
            },
            {
                "id": "finish", "operation": "image.edit", "executor": "fake-a",
                "inputs": [{"id": "$base", "role": "source"}],
                "prompt": "fail once", "parameters": {"seed": 2},
            },
        ]
    }


def test_failed_workflow_preserves_artifacts_and_retries_from_failed_step(make_engine, client_for):
    engine = make_engine()
    original_run = engine._run
    failures = 0

    def fail_once(job):
        nonlocal failures
        if job.request.prompt == "fail once" and failures == 0:
            failures += 1
            job.status = "error"
            job.error = {"code": "test_failure", "message": "planned failure"}
            job.report = {"test": True}
            return
        original_run(job)

    engine._run = fail_once
    client = client_for(engine)
    failed = client.post("/v1/workflows", json=_two_step_plan())
    assert failed.status_code == 500
    failed_job = failed.json()["job"]
    assert failed_job["report"]["failed_step_id"] == "finish"
    completed = failed_job["report"]["completed_steps"]
    assert [item["id"] for item in completed] == ["base"]
    base_result = completed[0]["result_id"]
    assert engine.results.exists(base_result)

    retried = client.post(f"/v1/jobs/{failed_job['id']}/retry", json={})
    assert retried.status_code == 200, retried.text
    retry_report = retried.json()["dmlx"]
    assert retry_report["workflow"]["route"] == {
        "retry_of": failed_job["id"], "retry_from_step": "finish"
    }
    final_id = retried.json()["data"][0]["result_id"]
    assert engine.results.meta(final_id)["sources"] == [base_result]
    assert not (engine.results.workflow_root / f"{failed_job['id']}.json").exists()
    assert not (engine.results.workflow_root / f"{retry_report['job_id']}.json").exists()


def test_engine_restart_recovers_failed_workflow_checkpoint(make_engine, client_for):
    engine = make_engine()
    original_run = engine._run

    def fail_finish(job):
        if job.request.prompt == "fail once":
            job.status = "error"
            job.error = {"code": "test_failure", "message": "planned failure"}
            return
        original_run(job)

    engine._run = fail_finish
    failed = client_for(engine).post("/v1/workflows", json=_two_step_plan()).json()["job"]
    checkpoint = next(item for item in engine.results.workflow_jobs() if item["id"] == failed["id"])
    checkpoint["status"] = "running"
    checkpoint["error"] = None
    checkpoint["report"].pop("failed_step_id", None)
    checkpoint["report"].pop("failed_step_index", None)
    checkpoint["report"]["current_step_id"] = "finish"
    checkpoint["report"]["current_step_index"] = 1
    engine.results.save_workflow_job(checkpoint)
    recovered_engine = make_engine()
    recovered = recovered_engine.job(failed["id"])
    assert recovered.status == "error"
    assert recovered.error["code"] == "engine_restarted"
    assert recovered.report["failed_step_id"] == "finish"
    assert recovered.report["completed_steps"][0]["id"] == "base"

    retried = client_for(recovered_engine).post(f"/v1/jobs/{failed['id']}/retry", json={})
    assert retried.status_code == 200, retried.text
