"""HTTP API: OpenAI-compatible images endpoints, job events over SSE, admin endpoints."""

from __future__ import annotations

import asyncio
import base64
import errno
import json
import os
import logging
import time

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from dmlx import __version__
from dmlx.engine import Engine, Job, RequestError
from dmlx.workflow import WorkflowError, WorkflowPlan

log = logging.getLogger("dmlx.api")

TERMINAL_EVENTS = {"done", "error", "cancelled"}
HEIF_BRANDS = (b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx", b"mif1", b"msf1", b"avif")


def _unreadable(name: str, data: bytes) -> RequestError:
    """The 415 for a file Pillow cannot open: it names the file, and says what HEIC needs (the app turns
    HEIC into PNG before it uploads; this is for other clients)."""
    lower = name.lower()
    heif = lower.endswith((".heic", ".heif", ".avif")) or (data[4:8] == b"ftyp" and data[8:12] in HEIF_BRANDS)
    if heif:
        message = f"{name} is a HEIC photo, which the engine can't open. Convert it to PNG or JPEG first."
    else:
        message = f"{name} is not an image dMLX can open. Use a PNG, JPEG or WebP image."
    return RequestError("unsupported_image", message, 415)


def _save_upload(engine: Engine, name: str | None, data: bytes) -> str:
    from PIL import Image, UnidentifiedImageError

    name = os.path.basename(name or "") or "The file"
    try:
        image = engine.results.decode_upload(data)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError, SyntaxError) as exc:
        log.info("upload %s refused: %s", name, exc)
        raise _unreadable(name, data) from None
    try:
        return engine.results.save_upload_image(image)
    except OSError as exc:
        # Saving failed, not reading: a full disk or a folder dMLX cannot write is not "not an image" (task 19).
        log.warning("could not save upload %s: %s", name, exc)
        full = exc.errno == errno.ENOSPC
        reason = "the disk is full" if full else "dMLX could not write to its folder"
        raise RequestError("upload_failed", f"Couldn't save {name}: {reason}.", 507 if full else 500) from None


def create_app(engine: Engine | None = None, token: str | None = None) -> FastAPI:
    engine = engine or Engine()
    token = token if token is not None else os.environ.get("DMLX_TOKEN") or None
    app = FastAPI(title="dMLX", version=__version__)
    app.state.engine = engine

    def auth(request: Request) -> None:
        if not token:
            return
        header = request.headers.get("authorization", "")
        if header != f"Bearer {token}" and request.query_params.get("token") != token:
            raise HTTPException(status_code=401, detail={"code": "unauthorized", "message": "Missing or bad token"})

    @app.exception_handler(RequestError)
    async def request_error(_request: Request, exc: RequestError):
        return JSONResponse(status_code=exc.status, content={"error": {"code": exc.code, "message": exc.message}})

    @app.exception_handler(KeyError)
    async def key_error(_request: Request, exc: KeyError):
        message = exc.args[0] if exc.args else str(exc)  # str() of a KeyError adds quotes
        return JSONResponse(status_code=404, content={"error": {"code": "not_found", "message": message}})

    # -- health / models --------------------------------------------------------------
    @app.get("/health")
    async def health():
        return {"ok": True, "version": __version__}

    @app.get("/v1/models", dependencies=[Depends(auth)])
    async def models():
        # In a thread: it reads the disk (partial downloads) and the memory manager; the event loop must
        # stay free for a Stop (fix round 2 of task 19: a poll during an eviction stalled every request).
        return {"object": "list", "data": await asyncio.to_thread(engine.models)}

    @app.get("/v1/capabilities", dependencies=[Depends(auth)])
    async def capabilities():
        return engine.capabilities()

    @app.post("/v1/workflows", dependencies=[Depends(auth)])
    async def workflows(request: Request):
        body = await request.json()
        try:
            plan = WorkflowPlan.from_dict(body)
        except WorkflowError as exc:
            raise RequestError(exc.code, str(exc), 422) from exc
        job = await asyncio.to_thread(engine.submit_workflow, plan)
        return await _respond(job, stream=bool(body.get("stream")), response_format=body.get("response_format"))

    @app.post("/v1/workflows/route", dependencies=[Depends(auth)])
    async def route_workflow(request: Request):
        body = await request.json()
        return await asyncio.to_thread(engine.route, body)

    @app.post("/admin/intelligence/router/test", dependencies=[Depends(auth)])
    async def test_workflow_router():
        return await asyncio.to_thread(engine.test_router)

    @app.post("/v1/prompts/compile", dependencies=[Depends(auth)])
    async def compile_prompt(request: Request):
        body = await request.json()
        return await asyncio.to_thread(engine.compile_prompt, body)

    # -- images -----------------------------------------------------------------------
    @app.post("/v1/images/generations", dependencies=[Depends(auth)])
    async def generations(request: Request):
        body = await request.json()
        ext = body.get("dmlx") or {}
        job = engine.submit(
            task="t2i",
            prompt=body.get("prompt", ""),
            negative_prompt=body.get("negative_prompt"),
            model=body.get("model"),
            size=body.get("size"),
            steps=body.get("steps") or ext.get("steps"),
            seed=body.get("seed", ext.get("seed")),
            n=body.get("n", 1),
            strategy=ext.get("strategy", "auto"),
            force=bool(ext.get("force", False)),
        )
        return await _respond(job, stream=bool(body.get("stream")), response_format=body.get("response_format"))

    @app.post("/v1/images/edits", dependencies=[Depends(auth)])
    async def edits(request: Request):
        ctype = request.headers.get("content-type", "")
        image_ids: list[str] = []
        if ctype.startswith("multipart/"):
            form = await request.form()
            body = {k: v for k, v in form.items() if isinstance(v, str)}
            for key in ("image", "image[]"):
                for upload in form.getlist(key):
                    if hasattr(upload, "read"):
                        image_ids.append(_save_upload(engine, getattr(upload, "filename", None), await upload.read()))
            ext = json.loads(body.get("dmlx", "{}") or "{}")
        else:
            body = await request.json()
            ext = body.get("dmlx") or {}
        source = body.get("source_id") or ext.get("source_id")
        refs = body.get("reference_ids") or ext.get("reference_ids") or []
        if isinstance(refs, str):
            refs = [r for r in refs.split(",") if r]
        image_ids = ([source] if source else []) + image_ids + list(refs)
        job = engine.submit(
            task="edit",
            prompt=body.get("prompt", ""),
            negative_prompt=body.get("negative_prompt"),
            model=body.get("model") or None,
            size=body.get("size"),
            steps=_int(body.get("steps") or ext.get("steps")),
            seed=_int(body.get("seed", ext.get("seed"))),
            n=_int(body.get("n")) or 1,
            image_ids=image_ids,
            strategy=ext.get("strategy", "auto"),
            force=bool(ext.get("force", False)),
        )
        stream = str(body.get("stream", "false")).lower() in ("1", "true")
        return await _respond(job, stream=stream, response_format=body.get("response_format"))

    @app.post("/v1/uploads", dependencies=[Depends(auth)])
    async def uploads(request: Request):
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise RequestError("missing_file", "Send the image as multipart field 'file'.")
        rid = _save_upload(engine, getattr(upload, "filename", None), await upload.read())
        return {"id": rid}

    @app.get("/v1/results/{result_id}", dependencies=[Depends(auth)])
    async def result(result_id: str):
        return FileResponse(engine.results.path(result_id), media_type="image/png")

    @app.get("/v1/results/{result_id}/meta", dependencies=[Depends(auth)])
    async def result_meta(result_id: str):
        return engine.results.meta(result_id)

    # -- jobs -------------------------------------------------------------------------
    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(auth)])
    async def job_status(job_id: str):
        return engine.job(job_id).as_dict()

    @app.get("/v1/jobs/{job_id}/events", dependencies=[Depends(auth)])
    async def job_events(job_id: str):
        return _sse(engine.job(job_id))

    @app.delete("/v1/jobs/{job_id}", dependencies=[Depends(auth)])
    async def job_cancel(job_id: str):
        return (await asyncio.to_thread(engine.cancel, job_id)).as_dict()

    @app.post("/v1/jobs/{job_id}/retry", dependencies=[Depends(auth)])
    async def job_retry(job_id: str, request: Request):
        body = await _json_or_empty(request)
        job = await asyncio.to_thread(engine.retry_workflow, job_id, body.get("from_step"))
        return await _respond(job, stream=bool(body.get("stream")), response_format=body.get("response_format"))

    # -- admin ------------------------------------------------------------------------
    @app.post("/admin/models/{model_id}/pull", dependencies=[Depends(auth)])
    async def pull(model_id: str):
        return engine.pull(model_id)

    @app.delete("/admin/models/{model_id}/pull", dependencies=[Depends(auth)])
    async def cancel_pull(model_id: str):
        return await asyncio.to_thread(engine.cancel_pull, model_id)

    @app.post("/admin/models/{model_id}/load", dependencies=[Depends(auth)])
    async def load(model_id: str, request: Request):
        body = await _json_or_empty(request)
        return await asyncio.to_thread(engine.load, model_id, bool(body.get("pin", False)))

    @app.post("/admin/models/{model_id}/unload", dependencies=[Depends(auth)])
    async def unload(model_id: str):
        return await asyncio.to_thread(engine.unload, model_id)

    @app.get("/admin/catalog/suggestions", dependencies=[Depends(auth)])
    async def catalog_suggestions():
        from dmlx import catalog

        try:
            repos = await asyncio.to_thread(catalog.suggestions)
        except Exception as exc:
            raise RequestError("catalog_unavailable", f"Could not reach Hugging Face: {exc}", 502) from exc
        added = {s.repo.lower() for s in engine.registry.all()}
        total_gb = engine.memory.total / 1024**3
        return {"data": [{**r, "added": r["repo"].lower() in added,
                          "fits_this_mac": r.get("min_ram_gb", 16) <= total_gb + 0.5} for r in repos]}

    @app.post("/admin/catalog/sizes", dependencies=[Depends(auth)])
    async def catalog_sizes(request: Request):
        """Download sizes of a few repos (a model's versions), before anything is downloaded."""
        from dmlx import catalog

        body = await _json_or_empty(request)
        repos = [str(r) for r in body.get("repos") or [] if isinstance(r, str)][:16]
        return {"sizes": await asyncio.to_thread(catalog.repo_sizes, repos)}

    @app.post("/admin/models/inspect", dependencies=[Depends(auth)])
    async def inspect_model(request: Request):
        body = await _json_or_empty(request)
        return await asyncio.to_thread(engine.inspect_model, str(body.get("repo", "")))

    @app.post("/admin/models", dependencies=[Depends(auth)])
    async def add_model(request: Request):
        body = await _json_or_empty(request)
        return await asyncio.to_thread(engine.add_model, str(body.get("repo", "")), bool(body.get("download", True)))

    @app.delete("/admin/models/{model_id}/files", dependencies=[Depends(auth)])
    async def delete_model_files(model_id: str):
        return await asyncio.to_thread(engine.delete_model_files, model_id)

    @app.delete("/admin/models/{model_id}", dependencies=[Depends(auth)])
    async def remove_model(model_id: str):
        return await asyncio.to_thread(engine.remove_model, model_id)

    @app.get("/admin/prompt-assistant", dependencies=[Depends(auth)])
    def prompt_assistant():
        return engine.prompt_assistant_status()

    @app.post("/admin/prompt-assistant/pull", dependencies=[Depends(auth)])
    async def pull_prompt_assistant(request: Request):
        body = await request.json() if await request.body() else {}
        return engine.pull_prompt_assistant(body.get("model") if isinstance(body, dict) else None)

    @app.post("/admin/prompt-assistant/select", dependencies=[Depends(auth)])
    async def select_prompt_assistant(request: Request):
        body = await request.json()
        model = body.get("model") if isinstance(body, dict) else None
        if not isinstance(model, str):
            raise RequestError("invalid_request", "Send {\"model\": \"<id>\"}.", 422)
        return await asyncio.to_thread(engine.select_prompt_assistant, model)

    @app.post("/admin/prompt-assistant/provider", dependencies=[Depends(auth)])
    async def set_agent_provider(request: Request):
        body = await request.json()
        if not isinstance(body, dict):
            raise RequestError("invalid_request", "Send {\"kind\": \"local\"} or a service.", 422)
        return await asyncio.to_thread(engine.set_agent_provider, body)

    @app.post("/admin/prompt-assistant/provider/models", dependencies=[Depends(auth)])
    async def agent_service_models(request: Request):
        body = await request.json()
        return await asyncio.to_thread(engine.agent_service_models, body if isinstance(body, dict) else {})

    @app.get("/admin/memory", dependencies=[Depends(auth)])
    async def memory():
        return await asyncio.to_thread(engine.status)

    # -- helpers ----------------------------------------------------------------------
    async def _respond(job: Job, stream: bool, response_format: str | None):
        if stream:
            return _sse(job)
        await asyncio.to_thread(job.finished.wait)
        if job.status != "done":
            status = 507 if (job.error or {}).get("code") in ("insufficient_memory", "memory_pressure") else 500
            if job.status == "cancelled":
                status = 409
            return JSONResponse(status_code=status, content={"error": job.error, "job": job.as_dict()})
        data = []
        for rid in job.result_ids:
            item = {"result_id": rid, "url": f"/v1/results/{rid}"}
            if response_format != "url":
                item["b64_json"] = base64.b64encode(engine.results.path(rid).read_bytes()).decode()
            data.append(item)
        return {"created": int(time.time()), "data": data, "dmlx": {"job_id": job.id, **job.report}}

    def _sse(job: Job) -> StreamingResponse:
        async def gen():
            q = job.subscribe()
            try:
                while True:
                    try:
                        event, payload = await asyncio.wait_for(q.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    yield f"event: {event}\ndata: {json.dumps(payload)}\n\n"
                    if event in TERMINAL_EVENTS:
                        break
            finally:
                job.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    return app


def _int(value):
    if value in (None, ""):
        return None
    return int(value)


async def _json_or_empty(request: Request) -> dict:
    try:
        return await request.json()
    except Exception:  # noqa: BLE001
        return {}
