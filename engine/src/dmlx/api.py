"""HTTP API: OpenAI-compatible images endpoints, job events over SSE, admin endpoints."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from dmlx import __version__
from dmlx.engine import Engine, Job, RequestError

TERMINAL_EVENTS = {"done", "error", "cancelled"}


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
        return JSONResponse(status_code=404, content={"error": {"code": "not_found", "message": str(exc).strip("'\"")}})

    # -- health / models --------------------------------------------------------------
    @app.get("/health")
    async def health():
        return {"ok": True, "version": __version__}

    @app.get("/v1/models", dependencies=[Depends(auth)])
    async def models():
        return {"object": "list", "data": engine.models()}

    # -- images -----------------------------------------------------------------------
    @app.post("/v1/images/generations", dependencies=[Depends(auth)])
    async def generations(request: Request):
        body = await request.json()
        ext = body.get("dmlx") or {}
        job = engine.submit(
            task="t2i",
            prompt=body.get("prompt", ""),
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
                        image_ids.append(engine.results.save_upload(await upload.read()))
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
        rid = engine.results.save_upload(await upload.read())
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
        return engine.cancel(job_id).as_dict()

    # -- admin ------------------------------------------------------------------------
    @app.post("/admin/models/{model_id}/pull", dependencies=[Depends(auth)])
    async def pull(model_id: str):
        return engine.pull(model_id)

    @app.post("/admin/models/{model_id}/load", dependencies=[Depends(auth)])
    async def load(model_id: str, request: Request):
        body = await _json_or_empty(request)
        return await asyncio.to_thread(engine.load, model_id, bool(body.get("pin", False)))

    @app.post("/admin/models/{model_id}/unload", dependencies=[Depends(auth)])
    async def unload(model_id: str):
        return await asyncio.to_thread(engine.unload, model_id)

    @app.get("/admin/memory", dependencies=[Depends(auth)])
    async def memory():
        return engine.status()

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
