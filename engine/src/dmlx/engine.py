"""Engine core: model catalog + memory manager + single GPU worker with a job queue."""

from __future__ import annotations

import asyncio
import logging
import queue
import random
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from dmlx import downloads, sysmem
from dmlx.adapters.base import GenRequest, JobAborted
from dmlx.memory import GB, InsufficientMemory, JobGuard, MemoryManager
from dmlx.registry import ModelSpec, Registry
from dmlx.results import ResultStore

log = logging.getLogger("dmlx.engine")

MAX_SIDE = 2048
MIN_SIDE = 256


class RequestError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


@dataclass
class Job:
    id: str
    model_id: str
    task: str
    request: GenRequest
    n: int = 1
    strategy: str = "auto"
    force: bool = False
    status: str = "queued"  # queued | running | done | error | cancelled
    created: float = field(default_factory=time.time)
    events: list[tuple[str, dict]] = field(default_factory=list)
    result_ids: list[str] = field(default_factory=list)
    report: dict = field(default_factory=dict)
    error: dict | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    _subscribers: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, event: str, data: dict) -> None:
        payload = {"job_id": self.id, **data}
        with self._lock:
            self.events.append((event, payload))
            subscribers = list(self._subscribers)
        for loop, q in subscribers:
            loop.call_soon_threadsafe(q.put_nowait, (event, payload))

    def subscribe(self) -> asyncio.Queue:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            for item in self.events:
                q.put_nowait(item)
            self._subscribers.append((loop, q))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers = [(l, s) for (l, s) in self._subscribers if s is not q]

    @property
    def terminal(self) -> bool:
        return self.status in ("done", "error", "cancelled")

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "model": self.model_id,
            "task": self.task,
            "status": self.status,
            "created": self.created,
            "result_ids": self.result_ids,
            "report": self.report,
            "error": self.error,
        }


class Engine:
    def __init__(self, registry: Registry | None = None, results: ResultStore | None = None, adapter_factory=None):
        self.registry = registry or Registry.load_builtin()
        self.results = results or ResultStore()
        self.memory = MemoryManager()
        self.downloader = downloads.Downloader()
        self._adapter_factory = adapter_factory or _default_adapter_factory
        self._adapters: dict[str, object] = {}
        self._jobs: dict[str, Job] = {}
        self._queue: "queue.Queue[Job]" = queue.Queue()
        self._lock = threading.Lock()
        self.started = time.time()
        for spec in self.registry.all():
            path = self._resolve_local(spec)
            if path:
                downloads.fill_component_sizes(spec, path)
        self._worker = threading.Thread(target=self._work, name="dmlx-gpu", daemon=True)
        self._worker.start()

    # -- catalog ----------------------------------------------------------------------
    def _resolve_local(self, spec: ModelSpec) -> Path | None:
        if spec.local_path is not None:
            return spec.local_path
        return downloads.local_path(spec)

    def is_downloaded(self, spec: ModelSpec) -> bool:
        return self._resolve_local(spec) is not None

    def models(self) -> list[dict]:
        total_gb = self.memory.total / GB
        resident = self.memory.resident_models()
        out = []
        for spec in self.registry.all():
            out.append(
                {
                    **spec.as_dict(),
                    "object": "model",
                    "downloaded": self.is_downloaded(spec),
                    "download": self.downloader.state(spec.id).as_dict(),
                    "resident": spec.id in resident,
                    "pinned": self.memory.is_pinned(spec.id),
                    "fits_this_mac": spec.min_ram_gb <= total_gb + 0.5,
                }
            )
        return out

    def adapter(self, spec: ModelSpec):
        with self._lock:
            if spec.id not in self._adapters:
                path = self._resolve_local(spec)
                if path is None:
                    raise RequestError("model_not_downloaded", f"{spec.name} is not downloaded yet.", 409)
                adapter = self._adapter_factory(spec, path)
                self._adapters[spec.id] = adapter
                self.memory.register_adapter(spec.id, adapter)
            return self._adapters[spec.id]

    def pull(self, model_id: str) -> dict:
        spec = self.registry.get(model_id)
        if self.is_downloaded(spec):
            state = self.downloader.state(spec.id)
            state.status = "done"
            return state.as_dict()
        return self.downloader.start(spec).as_dict()

    def load(self, model_id: str, pin: bool = False) -> dict:
        spec = self.registry.get(model_id)
        adapter = self.adapter(spec)
        self.memory.pin(spec.id, pin)
        plan = self.memory.plan(spec, spec.activation_bytes(1024, 1024), strategy="auto")
        if hasattr(adapter, "stream_mode"):
            adapter.stream_mode = plan.strategy == "streamed"
        names = list(spec.components) if plan.strategy == "eager" else [n for n, c in spec.components.items() if c.keep or n == "transformer"]
        for name in names:
            self.memory.ensure(spec.id, name)
        del adapter
        return {"model": spec.id, "strategy": plan.strategy, "resident": self.memory.resident_components()}

    def unload(self, model_id: str) -> dict:
        spec = self.registry.get(model_id)
        self.memory.pin(spec.id, False)
        self.memory.unload_model(spec.id)
        return {"model": spec.id, "resident": self.memory.resident_components()}

    # -- jobs -------------------------------------------------------------------------
    def submit(
        self,
        *,
        task: str,
        prompt: str,
        model: str | None = None,
        size: str | None = None,
        steps: int | None = None,
        seed: int | None = None,
        n: int = 1,
        image_ids: list[str] | None = None,
        strategy: str = "auto",
        force: bool = False,
    ) -> Job:
        if not prompt or not prompt.strip():
            raise RequestError("invalid_prompt", "Prompt is empty.")
        if task not in ("t2i", "edit"):
            raise RequestError("invalid_task", f"Unknown task {task}")
        if model:
            if model not in self.registry:
                raise RequestError("model_not_found", f"Unknown model '{model}'.", 404)
            spec = self.registry.get(model)
        else:
            spec = self.registry.default_for(task, self.memory.total / GB)
            if spec is None:
                raise RequestError("no_model", f"No model on this Mac can do {task}.", 409)
        if not spec.supports(task):
            raise RequestError("unsupported_task", f"{spec.name} cannot do {task}.")
        if not self.is_downloaded(spec):
            raise RequestError("model_not_downloaded", f"{spec.name} is not downloaded yet.", 409)

        image_paths: list[Path] = []
        for rid in image_ids or []:
            try:
                image_paths.append(self.results.path(rid))
            except KeyError:
                raise RequestError("image_not_found", f"No image with id {rid}.", 404) from None
        if task == "edit":
            if not image_paths:
                raise RequestError("missing_image", "Editing needs a source image.")
            if len(image_paths) > max(spec.max_references, 1):
                raise RequestError("too_many_images", f"{spec.name} accepts at most {spec.max_references} images.")

        width, height = self._parse_size(size, spec, image_paths)
        req = GenRequest(
            task=task,
            prompt=prompt.strip(),
            width=width,
            height=height,
            steps=int(steps or spec.defaults.get("steps", 8)),
            seed=int(seed) if seed is not None else random.randint(0, 2**31 - 1),
            guidance=spec.defaults.get("guidance"),
            image_paths=image_paths,
        )
        if not 1 <= req.steps <= 100:
            raise RequestError("invalid_steps", "steps must be between 1 and 100.")
        if strategy not in ("auto", "eager", "phase_free", "sequential", "streamed"):
            raise RequestError("invalid_strategy", f"Unknown strategy {strategy}")
        job = Job(id=f"job_{secrets.token_hex(6)}", model_id=spec.id, task=task, request=req,
                  n=max(1, min(int(n), 4)), strategy=strategy, force=force)
        with self._lock:
            self._jobs[job.id] = job
            position = sum(1 for j in self._jobs.values() if j.status == "queued")
        job.emit("queued", {"position": position, "model": spec.id, "width": width, "height": height,
                            "steps": req.steps, "seed": req.seed, "n": job.n})
        self._queue.put(job)
        return job

    @staticmethod
    def _parse_size(size: str | None, spec: ModelSpec, image_paths: list[Path]) -> tuple[int, int]:
        if size in (None, "", "auto"):
            if image_paths:
                from PIL import Image

                with Image.open(image_paths[0]) as im:
                    w, h = im.size
                scale = min(1.0, (1024 * 1024 / (w * h)) ** 0.5)
                w, h = int(w * scale), int(h * scale)
            else:
                w, h = int(spec.defaults.get("width", 1024)), int(spec.defaults.get("height", 1024))
        else:
            try:
                w, h = (int(v) for v in size.lower().split("x"))
            except ValueError:
                raise RequestError("invalid_size", f"size must look like 1024x1024, got {size!r}") from None
        if not (MIN_SIDE <= w <= MAX_SIDE and MIN_SIDE <= h <= MAX_SIDE):
            raise RequestError("invalid_size", f"Each side must be between {MIN_SIDE} and {MAX_SIDE} pixels.")
        return (w // 16) * 16, (h // 16) * 16

    def job(self, job_id: str) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise RequestError("job_not_found", f"No job {job_id}", 404) from None

    def cancel(self, job_id: str) -> Job:
        job = self.job(job_id)
        job.cancel_event.set()
        if job.status == "queued":
            job.status = "cancelled"
            job.emit("cancelled", {"message": "Cancelled before start"})
            job.finished.set()
        return job

    def queue_depth(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status in ("queued", "running"))

    # -- worker -----------------------------------------------------------------------
    def _work(self) -> None:
        while True:
            job = self._queue.get()
            if job.status == "cancelled":
                continue
            try:
                self._run(job)
            except Exception as exc:  # noqa: BLE001 - never kill the worker
                log.exception("job %s crashed", job.id)
                job.status = "error"
                job.error = {"code": "internal_error", "message": str(exc)}
                job.emit("error", job.error)
            finally:
                job.finished.set()

    def _run(self, job: Job) -> None:
        spec = self.registry.get(job.model_id)
        job.status = "running"
        started = time.monotonic()
        try:
            adapter = self.adapter(spec)
            cached = job.n == 1 and getattr(adapter, "has_embedding", lambda _p: False)(job.request.prompt)
            plan = self.memory.plan(
                spec,
                spec.activation_bytes(job.request.width, job.request.height, len(job.request.image_paths)),
                strategy=job.strategy,
                force=job.force,
                skip=("text_encoder",) if cached else (),
            )
        except InsufficientMemory as exc:
            job.status = "error"
            job.error = {"code": "insufficient_memory", "message": str(exc),
                         "need_gb": round(exc.need / GB, 2), "available_gb": round(max(exc.headroom, 0) / GB, 2)}
            job.emit("error", job.error)
            return
        except RequestError as exc:
            job.status = "error"
            job.error = {"code": exc.code, "message": exc.message}
            job.emit("error", job.error)
            return

        import mlx.core as mx

        mx.reset_peak_memory()
        guard = JobGuard(self.memory, spec, plan, job.cancel_event, job.emit)
        guard.size = (job.request.width, job.request.height)
        job.emit("plan", plan.as_dict())
        try:
            guard.start()
            for i in range(job.n):
                req = job.request if i == 0 else GenRequest(**{**job.request.__dict__, "seed": job.request.seed + i})
                gen_started = time.monotonic()
                image = adapter.generate(req, guard)
                meta = {
                    "kind": "generation",
                    "model": spec.id,
                    "task": job.task,
                    "prompt": req.prompt,
                    "width": req.width,
                    "height": req.height,
                    "steps": req.steps,
                    "seed": req.seed,
                    "sources": [p.stem for p in req.image_paths],
                    "strategy": plan.strategy,
                    "seconds": round(time.monotonic() - gen_started, 3),
                    "created": time.time(),
                }
                rid = self.results.save(image, meta)
                job.result_ids.append(rid)
                job.emit("image", {"index": i, "result_id": rid, "seed": req.seed})
        except JobAborted as exc:
            job.status = "cancelled" if exc.code == "cancelled" else "error"
            job.error = {"code": exc.code, "message": exc.message}
            if exc.code == "memory_pressure":
                self.memory.unload_model(spec.id)
            job.report = {**guard.report(), "total_seconds": round(time.monotonic() - started, 3)}
            job.emit("cancelled" if exc.code == "cancelled" else "error", {**job.error, "report": job.report})
            return
        job.report = {
            **guard.report(),
            "total_seconds": round(time.monotonic() - started, 3),
            "embed_cache": adapter.embed_cache_stats(),
        }
        job.status = "done"
        job.emit("done", {"result_ids": job.result_ids, "report": job.report})

    def status(self) -> dict:
        return {
            "uptime_s": round(time.time() - self.started, 1),
            "queue": self.queue_depth(),
            "memory": self.memory.status(),
            "ram_tier_gb": round(self.memory.total / GB),
            "emulated": bool(__import__("os").environ.get("DMLX_EMULATE_TOTAL_GB")),
            "real_total_gb": round(sysmem._sysctl_int("hw.memsize") / GB),
        }


def _default_adapter_factory(spec: ModelSpec, path: Path):
    if spec.adapter.startswith("fake"):
        from dmlx.adapters.fake import FakeAdapter

        return FakeAdapter(spec, path)
    from dmlx.adapters.mflux_adapter import ADAPTERS

    try:
        cls = ADAPTERS[spec.adapter]
    except KeyError:
        raise RequestError("adapter_missing", f"No adapter {spec.adapter}", 500) from None
    return cls(spec, path)
