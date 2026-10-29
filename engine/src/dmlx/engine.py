"""Engine core: model catalog + memory manager + single GPU worker with a job queue."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import random
import secrets
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from dmlx import catalog, downloads, paths, sysmem
from dmlx.adapters.base import GenRequest, JobAborted
from dmlx.intelligence.base import IntelligenceError
from dmlx.intelligence.openai_compatible import OpenAICompatibleProvider
from dmlx.intelligence.local_llm import LocalLLMDecisionProvider
from dmlx.memory import GB, InsufficientMemory, JobGuard, MemoryManager
from dmlx.prompting import OpenAICompatiblePromptProvider, agent, local_vlm
from dmlx.registry import ModelSpec, Registry
from dmlx.results import ResultStore
from dmlx.routing import WorkflowRouter
from dmlx.routing.router import DEFAULT_DESCRIPTIONS
from dmlx.workflow import WorkflowError, WorkflowPlan

log = logging.getLogger("dmlx.engine")

MAX_SIDE = 2048
MIN_SIDE = 256


def _router_from_environment() -> WorkflowRouter:
    """Build the optional sidecar router without coupling it to diffusion models.

    A partial or invalid configuration deliberately leaves routing manual. This
    keeps generation available even when an external intelligence service is
    unavailable or misconfigured.
    """
    kind = os.environ.get("DMLX_ROUTER_PROVIDER", "openai-compatible").strip().lower()
    if kind in {"disabled", "manual", "none"}:
        return WorkflowRouter()
    if kind in {"local-llm", "mlx-lm", "qwen"}:
        model_path = os.environ.get("DMLX_LOCAL_LLM_MODEL_PATH", "").strip()
        revision = os.environ.get("DMLX_LOCAL_LLM_REVISION", "").strip()
        if not model_path or not revision:
            return WorkflowRouter()
        try:
            return WorkflowRouter(LocalLLMDecisionProvider(
                model_path, revision=revision,
                max_tokens=int(os.environ.get("DMLX_LOCAL_LLM_MAX_TOKENS", "192")),
            ))
        except (TypeError, ValueError) as exc:
            log.warning("Ignoring invalid local LLM router configuration: %s", exc)
            return WorkflowRouter()
    if kind not in {"openai", "openai-compatible"}:
        log.warning("Ignoring unknown semantic router provider: %s", kind)
        return WorkflowRouter()
    base_url = os.environ.get("DMLX_ROUTER_BASE_URL", "").strip()
    model = os.environ.get("DMLX_ROUTER_MODEL", "").strip()
    if not base_url or not model:
        return WorkflowRouter()
    try:
        timeout = float(os.environ.get("DMLX_ROUTER_TIMEOUT", "15"))
        provider = OpenAICompatibleProvider(
            base_url, model, os.environ.get("DMLX_ROUTER_API_KEY") or None, timeout=max(0.1, timeout)
        )
        return WorkflowRouter(provider)
    except (TypeError, ValueError) as exc:
        log.warning("Ignoring invalid semantic router configuration: %s", exc)
        return WorkflowRouter()


def _prompt_provider_from_environment():
    base_url = os.environ.get("DMLX_PROMPT_BASE_URL", "").strip()
    model = os.environ.get("DMLX_PROMPT_MODEL", "").strip()
    if not base_url or not model:
        return None
    try:
        timeout = max(0.1, float(os.environ.get("DMLX_PROMPT_TIMEOUT", "30")))
        return OpenAICompatiblePromptProvider(
            base_url, model, os.environ.get("DMLX_PROMPT_API_KEY") or None, timeout=timeout
        )
    except (TypeError, ValueError) as exc:
        log.warning("Ignoring invalid prompt provider configuration: %s", exc)
        return None


def selected_assistant_model() -> str:
    """The agent model the user picked (saved by the app), or the default."""
    chosen = os.environ.get("DMLX_PROMPT_ASSISTANT_MODEL", "").strip()
    if not chosen:
        try:
            chosen = json.loads((paths.home() / "prompt-assistant.json").read_text()).get("model", "")
        except (OSError, ValueError, AttributeError):
            chosen = ""
    if chosen in local_vlm.MODELS:
        return chosen
    # Nothing picked yet: the default for this Mac, unless only another model is already downloaded.
    default = local_vlm.default_model(sysmem.total_memory())
    if local_vlm.locate(default) is None:
        return next((m for m in local_vlm.MODELS if local_vlm.locate(m) is not None), default)
    return default


def _prompt_assistant_from_disk(model_id: str | None = None):
    """The local image-aware prompt assistant, when its pinned weights and runtime are installed."""
    if os.environ.get("DMLX_PROMPT_ASSISTANT", "").strip().lower() in {"off", "0", "false", "disabled"}:
        return None
    model = local_vlm.MODELS[model_id or selected_assistant_model()]
    path = local_vlm.locate(model.id)
    if path is None:
        return None
    import importlib.util

    if importlib.util.find_spec("mlx_vlm") is None:
        log.warning("The prompt assistant is downloaded but mlx-vlm is not installed")
        return None
    return local_vlm.LocalVLMPromptProvider(path, model=model)


def prompt_assistant_spec(model_id: str | None = None) -> ModelSpec:
    """Download descriptor for an agent model; it is not an image model and never joins the registry."""
    model = local_vlm.MODELS[model_id or selected_assistant_model()]
    return ModelSpec(
        id=f"prompt-assistant:{model.id}", name=f"Prompt assistant ({model.name})", adapter="none", repo=model.repo,
        license=model.license, tasks=[], components={}, defaults={}, revision=model.revision,
    )


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
    workflow: dict | None = None
    workflow_plan: WorkflowPlan | None = field(default=None, repr=False)
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
    _forward: Callable[[str, dict], None] | None = field(default=None, repr=False)

    def emit(self, event: str, data: dict) -> None:
        payload = {"job_id": self.id, **data}
        with self._lock:
            self.events.append((event, payload))
            subscribers = list(self._subscribers)
        for loop, q in subscribers:
            loop.call_soon_threadsafe(q.put_nowait, (event, payload))
        if self._forward is not None:
            self._forward(event, payload)

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
    def __init__(
        self, registry: Registry | None = None, results: ResultStore | None = None, adapter_factory=None, router=None
    ):
        self.registry = registry or Registry.load_builtin(extra_dir=paths.models_dir())
        self.results = results or ResultStore()
        self.memory = MemoryManager()
        self.downloader = downloads.Downloader()
        self._adapter_factory = adapter_factory or _default_adapter_factory
        self.router = router or _router_from_environment()
        self.prompt_provider = _prompt_provider_from_environment()
        self.prompt_assistant = _prompt_assistant_from_disk()
        self._adapters: dict[str, object] = {}
        self._jobs: dict[str, Job] = {}
        self._queue: "queue.Queue[Job]" = queue.Queue()
        self._lock = threading.Lock()
        self._accelerator_lock = threading.Lock()
        self.started = time.time()
        for spec in self.registry.all():
            path = self._resolve_local(spec)
            if path:
                downloads.fill_component_sizes(spec, path)
        self._recover_workflow_jobs()
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

    def capabilities(self) -> dict:
        """Installed and catalogued executors grouped by workflow operation."""
        operations: dict[str, list[dict]] = {}
        for spec in self.registry.all():
            for name, capability in spec.capabilities.items():
                operations.setdefault(name, []).append(
                    {
                        "model": spec.id,
                        "name": spec.name,
                        "downloaded": self.is_downloaded(spec),
                        "fits_this_mac": spec.min_ram_gb <= self.memory.total / GB + 0.5,
                        **capability.as_dict(),
                    }
                )
        return {"schema_version": 1, "operations": operations}

    def route(self, state: dict) -> dict:
        if not isinstance(state, dict):
            raise RequestError("invalid_route", "Routing state must be a JSON object.", 422)
        message = str(state.get("message", "")).strip()
        if not message or len(message) > 8_000:
            raise RequestError("invalid_route", "Routing message must contain 1 to 8,000 characters.", 422)
        attachments = state.get("attachments") or []
        if not isinstance(attachments, list) or len(attachments) > 16 or not all(isinstance(item, str) for item in attachments):
            raise RequestError("invalid_route", "Routing attachments must be a list of at most 16 IDs.", 422)
        raw_history = state.get("history") or []
        if not isinstance(raw_history, list) or len(raw_history) > 24 or not all(isinstance(item, dict) for item in raw_history):
            raise RequestError("invalid_route", "Routing history must contain at most 24 objects.", 422)
        history = []
        for item in raw_history:
            input_ids = item.get("input_ids") or []
            if not isinstance(input_ids, list) or len(input_ids) > 16 or not all(isinstance(value, str) for value in input_ids):
                raise RequestError("invalid_route", "Routing history input_ids must contain at most 16 IDs.", 422)
            try:
                sequence = int(item.get("sequence", len(history)))
            except (TypeError, ValueError):
                raise RequestError("invalid_route", "Routing history sequence must be an integer.", 422) from None
            try:
                image_number = int(item["image_number"]) if item.get("image_number") is not None else None
            except (TypeError, ValueError):
                raise RequestError("invalid_route", "Routing history image_number must be an integer.", 422) from None
            kind = item.get("kind")
            text = item.get("text")
            operation = item.get("operation")
            result_id = item.get("result_id")
            history.append({
                "sequence": sequence,
                "kind": str(kind)[:32] if kind is not None else "",
                "text": str(text)[:2_000] if text is not None else "",
                "image_number": image_number,
                "operation": str(operation)[:128] if operation is not None else None,
                "result_id": str(result_id)[:256] if result_id is not None else None,
                "input_ids": [value[:256] for value in input_ids],
            })
        history.sort(key=lambda item: item["sequence"])
        current_prompt = str(state.get("current_prompt") or "")[:8_000]
        safe_state = {
            "message": message,
            "current_result_id": str(state.get("current_result_id") or "")[:256] or None,
            "current_prompt": current_prompt or None,
            "attachments": attachments,
            "history": history,
        }
        explicit_count = len(attachments)
        has_current = bool(safe_state["current_result_id"])
        available: set[str] = set()
        selected_model = str(state.get("selected_model", "")).strip() or None
        if selected_model:
            try:
                selected_specs = [self.registry.get(selected_model)]
            except KeyError as exc:
                raise RequestError("unknown_model", str(exc), 404) from exc
        else:
            selected_specs = [spec for spec in self.registry.all() if spec.auto_select]
        for spec in selected_specs:
            if not self.is_downloaded(spec):
                continue
            if selected_model is None and spec.min_ram_gb > self.memory.total / GB + 0.5:
                continue
            for operation, capability in spec.capabilities.items():
                constraint = capability.inputs.get("image")
                input_count = explicit_count if explicit_count else (1 if has_current and constraint else 0)
                if constraint:
                    if constraint.minimum <= input_count <= constraint.maximum:
                        available.add(operation)
                elif input_count == 0:
                    available.add(operation)
        requested = state.get("candidates")
        if requested is not None and not isinstance(requested, list):
            raise RequestError("invalid_route", "Routing candidates must be a list.", 422)
        candidates = [str(item) for item in requested] if requested else sorted(available)
        candidates = [item for item in candidates if item in available]
        # With no image and one possible workflow there is nothing to decide; the job's own rewrite
        # (Auto mode) still writes the prompt, so planning would only load the model for nothing.
        trivial = len(candidates) == 1 and not explicit_count and not has_current
        note = None
        if state.get("plan") and self.prompt_assistant is not None and candidates and not trivial:
            planned, note = self._plan(safe_state, candidates, wait=bool(state.get("wait")))
            if planned is not None:
                return planned
        decision = self._route_without_agent(safe_state, candidates)
        if note:
            decision["planning"] = note  # e.g. "skipped_busy": the client can tell the agent did not plan this
        return decision

    def _route_without_agent(self, safe_state: dict, candidates: list[str]) -> dict:
        deterministic = self.router.deterministic_decision(safe_state, candidates)
        if deterministic is not None:
            return deterministic.as_dict()
        provider = self.router.provider
        if provider is not None and getattr(provider, "uses_accelerator", False):
            if not self._accelerator_lock.acquire(blocking=False):
                return {
                    "intent": "ambiguous", "confidence": 0.0,
                    "probabilities": {name: 0.0 for name in candidates},
                    "provider": f"{provider.name}:busy",
                }
            try:
                return self.router.decide_with_provider(safe_state, candidates).as_dict()
            finally:
                self._accelerator_lock.release()
        return self.router.decide_with_provider(safe_state, candidates).as_dict()

    def test_router(self) -> dict:
        provider = self.router.provider
        if provider is None:
            raise RequestError("router_not_configured", "No semantic routing provider is configured.", 409)
        started = time.monotonic()
        acquired = False
        try:
            if getattr(provider, "uses_accelerator", False):
                acquired = self._accelerator_lock.acquire(blocking=False)
                if not acquired:
                    raise RequestError("router_busy", "The accelerator is currently running a generation job.", 409)
            decision = provider.decide(
                {"message": "Create a new image of a blue circle.", "current_result_id": "test-current"},
                {
                    "image.generate": "Create a new image independent of the current result.",
                    "image.edit": "Change the current image.",
                },
            )
        except RequestError:
            raise
        except Exception as exc:
            raise RequestError("router_unavailable", f"Routing provider test failed: {exc}", 502) from exc
        finally:
            if acquired:
                self._accelerator_lock.release()
        return {
            "ok": True,
            "provider": provider.name,
            "intent": decision.intent,
            "latency_ms": round((time.monotonic() - started) * 1_000, 1),
        }

    def compile_prompt(self, body: dict) -> dict:
        text = str(body.get("prompt", "")).strip()
        if not text or len(text) > 8_000:
            raise RequestError("invalid_prompt", "Prompt must contain 1 to 8,000 characters.", 422)
        provider = self.enhancer()
        if provider is None:
            raise RequestError("prompt_provider_not_configured", "No prompt enhancement provider is configured.", 409)
        operation = str(body.get("operation", "image.generate"))
        model_id = body.get("model")
        spec = self._spec(str(model_id)) if model_id else self.registry.default_for_operation(
            operation, self.memory.total / GB)
        capability = spec.capabilities.get(operation) if spec else None
        if capability is None:
            raise RequestError("unsupported_operation", f"No selected model supports {operation}.", 422)
        if provider is self.prompt_assistant:
            images = []
            for rid in body.get("image_ids") or []:
                try:
                    images.append(self.results.path(str(rid)))
                except KeyError:
                    raise RequestError("image_not_found", f"No image with id {rid}.", 404) from None
            compiled = self._assist(text, operation=operation, images=images if operation == "image.edit" else [],
                                    supports_negative=capability.negative_prompt)
        else:
            try:
                compiled = provider.enhance(text, supports_negative=capability.negative_prompt)
            except Exception as exc:
                raise RequestError("prompt_provider_unavailable", f"Prompt enhancement failed: {exc}", 502) from exc
        return {
            "user": text,
            "positive": compiled.positive,
            "negative": compiled.negative,
            "compiler": compiled.compiler,
            "negative_supported": capability.negative_prompt,
        }

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

    # -- adding and removing models ---------------------------------------------------
    def inspect_model(self, text: str) -> dict:
        """What a Hugging Face repo is, before anything is downloaded."""
        try:
            found = catalog.inspect(text)
        except catalog.InspectError as exc:
            raise RequestError("unsupported_model", str(exc), 422) from exc
        existing = next((s.id for s in self.registry.all() if s.repo.lower() == found.repo.lower()), None)
        return {**found.as_dict(), "already_added": existing}

    def add_model(self, text: str, download: bool = True) -> dict:
        try:
            found = catalog.inspect(text)
        except catalog.InspectError as exc:
            raise RequestError("unsupported_model", str(exc), 422) from exc
        existing = next((s for s in self.registry.all() if s.repo.lower() == found.repo.lower()), None)
        if existing is None:
            base, n = found.id, 2
            while found.id in self.registry:
                found.id, n = f"{base}-{n}", n + 1
            existing = catalog.save_spec(found)
            self.registry.add(existing)
            log.info("added model %s from %s", existing.id, existing.repo)
        path = self._resolve_local(existing)  # another tool may already have downloaded it
        if path:
            downloads.fill_component_sizes(existing, path)
        elif download:
            self.downloader.start(existing)
        return next(m for m in self.models() if m["id"] == existing.id)

    def _ensure_idle(self, spec: ModelSpec) -> None:
        if any(j.model_id == spec.id and not j.terminal for j in list(self._jobs.values())):
            raise RequestError("model_busy", f"{spec.name} is generating right now. Try again when it finishes.", 409)
        if self.downloader.state(spec.id).status == "downloading":
            raise RequestError("model_busy", f"{spec.name} is still downloading.", 409)

    def delete_model_files(self, model_id: str) -> dict:
        """Free the model from memory and delete its downloaded files; it can be downloaded again."""
        spec = self._spec(model_id)
        self._ensure_idle(spec)
        self.memory.unload_model(spec.id)
        with self._lock:
            self._adapters.pop(spec.id, None)
        self.memory.unregister_adapter(spec.id)
        freed = downloads.delete_local(spec)
        state = self.downloader.state(spec.id)
        state.status, state.done, state.error = "idle", 0, None
        log.info("deleted files of %s (%.2f GB)", spec.id, freed / GB)
        return {"model": spec.id, "freed_gb": round(freed / GB, 2)}

    def remove_model(self, model_id: str, delete_files: bool = True) -> dict:
        """Remove a model added from the app (built-in models can only have their files deleted)."""
        spec = self._spec(model_id)
        if not spec.user_added:
            raise RequestError("builtin_model", f"{spec.name} comes with dMLX; you can delete its files instead.", 409)
        freed = 0.0
        if delete_files:
            freed = self.delete_model_files(spec.id)["freed_gb"]
        else:
            self._ensure_idle(spec)
            self.memory.unload_model(spec.id)
        catalog.delete_spec(spec.id)
        self.registry.remove(spec.id)
        return {"model": spec.id, "removed": True, "freed_gb": freed}

    def _spec(self, model_id: str) -> ModelSpec:
        try:
            return self.registry.get(model_id)
        except KeyError as exc:
            raise RequestError("unknown_model", str(exc), 404) from exc

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
        negative_prompt: str | None = None,
        model: str | None = None,
        size: str | None = None,
        steps: int | None = None,
        seed: int | None = None,
        n: int = 1,
        image_ids: list[str] | None = None,
        strategy: str = "auto",
        force: bool = False,
        workflow: dict | None = None,
        operation: str | None = None,
        extras: dict | None = None,
        _enqueue: bool = True,
    ) -> Job:
        if not prompt or not prompt.strip():
            raise RequestError("invalid_prompt", "Prompt is empty.")
        legacy_call = operation is None
        if legacy_call and task not in ("t2i", "edit"):
            raise RequestError("invalid_task", f"Unknown task {task}")
        operation = operation or ("image.generate" if task == "t2i" else "image.edit")
        if model:
            if model not in self.registry:
                raise RequestError("model_not_found", f"Unknown model '{model}'.", 404)
            spec = self.registry.get(model)
        else:
            spec = self.registry.default_for_operation(operation, self.memory.total / GB)
            if spec is None:
                raise RequestError("no_model", f"No model on this Mac can perform {operation}.", 409)
        capability = spec.capabilities.get(operation)
        if capability is None:
            if legacy_call:
                raise RequestError("unsupported_task", f"{spec.name} cannot do {task}.")
            raise RequestError("unsupported_operation", f"{spec.name} cannot perform {operation}.", 422)
        if negative_prompt and (capability is None or not capability.negative_prompt):
            raise RequestError("negative_prompt_unsupported", f"{spec.name} does not support negative prompts.", 422)
        if not self.is_downloaded(spec):
            raise RequestError("model_not_downloaded", f"{spec.name} is not downloaded yet.", 409)

        image_paths: list[Path] = []
        for rid in image_ids or []:
            try:
                image_paths.append(self.results.path(rid))
            except KeyError:
                raise RequestError("image_not_found", f"No image with id {rid}.", 404) from None
        if legacy_call and task == "edit":
            if not image_paths:
                raise RequestError("missing_image", "Editing needs a source image.")
            if len(image_paths) > max(spec.max_references, 1):
                raise RequestError("too_many_images", f"{spec.name} accepts at most {spec.max_references} images.")
        input_constraint = capability.inputs.get("image")
        if input_constraint and not input_constraint.minimum <= len(image_paths) <= input_constraint.maximum:
            raise RequestError(
                "invalid_input_count",
                f"{spec.name} expects {input_constraint.minimum} to {input_constraint.maximum} image inputs for {operation}.",
                422,
            )
        if not input_constraint and image_paths:
            raise RequestError("unexpected_input", f"{operation} does not accept image inputs.", 422)

        width, height = self._parse_size(
            size, spec, image_paths, max_side=8192 if operation == "image.upscale" else MAX_SIDE)
        req = GenRequest(
            task=task,
            prompt=prompt.strip(),
            width=width,
            height=height,
            steps=int(steps or spec.defaults.get("steps", 8)),
            seed=int(seed) if seed is not None else random.randint(0, 2**31 - 1),
            negative_prompt=negative_prompt.strip() if negative_prompt else None,
            guidance=spec.defaults.get("guidance"),
            image_paths=image_paths,
            extras=dict(extras or {}),
        )
        if not spec.min_steps <= req.steps <= 100:
            raise RequestError(
                "invalid_steps", f"{spec.name} requires between {spec.min_steps} and 100 steps.")
        if strategy not in ("auto", "eager", "phase_free", "sequential", "streamed"):
            raise RequestError("invalid_strategy", f"Unknown strategy {strategy}")
        if workflow is None:
            workflow = WorkflowPlan.legacy(
                task=task,
                prompt=req.prompt,
                model=spec.id,
                input_ids=list(image_ids or []),
                parameters={"width": width, "height": height, "steps": req.steps, "seed": req.seed, "n": n},
                negative_prompt=req.negative_prompt,
            ).as_dict()
        job = Job(id=f"job_{secrets.token_hex(6)}", model_id=spec.id, task=task, request=req,
                  n=max(1, min(int(n), 4)), strategy=strategy, force=force, workflow=workflow)
        with self._lock:
            self._jobs[job.id] = job
            position = sum(1 for j in self._jobs.values() if j.status == "queued")
        job.emit("queued", {"position": position, "model": spec.id, "width": width, "height": height,
                            "steps": req.steps, "seed": req.seed, "n": job.n})
        if _enqueue:
            self._queue.put(job)
        return job

    def submit_workflow(self, plan: WorkflowPlan, _enqueue: bool = True) -> Job:
        """Validate a plan and enqueue it, or prepare one step for an active parent workflow."""
        if len(plan.steps) > 1:
            return self._submit_multi_workflow(plan)
        step = plan.steps[0]
        task = {"image.generate": "t2i", "image.edit": "edit"}.get(
            step.operation, step.operation.rsplit(".", 1)[-1])

        if step.executor:
            spec = self._spec(step.executor)
        else:
            spec = self.registry.default_for_operation(step.operation, self.memory.total / GB)
            if spec is None:
                raise RequestError(
                    "unsupported_operation", f"No executor is installed for operation '{step.operation}'.", 422)
        capability = spec.capabilities.get(step.operation)
        if capability is None:
            raise RequestError("unsupported_operation", f"{spec.name} cannot perform {step.operation}.", 422)
        if step.prompt.enhancement_mode == "auto" and self.prompt_assistant is None:
            provider = self.prompt_provider
            if provider is None:
                raise RequestError(
                    "prompt_provider_not_configured", "Automatic prompt enhancement is enabled but no provider is configured.", 409
                )
            try:
                compiled = provider.enhance(step.prompt.user, supports_negative=capability.negative_prompt)
            except Exception as exc:
                raise RequestError("prompt_provider_unavailable", f"Prompt enhancement failed: {exc}", 502) from exc
            prompt = replace(
                step.prompt, positive=compiled.positive, negative=compiled.negative, compiler=compiled.compiler
            )
            step = replace(step, prompt=prompt)
            plan = replace(plan, steps=(step,))
        if step.prompt.negative and not capability.negative_prompt:
            raise RequestError("negative_prompt_unsupported", f"{spec.name} does not support negative prompts.", 422)
        if any(item.media_type != "image" or item.role in ("mask", "sketch") for item in step.inputs):
            raise RequestError("unsupported_input", f"{spec.name} only accepts ordinary image inputs here.", 422)

        constraint = capability.inputs.get("image")
        count = len(step.inputs)
        if constraint and not constraint.minimum <= count <= constraint.maximum:
            raise RequestError(
                "invalid_input_count",
                f"{spec.name} expects {constraint.minimum} to {constraint.maximum} image inputs for {step.operation}.",
                422,
            )
        if not constraint and count:
            raise RequestError("unexpected_input", f"{step.operation} does not accept image inputs.", 422)

        p = step.parameters
        size = p.get("size")
        if not size and p.get("width") and p.get("height"):
            size = f"{int(p['width'])}x{int(p['height'])}"
        extras = {
            key: value for key, value in p.items()
            if key not in {"size", "width", "height", "steps", "seed", "n", "strategy", "force"}
        }
        if step.operation == "image.upscale":
            try:
                scale = int(extras.get("scale", 2))
                softness = float(extras.get("softness", 0.0))
            except (TypeError, ValueError):
                raise RequestError("invalid_upscale_parameters", "Upscale scale and softness must be numeric.", 422) from None
            if scale not in (2, 3, 4):
                raise RequestError("invalid_scale", "Upscale scale must be 2, 3, or 4.", 422)
            if not 0 <= softness <= 1:
                raise RequestError("invalid_softness", "Upscale softness must be between 0 and 1.", 422)
            source_path = self.results.path(step.inputs[0].id)
            from PIL import Image

            with Image.open(source_path) as source:
                target_width, target_height = source.width * scale, source.height * scale
            max_pixels = int(spec.defaults.get("max_output_pixels", 16_777_216))
            if target_width * target_height > max_pixels or max(target_width, target_height) > 8192:
                raise RequestError(
                    "upscale_too_large",
                    f"Requested upscale is {target_width}x{target_height}; this executor allows at most "
                    f"{max_pixels:,} output pixels and an 8192-pixel side.", 422,
                )
            size = f"{target_width}x{target_height}"
            extras["scale"] = scale
            extras["softness"] = softness
            normalized_parameters = dict(p)
            normalized_parameters.update({
                "width": target_width, "height": target_height,
                "scale": scale, "softness": softness,
            })
            step = replace(step, parameters=normalized_parameters)
            plan = replace(plan, steps=(step,))
        return self.submit(
            task=task,
            prompt=step.prompt.positive,
            negative_prompt=step.prompt.negative,
            model=spec.id,
            size=size,
            steps=p.get("steps"),
            seed=p.get("seed"),
            n=p.get("n", 1),
            image_ids=[item.id for item in step.inputs],
            strategy=p.get("strategy", "auto"),
            force=bool(p.get("force", False)),
            workflow=plan.as_dict(),
            operation=step.operation,
            extras=extras,
            _enqueue=_enqueue,
        )

    def _submit_multi_workflow(self, plan: WorkflowPlan) -> Job:
        if len(plan.steps) > 8:
            raise RequestError("too_many_steps", "A workflow can contain at most 8 steps.", 422)
        known_steps: set[str] = set()
        first_spec: ModelSpec | None = None
        for step in plan.steps:
            if step.id in known_steps:
                raise RequestError("duplicate_step", f"Workflow step id '{step.id}' is duplicated.", 422)
            spec = self._spec(step.executor) if step.executor else self.registry.default_for_operation(
                step.operation, self.memory.total / GB)
            if spec is None:
                raise RequestError("unsupported_operation", f"No executor is installed for '{step.operation}'.", 422)
            if first_spec is None:
                first_spec = spec
            capability = spec.capabilities.get(step.operation)
            if capability is None:
                raise RequestError("unsupported_operation", f"{spec.name} cannot perform {step.operation}.", 422)
            if not self.is_downloaded(spec):
                raise RequestError("model_not_downloaded", f"{spec.name} is not downloaded yet.", 409)
            if step.prompt.negative and not capability.negative_prompt:
                raise RequestError("negative_prompt_unsupported", f"{spec.name} does not support negative prompts.", 422)
            if step.prompt.enhancement_mode == "auto" and self.enhancer() is None:
                raise RequestError("prompt_provider_not_configured", "Automatic prompt enhancement has no provider.", 409)
            if int(step.parameters.get("n", 1)) != 1:
                raise RequestError("multi_output_step", "Multi-step workflows currently require n=1 per step.", 422)
            if any(item.media_type != "image" or item.role in ("mask", "sketch") for item in step.inputs):
                raise RequestError("unsupported_input", f"{spec.name} only accepts ordinary image inputs here.", 422)
            constraint = capability.inputs.get("image")
            count = len(step.inputs)
            if constraint and not constraint.minimum <= count <= constraint.maximum:
                raise RequestError("invalid_input_count", f"{spec.name} received {count} image inputs.", 422)
            if not constraint and count:
                raise RequestError("unexpected_input", f"{step.operation} does not accept image inputs.", 422)
            for item in step.inputs:
                if item.id.startswith("$"):
                    reference = item.id[1:]
                    if reference not in known_steps:
                        raise RequestError("invalid_step_reference", f"Step '{step.id}' references unavailable '{reference}'.", 422)
                else:
                    try:
                        self.results.path(item.id)
                    except KeyError:
                        raise RequestError("image_not_found", f"No image with id {item.id}.", 404) from None
            known_steps.add(step.id)

        assert first_spec is not None
        placeholder = GenRequest("workflow", "multi-step workflow", MIN_SIDE, MIN_SIDE, 1, 0)
        job = Job(
            id=f"job_{secrets.token_hex(6)}", model_id=first_spec.id, task="workflow", request=placeholder,
            workflow=plan.as_dict(), workflow_plan=plan)
        with self._lock:
            self._jobs[job.id] = job
            position = sum(1 for item in self._jobs.values() if item.status == "queued")
        job.emit("queued", {"position": position, "model": first_spec.id, "steps": len(plan.steps)})
        self._checkpoint_workflow(job)
        self._queue.put(job)
        return job

    @staticmethod
    def _parse_size(
        size: str | None, spec: ModelSpec, image_paths: list[Path], *, max_side: int = MAX_SIDE
    ) -> tuple[int, int]:
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
        if not (MIN_SIDE <= w <= max_side and MIN_SIDE <= h <= max_side):
            raise RequestError("invalid_size", f"Each side must be between {MIN_SIDE} and {max_side} pixels.")
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
        if job.workflow_plan is not None:
            self._checkpoint_workflow(job)
        return job

    def retry_workflow(self, job_id: str, from_step: str | None = None) -> Job:
        previous = self.job(job_id)
        plan = previous.workflow_plan
        if plan is None or len(plan.steps) < 2:
            raise RequestError("workflow_not_retryable", "Only multi-step workflow jobs can be retried.", 409)
        if not previous.terminal:
            raise RequestError("workflow_not_terminal", "Wait for the workflow to stop before retrying it.", 409)
        if previous.status == "done":
            raise RequestError("workflow_already_done", "This workflow already completed.", 409)

        completed = previous.report.get("completed_steps") or []
        outputs = {item["id"]: item["result_id"] for item in completed if item.get("id") and item.get("result_id")}
        target = from_step or previous.report.get("failed_step_id")
        ids = [step.id for step in plan.steps]
        if target not in ids:
            raise RequestError("invalid_retry_step", "Choose a step from the original workflow.", 422)
        start = ids.index(target)
        remaining_ids = set(ids[start:])
        retried_steps = []
        for step in plan.steps[start:]:
            inputs = []
            for item in step.inputs:
                reference = item.id[1:] if item.id.startswith("$") else None
                if reference and reference not in remaining_ids:
                    result_id = outputs.get(reference)
                    if not result_id or not self.results.exists(result_id):
                        raise RequestError(
                            "retry_artifact_missing", f"Completed artifact for step '{reference}' is unavailable.", 409
                        )
                    item = replace(item, id=result_id)
                inputs.append(item)
            retried_steps.append(replace(step, inputs=tuple(inputs)))
        route = dict(plan.route or {})
        route.update({"retry_of": previous.id, "retry_from_step": target})
        retry_plan = WorkflowPlan(tuple(retried_steps), plan.schema_version, route)
        if len(retry_plan.steps) == 1:
            job = self.submit_workflow(retry_plan, _enqueue=False)
            job.workflow_plan = retry_plan
            self._checkpoint_workflow(job)
            self._queue.put(job)
            return job
        return self.submit_workflow(retry_plan)

    def queue_depth(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status in ("queued", "running"))

    # -- worker -----------------------------------------------------------------------
    def _work(self) -> None:
        while True:
            job = self._queue.get()
            if job.status == "cancelled":
                continue
            try:
                with self._accelerator_lock:
                    provider = self.router.provider
                    if provider is not None and getattr(provider, "uses_accelerator", False):
                        getattr(provider, "unload", lambda: None)()
                    if self.prompt_assistant is not None:
                        self.prompt_assistant.unload()
                    if job.workflow_plan is not None and len(job.workflow_plan.steps) > 1:
                        self._run_multi(job)
                    else:
                        self._run(job)
            except Exception as exc:  # noqa: BLE001 - never kill the worker
                log.exception("job %s crashed", job.id)
                job.status = "error"
                job.error = {"code": "internal_error", "message": str(exc)}
                if job.workflow_plan is not None:
                    job.report.setdefault("failed_step_id", job.report.get("current_step_id"))
                    job.report.setdefault("failed_step_index", job.report.get("current_step_index"))
                job.emit("error", job.error)
            finally:
                if job.workflow_plan is not None:
                    self._checkpoint_workflow(job)
                job.finished.set()

    def _run_multi(self, job: Job) -> None:
        plan = job.workflow_plan
        assert plan is not None
        job.status = "running"
        started = time.monotonic()
        outputs: dict[str, str] = {}
        completed: list[dict] = []
        step_reports: list[dict] = []
        job.report = {
            "completed_steps": completed, "step_reports": step_reports,
            "workflow": job.workflow,
        }
        self._checkpoint_workflow(job)

        for index, step in enumerate(plan.steps):
            job.report.update({"current_step_id": step.id, "current_step_index": index})
            self._checkpoint_workflow(job)
            if job.cancel_event.is_set():
                job.status = "cancelled"
                job.error = {"code": "cancelled", "message": "Workflow cancelled."}
                job.report.update({
                    "total_seconds": round(time.monotonic() - started, 3),
                    "failed_step_id": step.id, "failed_step_index": index,
                })
                self._checkpoint_workflow(job)
                job.emit("cancelled", {**job.error, **job.report})
                return
            resolved_inputs = tuple(
                replace(item, id=outputs[item.id[1:]]) if item.id.startswith("$") else item
                for item in step.inputs
            )
            resolved_step = replace(step, inputs=resolved_inputs)
            single = WorkflowPlan((resolved_step,), plan.schema_version, plan.route)
            job.emit("workflow_step_started", {
                "workflow_step_id": step.id, "workflow_step_index": index,
                "workflow_step_total": len(plan.steps), "operation": step.operation,
            })
            try:
                child = self.submit_workflow(single, _enqueue=False)
            except RequestError as exc:
                job.status = "error"
                job.error = {"code": exc.code, "message": exc.message, "workflow_step_id": step.id}
                job.report.update({
                    "total_seconds": round(time.monotonic() - started, 3),
                    "failed_step_id": step.id, "failed_step_index": index,
                })
                self._checkpoint_workflow(job)
                job.emit("error", {**job.error, **job.report})
                return
            child.cancel_event = job.cancel_event

            def forward(event: str, payload: dict, step_id=step.id, step_index=index) -> None:
                if event in {"queued", "image", "done", "error", "cancelled"}:
                    return
                data = {key: value for key, value in payload.items() if key != "job_id"}
                job.emit(event, {**data, "workflow_step_id": step_id, "workflow_step_index": step_index})

            child._forward = forward
            self._run(child)
            with self._lock:
                self._jobs.pop(child.id, None)
            if child.status != "done" or not child.result_ids:
                job.status = "cancelled" if child.status == "cancelled" else "error"
                job.error = {
                    **(child.error or {"code": "step_failed", "message": "Workflow step failed."}),
                    "workflow_step_id": step.id,
                }
                job.report.update({
                    "total_seconds": round(time.monotonic() - started, 3),
                    "failed_step_id": step.id, "failed_step_index": index,
                    "failed_step_report": child.report,
                })
                self._checkpoint_workflow(job)
                job.emit("cancelled" if job.status == "cancelled" else "error", {**job.error, **job.report})
                return
            result_id = child.result_ids[-1]
            outputs[step.id] = result_id
            completed.append({"id": step.id, "operation": step.operation, "result_id": result_id})
            step_reports.append(child.report)
            self._checkpoint_workflow(job)
            job.emit("workflow_step_done", {
                "workflow_step_id": step.id, "workflow_step_index": index,
                "workflow_step_total": len(plan.steps), "result_id": result_id,
            })

        final_id = outputs[plan.steps[-1].id]
        job.result_ids = [final_id]
        job.report = {
            "total_seconds": round(time.monotonic() - started, 3),
            "completed_steps": completed,
            "step_reports": step_reports,
            "workflow": job.workflow,
        }
        job.status = "done"
        self._checkpoint_workflow(job)
        job.emit("image", {"index": 0, "result_id": final_id})
        job.emit("done", {"result_ids": job.result_ids, "report": job.report})

    def _run(self, job: Job) -> None:
        spec = self.registry.get(job.model_id)
        job.status = "running"
        started = time.monotonic()
        try:
            adapter = self.adapter(spec)
            run_request, rewrite = self._rewrite(job, spec)
            cached = job.n == 1 and getattr(adapter, "has_embedding", lambda _p: False)(run_request.prompt)
            skip = ("text_encoder",) if cached else ()
            plan = self.memory.plan(
                spec,
                spec.activation_bytes(job.request.width, job.request.height, len(job.request.image_paths)),
                strategy=job.strategy,
                force=job.force,
                skip=skip,
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
                req = run_request if i == 0 else GenRequest(**{**run_request.__dict__, "seed": run_request.seed + i})
                gen_started = time.monotonic()
                image = adapter.generate(req, guard)
                meta = {
                    "kind": "generation",
                    "model": spec.id,
                    "task": job.task,
                    "operation": ((job.workflow or {}).get("steps") or [{}])[0].get("operation"),
                    "prompt": (((job.workflow or {}).get("steps") or [{}])[0].get("prompt") or {}).get("user")
                    or job.request.prompt,
                    "negative_prompt": req.negative_prompt,
                    "width": req.width,
                    "height": req.height,
                    "steps": req.steps,
                    "seed": req.seed,
                    "sources": [p.stem for p in req.image_paths],
                    "parameters": req.extras,
                    "strategy": plan.strategy,
                    "seconds": round(time.monotonic() - gen_started, 3),
                    "created": time.time(),
                    "workflow": job.workflow,
                }
                rid = self.results.save(image, meta)
                job.result_ids.append(rid)
                job.emit("image", {"index": i, "result_id": rid, "seed": req.seed})
        except JobAborted as exc:
            job.status = "cancelled" if exc.code == "cancelled" else "error"
            job.error = {"code": exc.code, "message": exc.message}
            if exc.code == "memory_pressure" or job.task == "upscale":
                self.memory.unload_model(spec.id)
            job.report = {**guard.report(), **rewrite, "total_seconds": round(time.monotonic() - started, 3)}
            job.emit("cancelled" if exc.code == "cancelled" else "error", {**job.error, "report": job.report})
            return
        job.report = {
            **guard.report(),
            **rewrite,
            "total_seconds": round(time.monotonic() - started, 3),
            "embed_cache": adapter.embed_cache_stats(),
            "workflow": job.workflow,
        }
        job.status = "done"
        job.emit("done", {"result_ids": job.result_ids, "report": job.report})

    # -- prompt assistant ---------------------------------------------------------------
    def enhancer(self):
        """The local, image-aware assistant when installed; otherwise a configured text-only provider."""
        return self.prompt_assistant or self.prompt_provider

    def _plan(self, state: dict, candidates: list[str], wait: bool = False) -> tuple[dict | None, str | None]:
        """The agent's decision for a sent message, or None to fall back to rules and the router, with a
        note when it did not plan. While an image is being made it plans beside the job when both fit in
        memory, waits for the job when `wait` is set, and otherwise is skipped."""
        started = time.monotonic()
        acquired = self._accelerator_lock.acquire(blocking=False)
        beside = waited = False
        if not acquired:
            if self._fits_beside_job():
                beside = True
                log.info("agent planning beside the running image job")
            elif wait:
                self._accelerator_lock.acquire()
                acquired = waited = True
            else:
                log.warning("agent planning skipped: an image is being made")
                return None, "skipped_busy"  # the cheap path answers now
        try:
            # Beside a job, never free memory: the components to free would be the running job's.
            if not beside and not self.memory.make_room(self.prompt_assistant.need_bytes):
                # Frees idle models first, but never skips the agent: planning runs even when memory is tight.
                log.warning("agent planning with less free memory than it needs")
            workflows = {name: DEFAULT_DESCRIPTIONS.get(name, name) for name in candidates}
            result = agent.plan(self.prompt_assistant, state, workflows, image_path=self.results.path)
        except (IntelligenceError, KeyError, OSError) as exc:
            log.warning("agent planning failed, using the router: %s", exc)
            return None, "failed"
        finally:
            if not beside:
                self._release_assistant()
            if acquired:
                self._accelerator_lock.release()
        if result is None:
            log.warning("agent answer failed the checks; using the router")
            return None, "rejected"
        mode = "beside_job" if beside else "waited" if waited else None
        decision = {**result.as_decision(), "latency_ms": round((time.monotonic() - started) * 1_000, 1)}
        return (decision | {"planning": mode} if mode else decision), None

    def _fits_beside_job(self) -> bool:
        """Big Macs: the agent and the running image job fit in memory together, with the reserve kept.
        Measured beside Klein: replies identical, the plan ~3x slower and the image ~1.7x (docs/benchmarks/agent-models.md)."""
        return self.memory.total >= 32 * GB and self.prompt_assistant.need_bytes <= self.memory.headroom()

    def _release_assistant(self) -> None:
        """Between messages the assistant stays loaded where it leaves clear room (24 GB and up), so the
        next message is planned in seconds; the worker always unloads it before an image job."""
        if self.memory.total < 24 * GB:
            self.prompt_assistant.unload()

    def _assist(self, text: str, *, operation: str, images: list[Path], supports_negative: bool, emit=None):
        """Run the assistant alone on the accelerator and release it, so it never sits beside an image model."""
        assistant = self.prompt_assistant
        hit = assistant.cached(text, operation=operation, supports_negative=supports_negative, images=images)
        if hit is not None:
            return hit
        if emit is None and not self._accelerator_lock.acquire(blocking=False):
            raise RequestError("engine_busy", "An image is being made; try the prompt assistant again when it is done.", 409)
        try:
            if not self.memory.make_room(assistant.need_bytes):
                log.warning("prompt assistant running with less free memory than it needs")
            if emit:
                emit("loading", {"component": "prompt_assistant"})
                emit("rewriting", {})
            try:
                return assistant.enhance(text, supports_negative=supports_negative, operation=operation, images=images)
            except IntelligenceError as exc:
                raise RequestError("prompt_provider_unavailable", str(exc), 502) from exc
            finally:
                if emit is None:
                    self._release_assistant()
                else:
                    assistant.unload()
        finally:
            if emit is None:
                self._accelerator_lock.release()

    def _rewrite(self, job: Job, spec: ModelSpec) -> tuple[GenRequest, dict]:
        """Edits, and new images in Auto mode, are rewritten on this Mac before the image model loads.

        A prompt the user already enhanced (✨) is used as is. Without the assistant, or when it fails,
        the job runs with the user's own words. Runs on the worker, which holds the accelerator lock.
        """
        req = job.request
        step = ((job.workflow or {}).get("steps") or [{}])[0]
        prompt = step.get("prompt") or {}
        operation = step.get("operation")
        compiler = prompt.get("compiler", "passthrough@1")
        if compiler.startswith("agent:") and prompt.get("user") and prompt["user"] != req.prompt:
            # Written by the agent when the message was sent: record it like a rewrite, keep the user's words.
            return replace(req, extras={**req.extras, "rewritten_prompt": req.prompt, "prompt_assistant": compiler}), {
                "rewrite": {"provider": compiler, "planned": True}}
        is_edit = operation == "image.edit" and bool(req.image_paths)
        wanted = is_edit or (prompt.get("enhancement_mode") == "auto" and operation == "image.generate")
        if (self.prompt_assistant is None or not wanted or not req.prompt
                or prompt.get("compiler", "passthrough@1") != "passthrough@1"):
            return req, {}
        capability = spec.capabilities.get(operation)
        # Edits never gain a negative prompt by default: on CFG models it doubles the work.
        supports_negative = not is_edit and bool(capability and capability.negative_prompt) and not req.negative_prompt
        started = time.monotonic()
        try:
            compiled = self._assist(req.prompt, operation=operation, images=list(req.image_paths) if is_edit else [],
                                    supports_negative=supports_negative, emit=job.emit)
        except RequestError as exc:
            log.warning("prompt rewrite skipped for %s: %s", job.id, exc.message)
            return req, {"rewrite": {"skipped": exc.code, "message": exc.message}}
        job.emit("rewritten", {"prompt": compiled.positive})
        rewritten = replace(
            req, prompt=compiled.positive, negative_prompt=req.negative_prompt or compiled.negative,
            extras={**req.extras, "rewritten_prompt": compiled.positive, "prompt_assistant": compiled.compiler},
        )
        return rewritten, {"rewrite": {"provider": compiled.compiler, "seconds": round(time.monotonic() - started, 3)}}

    def prompt_assistant_status(self) -> dict:
        selected = selected_assistant_model()
        assistant = self.prompt_assistant
        models = []
        for model in local_vlm.MODELS.values():
            state = self.downloader.state(prompt_assistant_spec(model.id).id).as_dict()
            models.append({**model.as_dict(), "installed": local_vlm.locate(model.id) is not None, "download": state})
        current = next(m for m in models if m["id"] == selected)
        return {
            # The selected model, in the shape the app has always read.
            "repo": current["repo"], "revision": current["revision"], "license": current["license"],
            "installed": assistant is not None, "size_gb": current["size_gb"],
            "resident": assistant is not None and assistant.resident, "download": current["download"],
            "selected": selected, "models": models,
        }

    def pull_prompt_assistant(self, model_id: str | None = None) -> dict:
        if model_id is not None and model_id not in local_vlm.MODELS:
            raise RequestError("unknown_model", f"No prompt assistant model '{model_id}'.", 404)
        model_id = model_id or selected_assistant_model()

        def done(_spec=None, *_args) -> None:
            if model_id == selected_assistant_model():
                self.prompt_assistant = _prompt_assistant_from_disk(model_id)

        self.downloader.start(prompt_assistant_spec(model_id), on_done=done)
        return self.prompt_assistant_status()

    def select_prompt_assistant(self, model_id: str) -> dict:
        """Switch the agent to another model; it is used from the next message once downloaded."""
        if model_id not in local_vlm.MODELS:
            raise RequestError("unknown_model", f"No prompt assistant model '{model_id}'.", 404)
        with self._accelerator_lock:
            if self.prompt_assistant is not None:
                self.prompt_assistant.unload()
            (paths.home() / "prompt-assistant.json").write_text(json.dumps({"model": model_id}))
            self.prompt_assistant = _prompt_assistant_from_disk(model_id)
        return self.prompt_assistant_status()

    def status(self) -> dict:
        provider = self.router.provider
        router_status = (
            provider.status() if provider is not None and hasattr(provider, "status")
            else {"kind": "remote" if provider is not None else "manual",
                  "name": getattr(provider, "name", "manual"), "resident": False}
        )
        return {
            "uptime_s": round(time.time() - self.started, 1),
            "queue": self.queue_depth(),
            "memory": self.memory.status(),
            "ram_tier_gb": round(self.memory.total / GB),
            "emulated": bool(__import__("os").environ.get("DMLX_EMULATE_TOTAL_GB")),
            "real_total_gb": round(sysmem._sysctl_int("hw.memsize") / GB),
            "intelligence": {"router": router_status},
        }

    def _checkpoint_workflow(self, job: Job) -> None:
        if job.workflow_plan is None:
            return
        if job.status == "done":
            self.results.delete_workflow_job(job.id)
            retry_of = (job.workflow_plan.route or {}).get("retry_of")
            if retry_of:
                self.results.delete_workflow_job(str(retry_of))
            return
        self.results.save_workflow_job({
            "schema_version": 1,
            "id": job.id,
            "model_id": job.model_id,
            "task": job.task,
            "status": job.status,
            "created": job.created,
            "workflow": job.workflow,
            "result_ids": job.result_ids,
            "report": job.report,
            "error": job.error,
        })

    def _recover_workflow_jobs(self) -> None:
        for value in self.results.workflow_jobs():
            try:
                plan = WorkflowPlan.from_dict(value["workflow"])
                placeholder = GenRequest("workflow", "multi-step workflow", MIN_SIDE, MIN_SIDE, 1, 0)
                status = str(value.get("status", "error"))
                error = value.get("error")
                report = dict(value.get("report") or {})
                if status in {"queued", "running"}:
                    status = "error"
                    error = {
                        "code": "engine_restarted",
                        "message": "The engine stopped before this workflow completed. Retry from the recorded step.",
                    }
                    report.setdefault("failed_step_id", report.get("current_step_id") or plan.steps[0].id)
                    report.setdefault("failed_step_index", report.get("current_step_index", 0))
                job = Job(
                    id=value["id"], model_id=value["model_id"], task="workflow", request=placeholder,
                    workflow=value["workflow"], workflow_plan=plan, status=status,
                    created=float(value.get("created", time.time())),
                    result_ids=list(value.get("result_ids") or []), report=report,
                    error=error,
                )
                if status in {"done", "error", "cancelled"}:
                    job.finished.set()
                self._jobs[job.id] = job
                if status == "error" and error:
                    job.emit("error", error)
                if value.get("status") in {"queued", "running"}:
                    self._checkpoint_workflow(job)
            except (KeyError, TypeError, ValueError, WorkflowError) as exc:
                log.warning("Ignoring invalid workflow checkpoint: %s", exc)


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
