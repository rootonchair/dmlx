"""Memory manager: admission control, strategy selection, component residency, pressure guard.

Strategies (chosen per job from available memory, see spec §8):
  eager       all components resident; kept warm across requests
  phase_free  text encoder freed after encoding; transformer + VAE kept warm
  sequential  one big component resident at a time: encoder -> free -> transformer -> free -> VAE
  streamed    like sequential, but big components stream their blocks from disk one at a time
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from dataclasses import dataclass, field

import mlx.core as mx

from dmlx import sysmem
from dmlx.adapters.base import JobAborted
from dmlx.registry import GB, ModelSpec

log = logging.getLogger("dmlx.memory")

STRATEGIES = ("eager", "phase_free", "sequential", "streamed")
# Headroom must exceed a strategy's peak need by this factor before we pick it (spec §8 regimes):
# keeping more resident is only worth it when there is clearly room for it.
MARGIN = {"eager": 1.25, "phase_free": 1.5, "sequential": 1.0, "streamed": 1.0}
SWAP_GROWTH_CAP = int(4 * GB)  # backstop: abort a job that pushes this much new data to swap
SWAPIN_RATE_CAP = 64 * 1024**2  # bytes/s read back from swap, sustained over SWAPIN_WINDOW -> thrashing
SWAPIN_WINDOW = 3.0
CACHE_LIMIT = 512 * 1024**2


class InsufficientMemory(Exception):
    def __init__(self, need: int, headroom: int, model_id: str, model_name: str | None = None):
        self.need = need
        self.headroom = headroom
        self.model_id = model_id
        super().__init__(
            f"{model_name or model_id} needs about {need / GB:.1f} GB but only {max(headroom, 0) / GB:.1f} GB can be used "
            f"without slowing down your Mac. Close some apps, pick a smaller size, or choose a lighter model."
        )


@dataclass
class Resident:
    model_id: str
    name: str
    bytes: int
    last_used: float


@dataclass
class Plan:
    model_id: str
    strategy: str
    need: int
    headroom: int
    evicted: list[str] = field(default_factory=list)
    forced: bool = False

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "need_gb": round(self.need / GB, 2),
            "headroom_gb": round(self.headroom / GB, 2),
            "evicted": self.evicted,
            "forced": self.forced,
        }


class PressureMonitor:
    """Samples system pressure and swap in the background."""

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self._lock = threading.Lock()
        self._last = sysmem.snapshot()
        self._window_max_pressure = self._last.pressure
        self._history: list[sysmem.MemorySnapshot] = [self._last]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="dmlx-pressure", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                snap = sysmem.snapshot()
            except OSError:
                continue
            with self._lock:
                self._last = snap
                self._window_max_pressure = max(self._window_max_pressure, snap.pressure)
                self._history.append(snap)
                cutoff = snap.taken - 10.0
                while self._history and self._history[0].taken < cutoff:
                    self._history.pop(0)

    def last(self) -> sysmem.MemorySnapshot:
        with self._lock:
            return self._last

    def swapin_rate(self, window: float = SWAPIN_WINDOW) -> float:
        """Average bytes/s read back from swap over the last `window` seconds."""
        with self._lock:
            if len(self._history) < 2:
                return 0.0
            newest = self._history[-1]
            oldest = next((h for h in self._history if h.taken >= newest.taken - window), self._history[0])
            span = newest.taken - oldest.taken
            if span < window * 0.8:
                return 0.0
            return (newest.swapins - oldest.swapins) / span

    def start_window(self) -> None:
        with self._lock:
            self._window_max_pressure = self._last.pressure
            # Swap-in is a per-job guard. Do not let the tail of the previous
            # job's rolling window make a new job fail before it has done work.
            self._history = [self._last]

    def window_max_pressure(self) -> int:
        with self._lock:
            return self._window_max_pressure

    def stop(self) -> None:
        self._stop.set()


class MemoryManager:
    def __init__(self, monitor: PressureMonitor | None = None):
        self.monitor = monitor or PressureMonitor()
        self.total = sysmem.total_memory()
        self.reserve = sysmem.os_reserve(self.total)
        self._resident: dict[tuple[str, str], Resident] = {}
        self._pinned: set[str] = set()
        self._adapters: dict[str, object] = {}
        self._lock = threading.RLock()
        self.degraded_until = 0.0  # after real pressure, prefer slower strategies for a while
        mx.set_cache_limit(CACHE_LIMIT)

    # -- registration / bookkeeping -------------------------------------------------
    def register_adapter(self, model_id: str, adapter) -> None:
        with self._lock:
            self._adapters[model_id] = adapter

    def unregister_adapter(self, model_id: str) -> None:
        with self._lock:
            self._adapters.pop(model_id, None)

    def pin(self, model_id: str, pinned: bool = True) -> None:
        with self._lock:
            (self._pinned.add if pinned else self._pinned.discard)(model_id)

    def is_pinned(self, model_id: str) -> bool:
        return model_id in self._pinned

    def resident_bytes(self, model_id: str | None = None) -> int:
        with self._lock:
            return sum(r.bytes for r in self._resident.values() if model_id is None or r.model_id == model_id)

    def resident_components(self) -> list[dict]:
        with self._lock:
            return [
                {"model": r.model_id, "component": r.name, "gb": round(r.bytes / GB, 2), "pinned": r.model_id in self._pinned}
                for r in sorted(self._resident.values(), key=lambda r: (r.model_id, r.name))
            ]

    def resident_models(self) -> set[str]:
        with self._lock:
            return {r.model_id for r in self._resident.values()}

    def ensure(self, model_id: str, name: str) -> bool:
        """Load a component if needed. Returns True when a load actually happened."""
        with self._lock:
            adapter = self._adapters[model_id]
            key = (model_id, name)
            if key in self._resident and adapter.is_loaded(name):
                self._resident[key].last_used = time.monotonic()
                return False
        nbytes = adapter.load_component(name)
        with self._lock:
            self._resident[key] = Resident(model_id, name, nbytes, time.monotonic())
        log.info("loaded %s/%s (%.2f GB)", model_id, name, nbytes / GB)
        return True

    def free(self, model_id: str, name: str) -> None:
        with self._lock:
            adapter = self._adapters.get(model_id)
            self._resident.pop((model_id, name), None)
        if adapter is not None:
            adapter.free_component(name)
            log.info("freed %s/%s", model_id, name)

    def unload_model(self, model_id: str) -> None:
        for r in [r for r in list(self._resident.values()) if r.model_id == model_id]:
            self.free(r.model_id, r.name)
        gc.collect()
        mx.clear_cache()

    # -- planning ---------------------------------------------------------------------
    def headroom(self) -> int:
        snap = self.monitor.last()
        # MLX's cache is ours to give back; resident weights are accounted for separately.
        return snap.available + mx.get_cache_memory() - self.reserve

    def _missing(self, spec: ModelSpec, names: list[str]) -> int:
        return sum(spec.components[n].size for n in names if (spec.id, n) not in self._resident)

    def _needs(self, spec: ModelSpec, activation: int, skip: tuple[str, ...] = ()) -> dict[str, int]:
        """Extra memory each strategy needs at its peak, beyond what is already resident.

        eager:      everything missing loads and stays.
        phase_free: encoder and transformer are never resident together (encoder freed first).
        sequential: like phase_free, but resident big components are freed first, so they count as returned.
        streamed:   like sequential, but big components keep only their non-block weights and one block resident.
        """
        comps = {n: c for n, c in spec.components.items() if n not in skip}
        small_missing = self._missing(spec, [n for n, c in comps.items() if c.keep])
        big = [n for n in ("text_encoder", "transformer") if n in comps]
        eager = self._missing(spec, list(comps)) + activation
        phase_free = max((self._missing(spec, [n]) for n in big), default=0) + small_missing + activation
        resident_big = sum(comps[n].size for n in big if (spec.id, n) in self._resident)
        sequential = max(max((comps[n].size for n in big), default=0) - resident_big, 0) + small_missing + activation
        needs = {"eager": eager, "phase_free": phase_free, "sequential": sequential}
        if any(comps[n].stream_size for n in big):
            peak = max(comps[n].stream_size or comps[n].size for n in big)
            needs["streamed"] = max(peak - resident_big, 0) + small_missing + activation
        return needs

    def plan(
        self, spec: ModelSpec, activation: int, strategy: str = "auto", force: bool = False, skip: tuple[str, ...] = ()
    ) -> Plan:
        """`skip`: components this job will not need (e.g. the encoder when the prompt embedding is cached)."""
        with self._lock:
            needs = self._needs(spec, activation, skip)
            order = [s for s in STRATEGIES if s in needs]
            if time.monotonic() < self.degraded_until:
                order = [s for s in ("sequential", "streamed") if s in needs]
            if strategy != "auto":
                order = [strategy]
            evicted: list[str] = []

            def try_fit() -> Plan | None:
                room = self.headroom()
                for s in order:
                    if needs[s] * (MARGIN[s] if strategy == "auto" else 1.0) <= room:
                        return Plan(spec.id, s, needs[s], room, evicted)
                return None

            plan = try_fit()
            if plan is None:
                # Evict least recently used components of other, unpinned models.
                victims = sorted(
                    (r for r in self._resident.values() if r.model_id != spec.id and r.model_id not in self._pinned),
                    key=lambda r: r.last_used,
                )
                for victim in victims:
                    self.free(victim.model_id, victim.name)
                    evicted.append(f"{victim.model_id}/{victim.name}")
                    gc.collect()
                    mx.clear_cache()
                    time.sleep(0.05)
                    self.monitor._last = sysmem.snapshot()
                    plan = try_fit()
                    if plan:
                        break
            if plan is None:
                room = self.headroom()
                cheapest = order[-1]
                if not force:
                    raise InsufficientMemory(needs[cheapest], room, spec.id, spec.name)
                plan = Plan(spec.id, cheapest, needs[cheapest], room, evicted, forced=True)
            mx.set_memory_limit(max(int(plan.headroom + self.resident_bytes()), 2 * GB))
            return plan

    def make_room(self, need: int) -> bool:
        """Free idle components, least recently used first and never a pinned model's, until `need` fits."""
        with self._lock:
            if need <= self.headroom():
                return True
            victims = sorted(
                (r for r in self._resident.values() if r.model_id not in self._pinned), key=lambda r: r.last_used)
            for victim in victims:
                self.free(victim.model_id, victim.name)
                gc.collect()
                mx.clear_cache()
                time.sleep(0.05)
                self.monitor._last = sysmem.snapshot()
                if need <= self.headroom():
                    return True
            return False

    def degrade(self, seconds: float = 300.0) -> None:
        self.degraded_until = time.monotonic() + seconds

    def status(self) -> dict:
        snap = self.monitor.last()
        return {
            "system": snap.as_dict(),
            "reserve_gb": round(self.reserve / GB, 2),
            "headroom_gb": round(self.headroom() / GB, 2),
            "mlx_active_gb": round(mx.get_active_memory() / GB, 2),
            "mlx_cache_gb": round(mx.get_cache_memory() / GB, 2),
            "resident": self.resident_components(),
            "pinned": sorted(self._pinned),
            "degraded": time.monotonic() < self.degraded_until,
        }


class JobGuard:
    """Per-job phase hooks: loads/frees components per strategy and enforces the no-hang rules."""

    def __init__(self, manager: MemoryManager, spec: ModelSpec, plan: Plan, cancel_event: threading.Event, emit):
        self.m = manager
        self.spec = spec
        self.plan = plan
        self.cancel_event = cancel_event
        self.emit = emit
        self.loads: list[str] = []
        self.timings: dict[str, float] = {}
        self._phase_started: dict[str, float] = {}
        self.swap_start = manager.monitor.last().swap_used
        self.baseline_pressure = manager.monitor.last().pressure
        self.max_pressure = self.baseline_pressure
        self.size: tuple[int, int] | None = None
        self.decode_mode = "full"
        self.max_swapin_rate = 0.0
        manager.monitor.start_window()

    def _ensure(self, name: str) -> None:
        if name not in self.spec.components:
            return
        if not self.m._adapters[self.spec.id].is_loaded(name) or (self.spec.id, name) not in self.m._resident:
            self.emit("loading", {"component": name})
            started = time.monotonic()
            if self.m.ensure(self.spec.id, name):
                self.loads.append(name)
                self.timings["load"] = self.timings.get("load", 0.0) + time.monotonic() - started
        else:
            self.m.ensure(self.spec.id, name)

    def start(self) -> None:
        adapter = self.m._adapters[self.spec.id]
        if hasattr(adapter, "stream_mode"):
            adapter.stream_mode = self.plan.strategy == "streamed"
        if self.plan.strategy in ("sequential", "streamed"):
            # Establish the low-memory invariant at the job boundary as well as
            # at phase callbacks. A cached prompt can skip the encode callbacks,
            # otherwise leaving eager components from the preceding job resident
            # during reference-image preprocessing.
            self.m.free(self.spec.id, "text_encoder")
            self.m.free(self.spec.id, "transformer")
        for name, comp in self.spec.components.items():
            if comp.keep:
                self._ensure(name)
        if self.plan.strategy == "eager":
            for name in self.spec.components:
                self._ensure(name)

    def before_phase(self, phase: str) -> None:
        self.check()
        self._phase_started[phase] = time.monotonic()
        if phase == "encode":
            if self.plan.strategy in ("sequential", "streamed"):
                self.m.free(self.spec.id, "transformer")
            self._ensure("text_encoder")
            self.emit("encoding", {})
        elif phase == "denoise":
            # Image-conditioned pipelines may use the VAE before denoising.
            # In the low-memory strategies it must not overlap the transformer.
            self.after_phase("preprocess")
            if self.plan.strategy in ("sequential", "streamed"):
                self.m.free(self.spec.id, "vae")
            self._ensure("transformer")
        elif phase == "layout":
            # Some multimodal pipelines need transformer metadata before their
            # normal denoise callback, while the VAE is still encoding references.
            self._ensure("transformer")
        elif phase == "decode":
            self._ensure("vae")
            self._choose_decode()
            self.emit("decoding", {})

    def _choose_decode(self) -> None:
        """Decode in one piece when there is clearly room; otherwise banded (exact) or tiled, both far smaller."""
        adapter = self.m._adapters[self.spec.id]
        if not hasattr(adapter, "set_decode_mode") or self.size is None:
            return
        full = self.spec.decode_bytes(*self.size, "full")
        # Below 32 GB the one-piece decode's spike is never worth the risk; above, only with clear room.
        roomy = not self.plan.forced and self.m.total >= 32 * GB and full * 1.5 <= self.m.headroom()
        self.decode_mode = "full" if roomy else self.spec.low_memory_decode
        adapter.set_decode_mode(self.decode_mode)

    def after_phase(self, phase: str) -> None:
        started = self._phase_started.pop(phase, None)
        if started is not None:
            self.timings[phase] = self.timings.get(phase, 0.0) + time.monotonic() - started
        if phase == "encode" and self.plan.strategy in ("phase_free", "sequential", "streamed"):
            self.m.free(self.spec.id, "text_encoder")
        elif phase == "denoise" and self.plan.strategy in ("sequential", "streamed"):
            self.m.free(self.spec.id, "transformer")

    def on_step(self, step: int, total: int) -> None:
        self.emit("step", {"step": step, "total": total})

    def check(self) -> None:
        if self.cancel_event.is_set():
            raise JobAborted("cancelled", "Generation cancelled")
        snap = self.m.monitor.last()
        self.max_pressure = max(self.max_pressure, self.m.monitor.window_max_pressure(), snap.pressure)
        if snap.pressure >= sysmem.PRESSURE_CRITICAL:
            self.m.degrade()
            raise JobAborted("memory_pressure", "Stopped to keep your Mac responsive: memory pressure became critical.")
        rate = self.m.monitor.swapin_rate()
        self.max_swapin_rate = max(self.max_swapin_rate, rate)
        if rate > SWAPIN_RATE_CAP:
            self.m.degrade()
            raise JobAborted(
                "memory_pressure", "Stopped to keep your Mac responsive: apps were being read back from swap."
            )
        if snap.swap_used - self.swap_start > SWAP_GROWTH_CAP:
            self.m.degrade()
            raise JobAborted(
                "memory_pressure", "Stopped to keep your Mac responsive: the system started swapping heavily."
            )
        if snap.pressure > self.baseline_pressure:
            # Pressure rose because of us: prefer slower strategies for the next jobs.
            self.m.degrade()

    def report(self) -> dict:
        snap = self.m.monitor.last()
        return {
            **self.plan.as_dict(),
            "loads": self.loads,
            "decode_mode": self.decode_mode,
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
            "peak_mlx_gb": round(mx.get_peak_memory() / GB, 2),
            "max_pressure": sysmem.PRESSURE_NAMES.get(self.max_pressure, str(self.max_pressure)),
            "swap_delta_gb": round((snap.swap_used - self.swap_start) / GB, 2),
            "max_swapin_mb_s": round(self.max_swapin_rate / 1024**2, 1),
        }
