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

# Parts smaller than this are not checked after a free (the measurement is too noisy).
FREE_CHECK_MIN_BYTES = 256 * 1024**2

STRATEGIES = ("eager", "phase_free", "sequential", "streamed")
# Headroom must exceed a strategy's peak need by this factor before we pick it (spec §8 regimes):
# keeping more resident is only worth it when there is clearly room for it.
MARGIN = {"eager": 1.25, "phase_free": 1.5, "sequential": 1.0, "streamed": 1.0}
SWAP_GROWTH_CAP = int(4 * GB)  # backstop: abort a job that pushes this much new data to swap
SWAPIN_RATE_CAP = 64 * 1024**2  # bytes/s read back from swap, sustained over SWAPIN_WINDOW -> thrashing
SWAPIN_WINDOW = 3.0
# Reading back from swap alone is also what a Mac does while it recovers: apps the agent pushed out come
# back once it unloads, with pressure normal and swap shrinking (2026-11-07: every stop of the day was
# that, none a hang). Thrashing also pushes pages out, raises pressure or grows swap, so the swap-in
# rule needs one of these too.
SWAPOUT_RATE_SIGN = 32 * 1024**2  # bytes/s pushed out to swap over the same window
SWAP_GROWTH_SIGN = int(0.5 * GB)  # swap grown since the job started
CACHE_LIMIT = 512 * 1024**2
# After an eviction macOS hands the memory back with a delay: the room is polled until about what was freed
# has come back (or the job's need fits), for at most this long, before the next plan (task 19: re-planned
# after 50 ms, it evicted a warm Klein and still ran streamed).
EVICT_RETURN_S = 2.0
EVICT_POLL_S = 0.05
EVICT_RETURN_SHARE = 0.9  # "about what was freed"


# Above this many pixels a smaller size is worth suggesting; the app's smallest sizes are all about 1 MP
# (1024x1024, 1216x832, 832x1216), so below it the advice could not be followed (task 19).
SMALLER_SIZE_ABOVE = int(1.1 * 1024 * 1024)


class InsufficientMemory(Exception):
    def __init__(self, need: int, headroom: int, model_id: str, model_name: str | None = None,
                 pixels: int | None = None):
        self.need = need
        self.headroom = headroom
        self.model_id = model_id
        need_text, room_text = _apart(need / GB, max(headroom, 0) / GB)
        advice = ("Close some apps, pick a smaller size, or choose a lighter model."
                  if pixels is not None and pixels > SMALLER_SIZE_ABOVE else
                  "Close some apps, or choose a lighter model.")
        super().__init__(
            f"{model_name or model_id} needs about {need_text} GB but only {room_text} GB can be used "
            f"without slowing down your Mac. {advice}"
        )


def _apart(need: float, room: float) -> tuple[str, str]:
    """Both sizes with one decimal, or more when one decimal makes them look equal ("needs 3.3 GB but
    only 3.3 GB can be used" read as a contradiction; it is 3.34 against 3.26)."""
    for places in (1, 2, 3):
        a, b = f"{need:.{places}f}", f"{room:.{places}f}"
        if float(a) > float(b):
            return a, b
    return f"{need:.2f}", f"{room:.2f}"


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
    # Parts the need left out (the text encoder when the prompt's embedding is cached): the job must not
    # load them up front (fix round 2 of task 19: an eager plan counted without the encoder, then loaded it).
    skip: tuple[str, ...] = ()

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
        return self._rate("swapins", window)

    def swapout_rate(self, window: float = SWAPIN_WINDOW) -> float:
        """Average bytes/s pushed out to swap over the last `window` seconds."""
        return self._rate("swapouts", window)

    def _rate(self, counter: str, window: float) -> float:
        with self._lock:
            if len(self._history) < 2:
                return 0.0
            newest = self._history[-1]
            oldest = next((h for h in self._history if h.taken >= newest.taken - window), self._history[0])
            span = newest.taken - oldest.taken
            if span < window * 0.8:
                return 0.0
            return (getattr(newest, counter) - getattr(oldest, counter)) / span

    def start_window(self) -> None:
        with self._lock:
            self._window_max_pressure = self._last.pressure
            # Swap-in is a per-job guard. Do not let the tail of the previous
            # job's rolling window make a new job fail before it has done work.
            self._history = [self._last]

    def refresh(self) -> sysmem.MemorySnapshot:
        """A new sample now, rather than the last one (up to `interval` old)."""
        snap = sysmem.snapshot()
        with self._lock:
            self._last = snap
        return snap

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
        # `_lock` guards the bookkeeping and is only held for short reads and writes. `_plan_lock` keeps
        # plans one at a time; it is held while an eviction waits for its memory to come back (up to
        # EVICT_RETURN_S per part), so readers (GET /v1/models, a Stop) never wait on it (task 19).
        self._lock = threading.RLock()
        self._plan_lock = threading.RLock()
        self.degraded_until = 0.0  # after real pressure, prefer slower strategies for a while
        # The image job running now (jobs run one at a time): the room its plan had, dMLX's resident bytes
        # then, and its working memory. `could_fit` counts what it holds as coming back when it ends.
        self._job: tuple[int, int, int] | None = None
        # The model of the image job running now, from its plan to `job_ended`. A plan that is not a job's
        # (POST /admin/models/{id}/load) never evicts it: its pipeline still holds the weights (fix round 5).
        self._running_model: str | None = None
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
            resident = self._resident.pop((model_id, name), None)
        if adapter is not None:
            before = mx.get_active_memory()
            adapter.free_component(name)
            dropped = before - mx.get_active_memory()
            log.info("freed %s/%s (%.2f GB)", model_id, name, dropped / GB)
            expected = resident.bytes if resident else 0
            if expected > FREE_CHECK_MIN_BYTES and dropped < expected / 2:
                # Something still holds the weights (a lazy array, a cache): the guard's
                # accounting no longer matches memory, so the next phase may not fit.
                log.warning("freeing %s/%s released %.2f GB of %.2f GB; something still holds its weights",
                            model_id, name, dropped / GB, expected / GB)

    def unload_model(self, model_id: str) -> None:
        with self._lock:
            parts = [r for r in self._resident.values() if r.model_id == model_id]
        for r in parts:
            self.free(r.model_id, r.name)
        gc.collect()
        mx.clear_cache()

    # -- planning ---------------------------------------------------------------------
    def headroom(self) -> int:
        snap = self.monitor.last()
        # MLX's cache is ours to give back; resident weights are accounted for separately.
        return snap.available + mx.get_cache_memory() - self.reserve

    def _missing(self, spec: ModelSpec, names: list[str], resident=None) -> int:
        resident = self._resident if resident is None else resident
        return sum(spec.components[n].size for n in names if (spec.id, n) not in resident)

    def _needs(self, spec: ModelSpec, activation: int, skip: tuple[str, ...] = (), resident=None) -> dict[str, int]:
        """Extra memory each strategy needs at its peak, beyond what is already resident.

        eager:      everything missing loads and stays.
        phase_free: encoder and transformer are never resident together (encoder freed first).
        sequential: like phase_free, but resident big components are freed first, so they count as returned.
        streamed:   like sequential, but big components keep only their non-block weights and one block resident.
        `resident`: what counts as loaded already (default: what is); () for a model loaded from nothing.
        """
        resident = self._resident if resident is None else resident
        comps = {n: c for n, c in spec.components.items() if n not in skip}
        small_missing = self._missing(spec, [n for n, c in comps.items() if c.keep], resident)
        # The big parts by phase: the encoder side, then the denoise side (two transformers on Ideogram 4).
        groups = [g for g in ([n for n in spec.parts(p) if n in comps and not comps[n].keep]
                              for p in ("encode", "denoise")) if g]
        big = [n for g in groups for n in g]
        eager = self._missing(spec, list(comps), resident) + activation
        phase_free = max((self._missing(spec, g, resident) for g in groups), default=0) + small_missing + activation
        resident_big = sum(comps[n].size for n in big if (spec.id, n) in resident)
        sequential = (max(max((sum(comps[n].size for n in g) for g in groups), default=0) - resident_big, 0)
                      + small_missing + activation)
        needs = {"eager": eager, "phase_free": phase_free, "sequential": sequential}
        if any(comps[n].stream_size for n in big):
            peak = max(sum(comps[n].stream_size or comps[n].size for n in g) for g in groups)
            needs["streamed"] = max(peak - resident_big, 0) + small_missing + activation
        return needs

    def footprint(self, spec: ModelSpec) -> dict:
        """Memory a job of this model needs at its default size, loaded from nothing, in GB of 1024^3 bytes
        (the unit of the refusal messages and `need_gb`): `memory_gb` with every part resident (the eager
        plan, as when there is room: what the model holds), `memory_min_gb` by its leanest plan (below
        that it is refused). Before the download the parts' sizes are not known: `memory_gb` is then the
        download's size plus the working memory, `memory_min_gb` is None, and `memory_estimated` is True;
        with no download size either (a model added by hand), both are None."""
        width = int(spec.defaults.get("width", 1024))
        height = int(spec.defaults.get("height", 1024))
        activation = spec.activation_bytes(width, height)
        if any(not c.size for c in spec.components.values() if not c.external):
            if not spec.download_size:
                return {"memory_gb": None, "memory_min_gb": None, "memory_estimated": True}
            weights = spec.download_size + sum(c.size for c in spec.components.values() if c.external)
            return {"memory_gb": round((weights + activation) / GB, 2), "memory_min_gb": None,
                    "memory_estimated": True}
        needs = self._needs(spec, activation, resident=())
        return {"memory_gb": round(needs["eager"] / GB, 2), "memory_min_gb": round(min(needs.values()) / GB, 2),
                "memory_estimated": False}

    def plan(
        self, spec: ModelSpec, activation: int, strategy: str = "auto", force: bool = False, skip: tuple[str, ...] = (),
        pixels: int | None = None, cancel: threading.Event | None = None, for_job: bool = True,
    ) -> Plan:
        """`for_job`: False for a load outside a job, which must leave the running job's model alone.
        `skip`: components this job will not need (e.g. the encoder when the prompt embedding is cached).
        `pixels`: the job's image size, so a refusal suggests a smaller size only when there is one.
        `cancel`: the job's Stop; once set, no more models are evicted for it (raises JobAborted)."""
        skip = tuple(n for n in skip if n in spec.components)
        with self._plan_lock:
            with self._lock:
                needs = self._needs(spec, activation, skip)
            order = [s for s in STRATEGIES if s in needs]
            if time.monotonic() < self.degraded_until:
                order = [s for s in ("sequential", "streamed") if s in needs]
            if strategy != "auto":
                order = [strategy]
            evicted: list[str] = []
            margin = (lambda s: MARGIN[s]) if strategy == "auto" else (lambda s: 1.0)

            def stopped() -> None:
                if cancel is not None and cancel.is_set():
                    raise JobAborted("cancelled", "Generation cancelled")

            def evict_one(victims: list[Resident], need: float) -> None:
                stopped()
                victim = victims.pop(0)
                evicted.append(f"{victim.model_id}/{victim.name}")
                self._evict(victim, need, cancel)
                stopped()  # a Stop during the wait ends the job now, not after the wait (fix round 3)

            # Idle parts of other, unpinned models, least recently used first. (Jobs run one at a time, so
            # a job's plan has no other model running; a load outside a job keeps the running job's model.) Evicting them beats a slower strategy: after a Stop, the
            # warm Z-Image made FLUX.2 Klein run streamed beside it (task 19). Each strategy, fastest
            # first, is tried with what eviction could give back; the room is measured again after each
            # eviction, so a plan never takes more than is really there.
            with self._lock:
                kept = set(self._pinned) | {spec.id}
                if not for_job and self._running_model is not None:
                    kept.add(self._running_model)
                victims = sorted((r for r in self._resident.values() if r.model_id not in kept),
                                 key=lambda r: r.last_used)
            plan = None
            log.info("planning %s: room %.2f GB, needs %s, idle parts of other models %.2f GB", spec.id,
                     self.headroom() / GB, {k: round(v / GB, 2) for k, v in needs.items()},
                     sum(v.bytes for v in victims) / GB)
            for s in order:
                need = needs[s] * margin(s)
                while True:
                    room = self.headroom()
                    if need <= room:
                        plan = Plan(spec.id, s, needs[s], room, evicted, skip=skip)
                        break
                    if not victims or need > room + sum(v.bytes for v in victims):
                        break
                    evict_one(victims, need)
                if plan is not None:
                    break
            if plan is None and victims:
                # The estimate (bytes as loaded) can be short of what a free gives back: as before, free
                # every idle part and try once more before refusing.
                while victims:
                    evict_one(victims, needs[order[-1]] * margin(order[-1]))
                room = self.headroom()
                s = next((s for s in order if needs[s] * margin(s) <= room), None)
                if s is not None:
                    plan = Plan(spec.id, s, needs[s], room, evicted, skip=skip)
            if plan is None:
                room = self.headroom()
                cheapest = order[-1]
                if not force:
                    raise InsufficientMemory(needs[cheapest], room, spec.id, spec.name, pixels)
                plan = Plan(spec.id, cheapest, needs[cheapest], room, evicted, forced=True, skip=skip)
            stopped()
            if for_job:
                with self._lock:
                    self._running_model = spec.id
            if for_job or self._running_model is None:  # a load beside a job leaves the job's limit alone
                mx.set_memory_limit(max(int(plan.headroom + self.resident_bytes()), 2 * GB))
            return plan

    def job_started(self, plan: Plan, activation: int) -> None:
        with self._lock:
            self._job = (plan.headroom, self.resident_bytes(), activation)

    def job_ended(self) -> None:
        with self._lock:
            self._job = None
            self._running_model = None

    def _room_after_job(self) -> int:
        """Headroom once the running job ends, for a job queued behind it. Its working memory comes back,
        its weights stay resident (and count as evictable like any idle model's). Bounded twice so a
        guess never says more than is real: by the room the job's plan had less the weights it loaded
        since, and by now plus its planned working memory (it may hold less, early in the job)."""
        now = self.headroom()
        if self._job is None:
            return now
        room, resident, activation = self._job
        after = room - max(self.resident_bytes() - resident, 0)
        return max(now, min(after, now + activation))

    def could_fit(self, spec: ModelSpec, activation: int, returned: int = 0) -> bool:
        """Whether `plan` would admit this job now, by its cheapest strategy, once other unpinned models are
        evicted and `returned` bytes come back (the agent, unloaded before every image job). A dry run:
        nothing is freed. While an image job runs, this job runs after it, so the running job's working
        memory counts as returned too (test round 3 of task 17)."""
        with self._lock:
            cheapest = min(self._needs(spec, activation).values())
            evictable = sum(r.bytes for r in self._resident.values()
                            if r.model_id != spec.id and r.model_id not in self._pinned)
            return cheapest <= self._room_after_job() + evictable + returned

    def make_room(self, need: int) -> bool:
        """Free idle components, least recently used first and never a pinned model's, until `need` fits."""
        with self._plan_lock:
            if need <= self.headroom():
                return True
            with self._lock:
                victims = sorted(
                    (r for r in self._resident.values() if r.model_id not in self._pinned), key=lambda r: r.last_used)
            for victim in victims:
                self._evict(victim, need)
                if need <= self.headroom():
                    return True
            return False

    def _evict(self, victim: Resident, need: float | None = None, cancel: threading.Event | None = None) -> int:
        """Free an idle part, then wait for its memory to come back: until the room has grown by about
        what it held, or `need` fits, for at most EVICT_RETURN_S, or the job's `cancel` is set. Returns the
        room then."""
        before = self.headroom()
        self.free(victim.model_id, victim.name)
        gc.collect()
        mx.clear_cache()
        target = before + victim.bytes * EVICT_RETURN_SHARE
        began = time.monotonic()
        while True:
            try:
                self.monitor.refresh()
            except OSError:
                pass
            room = self.headroom()
            waited = time.monotonic() - began
            stop = cancel is not None and cancel.is_set()
            if room >= target or (need is not None and room >= need) or waited >= EVICT_RETURN_S or stop:
                log.info("evicted %s/%s (%.2f GB): room %.2f -> %.2f GB after %.2f s%s%s", victim.model_id,
                         victim.name, victim.bytes / GB, before / GB, room / GB, waited,
                         f", need {need / GB:.2f} GB" if need is not None else "", "; the job was stopped" if stop else "")
                return room
            if cancel is not None:
                cancel.wait(EVICT_POLL_S)
            else:
                time.sleep(EVICT_POLL_S)

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
        self.swapin_without_thrash = False
        self.unplanned: list[str] = []  # skipped parts the pipeline needed after all (see before_phase)
        self.started = False  # start() loaded everything; the pipeline runs from here
        manager.monitor.start_window()

    def _ensure_phase(self, phase: str) -> None:
        for name in self.spec.parts(phase):
            if not self.spec.components[name].keep:
                self._ensure(name)

    def _free_phase(self, phase: str) -> None:
        for name in self.spec.parts(phase):
            if not self.spec.components[name].keep:
                self.m.free(self.spec.id, name)

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

    def _stopped(self) -> None:
        if self.cancel_event.is_set():
            raise JobAborted("cancelled", "Generation cancelled")

    def start(self) -> None:
        """Load what the plan counted. A Stop is checked before each part (fix round 3 of task 19: an eager
        plan loaded every part, 2.1 s, before the first check, and the stopped model stayed resident)."""
        self._stopped()
        adapter = self.m._adapters[self.spec.id]
        if hasattr(adapter, "stream_mode"):
            adapter.stream_mode = self.plan.strategy == "streamed"
        if self.plan.strategy in ("sequential", "streamed"):
            # Establish the low-memory invariant at the job boundary as well as
            # at phase callbacks. A cached prompt can skip the encode callbacks,
            # otherwise leaving eager components from the preceding job resident
            # during reference-image preprocessing.
            self._free_phase("encode")
            self._free_phase("denoise")
        # Load exactly what the plan counted: never a part it skipped (fix round 2 of task 19: an eager plan
        # for a cached prompt left the 2.11 GB encoder out of its need and loaded it here anyway; on 16 GB
        # swap grew 8.26 -> 11.0 GB and the job stopped for memory pressure).
        for name, comp in self.spec.components.items():
            if comp.keep and name not in self.plan.skip:
                self._stopped()
                self._ensure(name)
        if self.plan.strategy == "eager":
            for name in self.spec.components:
                if name not in self.plan.skip:
                    self._stopped()
                    self._ensure(name)
        self._stopped()
        self.started = True

    def _encode_unplanned(self) -> None:
        """The pipeline encodes after all, with a part the plan skipped (its cache had the prompt with
        another negative prompt or guidance, say). That part was not counted, so make room as the
        low-memory strategies do: free the resident denoise parts first, unless there is clearly room;
        they load again for the denoise. The encoder is freed after the encode (see after_phase)."""
        missing = [n for n in self.spec.parts("encode")
                   if n in self.plan.skip and not self.spec.components[n].keep
                   and (self.spec.id, n) not in self.m._resident]
        if not missing:
            return
        self.unplanned += [n for n in missing if n not in self.unplanned]
        size = sum(self.spec.components[n].size for n in missing)
        if self.plan.strategy not in ("sequential", "streamed") and size * MARGIN["phase_free"] > self.m.headroom():
            self._free_phase("denoise")
            gc.collect()
            mx.clear_cache()
        log.info("%s: encoding with %s, which the plan skipped (%.2f GB)", self.spec.id, missing, size / GB)

    def before_phase(self, phase: str) -> None:
        self.check()
        self._phase_started[phase] = time.monotonic()
        if phase == "encode":
            if self.plan.strategy in ("sequential", "streamed"):
                self._free_phase("denoise")
            elif self.plan.skip:
                self._encode_unplanned()
            self._ensure_phase("encode")
            self.emit("encoding", {})
        elif phase == "denoise":
            # Image-conditioned pipelines may use the VAE before denoising.
            # In the low-memory strategies it must not overlap the transformer.
            self.after_phase("preprocess")
            if self.plan.strategy in ("sequential", "streamed"):
                self.m.free(self.spec.id, "vae")
            self._ensure_phase("denoise")
        elif phase == "preprocess":
            # Image-conditioned pipelines encode their source with the VAE before denoising.
            self._ensure("vae")
        elif phase == "layout":
            # Some multimodal pipelines need transformer metadata before their
            # normal denoise callback, while the VAE is still encoding references.
            self._ensure_phase("denoise")
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
        if phase == "encode" and (self.plan.strategy in ("phase_free", "sequential", "streamed") or self.unplanned):
            self._free_phase("encode")
        elif phase == "denoise" and self.plan.strategy in ("sequential", "streamed"):
            self._free_phase("denoise")

    def release_after_stop(self) -> None:
        """Free what the plan would have freed by the end of the job, for a job that stopped early (cancel,
        memory pressure, invalid numbers, an error): in sequential / streamed plans the encoder and denoise
        parts, in phase_free the encoder. Eager and phase_free keep the rest warm, as after a finished job.
        A job stopped before it ran (during start) frees every part it loaded, in every strategy: it made
        nothing, so nothing of it should stay warm (fix round 3 of task 19).
        Call it outside the `except` block: until that ends, the traceback holds the job's frames, and
        they hold the weights (task 17)."""
        strategy = self.plan.strategy
        phases = {"phase_free": ("encode",), "sequential": ("encode", "denoise"),
                  "streamed": ("encode", "denoise")}.get(strategy, ("encode",) if self.unplanned else ())
        adapter = self.m._adapters.get(self.spec.id)
        freed = False
        if not self.started:
            for name in self.loads:
                if (self.spec.id, name) in self.m._resident or (adapter is not None and adapter.is_loaded(name)):
                    self.m.free(self.spec.id, name)
                    freed = True
        for phase in phases:
            for name in self.spec.parts(phase):
                if self.spec.components[name].keep:
                    continue
                if (self.spec.id, name) in self.m._resident or (adapter is not None and adapter.is_loaded(name)):
                    self.m.free(self.spec.id, name)
                    freed = True
        if freed:
            gc.collect()
            mx.clear_cache()

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
        if rate > SWAPIN_RATE_CAP and self._thrashing(snap):
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

    def _thrashing(self, snap: sysmem.MemorySnapshot) -> bool:
        """Swap-in with a second sign that the Mac is short of memory, not coming back from it."""
        if snap.pressure > self.baseline_pressure or snap.swap_used - self.swap_start > SWAP_GROWTH_SIGN:
            return True
        if self.m.monitor.swapout_rate() > SWAPOUT_RATE_SIGN:
            return True
        self.swapin_without_thrash = True  # reported: the old rule would have stopped this job
        return False

    def report(self) -> dict:
        snap = self.m.monitor.last()
        return {
            **self.plan.as_dict(),
            "loads": self.loads,
            **({"skipped": list(self.plan.skip)} if self.plan.skip else {}),
            **({"unplanned_loads": self.unplanned} if self.unplanned else {}),
            "decode_mode": self.decode_mode,
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
            "peak_mlx_gb": round(mx.get_peak_memory() / GB, 2),
            "max_pressure": sysmem.PRESSURE_NAMES.get(self.max_pressure, str(self.max_pressure)),
            "swap_delta_gb": round((snap.swap_used - self.swap_start) / GB, 2),
            "max_swapin_mb_s": round(self.max_swapin_rate / 1024**2, 1),
            "swapin_while_recovering": self.swapin_without_thrash,
        }
