"""`dmlx bench-family`: measure the memory numbers a model spec needs (task 15).

Runs real jobs in the sequential strategy, so only the parts a phase needs are loaded. For each
phase it records the peak above what was resident when the phase started: that is the working
memory the planner adds on top of the weights. The decode runs once in the low-memory mode and
once in one piece. The result is printed as spec lines (per megapixel, with a margin).

    uv run dmlx bench-family --model ernie-image-turbo --spec path/to/ernie.yaml --sizes 512,1024
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path

GB = 1024**3
MARGIN = 1.10  # measured values vary a little between runs


def add_parser(sub) -> None:
    p = sub.add_parser("bench-family", help="Measure a model's working memory for its spec")
    p.add_argument("--model", required=True)
    p.add_argument("--spec", default=None, help="a spec YAML that is not installed yet (runs in a temp DMLX_HOME)")
    p.add_argument("--sizes", default="512,1024")
    p.add_argument("--edit", action="store_true", help="also measure an edit with one reference image")
    p.add_argument("--steps", type=int, default=None, help="fewer steps for speed; memory does not depend on them")
    p.add_argument("--prompt", default="A small wooden rowboat on a still mountain lake at golden hour, 35mm film photo")
    p.add_argument("--out", default=None, help="write the raw measurements as JSON")
    p.add_argument("--from-json", default=None, help="recompute the spec numbers from an earlier --out file")
    p.set_defaults(func=run)


class _PhasePeaks:
    """Wraps JobGuard so each phase records its peak above the memory resident at its start."""

    def __init__(self, decode_mode: str):
        import mlx.core as mx

        from dmlx import memory

        self.peaks: dict[str, float] = {}
        starts: dict[str, int] = {}
        guard = memory.JobGuard
        before, after = guard.before_phase, guard.after_phase
        peaks = self.peaks

        def before_phase(g, phase):
            before(g, phase)
            mx.synchronize()
            starts[phase] = mx.get_active_memory()
            mx.reset_peak_memory()

        def after_phase(g, phase):
            if phase in starts:
                mx.synchronize()
                extra = (mx.get_peak_memory() - starts.pop(phase)) / GB
                peaks[phase] = max(peaks.get(phase, 0.0), extra)
            after(g, phase)

        def choose_decode(g):
            g.decode_mode = decode_mode
            adapter = g.m._adapters[g.spec.id]
            if hasattr(adapter, "set_decode_mode"):
                adapter.set_decode_mode(decode_mode)

        self._restore = (guard, before, after, guard._choose_decode)
        guard.before_phase, guard.after_phase, guard._choose_decode = before_phase, after_phase, choose_decode

    def close(self) -> None:
        guard, before, after, choose = self._restore
        guard.before_phase, guard.after_phase, guard._choose_decode = before, after, choose


def _job(engine, model, size, decode_mode, prompt, task="t2i", image_ids=None, steps=None) -> dict:
    peaks = _PhasePeaks(decode_mode)
    try:
        job = engine.submit(task=task, prompt=prompt, model=model, size=f"{size}x{size}", seed=7,
                            image_ids=image_ids, steps=steps, strategy="sequential", force=True)
        job.finished.wait()
    finally:
        peaks.close()
    if job.status != "done":
        raise SystemExit(f"{model} {task} {size}: {job.error}")
    row = {"task": task, "size": size, "decode_mode": decode_mode, "phases_gb": {k: round(v, 3) for k, v in peaks.peaks.items()},
           "timings": job.report.get("timings"), "result_id": job.result_ids[0]}
    print(json.dumps(row), flush=True)
    return row


def _per_mp(rows, phase) -> float:
    # Small images have a fixed overhead that a per-megapixel number would blow up; count them as
    # 1 MP, so they get a safe estimate and the common 1024 px size an accurate one.
    return max(r["phases_gb"].get(phase, 0.0) / max(r["size"] ** 2 / 1_000_000, 1.0) for r in rows)


def _numbers(rows: list[dict], low: str) -> dict:
    t2i = [r for r in rows if r["task"] == "t2i" and r["decode_mode"] == low]
    return {
        "activation_gb_per_mp": _up(_per_mp(t2i, "denoise")),
        "decode_gb_per_mp": _up(_per_mp([r for r in rows if r["decode_mode"] == "full"], "decode")),
        f"decode_{low}_gb_per_mp": _up(_per_mp(t2i, "decode")),
        "encode_gb_measured": round(max(r["phases_gb"].get("encode", 0.0) for r in rows), 2),
    }


def _print(numbers: dict) -> None:
    print("\n# Measured with dmlx bench-family (sequential; 10% margin; sizes under 1 MP count as 1 MP)")
    for key, value in numbers.items():
        print(f"{key}: {value}")


def _up(value: float) -> float:
    return math.ceil(value * MARGIN * 10) / 10


def run(args) -> None:
    if args.from_json:
        rows = json.loads(Path(args.from_json).read_text())["rows"]
        _print(_numbers(rows, rows[0]["decode_mode"]))
        return
    if args.spec:
        home = tempfile.mkdtemp(prefix="dmlx-bench-")
        os.environ["DMLX_HOME"] = home
        Path(home, "models").mkdir()
        shutil.copy(args.spec, Path(home, "models", Path(args.spec).name))
    os.environ.setdefault("DMLX_ROUTER_PROVIDER", "disabled")
    from dmlx.engine import Engine

    engine = Engine()
    spec = engine.registry.get(args.model)
    low = spec.low_memory_decode
    sizes = [int(s) for s in args.sizes.split(",")]
    rows = []
    for size in sizes:
        rows.append(_job(engine, args.model, size, low, args.prompt, steps=args.steps))
        rows.append(_job(engine, args.model, size, "full", args.prompt, steps=args.steps))
    t2i = [r for r in rows if r["decode_mode"] == low]
    numbers = _numbers(rows, low)
    if args.edit and spec.supports("edit"):
        size = sizes[-1]
        source = next(r["result_id"] for r in rows if r["size"] == size)
        edit = _job(engine, args.model, size, low, "make it night, keep everything else the same", "edit", [source],
                    steps=args.steps)
        rows.append(edit)
        plain = next(r for r in t2i if r["size"] == size)["phases_gb"]["denoise"]
        numbers["reference_activation_factor"] = _up(max(edit["phases_gb"]["denoise"] / plain - 1, 0.0))
    _print(numbers)
    if args.out:
        Path(args.out).write_text(json.dumps({"model": args.model, "rows": rows, "numbers": numbers}, indent=2))
