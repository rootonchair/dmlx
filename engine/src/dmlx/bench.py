"""`dmlx bench`: run real generations in-process and report timings, memory, and responsiveness.

Each case runs while a probe process measures CPU timer lateness and tiny-GPU-job latency.
A case passes the no-hang gate when memory pressure never reaches critical, swap grows less
than the cap, and the probe's p99 GPU latency stays under the threshold.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

GPU_P99_LIMIT_MS = 250.0
TIMER_P99_LIMIT_MS = 50.0


def add_bench_parser(sub) -> None:
    bench = sub.add_parser("bench", help="Benchmark + no-hang check")
    bench.add_argument("--model", action="append", required=True)
    bench.add_argument("--task", choices=["t2i", "edit"], default="t2i")
    bench.add_argument("--strategy", action="append", default=None, help="eager|phase_free|sequential (repeatable)")
    bench.add_argument("--size", default="1024x1024")
    bench.add_argument("--runs", type=int, default=2, help="runs per case; run 2+ shows warm behaviour")
    bench.add_argument("--prompt", default="A small wooden rowboat on a still mountain lake at golden hour, 35mm film photo")
    bench.add_argument("--edit-prompt", default="make it night, keep the boat exactly the same")
    bench.add_argument("--out", default=None, help="write JSON results here")
    bench.add_argument("--no-probe", action="store_true")
    bench.set_defaults(func=run_bench)


def _start_probe():
    return subprocess.Popen(
        [sys.executable, "-m", "dmlx.nohang", "probe"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )


def _stop_probe(proc) -> dict:
    proc.stdin.close()
    out = proc.stdout.read().strip()
    proc.wait(timeout=10)
    return json.loads(out) if out else {}


def run_bench(args) -> None:
    from dmlx.engine import Engine

    engine = Engine()
    strategies = args.strategy or ["auto"]
    results = []
    for model in args.model:
        for strategy in strategies:
            engine.unload(model)
            source_id = None
            for run in range(args.runs):
                task = args.task
                prompt = args.prompt
                image_ids = None
                if task == "edit":
                    if source_id is None:
                        source_id = _first_image(engine, args, model)
                    image_ids = [source_id]
                    prompt = args.edit_prompt
                probe = None if args.no_probe else _start_probe()
                time.sleep(0.5)
                started = time.monotonic()
                job = engine.submit(task=task, prompt=prompt, model=model, size=args.size, seed=42 + run,
                                    image_ids=image_ids, strategy=strategy, force=True)
                job.finished.wait()
                wall = time.monotonic() - started
                responsiveness = _stop_probe(probe) if probe else {}
                report = job.report or {}
                gpu_p99 = responsiveness.get("gpu_ms", {}).get("p99", 0)
                timer_p99 = responsiveness.get("timer_late_ms", {}).get("p99", 0)
                passed = (
                    job.status == "done"
                    and report.get("max_pressure") != "critical"
                    and report.get("swap_delta_gb", 0) < 1.5
                    and (not responsiveness or (gpu_p99 <= GPU_P99_LIMIT_MS and timer_p99 <= TIMER_P99_LIMIT_MS))
                )
                row = {
                    "model": model, "task": task, "strategy_requested": strategy, "run": run + 1,
                    "status": job.status, "error": job.error, "wall_s": round(wall, 2),
                    "report": report, "responsiveness": responsiveness, "no_hang_pass": passed,
                    "result_ids": job.result_ids,
                }
                results.append(row)
                print(json.dumps(_summary(row)), flush=True)
    status = engine.status()
    out = {"machine": status, "cases": results}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
    failed = [r for r in results if not r["no_hang_pass"]]
    print(f"{len(results) - len(failed)}/{len(results)} cases passed the no-hang gate")


def _first_image(engine, args, model) -> str:
    job = engine.submit(task="t2i", prompt=args.prompt, model=model, size=args.size, seed=7, force=True)
    job.finished.wait()
    if job.status != "done":
        raise SystemExit(f"could not create a source image: {job.error}")
    return job.result_ids[0]


def _summary(row: dict) -> dict:
    r = row["report"]
    return {
        "model": row["model"], "task": row["task"], "run": row["run"], "status": row["status"],
        "strategy": r.get("strategy"), "loads": r.get("loads"), "wall_s": row["wall_s"],
        "timings": r.get("timings"), "peak_mlx_gb": r.get("peak_mlx_gb"),
        "pressure": r.get("max_pressure"), "swap_delta_gb": r.get("swap_delta_gb"),
        "gpu_p99_ms": row["responsiveness"].get("gpu_ms", {}).get("p99"),
        "timer_p99_ms": row["responsiveness"].get("timer_late_ms", {}).get("p99"),
        "pass": row["no_hang_pass"], "error": (row["error"] or {}).get("message"),
    }
