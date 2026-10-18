"""No-hang tooling.

probe    a separate process that measures how responsive the machine stays: CPU timer lateness
         and the latency of a tiny GPU job (what a UI compositor competes with).
ballast  holds incompressible memory to emulate a smaller Mac.

The probe prints one JSON line with percentiles when its stdin closes or after --seconds.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import statistics
import sys
import time


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, int(round(q * (len(values) - 1))))
    return values[idx]


def probe(seconds: float | None, interval: float = 0.05, gpu: bool = True) -> dict:
    mx = None
    if gpu:
        import mlx.core as mx  # noqa: PLC0415

        a = mx.ones((64, 64))
        mx.eval(a @ a)
    timer_late: list[float] = []
    gpu_lat: list[float] = []
    start = time.monotonic()
    next_tick = start + interval
    while True:
        if seconds is not None and time.monotonic() - start >= seconds:
            break
        if seconds is None:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if ready and not sys.stdin.read(1):
                break
        now = time.monotonic()
        if now < next_tick:
            time.sleep(next_tick - now)
        woke = time.monotonic()
        timer_late.append((woke - next_tick) * 1000)
        next_tick += interval
        if next_tick < woke:
            next_tick = woke + interval
        if mx is not None:
            t0 = time.monotonic()
            mx.eval(a @ a)
            gpu_lat.append((time.monotonic() - t0) * 1000)
    return {
        "samples": len(timer_late),
        "timer_late_ms": {"p50": round(_pct(timer_late, 0.5), 1), "p99": round(_pct(timer_late, 0.99), 1),
                          "max": round(max(timer_late, default=0), 1)},
        "gpu_ms": {"p50": round(_pct(gpu_lat, 0.5), 1), "p95": round(_pct(gpu_lat, 0.95), 1),
                   "p99": round(_pct(gpu_lat, 0.99), 1), "max": round(max(gpu_lat, default=0), 1),
                   "mean": round(statistics.fmean(gpu_lat), 1) if gpu_lat else 0.0},
    }


def ballast(gb: float, hold: float | None) -> None:
    block = os.urandom(1024 * 1024)
    chunks = []
    for _ in range(int(gb * 1024)):
        chunks.append(bytearray(block))
    print(json.dumps({"ballast_gb": gb, "pid": os.getpid()}), flush=True)
    end = time.monotonic() + hold if hold else None
    while end is None or time.monotonic() < end:
        # Touch a byte per MB now and then so pages stay resident.
        for c in chunks[:: max(1, len(chunks) // 64)]:
            c[0] = (c[0] + 1) % 256
        time.sleep(1.0)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m dmlx.nohang")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--interval", type=float, default=0.05)
    p.add_argument("--no-gpu", action="store_true")
    b = sub.add_parser("ballast")
    b.add_argument("--gb", type=float, required=True)
    b.add_argument("--hold", type=float, default=None)
    args = parser.parse_args()
    if args.cmd == "probe":
        print(json.dumps(probe(args.seconds, args.interval, gpu=not args.no_gpu)), flush=True)
    else:
        ballast(args.gb, args.hold)


if __name__ == "__main__":
    main()
