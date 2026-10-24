"""Opt-in SeedVR2 spike using local weights only.

This command intentionally requires --model-path. It never downloads weights or
adds the upscaler to the model catalog; enablement follows a successful no-hang
report on the target RAM tier.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from dmlx import sysmem
from dmlx.bench import GPU_P99_LIMIT_MS, TIMER_P99_LIMIT_MS, _start_probe, _stop_probe
from dmlx.memory import GB, PressureMonitor


def add_seedvr2_bench_parser(sub) -> None:
    bench = sub.add_parser("bench-seedvr2", help="Spike SeedVR2 from an existing local checkpoint")
    bench.add_argument("--model-path", type=Path, required=True)
    bench.add_argument("--image", type=Path, required=True)
    bench.add_argument("--variant", choices=["3b", "7b"], default="3b")
    bench.add_argument("--scale", action="append", choices=["2x", "3x", "4x"], default=None)
    bench.add_argument("--softness", type=float, default=0.0)
    bench.add_argument("--runs", type=int, default=1)
    bench.add_argument("--output-dir", type=Path, required=True)
    bench.add_argument("--out", type=Path, default=None, help="optional JSON report")
    bench.add_argument("--no-probe", action="store_true")
    bench.set_defaults(func=run_seedvr2_bench)


def run_seedvr2_bench(args) -> None:
    if not args.model_path.is_dir():
        raise SystemExit("--model-path must be an existing local checkpoint directory")
    if not args.image.is_file():
        raise SystemExit("--image must be an existing image")
    if not 0 <= args.softness <= 1:
        raise SystemExit("--softness must be between 0 and 1")

    import mlx.core as mx
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.seedvr2 import SeedVR2
    from mflux.utils.scale_factor import ScaleFactor

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = ModelConfig.seedvr2_3b() if args.variant == "3b" else ModelConfig.seedvr2_7b()
    monitor = PressureMonitor()
    load_started = time.monotonic()
    model = SeedVR2(model_path=str(args.model_path), model_config=config)
    load_seconds = time.monotonic() - load_started
    rows = []
    try:
        for scale in args.scale or ["2x", "3x", "4x"]:
            for run in range(args.runs):
                probe = None if args.no_probe else _start_probe()
                monitor.start_window()
                before = sysmem.snapshot()
                mx.reset_peak_memory()
                started = time.monotonic()
                generated = model.generate_image(
                    seed=42 + run, image_path=args.image,
                    resolution=ScaleFactor.parse(scale), softness=args.softness)
                mx.synchronize()
                wall = time.monotonic() - started
                output = args.output_dir / f"{args.image.stem}-{scale}-run{run + 1}.png"
                generated.save(output)
                after = sysmem.snapshot()
                responsiveness = _stop_probe(probe) if probe else {}
                gpu_p99 = responsiveness.get("gpu_ms", {}).get("p99", 0)
                timer_p99 = responsiveness.get("timer_late_ms", {}).get("p99", 0)
                passed = (
                    monitor.window_max_pressure() < 2
                    and max(0, after.swap_used - before.swap_used) < 4 * GB
                    and (not responsiveness or (gpu_p99 <= GPU_P99_LIMIT_MS and timer_p99 <= TIMER_P99_LIMIT_MS))
                )
                row = {
                    "variant": args.variant, "scale": scale, "run": run + 1,
                    "load_seconds": round(load_seconds, 3), "wall_seconds": round(wall, 3),
                    "peak_mlx_gb": round(mx.get_peak_memory() / GB, 3),
                    "pressure": monitor.window_max_pressure(),
                    "swap_delta_gb": round(max(0, after.swap_used - before.swap_used) / GB, 3),
                    "responsiveness": responsiveness, "no_hang_pass": passed,
                    "output": str(output),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    finally:
        monitor.stop()
    report = {"machine_total_gb": round(sysmem.total_memory() / GB), "cases": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
    failed = sum(not row["no_hang_pass"] for row in rows)
    print(f"{len(rows) - failed}/{len(rows)} SeedVR2 cases passed the no-hang gate")
