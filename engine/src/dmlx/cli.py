"""Command line: dmlx serve | pull | bench | models."""

from __future__ import annotations

import argparse
import logging
import os
import sys


def _watch_parent(parent_pid: int) -> None:
    """Exit when the app that launched us goes away, so no orphan keeps holding GPU memory."""
    import threading
    import time

    def loop():
        while True:
            time.sleep(1.0)
            try:
                os.kill(parent_pid, 0)
            except ProcessLookupError:
                os._exit(0)
            except PermissionError:
                pass

    threading.Thread(target=loop, name="dmlx-parent-watch", daemon=True).start()


def _serve(args) -> None:
    if args.parent_pid:
        _watch_parent(args.parent_pid)
    if args.short_buffers:
        # Shorter Metal command buffers. Measured: no responsiveness gain on M5 and ~4x more working
        # memory (docs/benchmarks/b1.md), so this is opt-in for experiments only.
        os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "32")
        os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "256")
    import uvicorn

    from dmlx.api import create_app

    app = create_app(token=args.token or os.environ.get("DMLX_TOKEN"))
    if args.port_file:
        # Tell the parent process which port we bound (useful with --port 0).
        config = uvicorn.Config(app, host=args.host, port=args.port, log_level=args.log_level)
        server = uvicorn.Server(config)

        import threading

        def announce():
            import time

            while not server.started:
                time.sleep(0.05)
            port = server.servers[0].sockets[0].getsockname()[1]
            with open(args.port_file, "w") as fh:
                fh.write(str(port))
            print(f"DMLX_READY port={port}", flush=True)

        threading.Thread(target=announce, daemon=True).start()
        server.run()
    else:
        print(f"DMLX_READY port={args.port}", flush=True)
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


def _pull(args) -> None:
    """Download a model as the app does: every repo it needs, resumable (task 19)."""
    from dmlx import downloads
    from dmlx.registry import Registry

    spec = Registry.load_builtin().get(args.model)
    state = downloads.DownloadState(spec.id)
    downloads.Downloader()._run(spec, state, None)
    if state.status != "done":
        raise SystemExit(state.error or state.status)
    print(downloads.local_path(spec))


def _models(_args) -> None:
    from dmlx import downloads
    from dmlx.registry import Registry

    for spec in Registry.load_builtin().all():
        path = downloads.local_path(spec)
        print(f"{spec.id:20s} {spec.license:12s} {'downloaded' if path else 'not downloaded'}  {spec.repo}")


def main(argv: list[str] | None = None) -> None:
    from dmlx import paths

    # Before anything imports huggingface_hub or mflux: dMLX keeps its models in its own folder.
    paths.use_own_caches()
    parser = argparse.ArgumentParser(prog="dmlx")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Run the HTTP server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8642)
    serve.add_argument("--port-file", default=None)
    serve.add_argument("--token", default=None)
    serve.add_argument("--log-level", default="info")
    serve.add_argument("--short-buffers", action="store_true", help="experimental; uses much more memory")
    serve.add_argument("--parent-pid", type=int, default=None, help="exit when this process is gone")
    serve.set_defaults(func=_serve)

    pull = sub.add_parser("pull", help="Download a model")
    pull.add_argument("model")
    pull.set_defaults(func=_pull)

    models = sub.add_parser("models", help="List models")
    models.set_defaults(func=_models)

    from dmlx.bench import add_bench_parser
    from dmlx.family_bench import add_parser as add_family_bench_parser
    from dmlx.seedvr2_bench import add_seedvr2_bench_parser

    add_bench_parser(sub)
    add_seedvr2_bench_parser(sub)
    add_family_bench_parser(sub)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stderr)
    args.func(args)


if __name__ == "__main__":
    main()
