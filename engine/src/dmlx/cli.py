"""Command line: dmlx serve | pull | bench | models."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys


def _serve(args) -> None:
    if args.responsive:
        # Keep Metal command buffers short so WindowServer is not starved while we compute.
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
    from huggingface_hub import snapshot_download

    from dmlx.registry import Registry

    spec = Registry.load_builtin().get(args.model)
    print(snapshot_download(spec.repo))


def _models(_args) -> None:
    from dmlx import downloads
    from dmlx.registry import Registry

    for spec in Registry.load_builtin().all():
        path = downloads.local_path(spec)
        print(f"{spec.id:20s} {spec.license:12s} {'downloaded' if path else 'not downloaded'}  {spec.repo}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="dmlx")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Run the HTTP server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8642)
    serve.add_argument("--port-file", default=None)
    serve.add_argument("--token", default=None)
    serve.add_argument("--log-level", default="info")
    serve.add_argument("--responsive", action=argparse.BooleanOptionalAction, default=True)
    serve.set_defaults(func=_serve)

    pull = sub.add_parser("pull", help="Download a model")
    pull.add_argument("model")
    pull.set_defaults(func=_pull)

    models = sub.add_parser("models", help="List models")
    models.set_defaults(func=_models)

    from dmlx.bench import add_bench_parser

    add_bench_parser(sub)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stderr)
    args.func(args)


if __name__ == "__main__":
    main()
