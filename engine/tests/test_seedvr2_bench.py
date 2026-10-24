import argparse

from dmlx.seedvr2_bench import add_seedvr2_bench_parser


def test_seedvr2_benchmark_requires_local_inputs():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    add_seedvr2_bench_parser(sub)
    args = parser.parse_args([
        "bench-seedvr2", "--model-path", "/tmp/model", "--image", "/tmp/image.png",
        "--output-dir", "/tmp/output", "--scale", "2x",
    ])
    assert args.variant == "3b"
    assert args.scale == ["2x"]
