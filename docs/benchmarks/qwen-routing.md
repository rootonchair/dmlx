# Local Qwen routing checkpoint evaluation

Evaluated on 2026-10-20 on an Apple M5 with 24 GB unified memory.

## Configuration

- Model: `mlx-community/Qwen3-4B-Instruct-2507-4bit`
- Model revision: `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`
- Weights SHA-256: `2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f`
- Runtime: `mlx-lm==0.32.2`
- Decoding: greedy, maximum 192 tokens
- Historical input resolution: a second maximum-96-token pass, used only when history exists

## Regression fixture results

The checked-in 26-case fixture includes generation, editing, upscale, restore,
enhance, variation, video, misleading keywords, short follow-ups, four
languages, and six chat-history cases with exact ordered result IDs, including
an implicit edit that must continue from the latest cumulative result.

| Metric | Result |
|---|---:|
| Workflow accuracy | 26/26 (100%) |
| Coverage | 100% |
| Exact ordered historical input selection | 6/6 (100%) |
| Mean latency | 1,813 ms |
| Minimum / maximum latency | 1,367 / 3,521 ms |
| Measured active MLX memory | 2,160 MB |

The prompt and few-shot examples were iterated against this regression fixture,
so these numbers are not a held-out generalization estimate. Before declaring
the model production-quality, run a larger independently authored set and the
separate 16 GB memory gate. The current result is sufficient to make Local Qwen
the on-device default when this exact checkpoint is installed; malformed output
or provider failure still falls back to manual workflow selection.

## Reproduce

```bash
cd engine
uv sync --extra local-llm
uv run --extra local-llm python -m dmlx.routing.evaluate \
  --local-llm-model-path "$HOME/Library/Application Support/dMLX/checkpoints/qwen3-4b-instruct-2507-4bit" \
  --local-llm-revision 50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b
```
