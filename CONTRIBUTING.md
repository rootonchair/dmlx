# Contributing to dMLX

Thanks for helping. dMLX is Apache-2.0; by contributing you agree your work is licensed the same way.

## Ground rules

1. **Never make the Mac hang.** Any change that loads more, keeps more resident, or changes how
   memory is estimated must come with numbers from `dmlx bench` (see `docs/benchmarks/b1.md`).
2. **No new model ports here.** Models come from engines (mflux today) through adapters in
   `engine/src/dmlx/adapters/`. Improve the engine upstream when you can.
3. Do not import code from GPL projects (ComfyUI, Draw Things). Interoperate over network APIs.
4. Never commit model weights, generated images, or benchmark raw output.

## Setup

```bash
cd engine && uv sync && uv run pytest            # unit tests, fake adapter, no GPU
uv run pytest -m integration                      # real models (downloads ~5 GB on first run)
cd ../app && swift test                           # routing, store, SSE parser
tools/dev/run_app.sh                              # run the app against ../engine
```

## Adding a model

1. Add `engine/src/dmlx/models/<id>.yaml` (components, tasks, defaults, license).
2. If its engine family is new, add an adapter implementing `load_component`, `free_component`,
   `generate` (calling `PhaseHooks` at phase boundaries), `has_embedding`, `set_tiled_decode`.
3. Measure per-phase working memory at 1024² and record `activation_gb_per_mp`,
   `decode_gb_per_mp`, `decode_tiled_gb_per_mp` in the YAML.
4. Run `uv run dmlx bench --model <id> --task t2i --runs 2` (and `--task edit` if supported) and
   include the summary lines in the PR.

## Scripted UI checks

`DMLX_HARNESS=/path/script.json tools/dev/run_app.sh` drives the app (send messages, wait,
variation, kill the engine, dump state, snapshot). See `app/Sources/dMLX/DevHarness.swift`.

## Commits and PRs

Short imperative subjects (`Fix edit routing for attachments`). PRs describe scope, list how you
validated (tests, bench numbers), and include screenshots for UI changes.
