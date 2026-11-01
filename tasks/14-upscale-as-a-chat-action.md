# 14 · Upscale and restore as a chat action

**Status:** Open (from the competitor comparison, 2026-11-02).

## Why
Draw Things ships SeedVR2 upscaling and ComfyUI has upscaling nodes; dMLX has none. "Make it 4K" or "make
it sharper" is a natural chat request, and the agent can route it like any other workflow.

## Today
There is a SeedVR2 adapter in the engine with tests (`engine/tests/test_seedvr2_adapter.py`,
`test_seedvr2_bench.py`); it is not offered in the app or to the agent.

## To do
- [ ] Check the adapter's state, memory need at 2× and 4×, and time on 16 / 24 GB.
- [ ] Expose an `image.upscale` workflow in the capabilities when the model is installed; the agent sees it
      in its workflow list and picks it ("make it 4K", "sharper", "print size").
- [ ] Add the upscaler to Add Model; the result is a new version of the turn.
- [ ] Eval cases: upscale requests, and edits that only sound like it ("make it bigger" = zoom?).

## Done when
"make it 4K" in the app gives an upscaled version, within the memory gate on 16 GB.
