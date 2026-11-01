# 13 · Faster turns: previews and a planner that stays loaded

**Status:** Open (from the competitor comparison, 2026-11-02).

## Why
Waiting is the visible weakness against Draw Things and the cloud: 3–10 s of planning plus ~20–40 s per
image (~37 s per edit at 1024² on a base M5), with only a progress bar. On 16 GB the agent is freed before
every image job, so every message loads it again.

## Options
- **Live previews:** decode cheap latent previews every few steps (a small linear latent-to-RGB map, or a
  tiny decoder) and stream them as `preview` events; show them in the image card. Measure the cost per step.
- **Planner residency on 16 GB:** keep Ministral 3B loaded when the memory plan says the image job still
  fits beside it (strategy streamed or sequential), instead of always freeing it; measure swap-in against
  the task 08 gate.
- **Shorter planning:** cache the system prompt's KV between messages; skip the describe step for simple
  edits (the agent already decides `describe`).

## Done when
A follow-up edit on 16 GB shows a first preview within a few seconds, and planning a message without a
reload where memory allows, with the 16 GB gate still passing.
