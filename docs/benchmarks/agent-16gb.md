# Planning agent on the 16 GB tier

Date: 2026-10-25  
Machine: Apple Silicon Mac, 24 GB, run with `DMLX_EMULATE_TOTAL_GB=16`; everyday apps open (editor,
browser, chat), which left **4.7 to 6.0 GB available** by the engine's count (free + file cache).  
Model: FLUX.2 Klein 4B q4, 1024×1024. Router: disabled; an unclear message is sent as an edit, as if
the user clicked Edit.  
Each turn = `POST /v1/workflows/route` with `plan: true`, then `POST /v1/workflows`, as the app does.
A sampler and the no-hang probe cover the whole turn, planning included.

Turns: a rowboat at golden hour, then "make it night with a full moon", "add a lighthouse on the far
shore", "more fog", "make the boat red".

| Turn | planned by | rewrite | strategy | peak MLX | swap-in | pressure | GPU p99 | time | pass |
|---|---|---|---|---|---|---|---|---|---|
| 0 new image | rules (nothing to decide) | — | sequential | 3.39 GB | 0.2 MB/s | normal | 1.6 ms | 19 s | ✅ |
| 1 edit | skipped: no room | skipped: no room | sequential | 4.09 GB | 0.2 MB/s | warn | 4.9 ms | 38 s | ✅ |
| 2 edit | skipped: no room | skipped: no room | sequential | 4.10 GB | 0.1 MB/s | warn | 4.3 ms | 40 s | ✅ |
| 3 edit | skipped: no room | skipped: no room | sequential | 4.11 GB | 0.0 MB/s | normal | 4.0 ms | 40 s | ✅ |
| 4 edit | skipped: no room | skipped: no room | sequential | 4.12 GB | 0.3 MB/s | normal | 6.1 ms | 41 s | ✅ |

**5/5 turns pass the no-hang gate, but the agent never ran.** The assistant needs 6.9 GB free
(its 5.4 GB of weights on disk plus 1.5 GB working memory) on top of the 1.5 GB kept for macOS, so about 8.4 GB available. With everyday apps open a 16 GB Mac rarely has that.
`_plan` and the edit rewrite both refuse instead of swapping, so the gate holds, but:

- every edit runs on the user's own words, with no rewrite (the NIGHT BLADE failure mode for Qwen
  Image 2.1), and an unclear message falls back to the router or the Edit/New buttons;
- nothing in the app says so: the engine logs the skip and records `rewrite.skipped` in the job
  report only.

Not measured: a 16 GB Mac with few apps open, where the assistant does fit and making room for it
evicts the cached image model each turn (a reload cost, not a hang risk, since loads are gated the
same way). Qwen Image 2.1 edits (about 9.5 GB) were also refused on the 24 GB Mac in this state.

Script: `mem16.py` (session scratchpad). Raw rows: turn reports with loads, timings and probe results.

## With the agent's free-memory check turned off

Same turns, `make_room` bypassed for the agent only (image jobs keep their check), and a ballast
holding 6.3 GB so about 5 GB stayed available, the state where the agent was skipped above.

| Turn | check would have | agent | swap-in | swap growth | pressure | GPU p99 | result |
|---|---|---|---|---|---|---|---|
| 0 new image | — | — | 6.9 MB/s | 0.03 GB | warn | 1.4 ms | ✅ |
| 1 edit | refused (need 6.87 GB, room 3.34 GB) | ran, 14.9 s | **327 MB/s** | **2.68 GB** | **critical** | 3.6 ms | ❌ |
| 2 edit | refused (need 6.87 GB, room 5.77 GB) | ran, 12.7 s | 8.6 MB/s | 0.62 GB | warn | 1.6 ms | ❌ image job refused: 1.6 GB left for a 3.1 GB need |

Without the check, the first planned edit pushed macOS to critical pressure and swapped 2.7 GB out,
with swap-in at five times the gate's 64 MB/s limit. The probe's tiny GPU job stayed fast, but other
apps page back in from disk when used. On the next turn the agent ran again, and the image model then
had no room left, so the edit was refused. The check prevented exactly this.

**Decision (2026-10-25): the agent's check was removed anyway.** The agent and the edit rewrite now
always run; the engine still frees idle models first and logs when room is short. Image jobs keep
their check. On a busy 16 GB Mac, expect the result in the table above.
For comparison, with 10.8 GB available the agent passed its check and all 5 turns passed
(planning 12 to 13 s per edit, peak swap-in 2.5 MB/s).

## Ministral 3 3B, the default under 24 GB (2026-10-30)

Same five turns on the 16 GB tier, ballast so about 4.2 GB stayed available (less than the 8B runs
above), no free-memory check for the agent (as shipped), and image jobs now wait up to 10 s for memory
that is still coming back after the agent unloads (`Engine._plan_memory`).

| Turn | planned by | plan | image | strategy | peak MLX | swap-in | pressure | GPU p99 | pass |
|---|---|---|---|---|---|---|---|---|---|
| 0 new image | router (agent answer failed the checks) | 11.6 s | 17.6 s | sequential | 3.39 GB | 0.5 MB/s | warn | 1.4 ms | ✅ |
| 1 edit | Ministral 3 3B | 9.8 s | 37.9 s | streamed | 2.02 GB | 5.0 MB/s | warn | 21.9 ms | ✅ |
| 2 edit | Ministral 3 3B | 9.5 s | 34.7 s | sequential | 4.10 GB | **131.7 MB/s** | warn | 3.6 ms | ❌ |
| 3 edit | Ministral 3 3B | 8.5 s | 38.8 s | streamed | 2.04 GB | 10.5 MB/s | warn | 20.8 ms | ✅ |
| 4 edit | Ministral 3 3B | 8.4 s | 35.0 s | sequential | 4.12 GB | 2.8 MB/s | warn | 3.0 ms | ✅ |

Every edit was planned by the agent and every image was made; pressure never reached critical. One
turn had a swap-in burst twice the gate's 64 MB/s (net swap over the turn fell by 0.34 GB). Compare the
8B without its check above: critical pressure, 327 MB/s, and the next image refused.
