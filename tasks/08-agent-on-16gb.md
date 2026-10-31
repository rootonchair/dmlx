# 08 · The agent on a busy 16 GB Mac

**Status:** Mostly done 2026-10-30: with Ministral 3 3B (the default under 24 GB) a busy 16 GB Mac
passes 4/5 turns, never critical, every image made; one swap-in burst (132 MB/s) remains.

## Problem
- The assistant (Qwen3-VL-8B) needs about 8.4 GB available: 6.9 GB for itself plus the 1.5 GB macOS
  reserve. A 16 GB Mac with everyday apps open has 4.7 to 6 GB.
- Before 2026-10-25 the engine skipped the agent and the edit rewrite then (5/5 turns passed the
  no-hang gate, but edits ran on the user's raw words, silently).
- The check is now removed by decision. With about 5 GB available the first planned edit reached
  critical pressure and 327 MB/s swap-in, and the next image job was refused (benchmark doc).

## Options
- [x] A smaller planner for the 16 GB tier: the user now picks the agent model (Settings › Intelligence ›
      Agent), and Macs under 24 GB default to Ministral 3 3B (4.2 GB peak, 35/36 eval, 6/8 real images;
      `docs/benchmarks/agent-models.md`). Re-measured: 4/5 turns pass, see below.
- [x] Image jobs wait up to 10 s for memory that is still coming back after the agent unloads
      (`Engine._plan_memory`); before, an edit right after planning was refused with 2.2 GB usable.
- Lower the planner's image size and token budget on 16 GB to cut its working memory.

## Done when
- On the 16 GB tier with everyday apps open, a planned edit stays under 64 MB/s swap-in.
- Measured again with few apps open, where the planner fits and evicts the image model each turn.

## Re-measured with Ministral 3 3B (2026-10-30, `docs/benchmarks/agent-16gb.md`)
About 4.2 GB available, five turns: every edit planned by the agent (8-10 s), every image made,
pressure warn at worst, 4/5 pass. One turn had a 131.7 MB/s swap-in burst.

## Remaining
- [ ] The one swap-in burst: find which step causes it (the agent's load, or the image model reloading
      after it) and whether waiting a little longer for memory, or the planner's smaller image size, avoids it.
