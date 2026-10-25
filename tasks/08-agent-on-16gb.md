# 08 · The agent on a busy 16 GB Mac

**Status:** Changed 2026-10-25: the agent's free-memory check was removed, so it always runs. On a
busy 16 GB Mac that means critical pressure and heavy swap on edits (measured, see benchmark doc).

## Problem
- The assistant (Qwen3-VL-8B) needs about 8.4 GB available: 6.9 GB for itself plus the 1.5 GB macOS
  reserve. A 16 GB Mac with everyday apps open has 4.7 to 6 GB.
- Before 2026-10-25 the engine skipped the agent and the edit rewrite then (5/5 turns passed the
  no-hang gate, but edits ran on the user's raw words, silently).
- The check is now removed by decision. With about 5 GB available the first planned edit reached
  critical pressure and 327 MB/s swap-in, and the next image job was refused (benchmark doc).

## Options
- A smaller planner for the 16 GB tier (e.g. Qwen3-VL-4B 4-bit, about 2.5 GB); needs the eval re-run.
- Lower the planner's image size and token budget on 16 GB to cut its working memory.

## Done when
- On the 16 GB tier with everyday apps open, a planned edit stays under 64 MB/s swap-in.
- Measured again with few apps open, where the planner fits and evicts the image model each turn.
