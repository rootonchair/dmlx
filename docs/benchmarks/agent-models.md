# Agent model candidates

Date: 2026-10-25 · Machine: Apple M5, 24 GB · mlx-vlm 0.7.6 · all 4-bit `mlx-community` builds, pinned
revisions in `engine/src/dmlx/prompting/local_vlm.py`.

Two checks per model:
- **Eval:** the 36-case routing eval (`python -m dmlx.prompting.agent_eval --model ID`): workflow and
  input images right. Images are placeholders.
- **Real images:** 8 cases on real pictures (`docs/benchmarks/raw/agent_candidates.py`): a boat and a
  teapot (answering "the boat" / "the teapot" to the agent's question, "make the boat red", "make the
  first one black and white", "make it red"), removing the text from a trading card, putting the title
  "NIGHT BLADE" back, and a new image. The descriptions were also read by hand.

| Model | Weights | Peak | Eval | Real | Median per message | Descriptions |
|---|---|---|---|---|---|---|
| **Qwen3-VL 8B** (default, 24 GB+) | 5.8 GB | 6.2 GB | **36/36** | **8/8** | 8.5 s | detailed; wrote `'NIGHT BLADE'` in single quotes and quoted the text to remove |
| **Qwen3.5 9B** | 6.0 GB | 6.8 GB | 35/36 | **8/8** | 9.9 s | detailed; `"NIGHT BLADE"` "at the top, above the character's head" |
| **Ministral 3 3B** (default, under 24 GB) | 2.8 GB | 4.2 GB | 35/36 | 6/8 | 6.3 s | good; `"NIGHT BLADE"` "at the top center" |
| **Qwen3-VL 4B** | 3.1 GB | 3.7 GB | 33/36 | 5/8 | 4.3 s | good; `"NIGHT BLADE"` but at the bottom |
| Gemma 4 E4B | 5.2 GB | 5.6 GB | 31/36 | 5/8 | 4.0 s | too short ("A close-up shot of a red fox in fresh snow.") |
| Gemma 4 E2B | 3.6 GB | 4.1 GB | 31/36 | 6/8 | 2.3 s | too short ("Remove all text from the image.") |
| Qwen3.5 4B | 3.0 GB | 4.2 GB | 32/36 | 5/8 | 6.0 s | described a glowing energy blade for "put a title back that says NIGHT BLADE" |
| Qwen3-VL 2B | 1.8 GB | 2.5 GB | 9/36 | 0/8 | — | invalid answers; fell back on 15 of 36 |
| MiniCPM-V 4.6 | 2.2 GB | 2.9 GB | 0/36 | 0/8 | — | no usable answer at all (not investigated) |

The app offers the first four. Peak = MLX peak over load plus the 8 real-image cases.

**Where the small models fail:** an older image the user names. Every model under 8B edited the image
in view for at least one of "the boat" (answering the question), "make the boat red", "the first one"
and "the one with the cat", because the agent sees only the current image and reads about older ones
(task 07). Ministral 3 3B also once asked a second question and once made a new image instead.

## Planning beside an image job

Qwen3-VL 8B planned "make the boat red" (real images) in one thread while Klein made a 1024 image in
another, in the same process (`docs/benchmarks/raw/concurrency.py`): 4/4 replies identical to planning
alone, no crash. The plan took about 28 s instead of 9 s, the image 34 s instead of about 20 s. The engine
does this only on Macs with 32 GB or more when both fit (`Engine._fits_beside_job`).
