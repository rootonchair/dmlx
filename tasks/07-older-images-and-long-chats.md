# 07 · The agent can't see older images, and long chats are cut

**Status:** Done 2026-11-01 (prompt-vs-pixels passes on the 8B, the 9B and Ministral; not the 4B)

## Problem
- `agent.shown_images` shows the attachments, or else only the current image (at most 3). For "the first
  one" or "the one with the cat", the agent picks by older prompts, not pixels, which contradicts its own
  "trust the image over the chat" rule.
- `agent.chat_lines` keeps only the last 16 lines, but `_resolve_id` looks up "image N" across the full
  history the app sends (up to 24 items). In a long chat the agent can pick an image it cannot see in
  the chat.

Seen in the app: after "make it red", a question, and the answer "the boat", the agent edited the teapot
it could see rather than the boat it could only read about (worked around for answers in task 04; a
message that names an older image still relies on the prompt text).

Every agent model under 8B misses these cases on real images (`docs/benchmarks/agent-models.md`),
so this decides whether the 16 GB default is good enough.

## Options
- Show small thumbnails of the images the message may refer to (the chat images it names, or the last
  few), within a token budget; measure the latency cost.
- Keep a short line for every image in the chat (number, id, short prompt) even when older messages
  are cut, like Agent Banana's "Context Folding" (arXiv 2602.09084).
- Reject resolved ids that are not in the lines the model saw.

## Done when
- Eval cases with long chats and with an image whose prompt disagrees with its pixels pass.
- Planning latency stays within about 12 s.

## What changed
- **The 'look' tool** (`agent.plan`, `agent.look_request`): the agent starts with only the image 'this'
  means (the attachments, or the latest image) and a list of every chat image. To see others it answers
  `{"look": ["image 1", ...]}` (at most 3, twice per message); they are shown as 384 px previews and it
  is asked again. Asking for nothing new ends the looking. Its instructions say when to look, with an
  example. The decision lists `looked`.
- **Long chats:** "Images in this chat" lists every image (number, id, prompt) even when older chat lines
  are cut, so any "image N" the agent names is one it was told about.
- A first version showed up to 4 previews chosen by a fixed rule (named, then recent). It helped the small
  models but made the 8B add images it could see as references; replaced by the tool, so the model decides.
- Eval: `long-image-n` and `long-the-cat-one` (10 images in the chat). Real-image checks: a
  `pixels-over-prompt` case (image 1's prompt says "a vintage car", its pixels are the rowboat).

## Results (real images 9 cases, eval 38 cases)
| Model | Real | Eval | Before (real 8 / eval 36) |
|---|---|---|---|
| Qwen3-VL 8B | 8/9 | 38/38 | 8/8, 36/36 |
| Qwen3.5 9B | 8/9 | 37/38 | 8/8, 35/36 |
| Ministral 3 3B | 8/9 | 36/38 | 6/8, 35/36 |
| Qwen3-VL 4B | 6/9 | 36/38 | 5/8, 33/36 (never uses the tool) |

Ministral and the 8B look for "the boat" and "make the boat red" and get them right; each look adds about
5 s. No model passes `pixels-over-prompt`: they look, then still edit the latest image (the 9B asks which
one is meant).

## Remaining
- [x] The prompt-vs-pixels case (2026-11-01). The raw replies showed why: the 8B looked at image 1, then
      asked to look at image 1 again. mlx-vlm puts every picture before the text, so the model could not
      tell which picture was "image 1". Now the message names pictures by position ("1st: image 2 … (latest);
      2nd: image 1 …, which you asked to look at"); asking for a picture already shown is answered with
      "What you asked to look at is already among these pictures"; after a look the agent is asked to
      start with `"seen"`, what each picture actually shows. The 8B then edits image 1.

## Results 2026-11-01 (real images 10 cases, eval 44 cases)
| Model | Real | Eval | Before (real / old 38) |
|---|---|---|---|
| Qwen3-VL 8B | 10/10 | 42/44 | 9/10, 37/38 → 37/38 |
| Qwen3.5 9B | 9/10 (ask-boat: adds the teapot as a reference) | 44/44 | 9/10, 38/38 → 38/38 |
| Ministral 3 3B | 10/10 | 41/44 | 9/10, 35/38 → 36/38 |
| Qwen3-VL 4B | 6/10 (asks instead of looking) | 40/44 | 6/10, 36/38 → 35/38 |

Medians 3.7-7.5 s per message; the slowest message 17.8 s (the 9B with an attachment, which adds a
confirmation round, task 09). Ministral's earlier misses were JSON with `// comments` after an invented
`"model"`; the reply parser now drops line comments outside strings (`local_vlm._without_line_comments`).
