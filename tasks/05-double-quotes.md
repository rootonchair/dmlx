# 05 · Text to draw in double quotes

**Status:** Open: the double-quote premise did not hold up in an end-to-end test.

## Problem
The agent answers in JSON, where double quotes must be escaped, and the instructions' own examples use
single quotes. It wrote `'NIGHT BLADE'`, and that edit drew the title tiny in a corner. Upstream mflux's
rewrite prompt (`mflux/models/qwen21/.../grounding.py`) says text to draw goes "in double quotes, verbatim".

## What changed
- `agent.INSTRUCTIONS` and `local_vlm.EDIT_INSTRUCTIONS` say: words to appear go in DOUBLE quotes
  (written `\"` inside the JSON string), never single quotes.
- New-image prompts: `agent._double_quote_user_text` switches single-quoted words the user typed to
  double quotes.
- Eval: `text-back` now gives `"NIGHT BLADE"`; `gen-misleading-keyword` gives `"MAKE IT COUNT"`.

## Remaining
- [ ] Edit descriptions still quote replaced text in single quotes (`text-sign`: "currently reading
      'DMLX'"). The rules say never to quote removed text; check whether the editor draws it.
- [x] Re-ran the NIGHT BLADE edit end to end (2026-10-25, Qwen Image 2.1 q4, seed 1206497696, 219 s).
      With the real card image, the agent's description still wrote `'NIGHT BLADE'` in single quotes,
      yet the title came out large, legible and gold at the top center (`img_20261025_415786bceea7`).
- [x] A/B with the same seed and the same prompt, only the quotes changed to double: **no title was drawn
      at all** (`img_20261025_4357a137ce04`).
- [ ] One seed is not evidence either way. Before enforcing double quotes (e.g. a `text` field the agent
      fills and the engine quotes), run a few seeds × {single, double, no quotes} on two or three text edits.
- [ ] 2026-10-25: tried 3 seeds × {single, double, none} on NIGHT BLADE and a boat name ("SEA WIND"),
      but all 18 edits were refused (Qwen Image 2.1 needs about 9.5 GB; about 5 GB was usable with
      everyday apps open). Re-run `quote_ab.py` with apps closed.
      2026-10-30: refused again (7.9 GB usable). The agent's quote rewrite (`_double_quote_user_text`) was
      removed meanwhile (task 09); the instructions still ask for double quotes.
