# 05 · Text to draw in double quotes

**Status:** Mostly done in the working tree, not committed.

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
- [ ] Re-run the NIGHT BLADE edit end to end and check that the title is large and at the top center.
