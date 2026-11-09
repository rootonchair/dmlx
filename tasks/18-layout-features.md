# 18 · Layout: next features

**Status:** Open (moved from task 17 on 2026-11-09; task 17 keeps the bug fixes and performance work).

## Why
Layout (task 17) works on MOSS Image, FIBO 2 and Ideogram 4: the agent writes the caption, the user drags
boxes in Crop / Draw → Layout and redraws. These are the features it does not have yet.

## To do
- [ ] **Moving a box in words** ("move the date to the bottom right"). Today the agent sees only the
      request, and the boxes go to `structure()` later; none of the three models edits, so the message makes
      a new image and the rest of the poster can change. The agent should plan with the boxes in view,
      change one box, and redraw from the old caption, as Redraw with layout does. A cheaper first step:
      answer such a message with where to drag the box.
- [ ] **Edits keep the caption** (`caption_of`, task 17 design 5): when the source was made by a model with
      the same format, `structure()` starts from its caption, changes only what was asked and adds
      `edit_instruction`. Needs FIBO Edit, the only format model that edits.
- [ ] **Follow-ups on Ideogram 4 and MOSS keep the boxes**: an edit or follow-up ("make it warmer") after a
      Layout poster carries the source's caption and layout, the user's boxes and descriptions included.
      Neither model edits, so the follow-up redraws from the source's caption with only the asked change
      (as Redraw with layout does), instead of handing the image to an edit model such as FLUX.2 Klein,
      which today keeps the elements but drops the boxes and changed MOSS's pink neon date to yellow.
- [ ] **FIBO Edit and FIBO 1 as families** (FIBO 2 is in). FIBO Edit (`briaai/Fibo-Edit`) takes an image
      and an optional `mask_path`, which also gives task 11 an exact-edit path.
- [ ] **Compare FIBO against FIBO-VLM's expansion** (mflux `--expand-prompt`): is the local agent's caption
      close enough, or worth one more model download?
- [ ] **Share `LayoutCanvas` with task 11's Mark area**: the same draw, move and resize code for a mask.

## Done when
"Move the date to the bottom right" typed after a Layout poster moves only the date, on Ideogram 4 and
MOSS Image; an edit with FIBO Edit keeps the source's caption and changes only what was asked. After a
Layout redraw on Ideogram 4 and on MOSS, "make it warmer" changes the colours and keeps the boxes and any
description the user wrote (moved from task 17's Done when; today an edit result carries no caption or
layout, see task 17's in-app check).
