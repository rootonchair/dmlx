# 06 · Image labels are misread

**Status:** Done 2026-10-25

## Problem
`agent._resolve_id` (`engine/src/dmlx/prompting/agent.py`) reads `image2` as the second image *shown* to
the agent (`<image2>`), but `image 2` as the chat's image number 2. The model sees both kinds of label:
`<imageN>` for the images in view and `image N id=…` in the chat. With one image shown, `image2` fails
the checks and the plan falls back to the router: safe, but it wastes a 6–12 s planning call.

## Options
- Give the model one kind of label: list ids only (`id=img_…`) and drop `<imageN>` from the view line,
  or name shown images by their chat number.
- When a label does not resolve as a shown image, try the chat number before rejecting.

## Done when
- `image2`, `<image2>`, `image 2` and `2` all resolve the same way, or the model is never shown two
  numbering schemes.
- Unit tests cover each form, with one and with several images shown.

## What changed
- The model sees one numbering: the images in view are named as the chat names them
  (`Images shown to you, in order: image 2 id=img_…`), attachments as `attachment 1 id=up_…`. No `<imageN>`.
- `agent._resolve_id`: `image2`, `<image2>`, `image 2`, `2`, `#2` and `img 2` all mean the chat's image 2;
  only when the chat has no image 2 do they mean the second image shown. `attachment N` is the N-th
  attachment. Ids wrapped in `<>` or quotes are unwrapped.
- Tests: every form with one image shown, with two attachments shown, a number the chat lacks, and the
  view line.
