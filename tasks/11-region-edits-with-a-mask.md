# 11 · Region edits: a mask brush the agent can use

**Status:** Open (from the competitor comparison, 2026-11-02).

## Why
Every serious competitor has masks / inpainting (ComfyUI, Draw Things, InvokeAI); it is the precision
feature users miss most. A mask fits the chat: paint an area, then "make this red" or "remove this".

## Today
- Crop / Draw (`app/Sources/dMLX/Editor/EditorView.swift`) can crop and paint, and attaches the painted copy
  as a new image; the paint is pixels, not a mask, and the agent cannot tell what was marked.
- No image model path takes a mask.

## Options
- **Mask as an attachment role:** Crop / Draw gets a "Mark area" tool that exports a mask next to the image;
  the route state carries it, and the agent sees the image with the marked area (outlined) and decides
  whether the message is about that area. A mask is a fact passed to the agent, not a rule.
- **How the edit uses it:** a model with native inpainting if one of the families supports it in mflux;
  otherwise edit the whole image and composite only inside the mask (feathered), which works with every
  edit model.

## Done when
"Mark the hat, say 'make this blue'" changes only the hat, in the app, on Klein; eval cases for a marked
area pass on the 8B and Ministral.
