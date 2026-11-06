# 17 · Structured JSON prompts (FIBO, Ideogram 4)

**Status:** Open (split out of task 15, 2026-11-06).

## Why
Two families in mflux are trained on structured JSON captions rather than plain sentences:

- **FIBO** (Bria): JSON captions of up to 1,000+ words (lighting, camera, composition, depth of field).
  mflux 0.24 can expand a short prompt with FIBO-VLM (`--expand-prompt`), but that is one more model to
  download and run before each image.
- **Ideogram 4**: plain text works but clearly underperforms. Its JSON schema covers text placement
  (bounding boxes), key order and colour palettes. mflux does not include Ideogram's prompt expander
  (Magic Prompt is a hosted API).

dMLX's prompt assistant writes plain sentences today, so both models would run below their quality.

## Idea
The local prompt assistant writes the JSON when the chosen model wants it. The model's spec says so
(for example `prompt_format: fibo-json` / `ideogram-json`) and carries the schema and a short example; the
agent's instructions use them only for that model. The engine checks the JSON before the model loads and
falls back to plain text (with a note) if it does not parse.

## To do
- [ ] Read both schemas (FIBO's caption format, Ideogram's prompting guide); pick the fields a local model
      can fill reliably.
- [ ] `prompt_format` in the spec; the agent prompt adds the schema only when that model is picked.
- [ ] Check the local planner models (Ministral, the 9B) produce valid JSON for these schemas; compare
      against FIBO-VLM's expansion for FIBO.
- [ ] Edits: FIBO Edit takes a JSON edit instruction; keep the source image's caption between turns.
- [ ] Add FIBO and Ideogram 4 as families (task 15 adapter; Ideogram 4 has two transformers and a
      non-commercial license).

## Done when
"a poster for a jazz night on Friday" with Ideogram 4 or FIBO gives a JSON prompt written by the local
agent, and the image is clearly better than with the plain sentence.
