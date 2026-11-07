# 17 · Structured JSON prompts (FIBO, Ideogram 4, MOSS Image)

**Status:** In progress: engine and app done for MOSS Image (2026-11-06) and FIBO 2 (2026-11-07);
Ideogram 4 runs and follows a layout; its redraw is not checked yet (the memory guard, see Still open). Design agreed 2026-11-06; demo:
https://claude.ai/artifact/1VnS2KwtZ1KrusrRGF7Co8 (hand-written mockup, no model runs).

## Why
Three families in mflux read structured JSON captions. Two are trained on them rather than on plain
sentences; the third takes them for placement:

- **FIBO** (Bria): JSON captions of up to 1,000+ words (lighting, camera, composition, depth of field).
  mflux can expand a short prompt with FIBO-VLM (`--expand-prompt`), but that is one more model to
  download and run before each image.
- **Ideogram 4**: plain text works but clearly underperforms. Its JSON schema covers text placement
  (bounding boxes), key order and colour palettes. mflux does not include Ideogram's prompt expander
  (Magic Prompt is a hosted API).
- **MOSS Image** (OpenMOSS, Apache 2.0, already in dMLX as `moss-image`): plain sentences and tags work
  well; a JSON layout adds exact placement of text and objects. Its model card names "user-friendly layout
  tools" as a use. It is the only one of the three dMLX runs today, so layouts can ship on it first.

dMLX's prompt assistant writes plain sentences today, so FIBO and Ideogram 4 would run below their quality,
and no model can be told where things go.

## Rule: the user never sees or writes JSON
Our users don't know how to write a JSON prompt, and they shouldn't have to.

- The user types a normal sentence, as for every other model.
- The chat shows the plain description of what was drawn, the same view for every model.
- Edit on the user's message (`UserBubble`, "Edit Prompt…") changes only their words; the agent rewrites
  the JSON from them.
- If the JSON can't be written, the note is in plain words ("FIBO works best with a detailed description,
  and one couldn't be written this time. The result may be weaker."). It never mentions JSON.
- The JSON is saved with the result (`parameters.rewritten_prompt`) so edits and variations reuse it, and
  is visible only in the developer view (DevHarness) and the API.

## Layout: bounding boxes in the current UI
Users who want to place things themselves get boxes, never JSON. Everything lives in views we already have.

**How ComfyUI does it (researched 2026-11-06).** Core ComfyUI has *Create Bounding Boxes* (draw, select,
resize and label regions over an optional background image; each box has type, text, description and a
palette) and *Build JSON Prompt (Ideogram)* (PR #14537, merged 2026-06-25). KJNodes' Ideogram 4 Prompt
Builder adds: the last image as the canvas background ("Grab BG"), import of an existing caption (captions
go both ways), a token counter that turns red at 2048, and per-box inputs an LLM can fill. For FIBO there is
no box editor: the Bria nodes return the JSON with the image (same JSON + seed rebuilds it), and the only
spatial controls are aspect ratio and a mask. Their advice: start with plain words; add boxes only where
placement matters; boxes are not pixel-perfect.

**Where boxes appear in dMLX:**
1. **Over the image in the chat** (`ImageTurnView`): the boxes from the caption are drawn on the result,
   with a Show boxes / Hide boxes button next to Variations. Dashed = text, solid = picture, amber = placed
   by the user.
2. **After an image: Crop / Draw → Layout.** A fourth tool in `EditorView` next to Crop, Paint and Erase.
   The caption's boxes open over the image. Drag to move, pull the corner to resize, drag on empty space to
   add. The primary button is "Redraw with layout": a new version on the same turn, same seed.
3. **Before an image: ▭ Layout in the composer** (`ComposerView`). Opens the same sheet on a blank canvas
   at the chosen aspect. "Use layout" puts a chip in the composer ("Layout · 3 boxes", Change, ✕), like an
   attachment. On send, the boxes go with the message; the agent writes the rest. The button shows only
   when the chosen model has a `prompt_format`; on a blank canvas Crop, Paint and Erase are greyed out.

**Layout in Auto.** The ▭ Layout button also shows in Auto. A layout narrows the models the agent may pick;
it is a filter, not a hint in the prompt, because the local planner can ignore hints:
- `_model_choices()` (`engine.py`) keeps only downloaded models with a `prompt_format` when the request
  carries a layout. The agent picks among them as today, in this order of preference: Ideogram 4 for
  posters and text (exact boxes); MOSS Image (exact boxes too, but about 13 minutes at 1024 px); FIBO
  only when rough placement is enough.
- If exactly one such model remains, it is selected directly (today `_model_choices()` drops a workflow
  with a single option, so this needs its own case).
- If none is downloaded, the button is greyed out in Auto, with the tooltip "Download Ideogram 4 or FIBO to
  place things yourself."
- `check()` rejects a plan that pairs a layout with a model without a format, so the router fallback can't
  drop the boxes silently.

**Per box** (bottom bar of the sheet, where Crop shows its presets): Picture | Text; a name, or the exact
text for a text box; a description in plain words ("what it looks like", or the lettering style); "Let the
agent write it"; Delete. Double-click a box to jump to its description.

**Rules:**
- The user owns geometry, names, exact text, and any description they write. The agent owns everything
  they left empty (descriptions, background, style).
- Boxes and descriptions the user set are **locked**: when the agent rewrites the caption on a later turn
  ("make it warmer"), it keeps them word for word and in place. Renaming a box clears only a description the
  agent wrote.
- Ideogram 4 gets each box as `bbox: [y_min, x_min, y_max, x_max]` in 0–1000 and the description as `desc`.
  MOSS Image gets the same order in pixels from the image centre, computed for the width and height the
  image is made at (stored boxes stay 0–1000, so a change of aspect just recomputes them), and the
  description as the item's `caption`.
  FIBO gets a position word on a 3 × 3 grid ("top center") plus a size word, and the description as the
  object's `description`; the editor shows that word on each box, because placement is rough.
- Models without `prompt_format` (Qwen, Klein…) have no Layout tool. Exact changes on FIBO go through a
  mask on an edit (task 11: FIBO Edit takes `mask_path`).
- The check counts tokens and refuses over 2048 for FIBO and Ideogram 4 (both read at most 2048). MOSS
  states no limit.

**Data:** add `layout: [{id, kind, label, desc, desc_by_user, box, by_user}]` to the result meta next to
`rewritten_prompt`, so the editor, the overlay and later turns read boxes without parsing the caption. The
composer sends a pre-drawn layout with the route request; `structure()` gets the locked boxes and fills only
the rest.

## What mflux expects (checked in mflux 0.23.0 and 0.24.0)
- **FIBO** (`fibo/model/fibo_text_encoder/prompt_encoder.py:19`): the prompt is a `str` that must parse as
  JSON; plain text raises. No schema check; up to 2048 tokens read. Keys: `short_description`,
  `objects[]` (≤5: `description`, `location`, `relative_size`, `shape_and_color`, `texture`,
  `appearance_details`, `relationship`, `orientation`; people add `pose`, `expression`, `clothing`,
  `action`, `gender`, `skin_tone_and_texture`), `background_setting`,
  `lighting{conditions,direction,shadows}`, `aesthetics{composition,color_scheme,mood_atmosphere}`,
  `photographic_characteristics{depth_of_field,focus,camera_angle,lens_focal_length}`, `style_medium`,
  `artistic_style`, `context`, `text_render[]{text,location,size,color,font,appearance_details}`.
- **FIBO Edit** (`variants/edit/util.py:80`): the same JSON plus a required non-empty `edit_instruction`;
  `{"edit_instruction": "..."}` alone is accepted. Takes one `image_path` and an optional `mask_path`.
  A separate checkpoint (`briaai/Fibo-Edit`), not a mode of FIBO.
- **Ideogram 4** (`ideogram4/variants/txt2img/ideogram4.py:50`, `caption.py:25`): prompt is `str | dict`;
  plain text and bad JSON are accepted with a warning. Root: `high_level_description`,
  `style_description`, `compositional_deconstruction`. `style_description` keys in this order:
  `aesthetics, lighting, photo_style|art_style (exactly one), medium, color_palette?` (≤16 upper-case
  `#RRGGBB`). `compositional_deconstruction`: `background`, then `elements[]` with `type` (`obj`|`text`),
  optional `bbox` [y_min,x_min,y_max,x_max] in 0–1000, `text`, `desc`, optional `color_palette` (≤5).
  No image input, no negative prompt.
- **MOSS Image** (`moss_image/README.md`, `moss_image_prompt_encoder.py`; format from OpenMOSS's
  `USERGUIDE.md`, "Layout Prompts (Experimental)", and the `MOSS-Image-SFT` model card): the prompt is a
  `str` passed straight to the Qwen3.5 chat template; mflux neither parses nor checks it, so bad JSON gives
  no warning. Plain sentences, tags and JSON layouts all work, and they can be mixed. Layout: `summary`
  (`meta` such as `["poster"]`, `canvas: "single"`, `caption` for the whole scene and style), `items[]`
  (`label`: `text`|`object`, `caption`, `text` for the exact words, `bbox`), `bbox_format: "yxyx"`,
  `bbox_coord_space: "abs_centralized"` (integer pixels, origin at the centre: x in −W/2…W/2, y in
  −H/2…H/2; the layout must match the generation size). Takes a negative prompt; no image input.
  OpenMOSS's own prompt enhancer writes a layout only for placed text, posters, panels or object
  arrangements, and a plain caption otherwise. dMLX ships the SFT checkpoint; the guide was written for
  Base, and the SFT card says both share the pipeline.

## Design
1. **Spec.** `prompt_format: fibo-json | ideogram-json | moss-layout | None` on `ModelSpec` (`registry.py`) and in the
   catalog family entries (`catalog.py`, copied in `to_spec_data`). Schema text, a short example, the check
   and the fallback live in code, in a new `prompting/formats.py`.
2. **A second, short agent step, in `Engine._rewrite`.** The main plan stays plain (in Auto the agent
   picks the model and writes the prompt in one reply, so it can't know the format in advance). The step
   runs on the worker once the model is certain: `assistant.structure(plain, fmt, layout)` with the schema,
   `max_tokens` 1200 (400 for other replies). Only these models pay for it, and the caption is cached so
   Variations and repeats skip it. (Planned for `agent.plan` at first; moved because the model the app
   runs is only certain at the job, and every path, chat, API, Edit Prompt and router fallback, reaches it.)
3. **Fill only what a small model does well.** FIBO: `short_description`, 1–3 `objects`,
   `background_setting`, `lighting`, `aesthetics`, `style_medium`, `text_render` with the user's exact
   words. Ideogram 4: `high_level_description`, `style_description`, `background`, `elements` with a coarse
   `bbox` (top band, centre, bottom band). MOSS Image: a layout only when the request has text, a poster
   or panels, or the user drew boxes; otherwise the plain sentence, as OpenMOSS's enhancer does. The
   agent writes every box as `[x_min, y_min, x_max, y_max]` in 0–1000: in the other order the Qwen-family
   planners swapped the axes (a title came back as a tall strip). The engine converts to the stored
   `[y_min, x_min, y_max, x_max]`, then to each format (centred pixels for MOSS), drops boxes that cover
   the whole image, matches a user box the agent copied without its id, fixes key order and hex case and
   drops unknown keys.
4. **Check and fallback in `Engine._rewrite`,** before `_plan_memory`, so every path (chat, API, Edit
   Prompt, router fallback) goes through it. On failure, call `structure()` once more; then fall back:
   Ideogram 4 and MOSS Image run on the plain sentence (for MOSS with no note, since it is trained on
   plain prompts too, unless the user drew boxes: "Your layout couldn't be used this time; the image
   follows your words."); FIBO gets `{"short_description": text}`; FIBO Edit gets
   `{"edit_instruction": text}`. Without a usable caption the user's boxes still go in, built from the
   plain sentence and the boxes (MOSS, Ideogram 4). Reported in `report["structure"]`; its `note` is the
   plain-words line the chat shows under the image.
5. **Edits keep the caption.** Give `agent.plan` a `caption_of(rid)` that reads `results.meta(rid)`. When
   the source was made by a model with the same format, `structure()` starts from its caption, changes
   only what was asked and adds `edit_instruction`. Uploads and images from other models have no caption,
   so the agent writes one from the image.

## To do
Done 2026-11-06 (engine 212 tests, app 25 tests):
- [x] `prompt_format` on `ModelSpec`, in `catalog.py` (MOSS family) and defaulted from the adapter for specs
      saved earlier; listed in `/v1/models`.
- [x] `prompting/formats.py`: schema text for the three formats, check and fix (`normalize`), the user's
      boxes kept exactly (`_merge_locked`), FIBO place and size words, MOSS centred pixels, token estimate
      (2048 for FIBO and Ideogram 4), fallbacks and plain-words notes. `tests/test_formats.py`.
- [x] `local_vlm.py`: `structure()`, `chat(max_tokens=)`.
- [x] `engine.py`: `_rewrite` → `_structure` with one retry, cache, fallback, `layout` and `plain_prompt` in
      the result meta; `layout` parameter checked at submit (`layout_unsupported`, `invalid_layout`); Auto
      with a layout filters `_model_choices()`, keeps a single choice, and `_layout_model` names the model
      (Ideogram 4, then MOSS, then FIBO) when the agent didn't.
- [x] App: `LayoutBox` (`DMLXKit/Layout.swift`); the Layout tool in `EditorView` (`Editor/LayoutTool.swift`:
      draw, move, resize, per-box bar, double-click to describe, undo), Redraw with layout (new version, same
      seed), ▭ Layout on a blank canvas from the composer with the chip; the overlay with Show / Hide boxes;
      the fallback note under the image; DevHarness layout steps.
- [x] MOSS Image first. Real check, 512 px, same seed: with three boxes the title went across the top, the
      described saxophonist into her box and the date to the bottom right (the plain sentence put the title
      down the right side). Agents: Qwen3-VL 8B, Ministral 3B and Qwen3.5 9B wrote valid captions for all
      three formats on the first try (7–37 s).

Done 2026-11-07 (engine 221 tests), after the first in-app run gave a black poster at 1024 px:
- [x] Cause: mflux's Gated DeltaNet (MOSS Image's Qwen3.5 text encoder) inverts (I - A) per 64-token chunk
      by repeated squaring. Against the token-by-token recurrence in float64 it is right only on very short
      prompts (27 tokens: 0.002 off; 56: 0.3; 148: 79 on outputs of about 2.7) and from about 250 tokens it
      overflows to NaN in layer 25: every embedding NaN, a black image. Plain prose did the same, so every
      long MOSS prompt was affected, not only layouts; layouts are just long. `adapters/mflux_fixes.py`
      swaps in the reference's row-by-row solve (0.002 off at every length, +0.1 s per encode), installed
      with the other mflux fixes. The black prompt then rendered with the layout followed.
      `tests/test_mflux_fixes.py`. To report upstream to mflux.
- [x] Insurance: the denoise loop stops at the first non-finite step (`NonFinite`), and a flat image is
      never handed back. On an agent caption the job draws again from the plain sentence with a note; on a
      plain prompt it is an `invalid_output` error.
- [x] The agent's boxes: no invented facts (it wrote "Friday, June 14" for "Friday"), no boxes for a mood,
      lighting or style, text boxes clear of each other and the subject, a short `name` per box. The engine
      also drops an agent box covering a third of the image that wraps two others.
- [x] Quoted words must come back as text items: `quoted_words()` (straight, curly and single quotes; not
      apostrophes) from the user's message and the plan, listed in the request, checked in `normalize`
      (retry, then the plain sentence). The planner turned "JAZZ NIGHT" into 'JAZZ NIGHT' and the caption
      dropped both phrases. Ministral 3B wrote `"label": "title"` for text: words now make an item text.
      All three local agents: both phrases on the first try, 3/3 each.
- [x] Redraw with layout: every box on the canvas counts as the user's (the redraw sent only the moved box
      and the agent dropped the title); moving a box clears a description the agent wrote, and the engine
      strips place words from the agent's descriptions ("below main title" kept the moved date under the
      title); the same seed only when no box moved: with the same seed MOSS kept the old composition, on
      seeds 42 and 7 the date went to its new box.
- [x] MOSS at 30 steps by default (was 50): about 5 minutes at 1024 px on the M5 instead of 13–14; the
      first poster followed its layout closely. 50 steps draws small text more cleanly (the moved date was
      garbled at 30, clean at 50, same seed) but does not change placement. Saved specs at 50 move to 30.

FIBO 2 and Ideogram 4 families, 2026-11-07 (engine 233 tests):
- [x] FIBO 2 (`mflux-community/fibo-2-0-dev-mflux-q6`, 10.3 GB, non-commercial): family `fibo2`, 30 steps
      (mflux's default is 50), guidance 5. Two adapter hooks: `init_config` and `pipe_setup` build its
      Qwen3-VL processor and find its fonts, which mflux's `Fibo2` sets up in a step the generic adapter
      skips. Measured (bench-family, M5 24 GB): encode 0.45 GB, denoise 1.3 GB at 1 MP, decode 1.3 GB in
      either mode; about 16 s a step at 1024 px, so about 8 minutes. The FIBO schema now matches FIBO 2's
      check (`context`; `color` and `font` on each text line), and a text box the user moved keeps the
      agent's font and colour, which FIBO 2 also uses to render the line for its encoder.
- [x] FIBO 2 follows a layout: the jazz poster through the engine (the agent's caption, 1024 px, 30 steps)
      put the title at the top and the date under it; with every box approved and the date moved to the
      bottom right, the same seed redrew the same scene with the date in the bottom right. Its lettering
      is the cleanest of the three so far.
- [x] Ideogram 4 (`mflux-community/ideogram-4-mflux-q4`, 13.9 GB, gated, non-commercial): two transformers in one denoise loop. The planner and the guard now load and free every
      part of a phase together instead of one part named "transformer" (a no-op for every other model);
      a `before_apply` hook runs Ideogram's rebuild of its quantized layers before the weights go in; a
      step count picks the nearest sampler preset (12, 20 or 48 steps, each with its own guidance).
      The q4 pack (`mflux-community/ideogram-4-mflux-q4`, 13.9 GB) keeps the conditional transformer in
      `transformer/`, which mflux's definition and the catalog already map. Its initializer needs the
      model folder (`model_path`), now passed to every family's `init_config`.
- [x] Ideogram 4 streams both transformers (0.15 GB resident each instead of 4.9). Same image as with them
      held whole, and faster (12 steps at 512 px: 62 s against 72 s). Measured streamed: encode 2.2 GB,
      denoise 1.9 GB at 512 px but 4.9 GB at 1024 (attention grows faster than the area: 5.2 per MP in the
      spec, from 1024), decode 1.3 GB. About 25 s a step at 1024 px: 20 steps take about 8.5 minutes.
- [x] Our Ideogram captions failed Ideogram's own check (mflux `Ideogram4CaptionVerifier`): the key is
      `photo`, not `photo_style`; `medium` comes before `art_style` (after `photo`); an element is
      type, bbox, text, desc; the caption built from boxes alone lacked the required style keys. Fixed, and
      `test_ideogram_captions_pass_ideograms_own_check` runs the verifier on ours.
- [x] Ideogram 4 follows a layout, and the most exactly of the three: the jazz poster (1024 px, 20 steps)
      put the title, the date, the saxophone (top right), the piano and the cocktail each in its box,
      with clean lettering.
- [x] Tests no longer read this Mac's real swap: a model loading in another process made fake jobs stop
      with `memory_pressure` (`conftest.py`).

Still open:
- [ ] The app crashes (AppKit "more Update Constraints in Window passes than there are views", from the
      chat's LazyVStack prefetch) when a message is sent in a window nearly as tall as the screen
      (1280×900 on a 949-point screen). On main too; 1000×700 is fine. Not task 17.
- [ ] Edits keep the caption (`caption_of`, design 5): needs FIBO Edit, the only format model that edits.
- [ ] `compile_prompt` (✨ preview) still shows the plain rewrite; the caption is written at the job.
- [ ] Moving a box in words ("move the date to the bottom right") and the agent planning with the boxes in
      view; today the agent sees only the request, and the boxes go to `structure()`.
- [ ] Task 11's Mark area could share `LayoutCanvas`.
- [ ] The editor's per-box bar on narrow sheets (it scrolls sideways today).
- [ ] FIBO Edit and FIBO 1 as families (FIBO 2 is in, below).
- [ ] In the app, FIBO 2 after the 8B agent was stopped twice by the guard ("apps were being read back
      from swap") on a Mac with 9 GB of swap in use and about 6 GB free: the transformer streams from disk
      at every step. The same job without the agent first ran fine. Check on a quieter Mac before calling
      it a FIBO problem. Ideogram 4's redraw with the date moved hit the same guard twice through the
      engine (64.6 MB/s read back against the 64 MB/s cap, with pressure normal and swap shrinking: other
      apps coming back, not dMLX pushing them out); the same size passed twice in bench-family. Still to
      see: the Ideogram redraw, and whether the swap-in check should tell apps coming back from apps
      being pushed out.
- [ ] Compare FIBO against FIBO-VLM's expansion.

## Risks
- "FIBO 2.0 dev" is not in mflux 0.23: the configs are `fibo`, `fibo-lite`, `fibo-edit`, `fibo-edit-rmbg`.
- mflux 0.23 supports only `ideogram-ai/ideogram-4-fp8`; sizes must fall in a set range and step.
- 1,000-word FIBO captions won't come from a local planner. Expect shorter captions and measure the cost.
- MOSS layouts are marked experimental by OpenMOSS, and MOSS is slow (about 13 minutes at 1024 px on the
  dev Mac), so it suits a layout the user drew more than Auto.

## Done when
"a poster for a jazz night on Friday" with Ideogram 4, and a scene such as "a red fox in a misty pine
forest at dawn" with FIBO (it is weaker at text in images), each get a JSON prompt written by the local
agent; the image is clearly better than with the plain sentence; and the user saw only plain words.
Layout: on Ideogram 4, moving "FRIDAY 9 PM" to the bottom right in Crop / Draw → Layout and redrawing puts
it there; then "make it warmer" changes the colours and keeps the box and any description the user wrote.
The same layout steps work on MOSS Image.
