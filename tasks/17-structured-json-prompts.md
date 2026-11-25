# 17 · Structured JSON prompts (FIBO, Ideogram 4, MOSS Image)

**Status:** Done (2026-11-09): engine and app for MOSS Image (2026-11-06), FIBO 2 (2026-11-07) and Ideogram 4,
whose redraw after a moved box passes in the app. Every bug and performance item below is ticked, and so are
the fixes from the 7 test rounds (the last found no code bugs). "make it warmer" keeping the boxes moved to [task 18](18-layout-features.md)
with the other new Layout features. Design agreed 2026-11-06; demo:
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
   with a Show boxes / Hide boxes button next to Variations. Dashed = text, solid = picture, blue = placed
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
  posters and text (exact boxes); MOSS Image (exact boxes too, but about 5 minutes at 1024 px); FIBO
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

Done 2026-11-08:
- [x] The guard stopped every in-app run with the agent on 2026-11-07 (FIBO 2 twice, MOSS once, Ideogram
      4's redraw twice): "apps were being read back from swap", with pressure normal and swap shrinking.
      The 8B agent pushes idle apps to swap, and they read back in during the image job. Swap-in above
      64 MB/s now stops a job only with a second sign of thrashing: pages pushed out at 32 MB/s or more,
      pressure above the job's start, or swap grown 0.5 GB since it (`JobGuard._thrashing`); critical
      pressure and 4 GB of new swap still stop it alone. The report says `swapin_while_recovering` when
      the old rule would have stopped the job. Then the in-app flow on main passed on MOSS (2026-11-08,
      8 GB in swap): the agent's boxes, the Layout tool, the date moved to the bottom right, the redraw.
- [x] `compile_prompt` (✨ preview) shows the plain rewrite: not a bug. The user never sees JSON, and an
      accepted suggestion still gets its caption at the job (`Engine._rewrite`). The preview can't show
      the boxes the agent will add; Show boxes shows them on the result.

New features for Layout (moving a box in words, edits that keep the caption, FIBO Edit and FIBO 1,
FIBO-VLM, sharing `LayoutCanvas` with task 11's Mark area) moved to
[task 18](18-layout-features.md). What stays here is bug fixes and performance.

### Bugs
- [x] The app crashes (AppKit "more Update Constraints in Window passes than there are views", from the
      chat's LazyVStack prefetch) when a message is sent in a window nearly as tall as the screen
      (1280×900 on a 949-point screen). On main too, with every model; 1000×700 is fine.
      Already fixed on main by 8839eb4. Cause: the window's minimum height is measured with the composer at
      its narrowest; the old FlowLayout then wrapped one chip per row, a send raised the minimum, and a window
      as tall as the screen could not grow to it. Checked with the harness (scratch DMLX_HOME): the 2fff19d
      build crashed 3 of 3 times on send at 1280 × full screen height (887 pt here) with the same stack, and
      not at 1000×700; that build plus only 8839eb4's FlowLayout and composer `minWidth: 300` did not crash
      (2 of 2); main did not crash (4 sends at 886–887 pt). Not checked at 949 pt: this screen gives 887 with
      the Dock shown. The composer comment now names the crash; the harness gained `hostWindow` and logs minSize.
- [x] After a stop the log says "released 0.00 GB of 4.86 GB; something still holds its weights": check
      whether a stopped job keeps its weights until the engine restarts (on a 16 GB Mac that memory would be
      missing for the next job), or whether the check misreads a stopped job.
      Neither, quite: the weights came back, but late. The engine freed the model inside
      `except JobAborted`, where the exception's traceback still holds the stopped job's frames, and they
      hold the weights. FLUX.2 Klein 4B stopped at step 2 for memory pressure (512 px): "released 0.00 GB
      of 2.03 GB", then 0.01 GB active once the block ended. So nothing was kept until a restart, but for
      that moment the guard's count was wrong. The free now runs after the block: 2.03 of 2.03 GB
      released, active 2.19 → 0.01 GB, no warning. A user's cancel keeps the model warm as before, and a
      later unload frees it all (2.19 → 0.01 GB). `tests/test_stop_frees.py` (fails on the old code).
      Script: `docs/benchmarks/raw/task17-perf/stop_repro.py` (its output was not kept); test round 1 ran it
      again as `stop_check.py` / `stop_check_encode.py`, logs kept: active 3.02 → 0.01 GB mid-denoise,
      3.08 → 0.01 GB during encoding, a cancel keeps 2.04 GB warm and an unload frees it.
- [x] The 8B agent still drew a box for lighting ("glowing spotlights") and let "musicians silhouettes"
      cover the date, against the box rules; only the instructions enforce them today. Enforce in
      `normalize` what can be checked: drop boxes named for light, mood or style, and keep agent boxes
      off text boxes.
      `normalize` now leaves out an agent's object box named for light, mood or style ("stage lights",
      "smoky haze", "art deco style"; "neon sign" and "musicians silhouettes" stay; see test round 1), and
      cuts an agent's object box back to its largest part clear of every text box, or leaves it out when
      under 40% of it is left. The user's boxes are never changed. `tests/test_formats.py` (6 tests,
      including Ideogram's own check on a cut caption). Measured on 3 jazz-poster requests × MOSS and
      Ideogram 4 × with and without the user's date box (12 captions per agent, temperature 0): captions
      breaking a rule before → after: Qwen3-VL 8B 1 → 0 (a "stage lights" box), Ministral 3B 2 → 0
      (saxophonist and pianist over the title, musicians over the date), Qwen3.5 9B 2 → 0 (trumpet player
      and musician silhouettes over the date). All 36 stayed valid. Script and log:
      `docs/benchmarks/raw/task17-perf/box_rules.py`, `box_rules.log` (the rule changed since, see test
      round 1; it gives the same answer on all 28 box names in that log).
- [x] The editor's per-box bar scrolls sideways on narrow sheets; it should fit without scrolling.
      Already fixed on main by 50eeb7b: a narrow sheet puts the box's fields on two rows. Checked with harness
      captures at 520 pt (the sheet's minimum), 680 and 1000 pt, Picture box with "Let the agent write it"
      and Text box: every control shows, no scroll. No code change.
- [x] Not yet checked (Done when): Ideogram 4's redraw after a moved box, in the app; "make it warmer"
      after a layout keeps the boxes and the user's descriptions, on Ideogram 4 and MOSS.
      Checked in the real app (DevHarness, Qwen3-VL 8B, 1024 px, 2026-11-09). Redraw with the date moved to
      the bottom right and described "hand-painted pink neon script": passes on Ideogram 4 (12 steps) and
      on MOSS, in the user's lettering, the other boxes kept. "make it warmer" does not keep the boxes: it
      is an edit, and an edit result carries no caption or layout. With Edit set to FLUX.2 Klein 9B the
      colours turned warmer with every element in place (73–74 s), but on MOSS the user's pink neon date
      became yellow, and on Ideogram 4's near-empty poster Klein painted a face into the background.
      Keeping the boxes and descriptions across a follow-up is design 5, now task 18. Found on the way, not
      fixed here: in Auto the agent picked Qwen Image 2.1 for the edit, which the planner refused (9.5 GB
      needed, 7.6 GB usable with nothing of dMLX's resident) while Klein 4B/9B would fit; and Ideogram 4
      drew a short garbled line into the empty middle of two-text-box posters on two seeds, at 12 and at
      20 steps alike (the caption has no such text). Table and screenshots: `docs/benchmarks/task17-perf.md`.
- [x] Auto picked a model the memory plan then refused: "make it warmer" after the Ideogram 4 poster planned
      the edit with Qwen Image 2.1 (needs 9.5 GB, 7.6 GB usable) while FLUX.2 Klein 4B and 9B would fit.
      The agent was offered every installed model, with only a "free memory" line to go by. Now a filter in
      code: `_model_choices` leaves out a model the plan would refuse now (`MemoryManager.could_fit`, a dry
      run of `plan` that counts other idle models as evictable and the local agent's weights as free, at
      the new image's size) while another one for the workflow fits, and `_fitting_model` swaps a pick, or a default,
      that will not fit for one that does. A model the user picked is never swapped and is still refused;
      when nothing fits the choice is left alone and the refusal says why. `tests/test_auto_fits.py`
      (5 tests; the Qwen-Image case and the default case fail on the old code). Real engine, real
      Ministral 3B agent, all 7 models installed, headroom capped at 7.6 GB (the rest real): offered Klein
      4B and 9B only, the agent picked Klein 4B, the 1024 px edit ran in 44 s, warmer with every element
      in place (Klein added a garbled line of its own under the title). Script and log:
      `docs/benchmarks/raw/task17-perf/auto_edit.py`, `auto_edit.log`.
- [x] Old timings for FIBO 2 and Ideogram 4 (about 8 / 8.5 minutes) were left in the README's layout table,
      the app's model descriptions and the website ("5–9 minutes"). Now about 5.5 / 5 minutes, as the
      catalog notes say, and "5–6 minutes" on the website. A grep of README, web/, docs/, app/ and the engine
      finds the old figures only in history: past task notes, the old benchmark rows, and
      `registry._RETIRED_NOTES` (the old notes it replaces).
- [x] Ideogram 4 draws a short garbled line into empty space on two-text-box posters. Tried one cheap fix:
      the caption's background ends "No other text or lettering anywhere." Same caption, same two seeds,
      12 steps (`docs/benchmarks/raw/task17-perf/stray_text_ab.py`): no help, the line stays in the same
      place on both ("PNESCHIR SIN" → "STTABIRY SCTILASIM" top left on 1975810295; "1GROJJOW GEISIPIL" →
      "HONICK TES TAH" in the middle on 199947752). Not kept: a known limitation of the model. A fuller
      caption (objects, a described background) did not show it in the step runs. Images:
      `docs/benchmarks/raw/task17-perf/ideogram4-no-other-text.jpg`.

### Test round 1 fixes (2026-11-09)
- [x] Auto's fit rule checked dMLX's default, but the app's Auto runs its own default (downloaded models
      only, the other task's model first): with New on a heavy auto-selectable model that can edit, or
      Klein 4B not downloaded, Auto could run a model that won't fit while another would. With the built-in
      models this is rare: Klein 9B and Qwen Image 2.1 are not auto-selectable, so Auto's edit model is
      Klein 4B whenever it is downloaded; it matters for models the user adds. And with one fitting model
      the agent is offered nothing, so nothing picked it. Now the app sends its Auto model per workflow as
      `auto_models` (apart from `selected_models`, the user's picks), and `_fitting_model` checks the model
      the app will run: the agent's pick, else the app's Auto model, else dMLX's default (older apps).
      When that one won't fit and another does, the decision names the one that fits, also when it is the
      only one. The app runs the decision's model on Auto (`model(for:decision:)`). A model the user picked
      is still never swapped. `tests/test_auto_fits.py` (4 new tests; 3 fail on the old code) and
      `RouteBodyTests` in the app. Real engine and agent (Ministral 3B), headroom capped at 3.3 GB so Klein
      4B fits an edit and Klein 9B doesn't, Klein 9B passed as the app's Auto model by hand (`auto_models`;
      the app itself never picks the built-in 9B): the agent was offered nothing, Klein 9B gave way to
      Klein 4B, the edit ran in 14 s (`raw/task17-perf/auto_default_check.py`, `.log`). Test round 3 showed
      it in the app with an auto-selectable copy of Klein 9B set as New: Klein 4B ran, agent on and off. The
      round 1 fix run's "user's own Klein 9B was kept" did not show it (its route was not planned and came back
      "ambiguous"); rerun in test round 2 fixes below.
- [x] The light/mood rule dropped real things: "traffic light", "street light", "manta ray", "painter's
      palette", "wooden beam", "cat shadow". Now a word that never names a thing ("atmosphere", "glow",
      "style", "lighting") counts at the end of any name, and one that can ("light", "ray", "beam",
      "shadow", "palette") only when every word before it is a light or mood word ("warm lights", "stage
      spotlights"). `tests/test_formats.py`: the six fail on the old rule; "glowing spotlights", "warm
      lighting", "moody atmosphere", "neon glow" still go.
- [x] The fit check assumed 1024×1024: the app now sends the new image's size (`size`) and the check uses
      it; an edit's source is scaled to 1 MP at most, so edits still count as 1024×1024. Test in
      `test_auto_fits.py` (fails on the old code).
- [x] MOSS's plain-sentence retry kept the NaN run's exception (`bad = exc`), and through its traceback the
      failed run's frames and arrays, the same leak as the stop. Now only the fact is kept.
      `test_the_plain_sentence_retry_does_not_keep_the_failed_run_alive` (fails on the old code).
- [x] A step count the user set equal to an old default moves to the new one too. Left so (no steps
      setting in the app, only hand-edited specs); a comment in `registry.py` says so.
- [x] Docs: Status here and in tasks/README.md; the stored scripts and logs above; `task17-perf.md`
      ("0.5 s of a 5–6 minute job", images in `raw/task17-perf/`); Ideogram 4 times agree (5.4 / 8.7
      minutes, mean of both seeds) in the doc and `catalog.py`.

### Test round 2 fixes (2026-11-09)
- [x] Without the agent, a send with one possible workflow (empty chat, a photo attached, "make it warmer")
      was decided in the app and never asked the engine, so Auto ran its own model (say a heavy
      auto-selectable model the user added and set as New) though it would not fit and Klein 4B would. Now that send, when its workflow is on Auto, asks
      the engine's route for that workflow only (`candidates`), with `auto_models` and `size`, and runs the
      model it names; if the engine can't be reached or disagrees, the app's own decision runs as before.
      A workflow the user chose a model for skips the call and keeps its model. `auto_models` now covers
      upscale, restore and enhance too. Tests: `test_a_one_workflow_send_without_the_agent_gets_the_swap`,
      app `RouteBodyTests` (2 new; they don't build on the old code). In the app (test rounds 3 and 4, run
      (i), agent off, cap 3.3 GB, New = an auto-selectable copy of Klein 9B): the send got Klein 4B
      (13.2 s in round 4).
- [x] The route hint named the app's Auto model, not the model the engine swapped in. It now names the
      model that runs (`model(for:decision:)`, the same call the send uses), and a one-workflow hint on
      Auto asks the engine too. Test round 3 found it right without the agent and wrong with it (fixed in
      test round 3 fixes below).
- [x] `docs/benchmarks/raw/` was ignored by git, so the files `task17-perf.md` and this task link to would
      not be committed. `.gitignore` now ignores `raw/*` but not `raw/task17-perf/`: `git status
      --ignored` shows only that folder newly visible (26 files with this round's check, 470 KB; no weights),
      and every other `raw/` entry is still ignored (`git check-ignore -v`).
- [x] "The user's own Klein 9B was kept" (test round 1 fixes) rerun properly: real engine, real agent
      (Ministral 3B) planning (`plan`), scratch home, GPU lock, headroom capped at 3.3 GB. Edit set to Klein
      9B: decision `image.edit`, no model named (kept), the job refused with `insufficient_memory`. The same
      message on Auto with Klein 9B passed by hand as the app's Auto model: decision Klein 4B, edit done in 14.6 s.
      `raw/task17-perf/user_pick_check.py`, `.log`; unit test
      `test_a_model_the_user_picked_is_never_swapped_and_still_refused`.
- [x] The mood rule still dropped "solar flare", "hair style" and "sound effects": "flare", "effects" and
      "style" now count only when every word before them is a light or mood word ("lens flare", "light
      effects", "retro style", "art deco style" still go). `tests/test_formats.py` (the three fail on the
      old rule).
- [x] The fit swap could pick FIBO 2, Ideogram 4 or MOSS (all auto-selectable) when Auto's model would not
      fit. A swap now takes only a model suited to Auto: auto-selectable, and without a layout no model
      with a prompt format (those are exactly the three: five minutes and non-commercial, or "only when the
      user asks"). When nothing suited fits, the decision is left alone and the job's refusal says why.
      `tests/test_auto_fits.py` (4 new; 3 fail on the old code, the layout one guards that a layout
      still gets a format model).
- [x] Task 18's Done when wants "make it warmer" to keep the boxes on Ideogram 4 and MOSS with no to-do for
      it: added "Follow-ups on Ideogram 4 and MOSS keep the boxes" under To do.

### Test round 3 fixes (2026-11-09)
- [x] With the agent on, a one-workflow hint never asked the engine and named the Auto model the engine then
      swapped out ("Klein9B AutoTest" named, Klein 4B ran). The hint now asks the route for that workflow
      (no plan, so no agent run) with the agent on too, and when the agent will pick the model on send among
      several, the route lists them (`agent_choices`) and the hint says "the agent picks A or B". The hint
      and the send use one rule (`RouteModel` in DMLXKit). Tests: app `RouteBodyTests` (4 new; they don't
      build on the old code), engine `test_the_hint_route_*` (2). In the app (DevHarness, scratch home,
      round 3's hub, cap 3.3 GB, agent on, New = the auto-selectable 9B copy): hint "Will edit the attached
      image · FLUX.2 Klein 4B", and Klein 4B ran the edit (13.4 s, planner Ministral 3B).
- [x] The hint's route sent no layout, so a layout send's hint could name a model that can't place things.
      It now sends the composer's boxes (as the send will take them), the layout is part of the hint's key,
      and a changed layout refreshes the hint. Test `testTheHintSendsTheComposersLayout`. In the app (test round
      4, run (ii), MOSS added, 3 boxes): the hint went from Klein 4B to MOSS once the boxes were set, agent
      off and on, and MOSS ran (139 / 143 s).
- [x] Docs claimed "the app's Auto model Klein 9B gave way": the built-in Klein 9B and Qwen Image 2.1 are
      not auto-selectable, so Auto's edit model is Klein 4B whenever it is downloaded; the checks passed
      Klein 9B by hand. Corrected above and in `task17-perf.md`.
- [x] An agent's object box around its own text ("neon sign" around "OPEN") was left out as covering the
      text. A box that holds a text box whole and is named for a surface (sign, banner, cover, label, tag, …)
      now keeps it; a box only partly over text, or not named for a surface (a crowd around the date), is
      still cut or left out. `tests/test_formats.py` (6 new fail on the old rule).
- [x] The fit check read free memory now, so while another chat's job ran its working memory counted as
      gone and Auto could swap to a smaller model for no reason (jobs run one at a time, so the routed job
      runs after it). `could_fit` now counts the running job's working memory as coming back, never more
      than the room its plan had less the weights it loaded since, nor more than now plus its planned
      working memory. The planner still checks at run time. `test_the_fit_check_counts_a_running_jobs_*`.
- [x] The refusal read "needs about 3.3 GB but only 3.3 GB can be used": it now shows more decimals when one
      makes them look equal ("3.34 … 3.26"). `test_the_refusal_never_reads_as_needing_what_it_has`.
- [x] Logs in `raw/task17-perf/` had local paths and progress bars: paths now read `<scratch>`, bars
      removed (`grep -rE '/private|/Users/|vincent|claude-501'` and `grep -rE '[0-9]it/s|s/it|[0-9]%\|'`
      over the folder find nothing); `task17-perf.md` lists `user_pick_check`.

### Test round 4 fixes (2026-11-09)
- [x] With the agent on and no layout, the hint listed every fitting model, MOSS and Qwen Image 2.1 too
      (five names). It now names two at most, the app's Auto model first, then models suited to Auto
      ("A, B or another model"). The agent is still offered every downloaded model that can run the
      workflow and fits, Qwen Image 2.1, FLUX.2 Klein 9B and MOSS too: it picks by each model's note
      (MOSS's says only when the user asks), and the user accepted their licenses on download. A first
      fix also filtered the agent's list by the swap's rule (`_suits_auto`), so the agent could never pick
      them; fixed in fix round 6. `_suits_auto` (auto-selectable, no prompt-format model without a layout)
      now limits only the swap, when neither the agent nor the user chose a model that fits. Tests:
      `test_the_agent_is_offered_fitting_experimental_models_without_a_layout`,
      `test_a_swap_never_picks_an_experimental_model_the_agent_did_not_choose`,
      `test_a_model_that_fits_is_still_offered_and_kept` (Qwen Image again), and the hint order in
      `test_the_hint_route_lists_the_models_the_agent_will_pick_from`: all four fail on the first fix;
      `test_with_a_layout_the_agent_is_offered_the_prompt_format_models`, app
      `testTheAgentChoicesHintIsShort`. Rechecked in the app in test round 5 (agent on, Auto, Klein 4B /
      9B, Qwen Image 2.1 and MOSS downloaded): "Will generate · the agent picks FLUX.2 Klein 4B, FLUX.2
      Klein 9B or another model", and Klein 4B ran.
- [x] Keeping a surface box ("neon sign" around "OPEN") depended on text-box order: cut off a second text
      first, the sign could end half over OPEN. The cut is now decided against all text boxes at once: the
      largest part that misses every other text and holds each of its own whole or misses it.
      `test_a_surface_cut_off_a_second_text_never_ends_half_over_its_own` (both orders; OPEN-first fails on
      the old code), `test_a_surface_cut_off_another_text_keeps_its_own_when_that_part_is_largest`.
- [x] While the engine checked a one-workflow hint, the hint named the app's Auto model, which the engine
      could then swap out. Until the answer arrives it now names no model ("Will edit the attached image").
      App `testTheHintNamesNoAutoModelUntilTheEngineChecksIt` (doesn't build on the old code); the moment is
      too short to catch in the harness.
- [x] Docs: test round 2 and 3 items above cite the app runs; "That run's" names the run; the log check
      gives the exact greps.

### Test round 5 fixes (2026-11-09)
- [x] On Macs under 24 GB the agent is unloaded right after it plans, and the fit check ran in that moment,
      before macOS hands its memory back: it read about 3 GB too little and could swap a fitting pick
      (Qwen Image 2.1, offered at 7.50 GB) for Klein 4B. The check now counts the memory still coming back:
      what MLX freed at the unload less what has come back since, never more than the headroom fell since
      just before the agent ran, and only for the 10 s a job's plan waits for it. Not the fall alone: up to
      0.7 GB of it never comes back, while MLX's count matched the return within 0.3 GB, under it. The job's
      plan stays the gate. Tests: `test_an_agent_pick_is_kept_while_the_unloaded_agents_memory_comes_back`
      (fails on the old code: swapped to Klein 4B), `test_the_returning_agent_memory_is_bounded_and_expires`,
      `test_without_the_credit_a_low_reading_still_swaps`, `test_on_24_gb_the_resident_agent_is_counted_as_before`.
      Real engine, real Ministral 3B, total forced to 16 GB (GPU lock, scratch home and hub; the reserve cut
      0.6 GB so free memory matched round 5's 7.50 GB): MLX freed 2.75 GB each time. Run 1: 4.00 GB just
      after planning + 2.75 = 6.75, so an agent pick of Qwen Image 2.1 (needs 6.49) is kept, and memory
      settled at 6.72-6.80 GB (without the credit: swapped). Run 2: 3.60 + 2.75 = 6.35, swapped to Klein 4B;
      it settled at 6.40-6.44 and the job's own plan refused Qwen too ("needs about 6.5 GB but only 6.4").
- [x] The composer's hint sends the image size for the fit check, but a size change did not refresh it. The
      size is now part of the hint's key and changing Square / Landscape / Portrait asks again. `swift build`,
      `swift test`. Checked in the app in test round 6 (cap 2.65 GB, agent on): Square "· FLUX.2 Klein 4B",
      Landscape / Portrait "the agent picks FLUX.2 Klein 4B, Klein9B AutoTest or another model", back to
      Square "· FLUX.2 Klein 4B".

### Test round 6 fixes (2026-11-09)
- [x] The credit for the agent's memory still coming back stayed for the whole 10 s, so memory taken after
      it had come back (the image job loading, or another app) still counted. Real run, 16 GB: during a
      Klein 4B job straight after planning, Qwen Image 2.1 fitted the hint at 6.6 GB against a real 6.31
      (the job's plan still refused it). The credit now ends once it reaches 0 (the memory is back) and when
      a job starts. Tests: `test_the_credit_ends_once_the_agents_memory_is_back`,
      `test_the_credit_ends_when_a_job_starts` (both fail on the old code: 3 GB still counted).
- [x] `test_the_returning_agent_memory_is_bounded_and_expires` read the freed bytes as the deadline, so the
      expiry check passed whatever the window. It now uses the deadline and checks both sides: still counted
      0.1 s before it, gone 0.1 s after. Fails with the expiry turned off, and with it 1 s early.

### Performance
- [x] The caption step runs before the model loads and adds 7–37 s to every new prompt (cached for
      Variations and repeats). Measure it per agent in the app, and check whether loading the image model
      can overlap it where memory allows.
      Measured with the engine's own calls (`structure()` + `normalize`, jazz poster, M5 24 GB): Qwen3-VL 8B
      7 s (MOSS), 15 s (FIBO), 12 s (Ideogram), 14 s (Ideogram with 3 user boxes); Ministral 3B 6, 12, 9,
      11 s; Qwen3.5 9B 11, 20, 15, 18 s; loading the agent 3–4 s more. Every caption valid on the first
      try. No overlap: the image model's loads add up to about 0.5 s a job (FIBO 2 0.55 s, Ideogram 4
      0.44 s, both streamed, from the job report), so there is almost nothing to hide, and overlapping would
      put the agent and image parts in memory together (never safe on 16 GB), before the plan can know
      whether the encoder is needed. Not implemented; reasons in `docs/benchmarks/task17-perf.md`.
- [x] Steps: MOSS went from 50 to 30 steps (13–14 to about 5 minutes at 1024 px) with placement unchanged.
      Do the same check for FIBO 2 (30 steps, about 8 minutes) and Ideogram 4 (20 steps, about
      8.5 minutes; the 12-step preset): fewest steps that keep placement and clean lettering.
      FIBO 2 now 20 steps (about 5.5 minutes) and Ideogram 4 12 steps, its turbo preset (about 5 minutes).
      The jazz poster through the engine with every box the user's and the date moved to the bottom right,
      1024 px, seeds 1857933066 and 42: FIBO 2 at 20 kept every box and clean lettering like 30 (333 / 348 s
      against 497 / 504 s; 16 steps also placed everything but drew the saxophone less cleanly); Ideogram
      4 at 12 gave nearly the same image as 20 (315 / 329 s against 532 / 514 s). Saved specs at the old
      steps and speed note move to the new ones, a user's own step count stays (`_RETIRED_DEFAULTS`).
      `tests/test_step_defaults.py`; images in `docs/benchmarks/raw/task17-perf/`.

## Risks
- "FIBO 2.0 dev" is not in mflux 0.23: the configs are `fibo`, `fibo-lite`, `fibo-edit`, `fibo-edit-rmbg`.
- mflux 0.23 supports only `ideogram-ai/ideogram-4-fp8`; sizes must fall in a set range and step.
- 1,000-word FIBO captions won't come from a local planner. Expect shorter captions and measure the cost.
- MOSS layouts are marked experimental by OpenMOSS, and MOSS is slow (about 5 minutes at 1024 px at 30 steps on the
  dev Mac), so it suits a layout the user drew more than Auto.

## Done when
"a poster for a jazz night on Friday" with Ideogram 4, and a scene such as "a red fox in a misty pine
forest at dawn" with FIBO (it is weaker at text in images), each get a JSON prompt written by the local
agent; the image is clearly better than with the plain sentence; and the user saw only plain words.
Layout: on Ideogram 4, moving "FRIDAY 9 PM" to the bottom right in Crop / Draw → Layout and redrawing puts
it there. The same layout steps work on MOSS Image. ("make it warmer" afterwards keeping the box and the
user's description was here too; it needs edits that keep the caption, so it moved to task 18's Done when.)
