# Workflow system implementation plan

Status: in progress  
Date: 2026-10-20  

Implemented in the initial slice:

- Versioned Python and Swift workflow types, capability declarations, and the
  `/v1/workflows`, `/v1/workflows/route`, and `/v1/capabilities` endpoints.
- Backward-compatible translation of the existing image endpoints.
- One ordered attachment path in the app, including drag-to-reorder.
- Capability-bounded semantic routing through a configurable
  OpenAI-compatible provider, confidence gating, stale-response rejection,
  manual override, a low-confidence chooser, and bounded chronological thread
  history for resolving references and follow-up intent.
- Routing settings with Keychain-backed credentials.
- Deterministic `Negative:` and `--negative` parsing with capability rejection
  when the selected executor cannot apply negative conditioning.
- A non-destructive crop/paint/erase editor with normalized coordinates,
  undo/redo, original comparison, persisted documents, and composite export
  through the ordered attachment workflow.
- Independent OpenAI-compatible prompt enhancement with Ask/Auto modes,
  reviewed suggestions, model-aware negative prompting, and reproducible
  compiler metadata.
- A local-checkpoint-only `dmlx bench-seedvr2` spike command that measures
  2x/3x/4x runtime, peak MLX allocation, pressure, swap growth, and UI probe
  latency without exposing or downloading the model.
- Sequential multi-step execution with `$step_id` artifact references,
  per-step admission control, live step events, cancellation, and preserved
  intermediate result metadata.
- Atomic multi-step checkpoints, restart recovery, and retry from a failed or
  selected step while reusing earlier persisted artifacts.
- Capability-driven execution for additional image operations. A custom
  validated upscaler can execute and compose with editing without new API
  branches; the built-in SeedVR2 catalog entry remains gated on benchmarks.
- EXIF-orientation normalization and shared editor geometry with scale, Retina,
  crop, and stroke-alignment coverage.
- Real-model integration checks and a ten-turn FLUX.2 no-hang run; see
  [`../benchmarks/workflow-10-turn.md`](../benchmarks/workflow-10-turn.md).
- A routing evaluator that reports coverage, selective accuracy, multiclass
  Brier score, ECE, confidence bins, and exact ordered input selection.
- An in-process Qwen MLX provider with strict JSON validation, bounded
  chat-history context, ordered historical input resolution, lazy loading, and
  eviction before image generation. The pinned 4-bit checkpoint passes the
  25-case regression fixture, including all five ordered history cases; see
  [`../benchmarks/qwen-routing.md`](../benchmarks/qwen-routing.md).

Still gated on the evidence called for below: a larger independently authored
router evaluation and 16 GB-machine run, the canvas mask/sketch executors,
SeedVR2 benchmarking and enablement, and model-backed prompt quality evaluation.

## 1. Objective

Evolve dMLX from two hard-coded tasks (`t2i` and `edit`) into a chat-driven,
capability-aware workflow system that supports:

1. Editing a source image after cropping, painting, sketching, or masking it.
2. Routing requests such as "enhance this" or "upscale this 2x" to the correct
   operation and model.
3. Ordered multi-image inputs that can be referenced as image 1, image 2, and so
   on in the prompt.
4. Optional prompt enhancement, explicit negative prompts, and automatic
   negative-prompt generation where the selected model supports it.
5. Adding future operations without duplicating new branches across the Swift
   app, HTTP client, API, scheduler, and every adapter.
6. Using either a bundled local decision model or a user-configured
   OpenAI-compatible LLM for routing and prompt compilation.

The implementation must preserve the existing OpenAI-compatible image API,
stored conversations, deterministic metadata, and the 16 GB no-hang guarantee.

## 2. Product rules

### 2.1 One attachment model

- Replace separate **Edit** and **Use as reference** actions with **Attach**.
- The paperclip attaches a local asset; **Attach** on a result attaches that
  result without saving and re-uploading it.
- Display ordered badges (`1`, `2`, `3`) and support drag-to-reorder.
- Preserve attachment order through storage, API serialization, routing, and
  adapter invocation.
- An explicit attachment is never silently discarded by the semantic router.

### 2.2 Continued chat still requires routing

With no explicit attachment:

- No current result: generate a new image.
- A current result exists: ask the semantic router whether to generate, edit,
  vary, enhance, upscale, or reject an unsupported request.
- Low confidence: let the user choose rather than executing an expensive wrong
  workflow.

### 2.3 Explicit editor state overrides inference

- A crop is a deterministic local transform.
- Painted pixels produce a composited source image.
- A mask means the user explicitly wants a region-limited edit.
- A guidance sketch is a separate input role, not a mask.
- The router may choose an operation, but it may not reinterpret or remove
  explicit editor layers.

### 2.4 Prompt changes remain visible

- Store the exact user prompt.
- Show an enhanced prompt before execution in **Ask** mode.
- Store the compiled positive and negative prompts in result metadata.
- Never claim that a model used a negative prompt if its adapter ignored it.

## 3. Current capability facts

- The existing FLUX.2 adapter already passes an ordered list of image paths to
  `Flux2KleinEdit`; this is enough for ordered multi-reference editing.
- The built-in FLUX.2 Klein model accepts up to three image inputs in dMLX.
- FLUX.2 does not support a separate negative prompt. Negative-prompt UI must be
  capability-gated rather than globally enabled.
- The pinned mflux package contains SeedVR2 3B/7B upscaling code. It is a
  candidate adapter, not an approved default, until download size, cold load,
  peak memory, output quality, and 16 GB behavior are measured.
- The current app and result store are image-specific. Video needs a later
  artifact-store migration and is not part of this implementation.

## 4. Target architecture

```text
Composer text + ordered attachments + editor document
                         |
                         v
             deterministic constraints
          (available inputs and capabilities)
                         |
                         v
               semantic route decision
       (local decision model or configured LLM)
                         |
                         v
                capability resolver
           (operation -> installed executor)
                         |
                         v
                  prompt compiler
        (passthrough / enhance / negative prompt)
                         |
                         v
                validated workflow plan
                         |
                         v
                 single GPU job queue
                         |
                         v
              versioned result + metadata
```

The decision model never returns executable arbitrary JSON. It selects among
candidate operations supplied by dMLX. The engine constructs and validates the
final plan.

### 4.1 Intelligence providers

Routing and prompt compilation are provider-backed services, not properties of
the diffusion adapters:

```text
DecisionProvider
|- LocalLLMDecisionProvider      in-process MLX language model
`- OpenAICompatibleProvider      local server or remote API

PromptProvider
|- PassthroughPromptProvider     no model
`- OpenAICompatibleProvider     local server or remote API
```

The user may select different providers and models for routing and prompt
enhancement, or use the same endpoint for both. A provider advertises its own
capabilities: structured outputs, tool calling, JSON mode, vision, and whether
per-option probabilities are available.

For broad self-hosted compatibility, start with `/v1/chat/completions`. Add the
Responses API as a separate provider mode rather than assuming every
OpenAI-compatible server implements it. Output negotiation follows this order:

1. Strict JSON Schema structured output.
2. A single constrained function/tool call.
3. JSON mode.
4. Plain JSON with local schema validation and at most one repair retry.

All returned operations, input IDs, parameters, and step counts are validated
against dMLX's candidate set and capability registry. Schema compliance is not
authorization to execute an unsupported plan.

Provider configuration includes base URL, model ID, API style, optional API
key, timeout, and whether the endpoint is local or remote. Store API keys in the
macOS Keychain, never SQLite, workflow metadata, or logs. Require HTTPS for
non-loopback endpoints. Routing and prompt enhancement send text and structured
metadata only; sending source pixels to a VLM requires a separate explicit
opt-in capability in a later release.

## 5. Domain model

Introduce matching Python and Swift wire types. Use strings for operation and
capability identifiers so adding an executor does not require extending a
closed Swift enum before the app can decode model metadata.

### 5.1 Media input

```json
{
  "id": "img_20261020_ab12cd",
  "media_type": "image",
  "role": "source",
  "order": 0,
  "metadata": {}
}
```

Initial values:

- `media_type`: `image`; reserve `video` for a later release.
- `role`: `source`, `reference`, `composite`, `mask`, or `sketch`.
- `order`: stable zero-based order from the composer.

Roles remain optional for normal attachments. Adapters that only understand an
ordered image list receive the inputs in `order` order.

### 5.2 Prompt specification

```json
{
  "user": "Make this sharper without text",
  "positive": "Make this sharper without text",
  "negative": null,
  "enhancement_mode": "ask",
  "compiler": "passthrough@1"
}
```

Modes: `off`, `ask`, and `auto`.

### 5.3 Workflow plan

```json
{
  "schema_version": 1,
  "steps": [
    {
      "id": "step_1",
      "operation": "image.edit",
      "executor": "flux2-klein-4b",
      "inputs": [
        {"id": "img_1", "media_type": "image", "role": "source", "order": 0},
        {"id": "img_2", "media_type": "image", "role": "reference", "order": 1}
      ],
      "prompt": {
        "user": "Put the glasses from image 2 on the person in image 1",
        "positive": "Put the glasses from image 2 on the person in image 1",
        "negative": null,
        "enhancement_mode": "off",
        "compiler": "passthrough@1"
      },
      "parameters": {"width": 1024, "height": 1024, "steps": 4}
    }
  ],
  "route": {
    "intent": "image.edit",
    "confidence": 0.94,
    "router": "local-llm:qwen3-4b-instruct-2507-4bit"
  }
}
```

The schema supports multiple steps. A later input may use an earlier result as
`{"id":"$step_1"}`; the worker persists that artifact and reruns admission
control before the next step. Local crop/composite preparation may also happen
before inference. Chained edit-then-upscale becomes available only after the
upscale executor independently passes its memory gate.

### 5.4 Capabilities

Replace the meaning of the registry's bare `tasks` list with declarative
capabilities while continuing to emit `tasks` for old clients:

```yaml
capabilities:
  image.generate:
    inputs: {}
    prompt: { positive: true, negative: false }
  image.edit:
    inputs:
      image: { min: 1, max: 3, ordered: true }
    prompt: { positive: true, negative: false }
```

Later executor examples include `image.inpaint`, `image.upscale`,
`image.restore`, and `image.enhance`. Capability validation belongs in the
engine, not only in the UI.

## 6. API strategy

Add native workflow endpoints while preserving existing clients:

- `POST /v1/workflows/route`: return candidate decision probabilities and the
  proposed operation; do not start work.
- `POST /v1/workflows`: validate a plan, create a job, and optionally stream the
  existing SSE event format.
- `GET /v1/capabilities`: return available operations and input constraints.

Keep `/v1/images/generations` and `/v1/images/edits`. Translate their requests
into one-step workflow plans internally so there remains one execution path.

Additive fields go into existing result metadata. Do not rewrite old SQLite
payloads or result JSON files in place.

## 7. Implementation milestones

### W0 — Contracts and regression fixtures

Goal: define the new boundary without changing behavior.

Engine:

- Add `workflow.py` with `MediaInput`, `PromptSpec`, `WorkflowStep`,
  `WorkflowPlan`, and validation errors.
- Add capability parsing to `registry.py`; synthesize capabilities from old
  `tasks` values when loading existing or user-added YAML.
- Add translators from existing generation/edit payloads into plans.
- Add golden request/response fixtures for generation, one-image editing, and
  ordered three-image editing.

App:

- Add Codable equivalents in `DMLXKit`.
- Preserve decoding of existing `Turn` and `ImageVersion` JSON. New fields must
  have defaults or custom decoding so old sessions continue to load.

Gate:

- Existing 44 engine unit tests and 9 Swift tests remain green.
- Old stored-turn fixtures decode and round-trip.
- Existing OpenAI endpoints produce the same results and metadata, plus
  additive workflow metadata.

### W1 — One attachment system and ordered inputs

Goal: simplify the UI and make input ordering an explicit contract.

- Replace `Attachment`'s bare result ID with `MediaInput`.
- Replace **Edit** and **Use as reference** with **Attach**.
- Add numbered attachment chips and drag-to-reorder.
- Keep the current maximum of three image inputs for FLUX.2.
- Send attached inputs through `/v1/workflows` from the app.
- Keep the old image endpoints for third-party callers.
- Store the ordered input list on the user turn and result version.

Gate:

- Generated results and disk images use the same attachment path.
- A three-image prompt reaches the adapter in exactly the displayed order.
- Removing or reordering an attachment updates the request deterministically.
- Existing chat history remains readable.

### W2 — Router abstraction and evaluation harness

Goal: replace keyword routing only after a measurable alternative exists.

- Add a `SemanticRouter` protocol and deterministic pre-router.
- Candidate choices come from installed model capabilities and current inputs.
- Add a checked-in JSONL evaluation set covering generate, edit, variation,
  upscale, enhance/restore, unsupported video, ambiguity, multiple languages,
  slang, and misleading keywords.
- Implement the current regex router as a baseline evaluator, not the final
  fallback.
- Implement an in-process MLX language-model provider with a pinned revision.
- Implement an OpenAI-compatible decision provider with a mock server contract
  test. Support user-supplied base URL and model ID rather than assuming a
  particular vendor or `/v1/models` implementation.
- Add Settings UI for separate routing and prompt providers. Store credentials
  in Keychain and provide a non-destructive **Test connection** action.
- Negotiate structured-output support and validate every response locally.
- Record label probabilities, model version, latency, and memory allocation.
- Select confidence thresholds from the evaluation results.

Fallback policy:

- Explicit attachments and impossible states resolve deterministically.
- If the semantic model is missing, unsafe to load, or below threshold, return
  `ambiguous`; the app asks the user to choose.
- If a configured remote or self-hosted provider is unavailable, timeouts or
  schema failures produce `ambiguous`; they never block manual execution.
- Do not silently return to keyword routing in production.

Memory integration:

- Track router residency in `MemoryManager` and `/admin/memory`.
- Make router weights evictable before diffusion work.
- Do not run router MLX inference concurrently with a diffusion pass until a
  contention benchmark proves that safe.

Gate:

- Router beats the regex baseline on the checked-in evaluation set.
- Per-class recall and calibration are reported, not only total accuracy.
- Provider contract tests cover strict schema, tool-call, JSON-mode, malformed,
  refusal, timeout, authentication, and unreachable-endpoint responses.
- No stale decision can be applied to a changed message or attachment order.
- Router-resident and router-evicted no-hang tests pass.

### W3 — Asynchronous route preview and confirmation

Goal: make probabilistic decisions visible and correctable.

- Debounce preview decisions after typing pauses.
- Tag each request with a hash of message, current result, attachment IDs, and
  attachment order; discard stale responses.
- Show the proposed operation and selected executor in the composer.
- Let the user override the proposal before sending.
- Present a compact chooser when confidence is below threshold.
- Persist the final route decision and override, if any, in metadata.
- On Send, use the exact decision shown for the same context hash; reroute if
  the context changed.

Gate:

- UI tests cover cancellation, stale responses, low confidence, user override,
  missing router, and engine restart.
- No request executes with a route computed for different inputs.

### W4 — Non-destructive image editor

Goal: support crop and drawing without requiring a new diffusion model.

Add an `EditorDocument` stored beside the session:

- Base result ID and base dimensions.
- Normalized crop rectangle.
- Raster paint overlay with transparency.
- Optional binary/greyscale mask layer.
- Optional guidance sketch layer.
- Undo/redo command history for the active editing session.

Initial tools:

- Crop with common aspect ratios and freeform mode.
- Brush, eraser, color, opacity, and size.
- Separate **Paint** and **Mask** modes so pixels and edit regions are not
  conflated.
- Reset and compare-to-original.

Submission behavior:

- Crop is rendered locally before upload.
- Paint produces a flattened composite input while preserving the original and
  layer metadata.
- Mask and sketch are sent as typed inputs only to executors declaring those
  roles. If none is installed, the app explains that the capability is missing;
  it must not silently flatten a mask into visible paint.

Gate:

- Pixel-coordinate tests cover scaled display, Retina backing scale, crop,
  rotation/orientation, and mask alignment.
- Editor state survives closing and reopening a session.
- Submitted composites match the preview pixel-for-pixel.
- Undo/redo does not mutate the stored original.

### W5 — Upscale and enhancement workflows

Goal: deliver the first operation beyond generate/edit.

Spike first:

- Benchmark mflux SeedVR2 3B and 7B on supported hardware.
- Measure download size, cold load, peak and retained memory, 2x/3x/4x runtime,
  tiling seams, and output limits.
- Determine whether per-component lifecycle and streamed/sequential execution
  can be integrated with the existing memory manager.

If the 3B variant passes:

- Add `SeedVR2Adapter` with `image.upscale` capability.
- Add model download/removal and memory estimates.
- Add scale and target-size parameters with a pixel-count admission limit.
- Route explicit "upscale" requests directly; use the semantic router for
  ambiguous "enhance" requests.

Keep these operations distinct:

- `image.upscale`: increase resolution while preserving content.
- `image.restore`: reduce blur, noise, or compression artifacts.
- `image.enhance`: generatively recreate or add detail.

Only expose operations backed by an installed, validated executor. SeedVR2 does
not automatically establish separate restore and enhance capabilities unless
quality evaluation supports those claims.

Gate:

- Upscaling passes the real or emulated 16 GB no-hang gate, or is explicitly
  hidden below the measured RAM tier.
- Output dimensions and metadata are correct for every supported scale.
- Cancelling an upscale frees its components and leaves the worker usable.

### W6 — Prompt compiler and negative prompts

Goal: make prompt assistance explicit, reproducible, and model-aware.

- Add `PromptCompiler` protocol with a zero-cost passthrough implementation.
- Add user preference: `off`, `ask`, or `auto`.
- Parse explicit structured syntax such as `Negative:` and `--negative`
  deterministically.
- Implement prompt enhancement through the configured OpenAI-compatible
  provider. It may point to a local self-hosted model or a remote API.
- Keep provider selection independent between routing and prompt compilation.
  The same compatible LLM may perform both jobs, but it is not required.
- Show positive and negative compiled fields in **Ask** mode.
- Store user, positive, negative, compiler ID/revision, and user edits.
- Include prompt fields in embedding-cache keys and deterministic metadata.

Capability behavior:

- If the executor supports negative conditioning, pass the negative field.
- If it does not, retain the user's text but show that a separate negative
  prompt is unsupported by that model.
- Do not silently drop the field or claim it was used.
- Do not automatically switch to a much larger model without showing the model
  and memory impact before execution.

Gate:

- Prompt compilation is opt-in until its quality evaluation is approved.
- Remote prompt compilation is visibly identified and requires an explicit user
  choice; local-only operation remains fully supported.
- Fixed compiler revision and settings reproduce the stored compiled prompt.
- FLUX.2 tests verify that negative prompts are rejected or surfaced, never
  ignored.
- Models with negative-prompt support receive the exact stored value.

### W7 — Multi-step execution and hardening

Goal: safely support requests such as edit-then-upscale.

- Extend the worker from one adapter call to a validated list of steps.
- Pass step outputs by artifact ID rather than in-memory objects across
  executors, allowing memory to be reclaimed between steps.
- Re-run admission control before every step.
- Emit `workflow_step_started`, existing phase events, and
  `workflow_step_done` while retaining terminal job events.
- Cancel the remaining plan if a step fails; preserve completed intermediate
  artifacts and identify them in the report.
- Add retry-from-step only after plans and artifacts are stable.

Gate:

- Edit-then-upscale completes with deterministic step ordering and metadata.
- Cancellation and pressure aborts leave no model unexpectedly pinned.
- Crash recovery marks the plan accurately and preserves completed artifacts.
- The full no-hang and 10-turn suites pass.

## 8. Expected file changes

Engine additions:

```text
engine/src/dmlx/workflow.py
engine/src/dmlx/capabilities.py
engine/src/dmlx/routing/base.py
engine/src/dmlx/routing/deterministic.py
engine/src/dmlx/intelligence/local_llm.py
engine/src/dmlx/intelligence/base.py
engine/src/dmlx/intelligence/openai_compatible.py
engine/src/dmlx/prompting/base.py
engine/src/dmlx/prompting/passthrough.py
engine/src/dmlx/adapters/seedvr2.py
engine/tests/fixtures/routing.jsonl
engine/tests/test_workflow.py
engine/tests/test_routing.py
```

Likely engine modifications:

```text
engine/src/dmlx/api.py
engine/src/dmlx/engine.py
engine/src/dmlx/memory.py
engine/src/dmlx/registry.py
engine/src/dmlx/results.py
engine/src/dmlx/adapters/base.py
engine/src/dmlx/adapters/mflux_adapter.py
engine/src/dmlx/models/*.yaml
```

App additions:

```text
app/Sources/DMLXKit/Workflow.swift
app/Sources/DMLXKit/Capabilities.swift
app/Sources/DMLXKit/IntelligenceProvider.swift
app/Sources/dMLX/Editor/EditorDocument.swift
app/Sources/dMLX/Editor/EditorView.swift
app/Sources/dMLX/Editor/CanvasView.swift
app/Sources/dMLX/Views/RouteChooser.swift
app/Tests/DMLXKitTests/WorkflowTests.swift
```

Likely app modifications:

```text
app/Sources/DMLXKit/Models.swift
app/Sources/DMLXKit/EngineClient.swift
app/Sources/DMLXKit/Routing.swift
app/Sources/dMLX/AppModel.swift
app/Sources/dMLX/Views/ComposerView.swift
app/Sources/dMLX/Views/TurnView.swift
app/Sources/DMLXKit/Store.swift
```

## 9. Testing matrix

### Contract and compatibility

- Old and new SQLite turn payloads.
- Old OpenAI image endpoints translated into plans.
- Unknown future operation/capability identifiers handled without a crash.
- Attachment ordering survives encode/decode and engine restart.

### Routing

- No current image, current image, and explicit attachments.
- Generate/edit/variation/upscale/enhance/unsupported/ambiguous classes.
- Confidence thresholds and user overrides.
- Local LLM and OpenAI-compatible provider parity on the same routing fixture.
- Provider failures, refusals, malformed output, and fallback to manual choice.
- Multilingual, typo, short-fragment, and adversarial keyword cases.

### Editor

- Orientation-normalized imports.
- Crop and mask coordinates at different display scales.
- Composite output equality.
- Undo/redo and persisted documents.

### Execution

- One and three ordered references.
- Unsupported role and operation errors.
- Prompt and negative-prompt capability enforcement.
- API keys never appear in logs, SQLite payloads, result metadata, or exported
  workflow plans.
- Upscale admission, progress, cancellation, and pressure abort.
- Multi-step intermediate artifacts and failure recovery.

### Memory

- Router cold, warm, and evicted.
- Router plus resident FLUX.2 components.
- Canvas export of a maximum-size image.
- Each new executor across eager, sequential, and streamed strategies where
  applicable.
- Real or emulated 16 GB no-hang run before enabling a workflow on that tier.

## 10. Release slices

Keep the work shippable in independent slices:

1. **Workflow foundation:** W0, with no visible behavior change.
2. **Unified references:** W1, ordered multi-image attachments.
3. **Semantic chat routing:** W2--W3, confidence-gated decisions through local
   an in-process local LLM or an OpenAI-compatible provider.
4. **Canvas editing:** W4, crop and paint first; mask only with a validated
   executor.
5. **Upscaling:** W5, separately downloadable and RAM-tier gated.
6. **Prompt assistance:** W6, opt-in first.
7. **Composed workflows:** W7 after all individual operations are stable.

Each slice must pass engine tests, Swift tests, stored-data compatibility, and
the relevant no-hang matrix before the next slice depends on it.

## 11. Decisions still requiring evidence

Do not lock these choices without prototypes and measurements:

- Whether the local LLM meets routing accuracy on an independent fixture and
  16 GB memory requirements.
- Whether the semantic router should execute on MLX GPU, CPU, or a validated
  Core ML/Neural Engine export.
- Which OpenAI-compatible local or remote models produce sufficiently reliable
  route decisions and prompt enhancements; dMLX should not hard-code one.
- Which compatibility level (`json_schema`, tool call, or validated JSON) each
  target self-hosted server actually supports.
- Whether SeedVR2 3B is safe and useful on 16 GB hardware.
- Which model supplies true mask-guided inpainting on each RAM tier.
- Whether generative "enhance" deserves its own executor or is best expressed
  as edit followed by upscale.

Until those gates pass, the UI should expose only capabilities the installed
executors actually provide.

## 12. Definition of done

This initiative is complete when:

- The app has one ordered attachment system.
- Continued chat routes semantically with confidence and a user override.
- Routing and prompt enhancement can use independently configured local or
  OpenAI-compatible providers without exposing credentials.
- Crop and paint edits round-trip through chat without damaging originals.
- At least one validated upscale workflow is available on its supported RAM
  tier.
- Prompt enhancement and negative prompts are transparent and capability-aware.
- Existing API callers and stored sessions remain compatible.
- New workflows can be added primarily through a capability declaration and an
  executor adapter rather than new branches throughout the app.
- Every enabled workflow passes its declared memory-tier no-hang gate.

## 13. References

- [mflux FLUX.2 documentation](https://github.com/mflux-community/mflux/blob/main/src/mflux/models/flux2/README.md)
- [Apple Foundation Models](https://developer.apple.com/documentation/FoundationModels/generating-content-and-performing-tasks-with-foundation-models)
