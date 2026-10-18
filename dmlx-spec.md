# dMLX — A Diffusion Serving Layer for Apple Silicon

> Working name: **dMLX** (diffusion + MLX). Spec v0.1 — October 2026.
> One line: *oMLX for image and video generation* — a persistent, MLX-native, OpenAI-compatible server that keeps diffusion models warm, manages unified memory per component, and folds in step caching, for both images and video.

---

## 1. Summary

Mac users can now run strong open image models (Z-Image, FLUX.2, Qwen-Image) and video models (Wan 2.2, LTX-2) natively on MLX. The **engines** exist. What is missing is the **serving layer**: something that loads models once, keeps the right parts resident, swaps between models without paying full reload cost, picks a memory strategy that fits the machine, applies speed-ups like TeaCache automatically, and exposes it all behind one stable API for apps and agents.

For LLMs, oMLX and vllm-mlx solved the equivalent problem (continuous batching, SSD-backed KV cache, multi-model LRU). For diffusion on Mac, that layer does not exist yet in MLX-native, permissively licensed form. dMLX fills it.

---

## 2. Problem

### 2.1 What users hit today

| Pain | Evidence |
|---|---|
| **Reload on every run.** The reference MLX engine starts from nothing each invocation and discards everything at exit; ten images means ten full load cycles. | mflux maintainer discussion #717 (Sep 2026) calls MFlux "one-and-done" and notes the OS page cache does not remove deserialization, model construction, or on-the-fly quantization costs. |
| **Peak memory during load is the whole model.** All components (text encoders, transformer, VAE) are read and held at once before generation starts. Fine on 128 GB, fatal on 16 GB. | Same discussion: peak during load is roughly the whole model; Flux.1-dev is ~9.5 GB T5 + ~24 GB transformer at bf16; Qwen-Image uses a 7B VLM as its text encoder. |
| **Unpredictable slowdowns under memory pressure.** Running FLUX twice can make the second run much slower even with the model loaded, mostly on 32–64 GB and M1-class machines. | MLX issue #2344. Suggested fix: tune MLX cache / wired limits or pin weights — a policy a server should own. |
| **PyTorch-on-Mac (MPS) path is slow and fragile.** ComfyUI is the de-facto persistent server, but on Mac it runs PyTorch MPS. BF16 runs ~2× slower than FP16 on M1–M3 (no BF16 hardware); forcing FP16 can produce black images; FP8 weights need conversion. | Documented Qwen Image Edit regression on M1 Max: 80 s → 10 min after a ComfyUI update switched to BF16. |
| **Speed-ups exist but are scattered.** Step caching (TeaCache), fast decoders (TAESD-family), and training-free step reduction (MrFlow) are separate packages with separate integration work. | mlx-teacache (first MLX TeaCache port; FLUX, Qwen-Image, Z-Image via mflux); mlx-taef; MrFlow. |
| **Video jobs are long and memory-spiky** with no queue, progress, resume, or memory policy. | Wan2.2-TI2V-5B q8 on a 64 GB Mac: 720p × 41 frames × 20 steps ≈ 15 min. Wan A14B needs mixed q8/BF16 because full quantization produced invalid video. |

### 2.2 Why this is different from the LLM problem

LLM servers win by not recomputing a long, repeated **prompt** (KV cache). Diffusion has short prompts and a compute-heavy denoising loop, so the wins are different:

1. **Don't reload** — keep components (text encoder, transformer, VAE, LoRAs) resident across requests.
2. **Don't hold what you don't need** — load and free per component when memory is tight.
3. **Don't run steps that add nothing** — step caching (TeaCache-style).
4. **Don't re-encode the same prompt** — cache text-encoder outputs.
5. **Don't block** — queue long jobs (especially video), stream progress, allow cancel.

---

## 3. Gap validation

### 3.1 Method

Searched (Oct 2026) for: MLX/Mac diffusion servers, mflux servers and wrappers, ComfyUI MLX backends, multi-modal MLX servers that include image gen, Draw Things server, MLX video ports. Read primary sources (repos, PyPI pages, maintainer discussions, issues) where available. Full list in §15.

### 3.2 Landscape

| Project | What it is | Persistent server | Multi-model + eviction | Per-component memory policy | Step caching built in | Video | MLX-native | License / API |
|---|---|---|---|---|---|---|---|---|
| **mflux** | Reference MLX engine (Z-Image, FLUX.1/2, Qwen-Image, Krea, ERNIE…) | ❌ one-and-done CLI | ❌ | Partial (`MemorySaver` frees after loading everything; `--low-ram`) | ❌ (external) | ❌ | ✅ | MIT / CLI + Python |
| **mlx-gen** | mflux fork adding Wan2.2 T2V/I2V/V2V, Bernini-R, mixed-quant packages, progress callbacks | ❌ CLI/library | ❌ | Partial (`--low-ram`) | ❌ | ✅ | ✅ | MIT / CLI + Python |
| **mlx-video** | MLX video engine (LTX-2, Wan2.1/2.2) | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | CLI |
| **mfluxible** | Streaming HTTP API over mflux | ✅ | ❌ **one model per process**, fixed at startup | ❌ | ❌ | ❌ | ✅ | OpenAI-compatible + A1111 shim |
| **vMLX** | LLM-first MLX server that also serves image gen via mflux | ✅ | ✅ gateway routes to loaded models | Not documented | ❌ | ❌ | ✅ | OpenAI/Anthropic/Ollama APIs; image set limited to Flux schnell/dev, Z-Image Turbo, Qwen Image Edit |
| **mlx-openai-server** | Multi-modal MLX server | ✅ | Not documented | Not documented | ❌ | ❌ | ✅ | OpenAI-compatible; Flux series, Qwen Image, Z-Image, edits, LoRA |
| **Draw Things gRPCServerCLI** | Open-source server from the Draw Things engine (own Swift + Metal FlashAttention stack) | ✅ | Not documented | Not documented | Not documented | Not documented | ❌ (own engine, `.ckpt` format) | **GPL-3.0**, custom gRPC protocol |
| **ComfyUI** (+ ComfyUI-MLX nodes) | Node-graph server; MLX nodes are a thin DiffusionKit port | ✅ | ✅ | ✅ (dynamic VRAM / block streaming on supported hardware, per mflux #717) | Via custom nodes | ✅ | ❌ core is PyTorch MPS | GPL-3.0 / own API |
| **oMLX / vllm-mlx** | LLM/VLM servers with SSD KV cache, batching | ✅ | ✅ | n/a | n/a | Video *understanding* only | ✅ | Not for generation |

### 3.3 Verdict

**The gap is real but narrower than "nothing exists."** Validated precisely:

> There is no **MLX-native, permissively licensed, multi-model diffusion server** that (a) manages unified memory **per component** with a strategy chosen from the machine's headroom, (b) keeps components warm across requests and models with LRU eviction, (c) integrates step caching and prompt-embedding caching by default, and (d) treats **video** as a first-class queued job — all behind an OpenAI-compatible API.

Each neighbor covers a slice:

- **mfluxible** has the API, but one model per process.
- **vMLX** has multi-model routing, but image gen is a side feature over mflux: a small model set (Flux schnell/dev, Z-Image Turbo, Qwen Image Edit), no documented per-component memory policy or step caching, and no video.
- **Draw Things gRPCServerCLI** is the strongest persistent Mac server, but it is a separate non-MLX engine, uses its own model format and gRPC protocol, and is GPL-3.0 — which forces derivative servers to be GPL and constrains bundling.
- **ComfyUI** has the memory machinery, but on Mac it runs PyTorch MPS, which is exactly the slow/fragile path users are trying to leave.
- **mflux maintainers themselves** proposed per-component lifecycles, runtime strategy selection, a cross-generation cache, and a possible `mflux-generate-server` in Sep 2026 — confirming the need — but noted a persistent server is a large investment and that ComfyUI already fills that role for many. That is our opening: build the server layer *on top of* the engines, not inside them.

### 3.4 Corrections to earlier assumptions

- *"Video lacks MLX ports"* — **false.** mlx-video (LTX-2, Wan2.1/2.2) and mlx-gen (Wan2.2 T2V/I2V/V2V, Bernini-R) exist. The gap for video is serving and memory management, not the engine.
- *"Draw Things is closed source"* — **partly false.** Its engine and gRPCServerCLI are open source under GPL-3.0; the app is not.

### 3.5 What could close the gap before we ship (threats)

- mflux lands component lifecycles + a server mode (discussion #717 roadmap).
- vMLX deepens image support (component caching, video).
- ComfyUI gains a mature MLX backend.

**Mitigation:** design dMLX as a thin serving layer over engine adapters, contribute component-lifecycle work upstream to mflux/mlx-gen, and compete on serving (memory policy, caching, video jobs, API, UX) rather than on model ports.

---

## 4. Goals and non-goals

### Goals

1. Load once, serve many: warm components across requests and across models.
2. Per-component memory management with automatic strategy selection by memory headroom.
3. Built-in acceleration: prompt-embedding cache, step caching, fast preview decoders.
4. Images **and** video as first-class workloads, with an async job queue, progress streaming, and cancel.
5. Drop-in API: OpenAI-compatible images endpoints plus an async video jobs API.
6. Coexist with LLM servers (oMLX, vMLX) on the same unified memory.
7. Fully open source — server **and** chat app — under Apache-2.0. Model weights are never bundled; each model's own license is shown in the catalog, and defaults are permissively licensed models.
8. **Never hang the Mac on 16 GB.** 16 GB is a hard target, and "runs" is not enough: the machine must stay usable while generating (see §6.2).

### Non-goals (v1)

- Writing new model ports — use mflux / mlx-gen / mlx-video via adapters.
- Node-graph workflow editor (ComfyUI territory).
- Training or LoRA training.
- Linux/Windows/Intel Mac support.
- Multi-Mac distributed inference (revisit later).

---

## 5. Users and use cases

| User | Scenario | What they need |
|---|---|---|
| App / agent developer | Agent calls `images.generate` dozens of times per session | Warm model, low latency per call, OpenAI API |
| Creator on 16–24 GB MacBook | Wants Qwen-Image or FLUX.2 9B that barely fits | Sequential load/free, streaming, no crashes |
| Power user on 64–128 GB Mac Studio | Switches between 3–4 image models + a video model + an LLM in oMLX | Multi-model LRU, pinning, no reload thrash |
| Video tinkerer | Runs Wan 2.2 I2V jobs that take 10–20 min | Queue, progress, cancel, previews, stable memory |
| Frontend authors (Open WebUI, SillyTavern, LM Studio plugins) | Point an existing client at a local endpoint | OpenAI-compatible + A1111-style shim |

---

## 6. Requirements

### 6.1 Functional

| ID | Requirement | Priority |
|---|---|---|
| F1 | Serve multiple image/video models from one process; route by `model` name | P0 |
| F2 | Component registry: each model declares components (text encoders, transformer, VAE, vision tower, LoRA slots), size, the phase each is needed, and whether it is required per task | P0 |
| F3 | Component cache with memory budget and LRU eviction; share identical components across models (e.g. same VAE or text encoder) | P0 |
| F4 | Runtime memory strategy (§8): eager / phase-free / sequential / stream, auto-selected, manual override per model | P0 |
| F5 | Pin / unpin / load / unload models via API and UI | P0 |
| F6 | Prompt-embedding cache keyed by (encoder id, prompt, negative prompt, params), RAM + optional SSD tier | P1 |
| F7 | Step caching (TeaCache-style) per model with calibrated defaults and a quality toggle | P1 |
| F8 | Fast preview decoding (TAESD-family) streamed as progress events | P1 |
| F9 | Async job queue for video and long image jobs: submit, status, SSE progress, cancel, result fetch | P0 (video) |
| F10 | LoRA hot-swap without reloading the base transformer | P1 |
| F11 | Image-to-image, edit (Qwen Image Edit / FLUX.2 edit), and I2V inputs | P1 |
| F12 | Request batching: same model + same resolution requests share forward passes where the engine allows | P2 |
| F13 | Benchmark command: reports load time, time-to-first-image, s/step, peak memory per strategy | P0 |
| F14 | Model download and preparation (quantized packages) with no network during generation | P1 |
| F15 | Coexistence mode: read system memory pressure; yield (evict/unload) when an LLM server needs memory | P2 |

### 6.2 Non-functional

- **No crashes from memory pressure** on supported model/RAM combinations; degrade to a slower strategy instead.
- **No-hang guarantee (16 GB and up).** While a job runs: system memory pressure never reaches *critical*; swap growth stays under a fixed cap per job; the foreground UI (dMLX and other apps) keeps responding to input. Enforced by: admission control against *available* memory with an OS headroom reserve, conservative MLX memory/wired limits, a memory-pressure listener that pauses → degrades → cancels the job, and splitting GPU work into short command buffers so WindowServer is not starved. Verified by an automated "no-hang" test on every release.
- **Warm second request** for a resident model: no measurable load cost.
- **Localhost by default**, optional bearer token for LAN.
- **Deterministic outputs** given seed + params + strategy (step caching off).
- **Observability:** per-request timing breakdown (encode / denoise / decode / load), memory high-water mark.

---

## 7. Architecture

```
            ┌──────────────── Clients ────────────────┐
            │ OpenAI SDK · agents · Open WebUI ·     │
            │ SillyTavern (A1111 shim) · menu-bar UI  │
            └───────────────────┬────────────────────┘
                                │ HTTP + SSE
┌───────────────────────────────▼───────────────────────────────┐
│ API layer (FastAPI)                                           │
│  /v1/images/* (sync)   /v1/videos/* (async jobs)   /admin/*   │
├───────────────────────────────────────────────────────────────┤
│ Scheduler / job queue                                         │
│  priority · per-model queues · cancel · batching (P2)         │
├───────────────────────────────────────────────────────────────┤
│ Memory manager                                                │
│  budget from system headroom · strategy selection (§8) ·      │
│  MLX cache/wired-limit tuning · pressure listener             │
├──────────────────────┬────────────────────────────────────────┤
│ Component cache      │ Accelerators                           │
│  LRU · pin · share   │  prompt-embed cache (RAM + SSD)        │
│  identical parts     │  step cache (TeaCache) · TAESD preview │
├──────────────────────┴────────────────────────────────────────┤
│ Engine adapters (thin, versioned)                             │
│  mflux · mlx-gen (incl. Wan video) · mlx-video                │
├───────────────────────────────────────────────────────────────┤
│ MLX · Metal · unified memory                                  │
└───────────────────────────────────────────────────────────────┘
```

### 7.1 Engine adapters

Each adapter exposes the same interface so the server owns lifecycle, not the engine:

```python
class EngineAdapter(Protocol):
    def describe(model_id) -> ModelSpec            # components, sizes, phases, tasks
    def load_component(model_id, name) -> Handle   # load one component only
    def encode(handles, prompt, ...) -> Embeds
    def denoise(handles, embeds, latents, step_hook) -> Latents
    def decode(handles, latents) -> Image | Frames
```

Where an engine only exposes "load everything then generate," the adapter wraps it (v1) and we upstream per-component loading (aligned with mflux discussion #717) to unlock §8 fully.

### 7.2 Model spec (declarative)

```yaml
id: qwen-image-q8
engine: mflux
components:
  text_encoder: { size_gb: 8.2, phase: encode, required_for: [t2i, edit] }
  vision_tower: { size_gb: 1.3, phase: encode, required_for: [edit] }   # skipped for t2i
  transformer:  { size_gb: 20.4, phase: denoise }
  vae:          { size_gb: 0.3, phase: decode, share_key: qwen-vae-v1 }
defaults: { steps: 50, guidance: 4.0 }
accelerators: { teacache: calibrated, preview: taef }
```

---

## 8. Memory strategy

Chosen per request from **ratio = available memory ÷ required components size** (after skipping components the task doesn't need). Thresholds below are starting points to calibrate in M0.

| Ratio | Regime | Strategy |
|---|---|---|
| > 2× | Comfortable | **Eager:** load all, keep resident across requests, cache text embeddings in RAM |
| 1.2–2× | Adequate | **Phase-free:** load all, free text encoder after encode unless more requests are queued for that model |
| 0.8–1.2× | Tight | **Sequential:** load encoder → encode → free → load transformer → denoise → free → load VAE → decode |
| < 0.8× | Constrained | **Stream:** sequential + transformer block streaming from mmap'd weights; slow but runs |

Rules:

- The queue informs the policy: if more requests for the same model are waiting, don't free components they need.
- "Available memory" counts other processes (e.g. oMLX) and keeps a safety margin; unified memory is shared.
- The server sets MLX cache and wired limits per strategy to avoid the second-run slowdown in MLX #2344.
- Every strategy change is logged and reported in the response metadata.

---

## 9. Caching layers

| Layer | Key | Store | Benefit |
|---|---|---|---|
| Component cache | component content hash | RAM (LRU) | No reload between requests/models; shared VAEs/encoders |
| Prompt-embedding cache | encoder id + prompt + params | RAM → SSD (safetensors) | Skip text encoder for repeated prompts/seeds; lets the encoder be evicted |
| Step cache | per-request (TeaCache) | RAM | Skip low-impact denoise steps (opt-in quality toggle) |
| Result cache (optional) | full request hash incl. seed | SSD | Instant replay for identical deterministic requests |

---

## 10. API

### 10.1 Images (OpenAI-compatible, sync with optional SSE)

```http
POST /v1/images/generations
{ "model": "z-image-turbo", "prompt": "...", "size": "1024x1024",
  "n": 1, "seed": 42, "steps": 8, "stream": true,
  "dmlx": { "strategy": "auto", "teacache": "balanced", "preview": true } }

POST /v1/images/edits        # multipart; Qwen Image Edit, FLUX.2 edit
```

SSE events: `queued`, `loading{component}`, `encoding`, `step{i,total,preview_b64?}`, `decoding`, `done{image, timings, peak_mem_gb, strategy}`.

### 10.2 Video (async jobs)

```http
POST   /v1/videos            -> { "id": "job_123", "status": "queued" }
GET    /v1/videos/job_123    -> status, progress, eta, preview frame
GET    /v1/videos/job_123/events   (SSE)
DELETE /v1/videos/job_123    -> cancel
GET    /v1/videos/job_123/content  -> mp4
```

### 10.3 Admin

```http
GET  /v1/models                 # catalog + resident state
POST /admin/models/{id}/load    { "pin": true }
POST /admin/models/{id}/unload
GET  /admin/memory              # budget, resident components, regime
POST /admin/benchmark           # F13
```

### 10.4 Compatibility shims

A1111-style `/sdapi/v1/txt2img` for SillyTavern and similar clients.

---

## 11. UX

The primary client is a **native chat app for image and video only** — the conversation *is* the edit history. Server controls exist but stay out of the main window.

### 11.1 Why chat, and why not what exists

Neighbors either bolt media onto an LLM chat (MLX Studio, MLXtra, Osaurus, mlx-serve) or use parameter forms. mlx-serve's media panes (Oct 2026) are representative: prompt box, source-image picker, model dropdown with GB sizes, quality tiers, resolution, **Generate**, and a single result with a filename. Edits are one-shot (pick file → Edit/Variation → Generate); iterating means re-picking the output as the source. Media requests in its chat go through an LLM tool call, which loads a chat model just to route (mlx-serve #452). Users are asked to think in settings before intent.

### 11.2 Principles

| Principle | Behavior |
|---|---|
| Prompt first, settings later | One composer. Model, size, steps, quality are auto-picked from the request and the memory regime (§8). Overrides appear as small chips (`Square · Fast · Z-Image`), never a form. |
| Every result is a turn | Images/videos render inline. A follow-up ("make it night") edits the last result by default — no file picking. |
| Actions on the result | Per-result actions: Edit, Variations, Animate (I2V), Upscale, Use as reference, Download. |
| Versions, not overwrites | Every result is a node; regenerations stack as `‹ v1 · v2 · v3 ›`; editing an older version branches the thread. Edits always start from the full-resolution original, never a re-encoded copy. |
| Point at what to change | Brush/circle a region on an inline image for a masked edit; drag in up to 3 reference images. |
| Video as a living card | A video turn is a job card: preview frames, step count, ETA, cancel. The user keeps chatting/editing while it renders (F9, F8). |
| No LLM required to route | Deterministic intent rules (attachment / previous result / "animate", "video" keywords / active brush) choose generate · edit · animate · upscale. A small local LLM for prompt rewriting is optional and never displaces a diffusion component the next turn needs. |
| Infra out of sight | No tok/s, GPU graphs, or endpoints in the main window. Memory surfaces only when it changes behavior ("Needs 24 GB — using FLUX.2 Klein 4B instead. [Use anyway]"). |

### 11.3 Turn → server mapping

| Turn | Route | Server behavior |
|---|---|---|
| Text, no prior image | `t2i` → `/v1/images/generations` | Default model per RAM tier |
| Follow-up on a result, or attached image | `edit` → `/v1/images/edits` with `source_id` | Edit model + encoder kept warm while the session is active (§8 queue-informed policy) |
| Brush mask present | `edit` with `mask` | Inpaint-capable model |
| "Animate…", or Animate action | `i2v` → `/v1/videos` | Async job; card subscribes to SSE |
| Regenerate | same request, new seed | Prompt-embed cache hit (F6) |

Results are referenced by server-side id (`source_id`, `reference_ids`) so turns never re-upload pixels. Session history (turns, version tree, params, seeds) lives client-side in SQLite; every result stores its full request so any version is reproducible.

### 11.4 Defaults by machine

| RAM | Generate | Edit | Video |
|---|---|---|---|
| 16 GB | Z-Image Turbo, FLUX.2 Klein 4B q4 | Klein 4B | Hidden (explain why) |
| 32 GB | + Qwen-Image q4, Klein 9B | + Qwen-Image-Edit q4 | LTX distilled, short clips, warned |
| 64 GB+ | + Qwen-Image q8 | + Qwen-Image-Edit q8 | Wan 2.2 5B, LTX-2 |

### 11.5 Secondary surfaces

- **Menu-bar extra:** server on/off, resident models with memory bars, pin/unload, queue.
- **Settings → Models:** catalog, downloads, license per model, benchmark runner (F13).
- **CLI:** `dmlx serve`, `dmlx pull <model>`, `dmlx bench <model>`.

### 11.6 Open questions

- Masked edits on models without a fill checkpoint: chain (as flux-2-swift-mlx does) or restrict brush to inpaint-capable models?
- How much drift is acceptable across long edit chains before suggesting "branch from original"? Measure in M3 alongside the SSIM gates.
- Whether to ship the optional prompt-rewrite LLM in v1 or rely on the user's prompt verbatim.

---

## 12. Success metrics

| Metric | Target (to validate in M0) |
|---|---|
| Warm request overhead vs. bare engine generate | ≈ 0 (no load cost) |
| Ten-image session time vs. ten mflux CLI runs | ≥ 40% faster (removes nine load cycles) |
| Peak memory, sequential vs. eager (Flux.1-dev bf16) | ≈ transformer size instead of the sum (≈ −10 GB) |
| Largest model that runs on 16 GB without swap/crash | Qwen-Image q4 / FLUX.2 9B q4 |
| Switching between two resident image models | No reload when both fit |
| Video job (Wan2.2-TI2V-5B q8, 64 GB) | Same s/step as mlx-gen, plus queue/progress/cancel |
| Step cache speedup at matched quality (SSIM gate) | Report per model; ship only calibrated presets |

---

## 13. Roadmap

| Milestone | Scope | Exit criteria |
|---|---|---|
| **M0 — Prove it** (2–3 wks) | Benchmark harness over mflux/mlx-gen on 16/32/64/128 GB machines: load time, TTFI, s/step, peak memory per strategy; reproduce MLX #2344 | Numbers confirming §12 targets are reachable; calibrated regime thresholds |
| **M1 — Warm server** | API + scheduler + component cache (wrap-whole-model adapters), multi-model LRU, pin/unload, SSE progress | Ten-image session target met; OpenAI client works unchanged |
| **M2 — Memory manager** | Declarative specs, per-component load/free, strategy auto-selection, MLX limit tuning; upstream per-component loading PRs to mflux | 16 GB targets met; no OOM on supported matrix |
| **M3 — Acceleration** | Prompt-embed cache (RAM+SSD), TeaCache integration, TAESD previews, LoRA hot-swap | Calibrated presets pass SSIM gates |
| **M4 — Video** | Async jobs API, Wan2.2 / LTX-2 adapters, chunked VAE decode, resumable jobs | Video job metrics met |
| **M5 — UX + ecosystem** | Chat app (§11; a thin internal build starts after M1 for dogfooding), menu-bar extra, A1111 shim, coexistence with oMLX/vMLX | Public beta |

---

## 14. Risks and open questions

| Risk / question | Notes / mitigation |
|---|---|
| Upstream moves first (mflux server mode, vMLX image depth) | Position as serving layer; collaborate upstream; keep adapters thin |
| Engines don't expose per-component APIs | v1 wraps whole-model; contribute lifecycle hooks upstream (mflux #717 aligns) |
| Step caching hurts quality on some prompts | Opt-in, per-model calibrated presets, SSIM gates (as mlx-teacache does) |
| Model licenses vary (e.g. FLUX dev non-commercial) | Surface license per model in catalog; never bundle weights |
| GPL contamination | Do not import Draw Things or ComfyUI code; interoperate over network APIs only |
| Video memory spikes in VAE decode | Chunked/streaming VAE (Wan 2.2 MLX VAE already supports streaming-cache mode) |
| Python GIL / single GPU queue limits concurrency | Batching only where shapes match; one denoise loop at a time is acceptable for v1 |
| Open: Python vs. Swift host? | Python first (engines are Python); native Swift menu-bar app talks to it |
| Open: integrate into oMLX/vMLX as a plugin instead of a new server? | Evaluate in M0 — a plugin may be the faster path to users |

---

## 15. Sources

- mflux maintainers — *Idea: smarter approaches to loading model weights in MFlux* (Discussion #717, Sep 2026): https://github.com/orgs/mflux-community/discussions/717
- MLX issue #2344 — FLUX second run slower: https://upd.dev/ml-explore/mlx/issues/2344
- mflux: https://github.com/omercelik/mflux
- mfluxible (PyPI): https://pypi.org/project/mfluxible/
- mlx-gen (PyPI): https://pypi.org/project/mlx-gen/0.29.0/
- mlx-video Wan support (commit notes): https://git.ddq.nl/n0p/mlx-video/commit/36189666251ad5508ea40b834a9c2021f6beacd1
- Wan2.2-TI2V-5B MLX q8 (timing on 64 GB): https://huggingface.co/Anes1032/Wan2.2-TI2V-5B-mlx-q8
- Wan2.2 I2V A14B mixed q8/BF16 for mlx-gen: https://huggingface.co/AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit
- Wan 2.2 VAE MLX port (streaming cache): https://huggingface.co/RockTalk/Wan2.2-VAE-MLX
- vMLX: https://github.com/jjang-ai/vmlx
- mlx-openai-server (PyPI): https://pypi.org/project/mlx-openai-server/1.5.2/
- Draw Things community repo (GPL-3.0, gRPCServerCLI): https://github.com/drawthingsai/draw-things-community
- ComfyUI-MLX nodes: https://comfydeploy.com/comfy-node/thoddnn/ComfyUI-MLX
- ComfyUI Qwen Image Edit MPS BF16 slowdown: https://lilting.ch/en/articles/comfyui-qwen-mps-bf16-slowdown
- mlx-teacache (PyPI): https://pypi.org/project/mlx-teacache/0.10.0/
- MrFlow: https://huggingface.co/Xingyu-Zheng/MrFlow
- vllm-mlx paper (multimodal caching for understanding, not generation): https://arxiv.org/abs/2601.19139v1
- oMLX: https://sourceforge.net/mirror/omlx/
