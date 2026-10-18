# dMLX — Plan to first beta (v0.1-beta)

Owner of all implementation: Claude. Owner of decisions, approvals, and hardware access: Vincent.
Companion docs: `dmlx-spec.md` (what and why), prototype: https://claude.ai/artifact/An8GqnnPUboCtw1JUASYEM

## 1. Beta definition

A Mac app a technical user can download from GitHub, open, and use to **create and iteratively edit images in a chat**, running fully locally, that **never hangs a 16 GB Mac**.

### In scope

| Area | Beta includes |
|---|---|
| Models | Z-Image Turbo (generate), FLUX.2 Klein 4B (generate + edit + up to 3 references). Qwen-Image-Edit q4 offered only on ≥32 GB. |
| Chat UI | Thread, composer, inline results, version strip, actions (Edit, Variations, Use as reference, Save), rule-based routing with live route preview, sessions sidebar, empty state, starter prompts. |
| Engine | Warm components across turns, admission control, memory-pressure guard, eager / phase-free / sequential strategies, prompt-embedding cache (RAM), step previews over SSE, result store with ids, cancel. |
| Onboarding | Detect RAM tier → recommend models → download with progress/resume → first image. |
| Reliability | Engine as child process; auto-restart with backoff; jobs fail cleanly with a readable message, never a frozen UI. |
| Distribution | GitHub release: zipped `.app` (ad-hoc signed, Gatekeeper workaround documented) + build-from-source. Apache-2.0. |

### Out of scope for beta (kept in spec for later)

Video, brush masks/inpainting (stretch only), upscaling, LoRA, mlx-gen / mlx-video adapters, A1111 shim, menu-bar extra, SSD embedding cache, batching, notarized DMG, auto-update.

### Beta exit checklist

- [ ] No-hang test passes on the 24 GB dev Mac **and** under 16 GB emulation for every default model/task.
- [ ] At least one run on a real 16 GB Mac (borrowed machine or tester) passes the no-hang test.
- [ ] Warm follow-up edit performs zero component loads (verified in response metadata).
- [ ] 10-turn edit session completes without engine restart on 16 GB emulation.
- [ ] Engine killed mid-job → app recovers, shows the failure in the turn, next request works.
- [ ] Fresh-machine install: download → onboarding → first image, with no terminal use.
- [ ] Unit + integration tests green; golden-image tests deterministic for fixed seeds.
- [ ] README, CONTRIBUTING, known issues, model license table.

## 2. Architecture and repo layout

```
dmlx/
├── dmlx-spec.md, PLAN.md, LICENSE (Apache-2.0), README.md
├── engine/                      # Python, uv project, package `dmlx`
│   ├── pyproject.toml / uv.lock
│   ├── src/dmlx/
│   │   ├── api/                 # FastAPI app, routes, SSE events, auth token
│   │   ├── scheduler.py         # single GPU worker, job queue, cancel
│   │   ├── memory/              # budget, admission, pressure monitor, strategies, MLX limits
│   │   ├── registry/            # declarative model specs (YAML), component catalog
│   │   ├── adapters/            # base protocol, mflux adapter, fake adapter (tests)
│   │   ├── cache/               # component cache (LRU, pin), prompt-embed cache
│   │   ├── results.py           # result store: ids, files, embedded params
│   │   ├── downloads.py         # HF downloads with progress + resume
│   │   └── cli.py               # dmlx serve | pull | bench | nohang
│   ├── models/*.yaml            # model specs
│   └── tests/                   # unit (fake adapter), integration (real models, marked)
├── app/                         # SwiftUI, macOS 15+, generated with XcodeGen
│   ├── project.yml
│   ├── Sources/dMLX/
│   │   ├── Engine/              # process lifecycle, API client, SSE client
│   │   ├── Store/               # SQLite (GRDB): sessions, turns, versions
│   │   ├── Routing/             # intent rules (pure, unit-tested)
│   │   ├── Views/               # Thread, Composer, ResultCard, VersionStrip, Onboarding, Settings
│   │   └── App.swift
│   └── Tests/
├── tools/
│   ├── nohang/                  # ballast (16 GB emulation), pressure/swap/jitter sampler
│   ├── bundle_engine.sh         # python-build-standalone + locked deps → app Resources
│   └── release.sh               # build, ad-hoc sign, zip
└── .github/workflows/           # unit tests (engine + app) on macOS runners
```

**Key choices**

- **Engine:** Python 3.12, FastAPI + SSE, one GPU worker thread (one denoise loop at a time), mflux pinned to an exact version.
- **App ↔ engine:** HTTP on `127.0.0.1:<random port>` + bearer token passed at launch; same API documented for third-party clients.
- **App launches the engine** as a child process (dev: `uv run`; release: bundled interpreter). Health check, restart with backoff, kill on quit.
- **Storage:** `~/Library/Application Support/dMLX/` — `db.sqlite`, `results/` (PNG with params embedded), models in the Hugging Face cache layout.

## 3. No-hang design (the core requirement)

| Mechanism | Detail |
|---|---|
| Admission control | Before any load: `need(task, strategy) ≤ available − reserve`. Reserve ≈ 4.5 GB on 16 GB, scaled by RAM. "Available" = free + inactive + purgeable from `host_statistics64`, minus other apps' growth trend. If nothing fits, try the next slower strategy; if none fits, refuse with a readable reason. |
| MLX limits | `mx.set_memory_limit` and `mx.set_cache_limit` per strategy; `mx.set_wired_limit` kept conservative; `mx.clear_cache()` after freeing components. |
| Pressure guard | Poll `kern.memorystatus_vm_pressure_level` + `vm.swapusage` every 250 ms during jobs. *Warn* → finish current step, free non-essential components, drop to slower strategy. *Critical* or swap delta over cap → cancel the job, free everything, report. |
| GPU fairness | Evaluate per step (and per block in sequential mode) to keep command buffers short; optional "Keep my Mac responsive" setting adds yields between steps. |
| Verification | `dmlx nohang <model> <task>`: runs a job while sampling pressure level, swap delta, and main-thread timer jitter of a probe process; emits pass/fail. 16 GB emulation = ballast process holding ~8 GB wired + engine told the machine has 16 GB. |

## 4. Milestones

Each milestone ends with a demo artifact and its gate. I report results (numbers, screenshots) at each gate before moving on.

### B0 — Foundations
- `git init`, Apache-2.0 LICENSE, README skeleton, `.gitignore` (models, results, build output).
- `engine/` uv project with pinned deps; `app/` XcodeGen project that builds an empty window.
- GitHub Actions: engine unit tests (fake adapter) + app build/test on macOS runner.
- **Gate:** `uv run pytest` and `xcodebuild test` green locally.

### B1 — Measure and de-risk (spec M0, image only)
- Download Z-Image Turbo and FLUX.2 Klein 4B (q4/q8); confirm licenses and record them.
- Read mflux internals: can text encoder / transformer / VAE be loaded and freed separately? Document the hooks needed.
- Build `tools/nohang` (ballast + sampler) and `dmlx bench` (load time, time to first image, s/step, peak memory, pressure, swap).
- Run the matrix: {model} × {t2i, edit} × {eager, sequential} × {24 GB native, 16 GB emulation}.
- Packaging spike: bundle python-build-standalone + mlx + mflux into a test `.app`, ad-hoc sign, launch from Finder; record size and cold start.
- **Gate (go/no-go):** both defaults complete t2i and edit under 16 GB emulation without critical pressure in at least one strategy. Output: `docs/benchmarks/b1.md`. If it fails, we stop and fix the strategy before any UI work.

### B2 — Engine v0
- Model registry from YAML specs; component cache with LRU + pin; mflux adapter (whole-model first, per-component where B1 found hooks); fake adapter for tests.
- Memory manager: admission, strategies, pressure guard, MLX limits.
- Scheduler: queue, single GPU worker, cancel, queue-aware "keep warm" for active sessions.
- API: `POST /v1/images/generations`, `POST /v1/images/edits` (`source_id`, `reference_ids`), `GET /v1/results/{id}`, SSE events (`queued`, `loading`, `encoding`, `step{preview}`, `decoding`, `done{timings, peak_mem, strategy, loads}`), `/v1/models`, `/admin/models/{id}/pull|load|unload`, `/admin/memory`, `/health`.
- Prompt-embedding cache (RAM); step previews via cheap latent→RGB projection.
- **Gate:** no-hang test passes for all defaults on both tiers; warm edit shows `loads: 0`; OpenAI Python SDK can call generations unchanged.

### B3 — App v0 (the prototype, real)
- Engine lifecycle (launch, health, restart, logs view in Settings).
- SQLite store: sessions, turns, versions (each version stores full request + seed + result id).
- Views: sidebar, thread (pinned to newest), composer with route preview, result card with live preview blur-to-sharp, version strip, actions, context chips, toasts, empty state.
- Routing rules as a pure, unit-tested module (same rules as the prototype).
- Onboarding: RAM tier detection, recommended models, download progress/resume, first image.
- Readable inline errors ("Not enough free memory — close some apps or switch to Klein 4B").
- **Gate:** I drive the app (launch, type, screenshot via `screencapture`) through a scripted 10-turn session on 16 GB emulation; screenshots shared with you for review.

### B4 — Hardening
- Phase-free and sequential strategies fully wired; strategy shown in result metadata.
- Crash recovery: kill engine mid-job → turn shows failure → auto-restart → next turn works.
- Qwen-Image-Edit q4 for ≥32 GB tier (hidden below).
- Golden-image tests (fixed seed) per model; memory regression thresholds stored in repo.
- Performance pass: cold start, first-image latency, UI responsiveness during jobs.
- Stretch: brush mask edits if Klein inpainting via mflux proves reliable in B1/B4.
- **Gate:** full beta exit checklist except real-16 GB and release items.

### B5 — Beta release
- `tools/bundle_engine.sh` + `release.sh`: bundled interpreter, ad-hoc signed `.app`, zip, checksums.
- Docs: README (install, Gatekeeper step, requirements), CONTRIBUTING, architecture overview, model license table, known issues, API reference.
- Issue templates (bug with RAM tier + logs, model request).
- Real 16 GB validation run.
- **Gate:** beta exit checklist complete → tag `v0.1.0-beta`, publish GitHub release.

## 5. How we work

- I implement, test, benchmark, and screenshot. I commit in small, scoped commits per milestone task.
- At each gate I report: what was built, test results, benchmark numbers, screenshots, open issues. You approve or redirect.
- I ask before: large model downloads, installing system tools (e.g. `brew install xcodegen`), creating/pushing the GitHub repo, publishing a release.

## 6. What I need from you

| When | What |
|---|---|
| B0 | Approval to `brew install xcodegen` and add GRDB via SwiftPM. GitHub repo name/owner (or keep local until B5). |
| B1 | Approval for ~25–35 GB of model downloads. |
| B3 | Screen Recording permission for the terminal so I can screenshot the running app. |
| B5 | A real 16 GB Mac run (borrowed machine or one tester). Optional: Apple Developer ID if you want a notarized build instead of the Gatekeeper workaround. |
| Any time | Product calls when a choice changes the UX (I'll bring a recommendation). |

## 7. Risks

| Risk | Plan |
|---|---|
| mflux has no per-component load/free hooks | Wrap whole-model in B2; implement component loading in our adapter using mflux modules directly; offer the change upstream. |
| 16 GB emulation is optimistic vs. real hardware | Conservative reserve; real-device run is a release gate. |
| Bundled Python breaks under Gatekeeper/hardened runtime | Packaging spike in B1, before the app depends on it. |
| GPU contention still stutters the UI | Per-step yields + responsiveness setting; measured by the jitter probe. |
| Upstream mflux changes break the adapter | Pin exact version; adapter contract tests; upgrade deliberately. |
| Scope creep (video, masks) | Out of scope until the beta tag; tracked as post-beta issues. |
