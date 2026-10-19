# dMLX

Create and edit images by chatting, fully on your Mac. dMLX is a native macOS app plus a local engine built on [MLX](https://github.com/ml-explore/mlx) and [mflux](https://github.com/filipstrand/mflux). Its main promise: **it never makes your Mac hang**, even on a 16 GB machine.

> Status: **v0.1 beta**. Images only (generate + edit). Video is planned, not included. See [known issues](docs/KNOWN_ISSUES.md).

## What it does

- **Chat to create.** Describe an image; follow up in plain words ("make it night") to edit the latest result. No settings to pick first: the model menu in the message box starts on **Auto**, and you can pick a specific model there.
- **Versions, not overwrites.** Variations stack as v1, v2, v3 on the same turn; edit any version.
- **References.** Attach up to 3 images (or use a previous result) to guide an edit.
- **Stays responsive.** The engine checks free memory before every job, loads model parts one at a time when memory is tight, and in the tightest case streams the model from disk block by block (a new 1024×1024 image in about 1.5 GB). It stops a job if macOS memory pressure turns critical.
- **Warm follow-ups.** Models stay loaded between turns and prompt embeddings are cached, so follow-up edits skip loading entirely.
- **Local API.** The engine speaks an OpenAI-compatible images API, so other apps and agents can use it too.

## Requirements

- Apple Silicon Mac (M1 or newer), macOS 15 or later
- 16 GB memory or more
- ~6 GB free disk for the default model (FLUX.2 Klein 4B, 4-bit)

## Install (beta)

1. Download `dMLX-<version>-macos-arm64.zip` from Releases and unzip it.
2. Move `dMLX.app` to Applications.
3. The beta is not notarized. The first time, **right-click the app → Open → Open**. (Or run `xattr -dr com.apple.quarantine /Applications/dMLX.app`.)
4. On first launch dMLX offers to download a model. After that it works offline.

## Models

| Model | Does | License | Download | Notes |
|---|---|---|---|---|
| FLUX.2 Klein 4B (4-bit) | generate, edit, up to 3 references | Apache-2.0 | 4.6 GB | Default for everything |
| Z-Image Turbo (4-bit) | generate | Apache-2.0 | 5.9 GB | Optional; finer detail, slower |

Weights are downloaded from Hugging Face into the standard cache (`~/.cache/huggingface`) and are never bundled.

**Adding and removing models.** Settings → Models → **Add Model…** lists the community versions of the supported families (FLUX.2 Klein 4B/9B, base and distilled; Z-Image Turbo and base, 3- to 16-bit) and accepts any Hugging Face repo or link in mflux format. dMLX checks the repo before downloading anything (family, size, license) and warns when a license may not allow commercial use (FLUX.2 Klein 9B is non-commercial). **Remove…** deletes a model's files from the Hugging Face cache; built-in models stay listed and can be downloaded again.

## How memory is handled

Before each job the engine compares what the job needs with memory that can be used *without* making macOS compress or swap other apps (minus a reserve for the system), then picks a strategy:

| Strategy | When | What stays loaded |
|---|---|---|
| eager | plenty of room | everything, kept warm |
| phase-free | some room | image model + decoder; the text encoder is freed after reading the prompt |
| sequential | tight | one large part at a time |
| streamed | very tight | one block of a large part at a time, read from disk just before it runs (~25% slower) |

Below 32 GB, FLUX.2 Klein decodes with an exact, banded decoder (about 1.1 GB at 1024×1024 instead of 8.6 GB). If nothing fits, the job is refused with a readable message ("Try anyway" runs it in low-memory mode and stops automatically if the Mac gets slow). While a job runs, the engine watches memory pressure and swap and stops before the Mac starts to hang. Details: [`dmlx-spec.md`](dmlx-spec.md) §8, [`docs/benchmarks/b1.md`](docs/benchmarks/b1.md) and [`docs/benchmarks/b2.md`](docs/benchmarks/b2.md).

## Build from source

```bash
# engine (Python 3.13, uv)
cd engine && uv sync && uv run pytest

# app (Swift 6 toolchain / Xcode 16+)
cd app && swift test
tools/dev/run_app.sh            # runs the app; the engine starts from ../engine via uv

# release bundle (relocatable Python + engine inside the .app, ad-hoc signed)
tools/bundle_app.sh --zip
```

Run only the engine:

```bash
cd engine && uv run dmlx serve --port 8642
curl -s localhost:8642/v1/images/generations -H 'content-type: application/json' \
  -d '{"prompt":"a rowboat on a lake at dawn","size":"1024x1024"}' | jq '.dmlx'
```

## API (engine)

| Method | Path | Notes |
|---|---|---|
| POST | `/v1/images/generations` | OpenAI-compatible; `stream: true` returns server-sent events |
| POST | `/v1/images/edits` | multipart (OpenAI style) or JSON with `source_id` / `reference_ids` |
| POST | `/v1/uploads` | store an image, get an id to reference later |
| GET | `/v1/results/{id}` · `/meta` | PNG with prompt/seed embedded · its metadata |
| GET/DELETE | `/v1/jobs/{id}` · `/events` | status · cancel · SSE |
| GET | `/v1/models` | catalog with downloaded / resident state |
| POST | `/admin/models/{id}/pull` · `load` · `unload` | download, pin, free |
| GET | `/admin/memory` | memory budget, pressure, resident components |

Events: `queued`, `plan`, `loading{component}`, `encoding`, `step{step,total}`, `decoding`, `image{result_id}`, `done{result_ids, report}`, `error{code,message}`, `cancelled`.

## Project layout

```
engine/   Python engine (FastAPI + MLX via mflux), tests
app/      SwiftUI app (DMLXKit library + app target), tests
tools/    bundle_app.sh (release), dev/ helpers
docs/     benchmarks and notes
```

## License

Apache-2.0 for all code in this repository. Model weights have their own licenses (both defaults are Apache-2.0).
