# dMLX

Use chat to make and edit images fully on your Mac. dMLX is a native macOS app and a local engine.
The engine uses [MLX](https://github.com/ml-explore/mlx) and [mflux](https://github.com/filipstrand/mflux).
A small agent on your Mac plans each message.

**You can use your Mac while dMLX works.**

> **Status: v0.1 beta 2.** dMLX makes only images (it generates and edits them). Video is planned.
> The app is not notarized yet.
> Refer to the [release notes](docs/RELEASE_NOTES.md) and the [known issues](docs/KNOWN_ISSUES.md).

<p align="center">
  <img src="docs/screenshots/chat-edit-versions.jpg" width="760"
       alt="A dMLX chat. The message 'make the boat red' changes a wooden rowboat on a misty mountain lake to red. The result is Image 2, 'Edited from image 1', with the versions v1 and v2. The screenshot also shows the Variations, Crop / Draw and Save buttons, the suggestion chips, and the message box with the size and model menus.">
</p>

## Highlights

- **Use chat to make and edit images.** Describe an image. Then write in simple words what to change
  ("make the boat red"). Each image has a number. Thus you can write "image 1" or "the first one".
- **An agent plans each message.** A vision-language model on your Mac reads the chat and examines the
  images. Then it decides: a new image or an edit, the image to change, the images to use as references,
  and the correct image model.
- **The agent replies in the chat.** If a message is not clear, the agent asks one short question. If you
  send a greeting or a question about the app, the agent replies and does not make an image.
- **Your Mac stays available for use.** The engine does a check of the memory before each image job. If
  memory is low, the engine loads models in parts. The engine stops a job before macOS starts to swap.
  dMLX is for Macs with 16 GB of memory or more.
- **The app keeps all versions.** It does not overwrite an image. Variations show as v1, v2, v3. You can
  edit each version. You can also edit your prompt and run it again.
- **Private and offline.** After you download the models, all work occurs on your Mac. You do not need an
  account or a cloud service.
- **A local API.** The engine has an OpenAI-compatible images API for other apps and agents.

## Edit with Crop / Draw

You can write in simple words how to change a result. You can also show your request on the image first.
Open **Crop / Draw** on an image. Crop the image. Paint marks on it. Erase the marks that are not correct.
Attach the copy. Then write what to do. The app does not change the original image.

| 1 · Crop | 2 · Paint a mark | 3 · Erase an incorrect mark |
|---|---|---|
| <img src="docs/screenshots/edit-1-crop.jpg" alt="The Crop / Draw editor with the Crop tool. A small dashed crop frame surrounds a white teapot on a wooden table. The presets are Full, Square, 3:2 and 2:3."> | <img src="docs/screenshots/edit-2-paint.jpg" alt="The Paint tool. A red circle surrounds the body of the teapot. An unwanted red line also goes onto the table."> | <img src="docs/screenshots/edit-3-erase.jpg" alt="The Erase tool. The tool erased the unwanted line, which shows faded. The red circle stays. 'Drag to erase paint. The original image is never erased.'"> |

**4 · Attach and ask.** **Attach edited image** puts the prepared copy in the message box. The agent reads
the marks together with your words.

<img src="docs/screenshots/edit-4-attach.jpg" alt="The message box with 'Prepared image 1' attached and the message 'paint a blue and white floral pattern on the teapot where I circled in red, and remove the red marks'. Above the message box: 'Will edit the attached image · FLUX.2 Klein 4B'.">

| Before | After: "paint a blue and white floral pattern on the teapot where I circled in red, and remove the red marks" |
|---|---|
| <img src="docs/screenshots/edit-before.jpg" alt="Image 1: a plain white ceramic teapot on a wooden table."> | <img src="docs/screenshots/edit-after.jpg" alt="Image 2: the same teapot with a closer crop. A blue and white floral pattern now covers it. No red marks remain."> |

## Tour of the app

All screenshots show the app on a 24 GB Mac. The agent model is Qwen3-VL 8B.

| **If the agent is not sure, it asks.** With two images in the chat, the message "make them match" gets a question. The agent does not guess. You can answer the question, use the shortcuts, or send a different request. | **Then it uses the two images.** The message "give the teapot the same weathered wood as the boat" edits the teapot. It uses the boat as a reference ("Edited from image 2 · + 1 reference"). |
|---|---|
| <img src="docs/screenshots/agent-asks.jpg" alt="The message 'make them match' gets this question from the agent: 'Could you clarify which two images you want to match, or if you mean to adjust the current image to match a previous one?'. Create new, Edit current and Cancel show above the message box."> | <img src="docs/screenshots/answer-with-reference.jpg" alt="The message 'give the teapot the same weathered wood as the boat' gives Image 3, edited from image 2 with one reference. The ceramic teapot is now made of carved, dark, weathered wood."> |
| **It finds older images.** The teapot is the latest image. The message "make the boat red" edits the boat in image 1. If the message refers to older images, the agent examines them. | **It replies to casual messages.** If you send a greeting or thanks, the agent replies in the chat. It does not make an image. |
| <img src="docs/screenshots/finds-older-image.jpg" alt="The message 'make the boat red' gives Image 4, 'Edited from image 1'. The wooden rowboat on the misty mountain lake is now red."> | <img src="docs/screenshots/agent-replies.jpg" alt="The message 'hello' gets a greeting from the agent. Then the app makes an image of a sleeping fox in the snow. The message 'thanks, that's lovely!' gets the reply 'You're welcome! I'm glad you like the image of the sleeping fox'."> |
| **Use your own pictures.** You can attach a maximum of 3 files with the paper clip or with drag and drop. The message "a lighthouse on a cliff at dusk, painted like this" uses the attached watercolor as the style. | **Use Crop / Draw on a result.** You can crop the image, paint marks, and erase marks. Then you attach the copy. For the full procedure, refer to [Edit with Crop / Draw](#edit-with-crop--draw). |
| <img src="docs/screenshots/attach-composer.jpg" alt="The message box with watercolor-forest.png attached and the message 'a lighthouse on a cliff at dusk, painted like this'."><br><img src="docs/screenshots/attach-result.jpg" alt="Image 1, 'Edited from the attached image': a lighthouse on a cliff at dusk in soft watercolor."> | <img src="docs/screenshots/edit-2-paint.jpg" alt="The Crop / Draw editor with a red circle painted around a teapot."> |
| **Auto selects the model, and the engine first does a memory check.** For a title in the image, the agent selected Qwen Image 2.1, because this model is the best at words. This model needed 9.5 GB, but only 9.2 GB was free. Thus the engine refused the job, and the Mac did not become slow. **Try anyway** runs the job in the mode that uses the least memory. | **The app keeps your message while the agent plans.** After you send a message, the chat shows it with "Planning…" while the agent decides. You can continue to type. |
| <img src="docs/screenshots/memory-check.jpg" alt="The message 'write the title NIGHT BLADE in the banner at the top' gives Image 2 with Qwen Image 2.1 Base Q4 and this text: 'needs about 9.5 GB but only 9.2 GB can be used without slowing down your Mac'. Retry and Try anyway show below the text."> | <img src="docs/screenshots/planning.jpg" alt="The message 'make them match' after you send it, with 'Planning…' below it."> |
| **Add models.** Use the search field to find community versions of the supported families. You can also paste any Hugging Face repo. dMLX does a check of the repo before it downloads it. | **Control the models.** You can see the installed models, remove models, and set the default model. If you set Auto, the agent selects the model. |
| <img src="docs/screenshots/add-model.jpg" alt="The Add a model window. It has a search field and community versions of FLUX.2 Klein, Z-Image Turbo and other models, with Add and Download."> | <img src="docs/screenshots/settings-models.jpg" alt="Settings, Models. FLUX.2 Klein 4B, Qwen Image 2.1 Base Q4 and Z-Image Turbo are downloaded, each with Remove. The default model is Auto. The window also shows Add Model."> |

## The agent

The agent plans each message that you send. The agent:

- **decides the type of request**: a new image, an edit, or no image (then the agent replies with text);
- **finds the correct image**: it sees the latest image. If your message refers to older items ("the boat",
  "the one with the cat", "undo the hat"), the agent can *look* at older images or *read* older messages;
- **writes the prompt**: a full description for a new image, or an edit instruction that is complete by
  itself. Optionally, the agent expands the edit instruction into a detailed description of the result.
  Edit models follow such a description better;
- **selects the image model** if the menu is set to **Auto**. It uses the needs of the message (exact words
  in the image, several references) and the free memory. The agent never overrides a model that you select.

If the agent is not sure, it asks one short question. You can answer the question or send a different
request. After you send a message, the chat shows it with **Planning…**. The app never discards the
message. You can continue to type.

The agent makes its decisions from its instructions and tools, not from fixed rules in the code. The code
only refuses a request that cannot run, for example an image that does not exist or an edit with no image.

<p align="center">
  <img src="docs/screenshots/settings-agent.jpg" width="560"
       alt="Settings, Intelligence, Agent. Qwen3-VL 8B is in use (5.8 GB, recommended for 24 GB and more). Qwen3.5 9B and Ministral 3 3B are ready (recommended for 16 GB).">
  <br><sub>Settings → Intelligence → Agent</sub>
</p>

| Agent model | Download | Peak memory | Per message | Real-image checks | Notes |
|---|---|---|---|---|---|
| Qwen3-VL 8B | 5.8 GB | 6.2 GB | ~7 s | 10/10 | Default for 24 GB and more |
| Qwen3.5 9B | 6.0 GB | 6.8 GB | ~7.5 s | 9/10 | Best result on the routing eval |
| Ministral 3 3B | 2.8 GB | 4.2 GB | ~5.5 s | 10/10 | Default for less than 24 GB |
| Qwen3-VL 4B | 3.1 GB | 3.7 GB | ~3.5 s | 6/10 | Uses the least memory; seldom examines older images |

These values are from an M5 with 24 GB (2026-11-01). The engine loads the agent only while the agent plans.
The engine removes the agent from memory before the image job starts. If there is no agent model, the engine
uses simple rules to route messages. For more data, refer to
[`docs/benchmarks/agent-models.md`](docs/benchmarks/agent-models.md) and [`tasks/`](tasks/).

## Requirements

- An Apple Silicon Mac (M1 or newer), with macOS 15 or later
- 16 GB of memory or more
- Disk space: ~5 GB for the default image model, and 2.8–6.0 GB for an agent model

## Install

1. Download `dMLX-<version>-macos-arm64.zip` from [Releases](https://github.com/rootonchair/dmlx/releases).
   Then unzip the file.
2. Move `dMLX.app` to Applications.
3. The beta is not notarized. The first time that you open the app, do this: **right-click the app → Open → Open**.
   As an alternative, run `xattr -dr com.apple.quarantine /Applications/dMLX.app`.
4. When you start dMLX for the first time, it offers to download an image model. Select an agent model in
   **Settings → Intelligence → Agent**. After this step, dMLX operates offline.

## How to use dMLX

- **To make an image:** describe it. **To change an image:** write what to change. **To start again:**
  describe a different image.
- **To use an older image:** refer to it in your message ("make image 1 black and white", "put the teapot
  from the first one in the boat").
- **To use your own pictures:** attach a maximum of 3 pictures from your Mac with the paper clip. As an
  alternative, drop the pictures on the message box.
- **Variations** adds v2, v3… to a result. **Crop / Draw** crops a result or paints on it, and then attaches
  the copy. The save button writes the image to Pictures/dMLX with its prompt and seed.
- **Edit Prompt…** on one of your messages runs the image of that message again with the new words.
- While the app makes an image, the send button changes to **Stop**. You can write your next message during
  this time.

## Image models

| Model | Does | License | Download | Notes |
|---|---|---|---|---|
| FLUX.2 Klein 4B (4-bit) | generate, edit, a maximum of 3 images | Apache-2.0 | 4.6 GB | The default; ~20–40 s per image |
| Z-Image Turbo (4-bit) | generate | Apache-2.0 | 5.9 GB | Optional; more photographic detail |
| Qwen Image 2.1 Base (4-bit) | generate, edit, a maximum of 10 images in a set order | Qwen Research License (non-commercial) | ~10.6 GB | Experimental; best for exact words in the image; ~4 min per image; needs ~9.5 GB of free memory |

**Settings → Models → Add Model…** shows community versions of the supported families (FLUX.2 Klein
4B/9B, Z-Image Turbo/base, Qwen Image 2.1 Base). It also accepts any Hugging Face repo in mflux format.
Before dMLX downloads a repo, it does a check of the family, size and license. If a license has
restrictions, dMLX asks you to accept it. **Remove…** deletes the files of a model. dMLX puts the weights in
the standard Hugging Face cache (`~/.cache/huggingface`). The app bundle never contains weights. In Auto,
the engine uses Qwen Image 2.1 only if the agent selects it for a message.

## How dMLX controls memory

Before each job, the engine calculates the memory that the job needs. The engine compares this value with
the memory that it can use *without* a need for macOS to compress or swap other apps. The engine also
keeps a reserve of memory for the system. Then it selects one of these strategies:

| Strategy | When | What stays loaded |
|---|---|---|
| eager | a large quantity of free memory | all parts; they stay in memory for the next requests |
| phase-free | some free memory | image model and decoder; the engine removes the text encoder after it reads the prompt |
| sequential | low free memory | one large part at a time |
| streamed | very low free memory | one block at a time; the engine reads each block from disk immediately before it runs (~25% slower; a 1024² image in ~1.5 GB) |

On Macs with less than 32 GB, FLUX.2 Klein decodes in exact bands (about 1.1 GB at 1024², not 8.6 GB). If
no strategy fits, the engine refuses the job and shows a clear message. **Try anyway** runs the job in the
mode that uses the least memory. While a job runs, the engine monitors memory pressure and swap-ins. The
engine stops the job before the Mac starts to hang. The engine removes the agent from memory before each
image job. Then the engine waits until that memory is free again. For more data, refer to
[`dmlx-spec.md`](dmlx-spec.md) §8, [`docs/benchmarks/b1.md`](docs/benchmarks/b1.md),
[`b2.md`](docs/benchmarks/b2.md) and [`agent-16gb.md`](docs/benchmarks/agent-16gb.md).

## Build from source

```bash
# engine (Python 3.13, uv)
cd engine && uv sync --extra local-llm && uv run pytest

# app (Swift 6 toolchain / Xcode 16+)
cd app && swift test
tools/dev/run_app.sh            # runs the app; the engine starts from ../engine via uv

# release bundle (relocatable Python and engine inside the .app, ad-hoc signed)
tools/bundle_app.sh --zip
```

To score the agent with the routing eval and placeholder images, run this command:

```bash
cd engine && uv run --extra local-llm python -m dmlx.prompting.agent_eval --model qwen3-vl-8b-instruct-4bit
```

## Engine API

To run the engine without the app, use these commands:

```bash
cd engine && uv run dmlx serve --port 8642
curl -s localhost:8642/v1/images/generations -H 'content-type: application/json' \
  -d '{"prompt":"a rowboat on a lake at dawn","size":"1024x1024"}' | jq '.dmlx'
```

| Method | Path | Notes |
|---|---|---|
| POST | `/v1/images/generations` | OpenAI-compatible; `stream: true` returns server-sent events |
| POST | `/v1/images/edits` | multipart (OpenAI style) or JSON with `source_id` / `reference_ids` |
| POST | `/v1/workflows` | workflow with validation and capability checks; supports streaming |
| POST | `/v1/workflows/route` | what to do with a chat message; `"plan": true` asks the agent |
| GET | `/v1/capabilities` | operations and input limits of each model |
| POST | `/v1/prompts/compile` | prompt enhancement preview |
| POST | `/v1/uploads` | stores an image and gives an id for later references |
| GET | `/v1/results/{id}` · `/meta` | PNG with the prompt and seed in the file · its metadata |
| GET/DELETE | `/v1/jobs/{id}` · `/events` | status · cancel · events |
| POST | `/v1/jobs/{id}/retry` | runs a failed workflow again from the failed step |
| GET | `/v1/models` | catalog with downloaded / loaded state |
| POST | `/admin/models/{id}/pull` · `load` · `unload` | download, pin, free |
| GET | `/admin/memory` | memory budget, pressure, loaded parts |
| GET/POST | `/admin/prompt-assistant` · `/select` · `/pull` | agent models: status, select, download |

Events: `queued`, `plan`, `loading{component}`, `encoding`, `step{step,total}`, `decoding`,
`image{result_id}`, `done{result_ids, report}`, `error{code,message}`, `cancelled`.

### Configuration

The app sets these values in **Settings → Intelligence**. If you use the engine without the app, set
these environment variables.

| Variable | Use |
|---|---|
| `DMLX_PROMPT_ASSISTANT_MODEL` | Agent model id (for example, `ministral-3-3b-instruct-2512-4bit`) |
| `DMLX_PROMPT_ASSISTANT=off` | Sets the agent to off |
| `DMLX_ROUTER_BASE_URL`, `DMLX_ROUTER_MODEL`, `DMLX_ROUTER_API_KEY`, `DMLX_ROUTER_TIMEOUT` | Routes messages without the agent, through any OpenAI-compatible chat server (HTTPS, or HTTP on loopback) |
| `DMLX_ROUTER_PROVIDER=local-llm`, `DMLX_LOCAL_LLM_MODEL_PATH`, `DMLX_LOCAL_LLM_REVISION` | Routes messages without the agent, through a local Qwen checkpoint ("Local Qwen MLX" in the app) |
| `DMLX_PROMPT_BASE_URL`, `DMLX_PROMPT_MODEL`, `DMLX_PROMPT_API_KEY`, `DMLX_PROMPT_TIMEOUT` | Prompt enhancement through an OpenAI-compatible server |

If you set API keys in the app, the app keeps them in the macOS Keychain. When the agent plans, the engine
also does a check of its answer. The images that the answer names must exist. An edit must use 1–3 images.
The answer can use only workflows that an installed model can run. If the answer is still not valid after
one retry, the app asks the user and does not guess.

## Project layout

```
engine/   Python engine (FastAPI + MLX via mflux), the planning agent, evals, tests
app/      SwiftUI app (DMLXKit library + app target), tests
tools/    bundle_app.sh (release), dev/ helpers (run_app.sh, the scripted dev harness)
docs/     release notes, known issues, benchmarks, screenshots
tasks/    open and finished agent tasks, with their results
```

## License

All code in this repository uses the Apache-2.0 license. Model weights have their own licenses. The default
models use Apache-2.0. The optional Qwen Image 2.1 uses the Qwen Research License (non-commercial).
