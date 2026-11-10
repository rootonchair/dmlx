# dMLX v0.1.0-beta5

Layout for posters and banners, the agent anywhere you like, and a long list of fixes.

## Highlights

- **Layout.** MOSS Image, FIBO 2 and Ideogram 4 can be added from Settings → Models. The agent writes each
  model's detailed caption for you (you never see JSON). In **Crop / Draw → Layout** you draw, move and
  describe boxes, then **Redraw with layout** puts the title, date or logo where you placed it.
- **Choose where the agent runs.** At setup, or later from **dMLX → Choose Models…**: on this Mac, or
  Claude, ChatGPT, Ollama or any OpenAI-compatible service. Keys stay in the Keychain.
- **FLUX.2 Klein 9B** as a built-in model, for new images and edits with up to 3 images.
- **Auto picks by strengths.** Photos go to Z-Image and words in images to Qwen Image when they are
  downloaded, and Auto never picks a model that will not fit in memory when another one would. The hint
  in the message box names the model that will run and about how much memory it needs.
- **Stop frees memory.** A stopped image frees the model's memory, and the next image no longer runs
  slowly because the old model is still loaded.
- **Downloads you can stop and resume.** Cancel a download, quit, and pick it up later from where it
  stopped. Each file is checked when it finishes, leftover files are cleaned up, and Qwen Image 2.1
  downloads all its parts up front. Sizes are shown in GB.
- **Models in dMLX's own folder**, and **Uninstall dMLX…** removes every model, chat and setting.
- **Attaching images.** iPhone photos (HEIC) attach, files dropped on the message box attach, a message
  sent while an image uploads waits for it, and files that are not images get a clear message.
- **Fixes.** Quit works while a setup sheet is open, and asks first during a download. The setup sheet
  fits above the Dock. No more clipped or hidden controls in short windows. Fewer steps for FIBO 2 and
  Ideogram 4 (about 3 minutes faster per poster). An image no longer stops while the Mac is only reading
  apps back from swap.

## Requirements

Apple Silicon, macOS 15+, 16 GB memory or more. About 7–11 GB of disk for the image model and the agent.

## Install

Open the `.dmg` and drag dMLX onto the Applications folder (a `.zip` of the same app is also attached), then
right-click → Open the first time (beta builds are not notarized). The first-run setup downloads the models.

## Known issues

See [Known issues](KNOWN_ISSUES.md).

---

# dMLX v0.1.0-beta4

A guide for new users.

## Highlights

- **Tips after the first-run setup.** Short tips point at what they explain, one at a time: the message
  box, the paper clip for your own images, the size menu, the control that chooses between a new image and
  an edit (and the model), and **New creation**. After your first image, tips explain versions,
  **Variations** and **Crop / Draw**, and the editor explains **Crop**, **Paint** and **Erase**. Each tip
  shows once.
- **Help menu.** **dMLX Guide** opens the online guide, and **Show Tips Again** starts the tips over.
- **The new image's buttons stay in view.** When an image is done, the chat scrolls so that
  **Variations**, **Crop / Draw** and **Save** show below it.

## Requirements

Apple Silicon, macOS 15+, 16 GB memory or more. About 7–11 GB of disk for the image model and the agent.

## Install

Open the `.dmg` and drag dMLX onto the Applications folder (a `.zip` of the same app is also attached), then
right-click → Open the first time (beta builds are not notarized). The first-run setup downloads the models.

## Known issues

See [Known issues](KNOWN_ISSUES.md).

---

# dMLX v0.1.0-beta3

A guided first start, six more image models, and a new look.

## Highlights

- **First-run setup.** On first start dMLX recommends an image model and an agent model for your Mac's
  memory, shows the total download, and checks that there is enough free disk space. You can change either
  model, or skip the agent. While the models download you can continue in the background; you can start
  as soon as the image model is ready. The setup shows each model's size before it is downloaded, and a
  message sent before the image model is ready stays in the box, with the download's progress.
- **More image models.** Settings → Models → Add Model now offers ERNIE Image Turbo, MOSS Image, FLUX.3
  Klein, Boogu Image Turbo, Iris 3B Turbo and Krea 2 Turbo. Every family runs through one adapter and
  gives the same images as mflux's own pipelines. Some of these take minutes per image.
- **New look.** The Still Water logo and app icon, blue colors, and **Settings → General → Appearance**:
  System, Light or Dark.
- **Crop / Draw.** Erasing paint on the canvas now matches the exported image. The agent knows what a
  prepared (marked) image is, and each edit is labeled with the image it was made from.
- **Chat.** A chat opens at the latest message, and it scrolls to a message being planned and to the
  agent's question.
- **Memory.** Sizes under 1 MP are planned as 1 MP, so small images no longer reserve too little memory.
  The text encoder is freed when encoding ends (lower peaks for FLUX.3 and MOSS).
- **mflux 0.24.**

## Requirements

Apple Silicon, macOS 15+, 16 GB memory or more. About 7–11 GB of disk for the image model and the agent.

## Install

Open the `.dmg` and drag dMLX onto the Applications folder (a `.zip` of the same app is also attached), then
right-click → Open the first time (beta builds are not notarized). The first-run setup downloads the models.

## Known issues

See [Known issues](KNOWN_ISSUES.md).

---

# dMLX v0.1.0-beta2

Adds a planning agent: a small vision-language model on your Mac reads every message, looks at the
images, and decides what to do.

## Highlights

- **Planning agent.** Decides whether a message asks for a new image or an edit, and which images to edit
  or use as references ("make image 1 black and white", "put the teapot from the first one in the
  boat"). It can look at older images in the chat and read further back when it needs to. For an edit it
  can write a detailed description of the result for the image model.
- **It talks in the chat.** A greeting or a question about the app gets a short reply; when a message is
  unclear the agent asks one question. Answer it, or just ask for something else.
- **Auto picks the image model.** In Auto the agent chooses among your installed models by what the
  message needs and the free memory; a model you pick yourself is never overridden.
- **Choose the agent model.** Settings → Intelligence → Agent: Qwen3-VL 8B (default from 24 GB), Qwen3.5 9B,
  Ministral 3 3B (default under 24 GB) or Qwen3-VL 4B; 2.8–6.0 GB downloads. It is loaded only while it
  plans and freed before the image job starts.
- **Messages are never dropped.** A sent message shows in the chat with "Planning…"; you can keep typing.
  A message sent from another chat waits for the running image, then is planned.
- **Qwen Image 2.1 (experimental, research license).** Can be added in Settings → Models: generate and
  edit with up to 10 ordered references, best at exact words in the image. It needs about 9.5 GB free
  and takes about 4 minutes per image; in Auto only the agent picks it.
- **Removed:** the per-image Attach button. Refer to images by number instead, or attach files from
  your Mac.

## Requirements

Apple Silicon, macOS 15+, 16 GB memory or more. Plus 2.8–6.0 GB disk for an agent model.

## Install

Open the `.dmg` and drag dMLX onto the Applications folder (a `.zip` of the same app is also attached), then
right-click → Open the first time (beta builds are not notarized).
Download an agent model in Settings → Intelligence → Agent; without one, messages are routed by rules.

## Known issues

See [Known issues](KNOWN_ISSUES.md).

---

# dMLX v0.1.0-beta1

First public beta: chat to create and edit images, fully on your Mac.

## Highlights

- **Chat-style creation.** Describe an image; follow up in plain words to edit the latest result.
  The composer shows what will happen ("Will edit image 2 · FLUX.2 Klein 4B") before you send.
- **Model menu in the message box.** Auto by default, or pick a model; the line above the box says
  which model the next message uses, and notes when it has to load a different one.
- **Add and remove models.** Any FLUX.2 Klein or Z-Image model in mflux format, from a list of
  community versions or a pasted Hugging Face link; checked (family, size, license) before download.
- **Versions.** Variations stack as v1, v2, v3 on the same turn; edit any version; use any result
  as a reference for the next edit.
- **Never hangs your Mac.** Memory is checked before every job; model parts load one at a time when
  memory is tight, and when it is very tight the model streams from disk block by block (a new
  1024×1024 image in about 1.5 GB). Below 32 GB the decoder runs in exact low-memory bands; jobs
  stop automatically if macOS memory pressure turns critical or apps start being read back from swap.
- **Warm follow-ups.** Models stay loaded between turns and prompt embeddings are cached; a
  follow-up edit does zero model loads.
- **Recovers by itself.** If the engine crashes it restarts; the interrupted turn offers Retry.
- **Local OpenAI-compatible API** for other apps and agents (`/v1/images/generations`, `/v1/images/edits`).

## Models

FLUX.2 Klein 4B (default, generate + edit, Apache-2.0) and Z-Image Turbo (optional, Apache-2.0),
4-bit packs from `mflux-community`.

## Requirements

Apple Silicon, macOS 15+, 16 GB memory or more.

## Install

Unzip, move to Applications, then right-click → Open the first time (beta builds are not notarized).

## Known issues

See [Known issues](KNOWN_ISSUES.md). Most important: the 16 GB target still needs validation on a real
16 GB Mac — reports welcome.
