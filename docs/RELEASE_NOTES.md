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

See `docs/KNOWN_ISSUES.md`. Most important: the 16 GB target still needs validation on a real
16 GB Mac — reports welcome.
