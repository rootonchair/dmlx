# Model families on the generic adapter (task 15, Nov 2026)

Machine: Apple M5, 24 GB, macOS 26.5, mflux 0.23.0 (0.24.0 from the Nov 6 sections). All models are 4-bit packs from mflux-community.
Prompt "a red fox sitting in fresh snow, morning light" (edits: "make it night with a full moon, keep the
fox the same"), seed 7, 512 px unless stated.

## Moving the built-in families to the generic adapter

Each case ran through the real engine before and after the change, and the images were compared.

| Model / task | Strategy | Pixels before vs after | Peak MLX before → after |
|---|---|---|---|
| FLUX.2 Klein 4B t2i | sequential | identical | 3.14 → 3.14 GB¹ |
| FLUX.2 Klein 4B t2i | streamed | identical | 0.74 → 0.76 GB |
| FLUX.2 Klein 4B edit | sequential | identical | same per run¹ |
| FLUX.2 Klein 4B edit | streamed | identical | 2.23 → 2.23 GB |
| Qwen Image 2.1 Base t2i | sequential | identical | 5.66 → 5.66 GB |
| Qwen Image 2.1 Base edit | sequential | identical | 5.60 → 5.60 GB |
| Z-Image Turbo t2i | sequential | mean 0.44/255, max 28 | 4.04 → 4.04 GB |
| Z-Image Turbo t2i | streamed | mean 0.44/255, max 28 | 2.14 → 0.79 GB |

¹ Fresh process per run, old code (a worktree at the previous commit) against new. FLUX.2 Klein t2i:
3.14 GB in all 3 + 3 runs. Edit, run by run: 4.05 / 4.05, 2.64 / 2.64, 4.05 / 4.05 GB (old / new). The
edit peak changes between runs because the engine's prompt assistant rewrites the edit prompt, and the
rewrite differs in length.

Z-Image now uses mflux's own VAE setup, which turns on its low-memory block decoder. The old adapter
built a plain VAE. The images differ by less than 1/255 on average, and the streamed peak is lower.

## `dmlx bench-family`

Runs jobs in the sequential strategy and records, per phase, the peak above what was resident when the
phase started. Sizes under 1 MP count as 1 MP (small images have a fixed overhead), and a 10% margin is
added.

**Check against the hand-measured FLUX.2 Klein 4B numbers (`docs/benchmarks/b1.md`):**

| Number | Spec (hand-measured) | bench-family |
|---|---|---|
| activation_gb_per_mp | 1.4 | 1.5 |
| decode_banded_gb_per_mp | 1.2 | 1.2 |
| reference_activation_factor | 0.75 | 0.7 |
| decode_gb_per_mp (full) | 8.6 | 1.4 |

The full decode no longer spikes: mflux 0.23 builds the FLUX.2 VAE with its low-memory block decoder. The
spec keeps 8.6 for now; full decode is used only on Macs with 32 GB or more, so the old number only
reserves too much there.

## ERNIE Image Turbo (8 steps)

| Run | Peak MLX | Denoise | Turn |
|---|---|---|---|
| mflux `ErnieImage`, all parts loaded, 512 px | 7.18 GB | 32.8 s | — |
| Generic adapter, sequential, 512 px | 5.23 GB | 30.8 s | 37.2 s |
| Generic adapter, streamed, 512 px | 0.73 GB | 32.5 s | 38.1 s |
| Generic adapter, sequential, 1024 px | 5.56 GB | 137.8 s | 146.4 s |
| Generic adapter, streamed, 1024 px | 1.60 GB | 148.8 s | 158.4 s |

The generic adapter's images are identical to mflux's own pipeline (max difference 0).

Working memory above the resident weights (bench-family): denoise +1.03 GB at 512 px, +1.35 GB at
1024 px; decode +0.44 GB and +1.31 GB, the same in full and tiled mode. Spec: activation 1.5, decode 1.4,
tiled 1.4 GB per MP. Parts: transformer 4.21 GB, text encoder 1.80 GB, VAE 0.15 GB. Streamed, the
transformer keeps 0.32 GB and the text encoder 0.33 GB resident.

## FLUX.3 Klein (16 steps) and MOSS Image (50 steps, guidance 3)

Images against mflux's own pipelines, all parts loaded, same seed and prompt:

| Case | dMLX vs mflux |
|---|---|
| FLUX.3 Klein t2i, sequential and streamed | identical |
| FLUX.3 Klein edit, sequential | identical (with the prompt the engine's prompt assistant wrote; mflux given the same prompt) |
| MOSS Image t2i, sequential | identical |
| MOSS Image t2i, streamed | mean 1.3/255; identical to sequential with `MLX_DISABLE_COMPILE=1` (streamed runs uncompiled) |

**A leak the first measurements found.** FLUX.3 and MOSS cache their prompt encodings without evaluating
them. A lazy MLX result keeps the text encoder's weights alive, so the encoder stayed resident through the
whole denoise loop after the guard had freed it. The generic adapter now evaluates the cached encodings
when the encode phase ends. Peak, sequential, 512 px: FLUX.3 4.83 → 3.70 GB, MOSS 6.94 → 4.37 GB.

Both read transformer settings before they encode (FLUX.3's `flow_map`, MOSS's layer count), so in the
low-memory strategies the transformer loads twice per job (about 1 s). Every load goes through the guard.

| Model | Peak MLX, 512 px | Denoise per step, 1024 px | mflux all loaded, 512 px |
|---|---|---|---|
| FLUX.3 Klein, sequential | 3.70 GB | ~6.4 s (about 2 min per image; edit about 4 min) | 5.25 GB (t2i), 5.83 GB (edit) |
| FLUX.3 Klein, streamed | 1.62 GB | | |
| MOSS Image, sequential | 4.37 GB | ~15.5 s (about 13 min per image) | 8.95 GB |
| MOSS Image, streamed | 2.06 GB | | |

Working memory above the resident weights (bench-family, 1024 px):

| Model | denoise | with one reference | decode full | decode tiled | Spec (activation / full / tiled per MP) |
|---|---|---|---|---|---|
| FLUX.3 Klein | +1.06 GB | +1.52 GB | +1.31 GB | +1.31 GB | 1.2 / 1.4 / 1.4, reference factor 0.5 |
| MOSS Image | +0.86 GB | — | +7.04 GB | +2.02 GB | 1.0 / 7.4 / 2.3 |

## Boogu Image Turbo (8 steps, Nov 6)

Boogu keeps its text encoder (Qwen3-VL's language model) in `mllm/`. Specs now carry a part's folder
(`subdir`) when it differs from the part's name; the catalog takes it from mflux's weight definition.

| Run, 512 px | Peak MLX | Denoise | Turn | Image vs mflux |
|---|---|---|---|---|
| mflux `BooguImage`, all parts loaded | 13.25 GB | — | 55.4 s | reference |
| Generic adapter, sequential | 6.96 GB | 39.7 s | 42.9 s | identical (max diff 0) |
| Generic adapter, streamed | 3.14 GB | 40.4 s | 43.5 s | identical (max diff 0) |

A second run of the same prompt skipped the text encoder (prompt cache hit). Denoise takes about 5 s per
step at 512 px and about 25 s at 1024 px, so a 1024 px image takes about 3.5 minutes.

Working memory above the resident weights (bench-family, 1024 px): denoise +2.41 to +3.05 GB (two runs),
decode +8.60 GB full and +2.98 GB tiled (the FLUX.1 VAE's tiles are a fixed size, so tiled decode needs the
same at 512 px). Spec: activation 3.3, decode 9.1, tiled 3.3 GB per MP. Parts: text encoder 4.41 GB,
transformer 5.99 GB, VAE 0.15 GB; streamed, 0.59 and 1.13 GB stay resident. At 1024 px the planner needs
8.2 GB for sequential and 3.4 GB for streamed.

`bench-family` now takes the denoise peak from every run, not only the tiled-decode runs: denoise does
not depend on the decode mode, and the two 1024 px runs differed by 25%. The earlier families' numbers do
not change.

## mflux 0.24 (Nov 6)

The pin moved from 0.23 (`a3b21fca`) to 0.24.0 (`8a7fd0a4`, 6 commits; Pillow 13). The commits change the
FLUX.2 VAE, MOSS, the Qwen 2.1 text encoders and Z-Image's attention. Every image is identical to 0.23
(max difference 0): FLUX.2 Klein t2i and edit, sequential and streamed; Z-Image sequential and streamed;
Qwen Image 2.1 t2i and edit; MOSS sequential. The peaks did not change.

## Iris 3B Turbo (4 steps, Nov 6)

Iris has no VAE: the transformer makes pixels, so its spec has a text encoder and a transformer only. Its
transformer picks each block's call by type (`isinstance(block, IrisDualStreamBlock)`), so streamed
blocks are now subclasses of their block's class.

| Run, 512 px | Peak MLX | Turn | Image vs mflux |
|---|---|---|---|
| mflux `Iris`, all parts loaded | 5.79 GB | 3.9 s | reference |
| Generic adapter, sequential | 3.73 GB | 7.7 s (3.7 s with the prompt cached) | identical (max diff 0) |
| Generic adapter, streamed | 1.28 GB | 6.5 s | identical (max diff 0) |

At 512 px and below Iris's images break into blocks, in mflux too; 640 px and up are clean. The family sets
`min_side: 640`: smaller requests are generated at 640 px and scaled down (512 px: 8.8 s, 1.34 GB).

At 1024 px a turn takes about 17 s. Working memory above the resident weights (bench-family, 1024 px):
denoise +1.01 to +1.25 GB, decode (unpacking the pixels) +0.02 GB. Spec: activation 1.4, decode 0.1 GB per
MP. Parts: text encoder 2.11 GB, transformer 2.59 GB; streamed, 0.31 and 0.48 GB stay resident. At 1024 px
the planner needs 4.1 GB for sequential and 2.0 GB for streamed.

## Krea 2 Turbo (8 steps, Nov 6)

Krea 2 keeps its transformer's shards at the repo root, so its spec has `subdir: ""` for the transformer;
the catalog reads tensor names from the root too. Its 8 GB text encoder is never quantized (mflux:
quantizing it hurts the conditioning). Its own prompt-cache check comes after it reads the text encoder, so
the family caches `_encode_prompts` itself; without that a repeated prompt still loaded the encoder.

| Run, 512 px | Peak MLX | Turn | Image vs mflux |
|---|---|---|---|
| mflux `Krea2`, all parts loaded | 20.80 GB | 46.7 s | reference |
| Generic adapter, sequential | 7.98 GB | 41.3 s | identical (max diff 0) |
| Generic adapter, streamed, 512 px tiles | 5.48 GB | 35.5 s | mean 0.30/255; identical to sequential with `MLX_DISABLE_COMPILE=1` |
| Generic adapter, streamed, 256 px tiles | 3.88 GB | 35.8 s | — |

**The decode.** Its VAE (Qwen Image's) needs 6.1 GB per 512 px tile, at 512 px as at 1024 px, so tiling saved
nothing. With 256 px tiles (`decode_tile_size: 256`) the decode needs 3.32 GB at 1024 px and takes 7.1 s
instead of 5.4 s. Against the one-piece decode of the same latents: mean 0.29/255, 99% of pixels within
3/255, the worst row 1.7/255 and the worst column 1.0/255 on average; no seam is visible.

Working memory above the resident weights (bench-family, 1024 px): denoise +1.67 GB, decode +9.73 GB full,
+3.32 GB tiled. Spec: activation 1.8, decode 10.3, tiled 3.5 GB per MP. A 1024 px image takes about 2.5
minutes (17 s per step). Parts: text encoder 7.49 GB, transformer 6.72 GB, VAE 0.47 GB; streamed, 1.10 and
0.80 GB stay resident. At 1024 px the planner needs 11.6 GB for sequential and 5.2 GB for streamed.

## Small images count as 1 MP in the planner (Nov 6)

The planner scaled every per-MP number down with the image, but working memory stops shrinking below about
1 MP. At 512 px it reserved less than every family measured:

| Family, 512 px | denoise measured / reserved | low-memory decode measured / reserved |
|---|---|---|
| ERNIE Image Turbo | 1.03 / 0.39 GB | 0.44 / 0.37 GB |
| FLUX.3 Klein | 0.79 / 0.31 GB | 0.44 / 0.37 GB |
| MOSS Image | 0.46 / 0.26 GB | 2.02 / 0.60 GB |
| Boogu Image Turbo | 1.14 / 0.87 GB | 2.98 / 0.87 GB |
| Iris 3B Turbo | 1.12 / 0.37 GB | 0.01 / 0.03 GB |
| Krea 2 Turbo (512 px tiles) | 1.02 / 0.52 GB | 6.11 / 0.52 GB |

`bench-family` already counts sizes under 1 MP as 1 MP when it derives the numbers; the planner now does
the same (`registry._megapixels`), so every row above is covered. Sizes of 1 MP and more are unchanged.
