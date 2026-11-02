# Model families on the generic adapter (task 15, Nov 2026)

Machine: Apple M5, 24 GB, macOS 26.5, mflux 0.23.0. All models are 4-bit packs from mflux-community.
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
