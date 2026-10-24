# Qwen Image 2.1 Base Q4 integration gate (Oct 2026)

Machine: Apple M5, 24 GB, macOS 26.5 under everyday application load.

Checkpoint: `OsaurusAI/Qwen-Image-2.1-mflux-4bit` pinned to
`235fd28de40d745382e9d7a33f008c7735ec21c2`. dMLX uses its Q4 transformer and
FP32 VAE, but replaces its 4.8 GB Qwen3-VL-8B encoder with MFLUX's internal
Qwen3.5-0.8B + zen-image-edit fusion encoder. The active downloads are about
7.3 GB rather than 9.9 GB. The weights use the Qwen Research License and are
only available through explicit model selection after acknowledgment.

## Integration checks

- The installed mflux 0.23 runtime constructs the unified Qwen Image 2.1 edit
  pipeline from the pinned pack.
- Generation and ordered editing share that pipeline; zero to ten input images
  are supported.
- A negative prompt enables true CFG. Without one, dMLX uses guidance 1.
- The model requires at least two inference steps; dMLX validates that before a
  job is queued.
- Nested block streaming was implemented and measured, then rejected for this
  model because runtime allocations accumulated far beyond its static estimate.

## 24 GB no-hang gate

A 256×256, two-step streamed generation entered the real transformer, then the
runtime guard stopped it during the first step:

| Metric | Result | Gate |
|---|---:|---:|
| Peak reported MLX allocation | 26.81 GiB | must remain safe on 24 GB |
| Peak memory pressure | critical | below critical |
| Maximum swap-in | 239.4 MB/s | ≤ 64 MB/s |
| Time to abort | 93.86 s | — |

This failure was isolated to dMLX's custom block-streaming path, not a universal
hardware requirement. The same pinned Q4 checkpoint succeeds when components
run sequentially and are released at phase boundaries:

| Workload, two steps | Peak MLX | Swap growth | Result |
|---|---:|---:|---|
| Generate 256×256 | 6.05 GiB | 0 | pass |
| Generate 512×512 | 6.06 GiB | 0 | pass |
| Generate 1024×1024 | 6.79 GiB | 0 | pass |
| Edit 512×512, one reference | 7.09 GiB | 0.30 GiB | pass |
| Edit 512×512, two ordered references | 6.66 GiB | 0 | pass |
| Edit 512×512, one reference + true CFG | 7.09 GiB | -0.12 GiB | pass |

Qwen block streaming is therefore disabled. Sequential execution loads the
Q4 VAE and encoder for conditioning, releases the encoder, loads the Q4
transformer, releases it after denoising, then reloads the VAE for tiled decode.
For edits, the transformer metadata is loaded while the VAE finishes reference
preprocessing because Qwen constructs its layout before its normal loop
callback. Peak pressure stayed below critical in all successful runs.

Qwen remains visible as an explicitly selected **Experimental** model and is
excluded from Auto while higher reference counts remain unmeasured. Every job
still goes through the live memory preflight; the reference multiplier is
calibrated conservatively from the edit runs.

## Smaller encoder revalidation

The existing Qwen entry now uses the smaller encoder internally; it is not a
second user-facing model. MFLUX is pinned to commit
`a3b21fcaa88726b2267bd42bfc2e2d722f7d5a47`, the first tested upstream state
containing this path.

| Workload, two steps | Previous peak | Smaller encoder | Swap-in | Result |
|---|---:|---:|---:|---|
| Generate 1024×1024 | 6.79 GiB | **6.23 GiB** | 4.0 MB/s | pass |
| Edit 512×512, one reference | 7.09 GiB | **6.72 GiB** | 3.9 MB/s | pass |

The encoder itself falls from 4.8 GB to 2.37 GB resident. Whole-job peak falls
less because the Q4 transformer plus denoising activations becomes the limiting
phase. The smaller encoder supports ordered reference editing, but upstream
documents English-only prompts, possible fine-detail drift, and weaker rendered
numerals. Native Qwen3-VL prompt rewriting and verification are not used by
this integration; edits are rewritten by dMLX's prompt assistant instead (below).

Fresh downloads omit the unused 4.8 GB encoder folder from the base checkpoint.
Existing installations may retain that old folder in the Hugging Face disk
cache, but it is never loaded into memory.

## Edit instruction rewriting (Oct 24)

With a terse instruction the model reconstructs the reference instead of editing
it. "remove text from this image" on a 704×992 trading card (seed 1206497696,
40 steps) left every line of text in place, in dMLX and in upstream
`mflux-generate-qwen-2.1-edit --small-text-encoder` alike (PSNR 46 dB between
the two outputs). Guidance 4 with a negative prompt changed nothing. A detailed
description of the edited image removed all of it.

A first integration loaded the pack's own Qwen3-VL-8B to rewrite edit
instructions (upstream's recipe). On the same edit it took 1.8 s to load and
9.9 s to rewrite, peaked at 5.81 GB in that phase and 8.31 GB for the whole job
(pressure normal), and removed every line of text but also blanked the two
round emblem icons.

Edits for every model are now rewritten by dMLX's prompt assistant, a standalone
`mlx-community/Qwen3-VL-8B-Instruct-4bit` (Apache-2.0, 5.4 GB, pinned) run with
mlx-vlm, so this pack no longer needs its 4.8 GB encoder. Note that neither is
Qwen's dedicated rewriter: Qwen ships fine-tuned Qwen3.5-VL 9B models for this
(Qwen-Image-2.1-PE-I2I / PE-T2I), which MFLUX does not support (issue #809).

The implementation and contract tests pass. Higher reference counts still need
their own 24 GB measurements before removing the Experimental label.
