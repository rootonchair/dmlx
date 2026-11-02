# 15 · New model families faster

**Status:** Done for ERNIE Image Turbo, MOSS Image and FLUX.3 Klein (2026-11-02). More families open below.

## Why
dMLX runs three families in mflux format (FLUX.2 Klein, Z-Image, Qwen Image 2.1). Draw Things already ships
FLUX.3 Klein and Qwen Image Edit 2511 and adds families within weeks; ComfyUI runs nearly everything.
Model choice is not a strength today.

## What we found (2026-11-02)

**mflux already has the models.** The installed mflux 0.23.0 includes FLUX.3 Klein, Krea 2, FIBO, Ideogram 4,
Iris, ERNIE Image, MOSS Image, Boogu, Ming Image and Qwen Image Edit. PyPI has 0.24.0. What is missing is
dMLX's layer: an adapter, memory numbers, detection and an `agent_note` for each family.

**The families have the same shape.** Each one has:
- an `XInitializer` with `_init_config`, `_init_tokenizers` and `_init_models` (which builds the modules
  and sets `model.transformer`, `model.text_encoder`, `model.vae`);
- an `XWeightDefinition` with `get_components()` and `get_tokenizers()`;
- a pipeline class with `generate_image(seed, prompt, num_inference_steps, height, width, guidance, …)`;
- a `prompt_cache = {}` set in `_init_config`, and `callbacks.start(...)`, `before_loop` and `after_loop`
  around the denoise loop.

Our three adapters (`engine/src/dmlx/adapters/mflux_adapter.py`) repeat this by hand. The family-specific
parts are the class names, the encode method name, and the place where the phase hooks go.

**The differences that matter:**

| Family | Difference | Effect on dMLX |
|---|---|---|
| MOSS Image | reads `self.transformer.time_signal` before the loop starts | the transformer must load before `before_loop` (like Qwen 2.1's `axes`) |
| FLUX.3 Klein | reads `self.transformer.flow_map` before the loop; tokenizer is in `text_encoder/` | same; our catalog check wants `tokenizer/` or `processor/` and rejects the repo |
| Iris 3B | no VAE (it works in pixels); repo saved with mflux 0.24.0 | no decode phase; needs mflux 0.24 |
| Boogu | the text side's folder is `mllm/` | only a path: the part is still `text_encoder` (checked) |
| Ideogram 4 | two transformers (`conditional_transformer`, `unconditional_transformer`) | the engine assumes one `transformer` |
| Krea 2 | the transformer shards are at the repo root, not in `transformer/` | detection and loading need the definition's paths |
| FIBO | prompts are structured JSON | needs agent guidance for the prompt format |
| `_init_models` | three signatures: `(model)`, `(model, model_config)`, `(model, model_path)` | handle by inspection |

The saved weights carry only `mflux_version` and `quantization_level`. No family name, so detection still
uses the repo name and the tensor layout. mflux's `ModelConfig` aliases can supply the names.

## Candidates (mflux-community, 4-bit)

With the sequential and streamed strategies, the peak is about the largest part plus activations, not the sum.

| Family | Text side | Transformer | VAE | License | Edit | Verdict |
|---|---|---|---|---|---|---|
| ERNIE Image Turbo | 1.9 GB | 4.5 GB | 0.2 GB | Apache-2.0 | no | **pilot**: standard layout |
| MOSS Image SFT | 4.5 GB | 2.9 GB | 0.1 GB | Apache-2.0 | no | second: tests the early transformer read |
| FLUX.3 Image Klein | 2.0 GB | 3.1 GB | 0.2 GB | FLUX.3 Klein license | yes | third: edit, license acceptance flow |
| Iris 3B Turbo | 2.3 GB | 2.8 GB | none | Apache-2.0 | no | after the mflux 0.24 update |
| Boogu Image Turbo | 4.7 GB (`mllm`) | 6.4 GB | 0.2 GB | Apache-2.0 | no | needs component roles |
| FIBO 2.0 dev | 2.4 GB | 4.6 GB | 0.2 GB | FIBO 2.0 dev license | yes (FIBO Edit) | later: JSON prompts |
| Krea 2 Turbo | 8.0 GB | 7.2 GB | 0.5 GB | Krea 2 community license | no | later: tight on 16 GB |
| Ideogram 4 | 4.3 GB | 2 × 5.2 GB | 0.2 GB | non-commercial | no | later: two transformers, license |
| Qwen Image Edit 2511 | 15.5 GB | 13.2 GB | 0.3 GB | Apache-2.0 | yes | skip: does not fit 16 GB; Qwen Image 2.1 already edits |

## Design

1. **One generic adapter**, `MfluxFamilyAdapter`, configured by a family entry: the initializer, the weight
   definition, the pipeline class per task (`t2i`, `edit`), the mflux config name and the generate arguments.
   It builds the empty pipeline with `_init_config` and `_init_tokenizers`. It builds one part with
   `_init_models` on a scratch object, then keeps only the part it needs. MLX is lazy, so the unused parts
   should cost little; the prototype must measure this.
2. **Load a part when the pipeline first uses it**, not at hand-placed hooks. Each part on the pipeline is a
   small stand-in object. On first use it calls the job guard: the text side gets `encode`, the transformer
   gets `denoise` (which frees the text side), and the VAE gets `preprocess` or `decode`. This covers MOSS,
   FLUX.3 and Qwen 2.1 reading the transformer before the loop, with no per-family code. `after_loop` still
   starts `decode`.
3. **Bounded prompt cache.** Replace each pipeline's `prompt_cache = {}` with a dict backed by our
   `_EmbedCache`. Then embedding reuse works for every family, and the cache cannot grow without limit.
4. **Component roles in the spec.** Each part gets a role: `encode`, `denoise` or `decode`. `JobGuard` uses
   roles, not the names `text_encoder`, `transformer` and `vae`. This covers `mllm`, two transformers, and no VAE.
5. **Detection from the definition.** Match the repo name to mflux's `ModelConfig` aliases. Then check the
   files against the weight definition's `get_components()` paths and `get_tokenizers()` paths. This fixes
   FLUX.3's tokenizer and Krea 2's root shards.
6. **Measured memory by script.** Streaming lists come from the numbered stacks in the weight index (the
   catalog counts them already). Activation and decode numbers come from a benchmark script
   (`tools/bench_family.py`): 512 and 1024 px runs in the streamed strategy, peak recorded per phase, values
   written into the family entry. The no-hang gate still needs these numbers before a family is offered.

## Prototype results (2026-11-02, M5 24 GB)

`engine/src/dmlx/adapters/mflux_family.py`: `MfluxFamilyAdapter` plus a `FAMILIES` table (ERNIE only so far).
Parts are properties on the pipeline that enter their phase on first read. `stream: [auto]` streams every
block stack the part has (`streaming.find_stacks`; ERNIE's text encoder stack is `language_model.model.layers`).

**Building parts on a scratch object is free.** All parts of ERNIE, Z-Image, FLUX.3, MOSS, Iris, Boogu, FIBO
and Krea 2: 11-27 ms, under 1 MB. FLUX.2 needs `small_text_encoder` set first (a family default).

**ERNIE Image Turbo through the real engine** (seed 7, 8 steps, "a red fox sitting in fresh snow, morning light"):

| Run | Peak MLX | Denoise | Turn | Image vs mflux's own pipeline |
|---|---|---|---|---|
| mflux `ErnieImage`, all parts loaded, 512 px | 7.18 GB | 32.8 s | — | reference |
| Generic, sequential, 512 px | 5.23 GB | 30.8 s | 37.2 s | identical (max diff 0) |
| Generic, streamed, 512 px | 0.73 GB | 32.5 s | 38.1 s | identical (max diff 0) |
| Generic, sequential, 1024 px | 5.56 GB | 137.8 s | 146.4 s | — |
| Generic, streamed, 1024 px | 1.60 GB | 148.8 s | 158.4 s | — |

A second run of the same prompt skips the text encoder (the bounded prompt cache hit). ERNIE is slow: about
17 s per step at 1024 px, against about 3.2 s for Klein 4B. A 1024 px turn takes about 2.5 minutes, so its
`agent_note` must say so.

**No change to the current families.** The base adapter now reaches parts through `_module` and `_install`.
FLUX.2 Klein 512 px gives identical images before and after, sequential and streamed.

## Done (2026-11-02)
- [x] Prototype the generic adapter on ERNIE Image Turbo; measure the scratch build cost.
- [x] `dmlx bench-family` (`engine/src/dmlx/family_bench.py`): per-phase peaks above the resident weights,
      sizes under 1 MP counted as 1 MP, 10% margin. Checked against FLUX.2 Klein's hand-measured numbers.
- [x] ERNIE Image Turbo in the catalog with measured numbers and an `agent_note`.
- [x] Detection from the weight definition (every part and tokenizer must be in the repo); suggestions from
      the supported family list; unsupported versions (ERNIE base, FLUX.3 Klein Base) are named.
- [x] FLUX.2 Klein, Z-Image and Qwen Image 2.1 run on the generic adapter (`mflux_family.py`; Qwen 2.1 is a
      small subclass). The old per-family classes are gone. Results: `docs/benchmarks/families.md`.
- [x] MOSS Image and FLUX.3 Klein (see the benchmark doc). Their measurements found that a lazily cached
      prompt encoding kept the text encoder alive; the adapter now evaluates cached encodings when the
      encode phase ends (FLUX.3 peak 4.83 → 3.70 GB, MOSS 6.94 → 4.37 GB at 512 px).
- [x] `stream: [auto]`: the adapter finds the stacks from the module, the planner from the weight index.

Component roles (design point 4) turned out not to be needed for these families: specs already give each
part a phase, and Boogu's `mllm` is only a folder name. They come back with Ideogram 4 (two transformers)
and Iris (no VAE).

## Still open
- [ ] The guard trusts `free_component`. Check active memory after a free and log when it does not drop,
      so a leak like the one above shows up in any family.
- [ ] Update to mflux 0.24 (check for API changes), then Iris (no VAE).
- [ ] Boogu, FIBO (JSON prompts), Krea 2 (root shards, 16 GB), Ideogram 4 (two transformers, license).
- [ ] Re-measure FLUX.2 Klein's and Z-Image's full decode with `bench-family` (mflux's block decoder made
      the old 8.6 / 6.7 GB per MP spikes much smaller); only Macs with 32 GB or more use full decode.

## Done when
A new mflux family is added with a family entry and a benchmark run, with no new engine code beyond hooks.
