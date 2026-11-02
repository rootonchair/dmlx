# 15 · New model families faster

**Status:** Researched 2026-11-02; not started.

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
| Boogu | the text side is a component named `mllm` | the engine assumes `text_encoder` |
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

## To do
- [ ] Prototype the generic adapter on ERNIE Image Turbo. Measure the cost of building parts on a scratch
      object. Check the stand-ins with `nn.Module` attribute rules, `tree_flatten` and `mx.compile`.
- [ ] Move FLUX.2 Klein, Z-Image and Qwen Image 2.1 to the generic adapter. All three must give the same
      images (same seed) and the same peaks as today.
- [ ] Component roles in `ModelSpec` and `JobGuard`.
- [ ] Detection from the weight definition; catalog suggestions from the supported family list.
- [ ] `tools/bench_family.py`; add ERNIE, MOSS and FLUX.3 Klein with their notes and the license flow.
- [ ] Update to mflux 0.24 (check for API changes), then Iris.

## Done when
A new mflux family is added with a family entry and a benchmark run, with no new engine code beyond hooks.
