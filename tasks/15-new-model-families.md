# 15 · New model families faster

**Status:** Open (from the competitor comparison, 2026-11-02).

## Why
dMLX runs three families in mflux format (FLUX.2 Klein, Z-Image, Qwen Image 2.1). Draw Things already ships
FLUX.3 Klein and Qwen Image Edit 2511 and adds families within weeks; ComfyUI runs nearly everything.
Model choice is not a strength today.

## To do
- [ ] List what each new family needs from dMLX: an adapter (load, encode, denoise, decode), memory
      estimates for the planner, an `agent_note`, catalog entries.
- [ ] Reduce that to one generic adapter over mflux's model interface where possible, so a new mflux family
      is mostly a YAML spec plus measured memory.
- [ ] Candidates, by demand: Qwen Image Edit 2511, FLUX.3 Klein (check mflux support and licenses).

## Done when
A new mflux family is added with a spec file and a benchmark run, no new engine code beyond hooks.
