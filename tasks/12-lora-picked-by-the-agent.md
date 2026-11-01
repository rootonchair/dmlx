# 12 · LoRA import, picked by the agent

**Status:** Open (from the competitor comparison, 2026-11-02).

## Why
LoRA is table stakes in ComfyUI, Draw Things (which also trains LoRAs on-device) and InvokeAI. dMLX can
add something none of them do: the agent chooses the LoRA (and its strength) from the message, like it
chooses the image model in Auto.

## Today
`engine/src/dmlx/catalog.py` refuses LoRA repos: "ControlNet and LoRA repos are add-ons, not full models;
dMLX does not support them yet."

## To do
- [ ] Check mflux's LoRA support for FLUX.2 Klein and Z-Image, and its memory cost.
- [ ] Add Model accepts a LoRA repo or file: base family, trigger words, license, size; listed under its
      base model in Settings → Models.
- [ ] The engine applies a LoRA per job (and counts it in the memory plan).
- [ ] The agent sees installed LoRAs with a one-line note each (like the models' `agent_note`) and may set
      `"lora": {"id": …, "strength": …}`; a LoRA the user picked is never overridden. Code only refuses a
      LoRA that does not fit the chosen model.
- [ ] Eval cases: a style LoRA asked for by name, by description, and not wanted.

## Done when
A style LoRA added from Hugging Face is used when the message asks for that style, and not otherwise.
