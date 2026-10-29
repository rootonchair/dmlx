# 09 · Decisions in code that belong to the agent

**Status:** Done 2026-10-30 (one case open: a style reference from an attached image).

Found by checking the agent, the engine's routing and the app for places where code decided what the
model should. Each became a tool, a skill or written guidance; what stays in code only refuses what
cannot run.

| Was a fixed rule | Now |
|---|---|
| Auto used Klein, then Z-Image | **Model choice:** in Auto the agent sees the installed models (each with an `agent_note` in its YAML) and the free memory, and may set `"model"`. A model the user picked is never overridden. Experimental models are offered (license accepted on download). |
| The first message of a chat skipped the agent | Every sent message is planned; a message that is not a request for an image gets a reply in `"ask"`. |
| The agent's prompt for new images was used only in Auto enhancement | Always used (`AppModel.execute`); a detailed prompt is kept as written. |
| Every edit got a second "describe" pass | **Describe skill:** `"describe": true/false` per edit. |
| The chat was cut to the last 16 lines | **read_chat tool:** `{"read_chat": true}` shows the whole chat (uses a look). |
| Attachments forced into the edit, first; generate became edit when anything was attached | The agent's choice is kept. |
| An edit naming no image got the latest image | `agent.check` tells the agent what cannot run ("image.edit needs the image to change") and it answers once more; only then the router. |
| Single quotes rewritten to double by a regex | Removed (task 05 tests whether quotes matter). |

Still code, as checks: an image the agent names must exist; a new image takes no images; an edit takes
1-3; only workflows an installed model can run are offered; an unusable answer falls back to the router.

## Results (real images 10 cases, eval 38 cases; before = with the rules, after task 07)
| Model | Real | Eval | Before | Model picks |
|---|---|---|---|---|
| Qwen3-VL 8B | 9/10 | 37/38 | 38/38 | Qwen Image 2.1 for NIGHT BLADE and removing text; the poster went to Klein |
| Qwen3.5 9B | 9/10 | 38/38 | 37/38 | Qwen Image 2.1 for all three text cases, Klein otherwise |
| Ministral 3 3B | 9/10 | 35/38 | 36/38 | Qwen Image 2.1 for NIGHT BLADE only |
| Qwen3-VL 4B | 6/10 | 36/38 | 36/38 | always Klein |

`attach-style-2` now accepts editing the boat with the painting as the style (the old expected order came
from the forced rule). A first version of the model notes made the 9B pick Qwen Image 2.1 for a fox in
the snow; the notes now say Klein is the usual choice and Qwen Image 2.1 is worth its 4 minutes only for
exact words or precise multi-image edits.

## Remaining
- [ ] "a dragon in the style of this card": the 8B and the 4B make a new image and drop the card (the
      removed rule used to turn it into an edit). The instruction says so; they do not follow it.
