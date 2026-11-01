# Tasks

Open problems in the planning agent (`engine/src/dmlx/prompting/agent.py`), from the review on 2026-10-25
(03–09), and product gaps from the competitor comparison on 2026-11-02 (10–16, in priority order).
One file per task; update its **Status** line as work lands.

| # | Task | Status |
|---|------|--------|
| 03 | [Messages sent during an image job skip the agent](03-planning-during-image-job.md) | Done, checked in the app |
| 04 | [The agent's question works as a conversation](04-question-flow.md) | Done: replies in the chat; a new request after a question is often asked about again |
| 05 | [Text to draw in double quotes](05-double-quotes.md) | Done: quote style makes no difference on Qwen Image 2.1 |
| 06 | [Image labels are misread](06-image-label-resolution.md) | Done |
| 07 | [The agent can't see older images, and long chats are cut](07-older-images-and-long-chats.md) | Done: 'look' tool; prompt-vs-pixels passes except on the 4B |
| 08 | [The agent on a busy 16 GB Mac](08-agent-on-16gb.md) | Done: Ministral plans every edit; the swap burst did not come back |
| 09 | [Decisions in code that belong to the agent](09-fixed-rules.md) | Done: style reference via a feedback step (9B, Ministral) |
| 10 | [Show 16 GB works by simulation only, then notarize](10-16gb-without-a-16gb-mac.md) | Open: a wired ballast; notarizing needs a Developer ID |
| 11 | [Region edits: a mask brush the agent can use](11-region-edits-with-a-mask.md) | Open |
| 12 | [LoRA import, picked by the agent](12-lora-picked-by-the-agent.md) | Open |
| 13 | [Faster turns: previews and a planner that stays loaded](13-faster-turns.md) | Open |
| 14 | [Upscale and restore as a chat action](14-upscale-as-a-chat-action.md) | Open: a SeedVR2 adapter exists |
| 15 | [New model families faster](15-new-model-families.md) | Open |
| 16 | [dMLX as a tool server for other local agents](16-tool-server-for-other-agents.md) | Open |
