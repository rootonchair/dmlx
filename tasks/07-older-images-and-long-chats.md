# 07 · The agent can't see older images, and long chats are cut

**Status:** Open

## Problem
- `agent.shown_images` shows the attachments, or else only the current image (at most 3). For "the first
  one" or "the one with the cat", the agent picks by older prompts, not pixels, which contradicts its own
  "trust the image over the chat" rule.
- `agent.chat_lines` keeps only the last 16 lines, but `_resolve_id` looks up "image N" across the full
  history the app sends (up to 24 items). In a long chat the agent can pick an image it cannot see in
  the chat.

Seen in the app: after "make it red", a question, and the answer "the boat", the agent edited the teapot
it could see rather than the boat it could only read about (worked around for answers in task 04; a
message that names an older image still relies on the prompt text).

## Options
- Show small thumbnails of the images the message may refer to (the chat images it names, or the last
  few), within a token budget; measure the latency cost.
- Keep a short line for every image in the chat (number, id, short prompt) even when older messages
  are cut, like Agent Banana's "Context Folding" (arXiv 2602.09084).
- Reject resolved ids that are not in the lines the model saw.

## Done when
- Eval cases with long chats and with an image whose prompt disagrees with its pixels pass.
- Planning latency stays within about 12 s.
