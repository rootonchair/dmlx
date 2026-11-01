# 04 · The agent's question works as a conversation

**Status:** Done; checked in the app on 2026-10-25.

## Problem
- When the agent asks something like "Which image do you mean?", the app showed it above the workflow
  buttons, which cannot answer "which image".
- A typed reply was sent as a new message without the original one.
- The question was never added to the history, so the next plan did not know it was asked.

## What changed
- `AppModel.handle` holds the message with the question and clears the composer ("Type your answer…");
  typing no longer clears the question (`scheduleRoutePreview`). Cancel gives the message back.
- `AppModel.answer` plans the reply with the original message and the question in the history, then
  records the message, the question (a notice turn) and the answer in the chat.
- `agent.chat_lines` renders notice turns as `app: …`; the instructions say the NEW message answers the
  app's last question.
- Eval cases `ask-answer-new` and `ask-answer-image` in `engine/evals/agent-routing.jsonl` (both pass).

## Checked in the app (dev harness `ask` step, 2026-10-25)
Boat, then teapot, then "make it red", then the question "the boat or the teapot?", then the answer
"the boat": the chat shows the message, the question, the answer and the image.
- First run: the agent edited the teapot. With the real photo of the teapot in view, a short answer lost
  to the image (the eval passed only because its images are placeholders).
- Fix: `agent._new_message` writes the answer next to the request it answers ("make it red (asked …,
  the user answered: the boat)"). With real images: 5/5 on boat/teapot answers, naming, "the first
  one" and "it"; full eval still 36/36.

## Remaining
- [x] An answer sent during an image job: blocked in that chat; from another chat it waits for the job (task 03).

## Follow-up 2026-11-01: replies in the chat, and a new request while a question is open
- The agent's question or reply is a `reply` turn (`TurnKind.reply`), shown in the chat as its own bubble
  (`AgentReplyBubble`), not as a caption above the message box. While it is open it shows under the held
  message; the bar above the box keeps only the shortcuts ("Reply in the chat, or: Create new · Cancel").
- The next message is no longer forced to be the answer. `agent._new_message` shows the open question
  and the message it was about, and the agent decides: an answer ("the boat", even a word or two) plans
  the earlier message with it; a new request is planned alone.
- Greetings, thanks and questions about the app get a reply in the agent's own words (the instruction
  no longer has an example sentence; the 8B copied it and offered "image 2" in an empty chat).
- Eval: `greeting`, `greeting-with-image`, `thanks`, `greeting-then-edit`, `ask-then-new-request`.
- Checked in the app (harness, 8B): "hello" → reply bubble; the fox image; "thanks, that's lovely!" →
  "You're welcome! I'm glad you like the image of the sleeping fox…"; "now make it black and white" →
  image 2 edited from image 1. Screenshot `docs/screenshots/agent-replies.jpg`.

### Remaining
- [ ] `ask-then-new-request` ("actually, a lighthouse on a cliff at night…" after "which image?"): only the
      9B plans it as a new image; the 8B, Ministral and the 4B ask again ("…or is this a new request?").
      Safe, but one extra turn.
