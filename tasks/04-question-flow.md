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
