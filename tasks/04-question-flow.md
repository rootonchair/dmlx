# 04 · The agent's question works as a conversation

**Status:** Done in the working tree, not committed. Not yet clicked through in the app.

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

## Remaining
- [ ] Drive it once in the app (dev harness): ask, answer, check the chat and the job's inputs.
- [ ] An answer sent during an image job falls back to the router without the original message (task 03).
