# 03 · Messages sent during an image job skip the agent

**Status:** Open

## Problem
`Engine._plan` (`engine/src/dmlx/engine.py`) takes the accelerator lock without waiting. While an image
is being made the lock is held, so planning is skipped and the message goes through the old rule + router
path. Only a log line (`agent planning skipped: an image is being made`) records it, so the same message
is planned differently depending on timing.

It also affects answers to the agent's question (task 04): the router sees only the answer ("the boat"),
not the message it answers.

## Options
- The app waits: hold the sent message as pending ("Planned after the current image") and route it when
  the job finishes. Simple, keeps memory safe; the user waits.
- The engine queues the plan behind the job instead of falling back.
- Plan beside the job when memory clearly allows it (not on 16 GB; see the 16 GB gate).

## Done when
- A message sent during a job is planned by the agent, or the user is told plainly that it was not.
- A test covers a send during a running job.
