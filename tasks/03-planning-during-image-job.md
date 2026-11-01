# 03 · Messages sent during an image job skip the agent

**Status:** Done; checked in the app on 2026-10-29.

## Problem
`Engine._plan` took the accelerator lock without waiting. While an image was being made, planning was
skipped and the message went through the rule + router path, recorded only in a log line. The app also
took "the current image" when the message was sent, so a message sent during a job was planned against
the image before the one being made.

## What changed
- **The chat whose image is being made:** sending is blocked; the send button is Stop
  (`AppModel.isGeneratingHere`, `stopGeneratingHere`). The message box stays editable ("Write your next
  message; send it when the image is done…"). Its next message may refer to the image being made.
- **Other chats:** sending works. The route request carries `wait: true`; the composer shows "Waiting for
  the image in “…” to finish, then planning this." (`isWaitingForImage`).
- **Engine (`_plan`):** while a job runs it plans beside the job when the Mac has 32 GB or more and both
  fit (`_fits_beside_job`, never freeing memory then), waits for the job when `wait` is set, and otherwise
  skips. The decision says which: `planning` = `beside_job`, `waited`, `skipped_busy`, `failed` or `rejected`.
- **Chats off screen keep their job:** a job's progress and result used to be dropped when its chat was not
  shown, and it came back as "Stopped when the app closed". Turns with a job in flight now move to
  `offscreenTurns` when another chat is selected, keep updating, and are saved when the job ends.
- Tests: skipped while busy (and reported), waiting for the job, planning beside the job without freeing memory.

## Planning beside a job (measured)
Qwen3-VL 8B planning while Klein made a 1024 image, one process: 4/4 replies identical to planning
alone, no crash; the plan took about 28 s instead of 9 s and the image 34 s instead of about 20 s.
Worth it for long jobs (Qwen Image 2.1, about 220 s), roughly even for Klein.

## Checked in the app (dev harness, 2026-10-29, `docs/benchmarks/raw/harness-03/script.json`)
Chat B: a rowboat. Chat A: a teapot, then "make it green" while the teapot was being made. Then back to
chat B, which sent "make the boat red" during A's job.
- In chat A the send was refused and the text stayed in the message box; the button was Stop.
- In chat B the message waited ("waiting" while A's job ran), then the agent planned it as an edit of the
  rowboat (`agent:local-vlm:qwen3-vl-8b…`).
- Chat A's teapot finished while A was off screen and was saved as done with its image.
- The harness's `waitIdle` now also waits while a message is planned or any job runs.

## Remaining
- [x] Chat B's planned edit was refused for memory: 2.2 GB usable just after the agent unloaded, against
      3.1 GB needed (about 5.5 GB was free before). macOS takes a moment to hand back the agent's memory;
      the image job could wait a few seconds for it before refusing (task 08).
      Done: image jobs wait up to 10 s for it (`Engine._plan_memory`, task 08).
