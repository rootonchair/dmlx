"""Task 17 fix round 3, item 4, in the real engine: a model the user picked (Edit set to FLUX.2 Klein 9B) is
kept, planned by the real agent (Ministral 3B, plan=True, so the decision is a real image.edit, not
"ambiguous"), though it won't fit, and the job is refused; the same message on Auto gives way to Klein 4B.
Headroom capped so that Klein 9B won't fit for an edit and Klein 4B will (as in auto_default_check.py)."""
import logging, time
from dmlx.engine import Engine
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
GB = 1024**3
e = Engine()
print("agent", e.prompt_assistant and e.prompt_assistant.model.id)
print("downloaded", [s.id for s in e.registry.all() if e.is_downloaded(s)])
src = e.submit(task="t2i", prompt="a jazz night poster with a saxophone", model="flux2-klein-4b", size="512x512",
               steps=4, seed=3)
src.finished.wait(300); print("source", src.status, src.result_ids)
e.memory.unload_model("flux2-klein-4b")
real = e.memory.headroom
agent = e.prompt_assistant
def capped(cap):
    # The cap stands for free memory with the agent unloaded: while the agent is loaded, it takes its share.
    return lambda: min(real(), int(cap * GB)) - (agent.weights_bytes if getattr(agent, "resident", False) else 0)
def fits(sid, cap):
    e.memory.headroom = capped(cap)
    return e._fits_now(e.registry.get(sid), "image.edit", 0, True)
lo = next(c / 10 for c in range(10, 240) if fits("flux2-klein-4b", c / 10))
hi = next(c / 10 for c in range(10, 240) if fits("flux2-klein-9b", c / 10))
cap = round((lo + hi) / 2, 1)
e.memory.headroom = capped(cap)
print(f"klein-4b fits from {lo} GB, klein-9b from {hi} GB; cap {cap} GB; real {real() / GB:.2f} GB")
state = {"message": "make it warmer", "current_result_id": src.result_ids[0], "attachments": [],
         "plan": True, "wait": True, "candidates": ["image.edit"]}
for label, extra in (("user picked klein-9b", {"selected_models": {"image.edit": "flux2-klein-9b"}}),
                     ("auto, app's model klein-9b", {"auto_models": {"image.edit": "flux2-klein-9b"}})):
    t = time.monotonic()
    d = e.route({**state, **extra})
    print(label, "DECISION", {k: d.get(k) for k in ("intent", "model", "planning", "provider")},
          round(time.monotonic() - t, 1), "s")
    # What the app runs: the user's pick; on Auto the decision's model, else its own Auto model.
    model = "flux2-klein-9b" if "selected_models" in extra else d.get("model") or "flux2-klein-9b"
    job = e.submit(task="edit", prompt=d.get("prompt") or "Warmer colours.", model=model, image_ids=src.result_ids,
                   size="512x512", seed=3)
    job.finished.wait(600)
    print(label, "JOB", model, job.status, (job.error or {}).get("code"), (job.report or {}).get("total_seconds"))
