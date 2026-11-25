"""Task 17 fix round 2, item 1, in the real engine: the app's own Auto model (sent as auto_models) gives way
when it won't fit, also when only one model fits and the agent is offered none. Real agent (Ministral 3B),
real models; headroom capped so that FLUX.2 Klein 9B won't fit for an edit and Klein 4B will."""
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
for sid in ("flux2-klein-4b", "flux2-klein-9b", "qwen-image-2.1-q4"):
    print("fits_now", sid, e._fits_now(e.registry.get(sid), "image.edit", 0, True))
orig_plan = e._plan
def spy(state, candidates, wait=False, models=None):
    print("OFFERED", {op: [m for m, _ in ms] for op, ms in (models or {}).items()})
    return orig_plan(state, candidates, wait=wait, models=models)
e._plan = spy
for auto in ({"image.edit": "flux2-klein-9b"}, {}):
    t = time.monotonic()
    d = e.route({"message": "make it warmer", "current_result_id": src.result_ids[0], "attachments": [],
                 "plan": True, "wait": True, "candidates": ["image.edit"], "auto_models": auto})
    print("auto_models", auto, "DECISION", {k: d.get(k) for k in ("intent", "model", "planning", "provider")},
          round(time.monotonic() - t, 1), "s")
d = e.route({"message": "make it warmer", "current_result_id": src.result_ids[0], "attachments": [],
             "candidates": ["image.edit"], "selected_models": {"image.edit": "flux2-klein-9b"}})
print("user picked klein-9b: DECISION model", d.get("model"))
print("agent resident", getattr(agent, "resident", None))
model = e.route({"message": "make it warmer", "current_result_id": src.result_ids[0], "attachments": [],
                 "candidates": ["image.edit"], "auto_models": {"image.edit": "flux2-klein-9b"}}).get("model")
model = model or "flux2-klein-9b"  # what the app runs: the decision's model, else its own Auto model
job = e.submit(task="edit", prompt="Warmer colours.", model=model, image_ids=src.result_ids, size="512x512", seed=3)
job.finished.wait(600)
print("EDIT", model, job.status, job.error, (job.report or {}).get("strategy"), (job.report or {}).get("total_seconds"))
