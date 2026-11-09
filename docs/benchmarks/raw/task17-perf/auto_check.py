"""(b) Auto with headroom capped at 7.6 GB: Qwen Image 2.1 not offered for the edit, a Klein picked, edit runs."""
import logging, time
from dmlx.engine import Engine
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
GB = 1024**3
e = Engine()
print("agent", e.prompt_assistant and e.prompt_assistant.model.id)
print("downloaded", [s.id for s in e.registry.all() if e.is_downloaded(s)])
src = e.submit(task="t2i", prompt="a jazz night poster with a saxophone", model="flux2-klein-4b", size="512x512", steps=4, seed=3)
src.finished.wait(300); print("source", src.status, src.result_ids)
e.memory.unload_model("flux2-klein-4b")
real = e.memory.headroom
e.memory.headroom = lambda: min(real(), int(7.6 * GB))
print("headroom now GB", round(e.memory.headroom()/GB, 2), "real", round(real()/GB, 2))
for sid in ("flux2-klein-4b", "flux2-klein-9b", "qwen-image-2.1-q4", "z-image-turbo"):
    s = e.registry.get(sid)
    print("fits_now", sid, e._fits_now(s, "image.edit", 0, True) if "image.edit" in s.capabilities else "no edit")
orig_plan = e._plan
def spy(state, candidates, wait=False, models=None):
    print("OFFERED", {op: [m for m, _ in ms] for op, ms in (models or {}).items()})
    return orig_plan(state, candidates, wait=wait, models=models)
e._plan = spy
t = time.monotonic()
d = e.route({"message": "make it warmer", "current_result_id": src.result_ids[0], "attachments": [], "plan": True,
             "wait": True, "candidates": ["image.edit"]})
print("DECISION", {k: d.get(k) for k in ("intent", "model", "planning", "provider")}, round(time.monotonic() - t, 1), "s")
model = d.get("model") or "flux2-klein-4b"
job = e.submit(task="edit", prompt=d.get("prompt") or "Warmer colours.", model=model, image_ids=src.result_ids,
               size="512x512", seed=3)
job.finished.wait(600)
print("EDIT", model, job.status, job.error, job.result_ids, (job.report or {}).get("strategy"), (job.report or {}).get("total_seconds"))
