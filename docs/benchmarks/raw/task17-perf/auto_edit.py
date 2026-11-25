"""Item 1 in the real engine: after the Ideogram 4 poster, "make it warmer" on Auto with the real agent and
the installed models. Headroom is capped at 7.6 GB (what the app had when the bug was seen); everything else
is real. Prints the models offered to the agent, its decision, and runs the edit the app would send."""
import json
import logging
import shutil
import sys

from dmlx.engine import Engine
from dmlx.memory import GB

logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
out, source = sys.argv[1], "img_20261109_354c54fbed4c"
engine = Engine()
real = engine.memory.headroom
engine.memory.headroom = lambda: min(real(), int(7.6 * GB))
print("downloaded:", [s.id for s in engine.registry.all() if engine.is_downloaded(s)], flush=True)
print("agent:", engine.prompt_assistant and engine.prompt_assistant.name, flush=True)
print("offered for image.edit:", engine._model_choices(["image.edit"], 0, True).get("image.edit"), flush=True)
state = {"message": "make it warmer", "current_result_id": source, "attachments": [], "plan": True, "history": [
    {"sequence": 0, "kind": "user", "text": 'a poster for a jazz night with the words "JAZZ NIGHT" and "FRIDAY 9 PM"',
     "input_ids": []},
    {"sequence": 1, "kind": "image", "text": "a jazz night poster", "image_number": 1, "operation": "image.generate",
     "result_id": source, "input_ids": []}]}
decision = engine.route(state)
print("decision:", json.dumps({k: decision.get(k) for k in ("intent", "model", "provider", "planning")}), flush=True)
model = decision.get("model") or engine.registry.default_for_operation("image.edit", engine.memory.total / GB).id
job = engine.submit(task="edit", prompt=decision.get("prompt") or "Make it warmer.", model=model,
                    image_ids=[source], size="1024x1024")
job.finished.wait()
print("job:", json.dumps({"model": model, "status": job.status, "error": job.error,
                          "seconds": (job.report or {}).get("total_seconds")}), flush=True)
if job.status == "done":
    shutil.copy(engine.results.path(job.result_ids[0]), f"{out}/auto-warmer-{model}.png")
