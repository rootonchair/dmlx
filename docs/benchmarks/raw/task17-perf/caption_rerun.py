"""Draw a saved result's exact caption again at other step counts (same seed), through the engine.
Usage: uv run python caption_rerun.py <result.json> <out dir> <steps> [<steps>...]"""
import json
import shutil
import sys

from dmlx.engine import Engine
from dmlx.prompting import formats

meta = json.load(open(sys.argv[1]))
out, step_counts = sys.argv[2], [int(s) for s in sys.argv[3:]]
caption = meta["parameters"]["rewritten_prompt"]
layout = meta["parameters"].get("layout") or []
engine = Engine()


class _Stub:
    resident = False

    def unload(self):
        pass


engine.prompt_assistant = _Stub()
engine._write_caption = lambda job, fmt, plain, boxes, images, req, op: (
    formats.Structured(prompt=caption, layout=list(boxes)), None)
for steps in step_counts:
    job = engine.submit(task="t2i", prompt=meta["parameters"].get("plain_prompt") or meta["prompt"], model=meta["model"],
                        size=f"{meta['width']}x{meta['height']}", seed=meta["seed"], steps=steps,
                        extras={"layout": layout})
    job.finished.wait()
    print(json.dumps({"steps": steps, "seed": meta["seed"], "status": job.status, "error": job.error,
                      "timings": (job.report or {}).get("timings")}), flush=True)
    if job.status == "done":
        shutil.copy(engine.results.path(job.result_ids[0]), f"{out}/rerun-{meta['seed']}-{steps}.png")
