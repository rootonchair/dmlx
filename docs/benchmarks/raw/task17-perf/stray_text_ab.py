"""Ideogram 4 stray text (task 17): draw a saved result's caption again with "No other text or lettering
anywhere." added to its background, same seed and steps, through the engine. (Tried and not kept: the stray
line stayed on both seeds; ideogram4-no-other-text.jpg.)
Usage: uv run python stray_text_ab.py <result.json> <out dir>"""
import json
import shutil
import sys

from dmlx.engine import Engine
from dmlx.prompting import formats

meta = json.load(open(sys.argv[1]))
out = sys.argv[2]
layout = formats.parse_layout(meta["parameters"].get("layout") or [])
value = json.loads(formats.normalize(formats.IDEOGRAM, meta["parameters"]["rewritten_prompt"], layout=layout).prompt)
value["compositional_deconstruction"]["background"] += ". No other text or lettering anywhere."
caption = json.dumps(value, ensure_ascii=False)
print(json.loads(caption)["compositional_deconstruction"]["background"], flush=True)
engine = Engine()


class _Stub:
    resident = False

    def unload(self):
        pass


engine.prompt_assistant = _Stub()
engine._write_caption = lambda job, fmt, plain, boxes, images, req, op: (
    formats.Structured(prompt=caption, layout=list(boxes)), None)
job = engine.submit(task="t2i", prompt=meta["prompt"], model=meta["model"], size=f"{meta['width']}x{meta['height']}",
                    seed=meta["seed"], steps=meta["steps"], extras={"layout": [b.as_dict() for b in layout]})
job.finished.wait()
print(json.dumps({"seed": meta["seed"], "steps": meta["steps"], "status": job.status, "error": job.error,
                  "timings": (job.report or {}).get("timings")}), flush=True)
if job.status == "done":
    shutil.copy(engine.results.path(job.result_ids[0]), f"{out}/notext-{meta['seed']}-{meta['steps']}.png")
