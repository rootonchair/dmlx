"""Task 17 performance: fewest steps that keep the layout and clean lettering (FIBO 2, Ideogram 4).

The jazz poster through the real engine path (normalize, the adapter, the guard), with a fixed agent reply
so no agent loads: every box is the user's and the date is moved to the bottom right, the hardest case the
app's redraw sends. Same seeds at each step count.
Usage: uv run python steps_compare.py <fibo2|ideogram4> <out dir> <seed> <steps> [<steps>...]
"""
import json
import shutil
import sys
import time

from dmlx.engine import Engine
from dmlx.prompting import formats

family, out, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
step_counts = [int(s) for s in sys.argv[4:]]
plain = 'a poster for a jazz night with the words "JAZZ NIGHT" and "FRIDAY 9 PM"'
REPLIES = {
    "fibo2": json.dumps({
    "short_description": "A retro-modern 1950s jazz night poster: a dim, smoky jazz club with warm golden light, "
                         "a grand piano and a brass saxophone, with large elegant gold lettering and a smaller date line.",
    "objects": [
        {"description": "a black grand piano with a glossy finish, partly veiled by smoke", "location": "center",
         "relative_size": "large"},
        {"description": "a vintage brass saxophone resting on a stand", "location": "center-right",
         "relative_size": "medium"},
        {"description": "a cocktail glass with a red drink catching the light", "location": "bottom-center",
         "relative_size": "small"},
    ],
    "background_setting": "a moody jazz club interior, dark navy walls, drifting smoke",
    "lighting": {"conditions": "warm golden spotlight", "direction": "from above", "shadows": "deep, soft"},
    "aesthetics": {"composition": "poster layout, title at the top", "color_scheme": "navy, gold and cream",
                   "mood_atmosphere": "intimate, nostalgic"},
    "style_medium": "vintage screen-printed poster illustration",
    "artistic_style": "retro-modern, art deco",
    "context": "an event poster for a jazz club",
    "text_render": [
        {"text": "JAZZ NIGHT", "location": "top-center", "size": "large",
         "font": "bold art deco display serif", "color": "gold"},
        {"text": "FRIDAY 9 PM", "location": "center", "size": "medium",
         "font": "condensed sans-serif", "color": "cream"},
    ],
}),
    "ideogram4": json.dumps({
    "high_level_description": "A retro-modern 1950s jazz night poster: a dim, smoky jazz club with a grand piano "
                              "and a brass saxophone under warm golden light, with large gold art deco lettering.",
    "style_description": {"aesthetics": "intimate, nostalgic, bold poster layout", "lighting": "warm golden spotlight from above, deep soft shadows",
                          "medium": "screen-printed poster illustration", "art_style": "retro-modern art deco",
                          "color_palette": ["#1A1F3C", "#C9A227", "#F2E6C9"]},
    "compositional_deconstruction": {
        "background": "a moody jazz club interior, dark navy walls, drifting smoke",
        "elements": [
            {"type": "text", "bbox": [150, 100, 850, 350], "text": "JAZZ NIGHT", "desc": "large gold art deco lettering"},
            {"type": "text", "bbox": [250, 370, 750, 480], "text": "FRIDAY 9 PM", "desc": "cream condensed sans"},
            {"type": "obj", "name": "grand piano", "bbox": [300, 500, 600, 750], "desc": "black grand piano, glossy, veiled by smoke"},
            {"type": "obj", "name": "saxophone", "bbox": [650, 300, 850, 500], "desc": "vintage brass saxophone on a stand"},
            {"type": "obj", "name": "cocktail glass", "bbox": [500, 650, 650, 800], "desc": "cocktail glass with a red drink"},
        ]},
}),
}
MODELS = {"fibo2": "fibo-2-0-dev-mflux-q6", "ideogram4": "ideogram-4-mflux-q4"}
agent_boxes = [  # what the agent planned for MOSS in the app (y0, x0, y1, x1)
    ("a1", "text", "JAZZ NIGHT", [100, 150, 350, 850], "large gold art deco lettering"),
    ("a2", "text", "FRIDAY 9 PM", [880, 560, 960, 960], "cream condensed sans"),  # moved to the bottom right
    ("a3", "obj", "grand piano", [500, 300, 750, 600], "black grand piano, glossy, veiled by smoke"),
    ("a4", "obj", "saxophone", [300, 650, 500, 850], "vintage brass saxophone on a stand"),
    ("a5", "obj", "cocktail glass", [650, 500, 800, 650], "cocktail glass with a red drink"),
]
boxes = [{"id": i, "kind": k, "label": label, "box": box, "desc": desc, "desc_by_user": False, "by_user": True}
         for i, k, label, box, desc in agent_boxes]

engine = Engine()


class _Stub:
    resident = False

    def unload(self):
        pass


engine.prompt_assistant = _Stub()


def write_caption(job, fmt, plain_, layout, images, req, operation):
    return formats.normalize(fmt, REPLIES[family], layout=layout, width=req.width, height=req.height,
                             operation=operation, words=formats.quoted_words(plain_)), None


engine._write_caption = write_caption
for steps in step_counts:
    t = time.monotonic()
    job = engine.submit(task="t2i", prompt=plain, model=MODELS[family], size="1024x1024", seed=seed, steps=steps,
                        extras={"layout": boxes})
    job.finished.wait()
    seconds = round(time.monotonic() - t)
    row = {"family": family, "steps": steps, "seed": seed, "status": job.status, "seconds": seconds,
           "error": job.error, "timings": (job.report or {}).get("timings"), "strategy": (job.report or {}).get("strategy")}
    print(json.dumps(row, default=str), flush=True)
    if job.status == "done":
        shutil.copy(engine.results.path(job.result_ids[0]), f"{out}/{family}-s{seed}-{steps}.png")
    with open(f"{out}/{family}-steps.jsonl", "a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
