"""Jazz-poster captions from each local agent: box-rule breaks with the new checks off (before) and on (after)."""
import json, re, sys, time
from dmlx import paths; paths.use_own_caches()
from dmlx.prompting import formats, local_vlm

PROMPTS = [
    'A poster for a jazz night on Friday with the title "JAZZ NIGHT" and the line "FRIDAY 9 PM".',
    'Vintage art-deco jazz club poster: a saxophonist and a pianist under glowing spotlights, musicians in '
    'silhouette behind them, the title "JAZZ NIGHT" at the top and "FRIDAY 9 PM" at the bottom.',
    'Moody jazz night poster with a smoky atmosphere and warm stage lights, a trumpet player and a crowd of '
    'musicians silhouettes, "JAZZ NIGHT" and "FRIDAY 9 PM".',
]
USER_DATE = {"id": "date", "kind": "text", "label": "FRIDAY 9 PM", "box": [880, 520, 960, 960], "by_user": True}
FMTS = [formats.MOSS, formats.IDEOGRAM]

real_mood, real_clear = formats._is_mood, formats._clear_text

def breaks(layout):
    mood = [b.label for b in layout if real_mood(b)]
    texts = [b for b in layout if b.kind == "text"]
    cover = [f"{b.label}>{t.label}" for b in layout if b.kind == "obj" and not b.by_user for t in texts
             if min(b.box[2], t.box[2]) - max(b.box[0], t.box[0]) > 0 and min(b.box[3], t.box[3]) - max(b.box[1], t.box[1]) > 0]
    return mood, cover

def norm(fmt, reply, layout, checks):
    if not checks:
        formats._is_mood, formats._clear_text = (lambda b: False), (lambda out, fixes: out)
    try:
        return formats.normalize(fmt, reply, layout=layout)
    finally:
        formats._is_mood, formats._clear_text = real_mood, real_clear

totals = {}
for model_id in sys.argv[1:] or list(local_vlm.MODELS):
    path = local_vlm.locate(model_id)
    if path is None:
        print(f"## {model_id}: not downloaded", flush=True); continue
    provider = local_vlm.LocalVLMPromptProvider(path, model=local_vlm.MODELS[model_id])
    t = totals.setdefault(model_id, {"captions": 0, "invalid": 0, "before": 0, "after": 0})
    for p_i, plain in enumerate(PROMPTS):
        for fmt in FMTS:
            for with_layout in (False, True):
                layout = formats.parse_layout([USER_DATE]) if with_layout else []
                started = time.monotonic()
                reply = provider.structure(plain, fmt=fmt, layout=layout)
                secs = time.monotonic() - started
                t["captions"] += 1
                try:
                    before = norm(fmt, reply, layout, False)
                    after = norm(fmt, reply, layout, True)
                except formats.FormatError as exc:
                    t["invalid"] += 1
                    print(f"{model_id} p{p_i} {fmt} layout={with_layout}: invalid ({exc}) {secs:.0f}s", flush=True)
                    continue
                b, a = breaks(before.layout), breaks(after.layout)
                t["before"] += bool(b[0] or b[1]); t["after"] += bool(a[0] or a[1])
                print(f"{model_id} p{p_i} {fmt} layout={with_layout} {secs:.0f}s boxes={[x.label for x in before.layout]}"
                      f" before mood={b[0]} cover={b[1]} | after mood={a[0]} cover={a[1]} fixes={[f for f in after.fixes if 'box' in f]}",
                      flush=True)
    provider.unload()
    del provider
print(json.dumps(totals, indent=1))
