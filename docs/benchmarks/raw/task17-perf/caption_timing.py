"""Task 17 performance: how long the caption step takes per local agent (load + structure + check).

Runs `LocalVLMPromptProvider.structure` and `formats.normalize` exactly as `Engine._write_caption` does,
for the jazz poster in the three formats, plus a redraw with the user's boxes (Ideogram 4). One agent per
process run (pass its id); the first call includes loading the agent.
Usage: uv run python caption_timing.py <agent-id> [out.json]
"""
import json
import os
import sys
import time
from pathlib import Path

# The agents live in dMLX's own cache (paths.hf_cache_dir), set before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_CACHE", str(Path.home() / "Library/Application Support/dMLX/models-cache"))

from dmlx.prompting import formats, local_vlm  # noqa: E402

agent_id = sys.argv[1]
plain = 'a poster for a jazz night with the words "JAZZ NIGHT" and "FRIDAY 9 PM"'
boxes = formats.parse_layout([
    {"id": "a1", "kind": "text", "label": "JAZZ NIGHT", "box": [100, 150, 350, 850], "by_user": True},
    {"id": "a2", "kind": "text", "label": "FRIDAY 9 PM", "box": [880, 560, 960, 960], "by_user": True},
    {"id": "a3", "kind": "obj", "label": "saxophonist", "box": [380, 250, 860, 750], "by_user": True,
     "desc": "a saxophonist in a dark suit, lit from the side", "desc_by_user": True},
])
cases = [("moss-layout", []), ("fibo-json", []), ("ideogram-json", []), ("ideogram-json", boxes),
         ("moss-layout", [])]  # the last one again: the agent already loaded
provider = local_vlm.LocalVLMPromptProvider(local_vlm.locate(agent_id), model=local_vlm.MODELS[agent_id])
words = formats.quoted_words(plain)
rows = []
for i, (fmt, layout) in enumerate(cases):
    started = time.monotonic()
    loaded_before = provider.resident
    t_load = 0.0
    if not loaded_before:
        provider._model, provider._processor = provider._loader(provider.model_path)
        t_load = time.monotonic() - started
    tries, reason, ok, fixes = 0, None, False, []
    gen_started = time.monotonic()
    for _ in range(2):
        tries += 1
        reply = provider.structure(plain, fmt=fmt, layout=layout, retry=reason)
        try:
            result = formats.normalize(fmt, reply, layout=layout, width=1024, height=1024, words=words)
            ok, fixes = True, result.fixes
            break
        except formats.FormatError as exc:
            reason = str(exc)
    t_gen = time.monotonic() - gen_started
    row = {"agent": agent_id, "format": fmt, "boxes": len(layout), "load_s": round(t_load, 1),
           "caption_s": round(t_gen, 1), "total_s": round(time.monotonic() - started, 1), "tries": tries,
           "valid": ok, "reply_chars": len(reply), "reason": reason if not ok else None}
    rows.append(row)
    print(json.dumps(row), flush=True)
provider.unload()
if len(sys.argv) > 2:
    json.dump(rows, open(sys.argv[2], "w"), indent=1)
