"""(a) memory_pressure stop mid-denoise (and during encode) frees Klein 4B at once; cancel keeps it warm."""
import logging, sys, threading, time, dataclasses
import mlx.core as mx
from dmlx.engine import Engine
from dmlx import sysmem

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
GB = 1024**3
e = Engine()
mon = e.memory.monitor
orig_last = mon.last
force = threading.Event()
def last():
    s = orig_last()
    return dataclasses.replace(s, pressure=sysmem.PRESSURE_CRITICAL) if force.is_set() else s
mon.last = last

def run(stop_at, code, prompt="a red fox in a misty forest"):
    force.clear()
    job = e.submit(task="t2i", prompt=prompt, model="flux2-klein-4b", size="512x512", steps=8, seed=1)
    def watch():
        seen = set()
        while not job.finished.is_set():
            for ev, payload in list(job.events):
                key = (ev, payload.get("step"))
                if key in seen: continue
                seen.add(key)
                hit = (stop_at == "encode" and ev == "encoding") or \
                      (ev == "step" and payload.get("step") == stop_at)
                if hit:
                    print("STOP at", ev, payload, "active", round(mx.get_active_memory()/GB, 2), flush=True)
                    (force.set() if code == "memory_pressure" else job.cancel_event.set())
            time.sleep(0.05)
    threading.Thread(target=watch, daemon=True).start()
    job.finished.wait(600)
    force.clear()
    print("RESULT", stop_at, code, job.status, job.error and job.error["code"],
          "active after", round(mx.get_active_memory()/GB, 2), "resident", e.memory.resident_components(),
          "events", sorted({ev for ev, _ in job.events}), flush=True)

run("encode", "memory_pressure", prompt="a lighthouse on a cliff at night")
run("encode", "cancelled", prompt="a bowl of ramen on a wooden table")
