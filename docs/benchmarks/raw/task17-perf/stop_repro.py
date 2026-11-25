"""Stop a real job mid-denoise and measure what MLX still holds."""
import gc, logging, sys, time
from dmlx import paths; paths.use_own_caches()
import mlx.core as mx
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
from dmlx import memory
from dmlx.adapters.base import JobAborted
from dmlx.engine import Engine

model = sys.argv[1] if len(sys.argv) > 1 else "flux2-klein-4b"
code = sys.argv[2] if len(sys.argv) > 2 else "memory_pressure"
GB = 1024**3
orig_step = memory.JobGuard.on_step
def on_step(self, step, total):
    orig_step(self, step, total)
    if step == 2:
        print(f"[step 2] active {mx.get_active_memory()/GB:.2f} GB; resident {self.m.resident_components()}", flush=True)
        if code == "cancelled":
            self.cancel_event.set()
        else:
            raise JobAborted(code, "forced stop for the test")
memory.JobGuard.on_step = on_step

e = Engine()
print(f"start active {mx.get_active_memory()/GB:.2f} GB", flush=True)
job = e.submit(task="t2i", prompt="a jazz poster with a saxophonist", model=model, size="512x512", steps=6, seed=1,
               strategy=sys.argv[3] if len(sys.argv) > 3 else "auto")
job.finished.wait()
print("status", job.status, job.error, flush=True)
print(f"after stop: active {mx.get_active_memory()/GB:.2f} GB, cache {mx.get_cache_memory()/GB:.2f} GB, resident {e.memory.resident_components()}", flush=True)
time.sleep(0.5)
gc.collect(); mx.clear_cache()
print(f"after gc: active {mx.get_active_memory()/GB:.2f} GB", flush=True)
ad = e._adapters.get(model) if hasattr(e, "_adapters") else None
if code == "cancelled":
    e.unload(model)
    print(f"after unload: active {mx.get_active_memory()/GB:.2f} GB", flush=True)
