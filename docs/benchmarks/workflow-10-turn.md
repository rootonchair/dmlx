# Workflow 10-turn no-hang run

Date: 2026-10-20  
Machine: Apple Silicon Mac, 24 GB unified memory  
Model: FLUX.2 Klein 4B q4  
Workload: ten consecutive 512×512, four-step generations with the independent
timer/GPU responsiveness probe active.

Command:

```bash
cd engine
DMLX_HOME=<isolated-temporary-directory> uv run dmlx bench \
  --model flux2-klein-4b --strategy auto --size 512x512 --runs 10
```

Result: **10/10 cases passed the no-hang gate**.

| Metric | Worst observed |
|---|---:|
| Peak MLX allocation | 3.14 GB |
| Swap-in rate | 0.2 MB/s |
| GPU probe p99 | 2.0 ms |
| Timer lateness p99 | 8.3 ms |
| Wall time | 7.75 s |

The planner selected `phase_free` for the cold first turn and `sequential` for
the remaining turns as available memory changed. Memory pressure reached
`warn`, never `critical`; swap growth was 0 GB in every case. The configured
limits were 64 MB/s swap-in, 250 ms GPU p99, and 50 ms timer p99.

This run validates the existing generation executor and composed-workflow
scheduler environment. It does not substitute for the separate SeedVR2 or
the local workflow router, which is evaluated separately.
