# Profiling tools

CPU-only analysis helpers used by `V031-DIAG-WINDOW.sh`; they read the dumps and traces the
server produces, they do not run on the serving path.

| File | Input | Purpose |
|---|---|---|
| `step_decomp.py` | live server `/metrics` | tokens/step, ms/step, steps and TTFT for the coding workload |
| `boundary_report.py` | torch-profiler `*.json.gz` | per-boundary timing of a decode layer, ranked by chain cost |
| `tl_summary.py` | `w8oproj_timeline_*.pt` | W8 post-attention stage timeline per layer/width |
| `stamp_merge.py` | W1 stamps + W8 timeline `.pt` | merges unprofiled in-kernel stamps on `%globaltimer` |

All four are our original work, **MIT**, copyright 2026 Garner McCloud
(SPDX headers in each file). They import only the Python standard library and, for the
two `.pt` readers, PyTorch from the serving image.
