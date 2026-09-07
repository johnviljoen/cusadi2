# MIT Humanoid dynamics functions (from hojae-io/cusadi, branch dev_RAL2025)

`humanoid_fns/*.casadi`: the 12 CasADi functions (M, C, G, CoM, CMM, dCMM, CM, dCM,
base pose, foot Jacobians; nq=25, nv=24) that the coauthor fork of the original
CusADi code-generates for its humanoid MPC / RL work. Copied verbatim from
https://github.com/hojae-io/cusadi (branch dev_RAL2025, commit d39f08e, 2025-09-21).

`humanoid_build.py` writes cusadi2 plans and original-cusadi unrolled kernels
(via cusadi2.unrolled) for fp64 and fp32 and builds latency_lib.so from
examples/paper_results/latency.cu. `humanoid_sweep.py` sweeps B in
{1..16384} for `lp` (cusadi2 level kernel) vs `batch` (original cusadi
kernel), then the CasADi-C-codegen CPU reference. Results: `humanoid_latency.csv`,
`humanoid_cpu.json` (RTX 4080, 2026-09-07). `summarize.py` / `totals.py` print tables.
Scripts expect to run from a directory containing `humanoid_fns/`; paths to the
cusadi2 repo are absolute (edit HERE/REPO to relocate).
