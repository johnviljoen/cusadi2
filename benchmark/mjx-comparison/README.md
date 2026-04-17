# MJX comparison benchmark

Compares three GPU backends for ABA-Jacobian evaluation on two robots:

- **Level-parallel CusADi** (this fork) — analytical Pinocchio derivatives, fp64, shared-memory kernel.
- **Sequential CusADi** — same analytical derivatives, one thread per batch instance, fp64.
- **MJX** (MuJoCo's JAX backend) — `jax.jacfwd` through `mjx.forward`, fp32 or fp64.
- **mujoco-warp** (forward dynamics only) — no differentiation support.

Robots (loaded via `robot_descriptions`):

- Unitree A1 (`a1_mj_description`): nq=19, nv=18
- Unitree H1 (`h1_mj_description`): nq=26, nv=25

Collision contact detection is disabled for both (we measure smooth dynamics only; unsupported cylinder/box pairs in MJX are then irrelevant).

## Files

- `bench_mjx_jacobian.py` — MJX autodiff Jacobian benchmark (one dtype per process).
- `bench_lp_jacobian.py` — LP / sequential CusADi Jacobian benchmark.
- `bench_forward_dynamics.py` — Forward-dynamics-only benchmark across all four backends.
- `compare.py` — Join JSON outputs into a Markdown table.
- `run_all.sh` — Driver that runs every configuration serially on GPU 5, skipping configs whose JSON already exists.
- `results/*.json` — Raw timing JSONs.
- `results/*.log` — Per-run stdout/stderr.
- `RESULTS.md` — Summary and final comparison table.

## Running

```bash
# Full sweep (takes ~3 hours on L40S due to MJX JIT compile times)
CUDA_VISIBLE_DEVICES=5 bash benchmark/mjx-comparison/run_all.sh

# Rebuild summary from existing JSONs
python benchmark/mjx-comparison/compare.py > RESULTS_auto.md
```

## Caveats

- MJX defaults to fp32; we report both fp32 and fp64. LP is always fp64.
- MJX's JIT compile time is reported alongside evaluation time — the first call per batch size includes ~30-60 s of XLA compilation.
- Outputs: LP kernels return sparse COO nonzero values; MJX returns dense Jacobians. For downstream IPM/KKT usage, sparse is preferred.
- Contact collisions disabled in MJX and mujoco-warp (both require contact pair compatibility; cylinder-box contacts are unsupported in MJX). Since we're benchmarking smooth forward dynamics / Jacobians, this doesn't affect the arithmetic cost — the measurement is of `qacc = M^{-1}(tau - C v - g)` at a given state.
- `jax.jacfwd` is used (not `jacrev`) because MJX's constraint solver contains a `while_loop` which only supports forward-mode differentiation.
