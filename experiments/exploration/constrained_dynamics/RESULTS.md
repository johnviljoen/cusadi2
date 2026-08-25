# Constrained ABA-Jacobian: LP vs MJX autodiff (with contact)

Extends the MJX-comparison benchmark (`benchmark/mjx-comparison/`) to the **constrained** ABA Jacobian — Pinocchio's `computeConstraintDynamicsDerivatives` with contacts active — compiled through the level-parallel pipeline, and compared apples-to-apples against MJX with MuJoCo contacts enabled.

Hardware: NVIDIA L40S. GPU 5, verified idle before each run (prior measurements were distorted by shared-GPU contention — see "Rerun note"). fp64. 10 warmup + 100 timed iterations, median reported.

**This measures three 25×25 (H1) or 18×18 (A1) Jacobians: `ddq/dq`, `ddq/dv`, `ddq/dtau`. Not forward dynamics.**

## Setup

Two contact configurations:

- **A1 quadruped**: 4× CONTACT_3D at `FL_foot`, `FR_foot`, `RL_foot`, `RR_foot`, `LOCAL_WORLD_ALIGNED` — classical 12-DOF quadruped contact.
- **H1 humanoid**: 2× CONTACT_6D at `left_ankle_link`, `right_ankle_link`, `LOCAL_WORLD_ALIGNED` — flat-foot wrench contact, 12-DOF total.

MJX-with-contact required one workaround: the default MJCF in `robot_descriptions` contains cylinder geoms on both A1 legs and H1 legs. MJX rejects cylinder-box collision pairs at `put_model` time (`NotImplementedError`). Swapping `geom_type == CYLINDER (5)` to `CAPSULE (3)` is a drop-in fix because MuJoCo parametrises both as `(radius, half_length)` — same `geom_size` columns, same principle axis. After the swap, MJX `put_model` + `mjx.forward` + `jax.jacfwd` run end-to-end with contact active.

## DAG characterization

| Function | instrs | levels | avg width | peak_live | smem | LP mode |
|---|---:|---:|---:|---:|---:|---|
| A1 unconstrained (ref)      |  40,969 | 226 | 176.3 |   4,402 |  34.4 KB | `lp` |
| **A1 constrained (4×3D)**   | 102,131 | 435 | 232.2 |  10,385 |  81.1 KB | `lp` |
| H1 unconstrained (ref)      |  78,290 | 281 | 270.8 |   8,191 |  64.0 KB | `lp` |
| **H1 constrained (2×6D)**   | 169,562 | 538 | 311.1 |  18,939 | **148.0 KB** | `tiled` (99 KB cap → 235 chunks, 91.8 KB smem, 1.6 MB/inst spill) |

Adding contacts widens rather than deepens the DAG — avg width goes 176 → 232 (A1) and 271 → 311 (H1). On A1 the constrained DAG fits the 99 KB per-block shared memory ceiling and the unified `CusadiFunction2` router picks the plain LP kernel. On H1 the peak live slots exceed 99 KB (148 KB) and the router auto-selects the **tiled LP kernel** (235 chunks, 91.8 KB in-block smem, 1.6 MB/instance global-memory spill buffer).

## Validation (vs CasADi CPU oracle)

- **A1 constrained (LP mode):** max |err| = 3.4e-12 / 2.3e-14 / 3.4e-13 on ddq_dq / ddq_dv / ddq_dtau — PASS
- **H1 constrained (tiled mode):** max |err| = 4.1e-9 / 3.3e-12 / 2.7e-10 — PASS

## Per-evaluation Jacobian timing (median, µs)

### A1 — LP mode, 4× 3D foot contacts

| batch | **LP constrained** | LP unconstrained | MJX fp64 no-contact | **MJX fp64 WITH contact** | MJX/LP (contact) |
|---:|---:|---:|---:|---:|---:|
|   1 |     991 |   159 |  14,065 |    20,286 |    **20×** |
|   8 |   1,008 |   161 |  26,321 |   134,317 |   **133×** |
|  32 |   1,009 |   160 |  35,307 |   323,674 |   **321×** |
| 128 |   1,011 |   162 |  69,956 | 1,189,748 | **1,177×** |
| 256 |   1,960 |   167 |  87,839 | 2,492,520 | **1,272×** |

A1 LP constrained stays flat at ~1.0 ms up through batch=128 (no-op between batch=1 and batch=128 — every block fits on its own SM). It doubles at batch=256 because L40S has 142 SMs so 256/142 ≈ 1.8 blocks/SM forces serialization.

MJX-with-contact at batch=128 is **1.19 seconds per evaluation**; LP does the same computation in 1.01 milliseconds. **1,177× speedup.** MJX's contact solver (`lax.while_loop` inside `mjx.forward`) dominates the evaluation — contact adds a 17× penalty on MJX at batch=128 (70 ms → 1,190 ms) while adding only 6× to LP (162 µs → 1,011 µs).

### H1 — tiled mode, 2× 6D flat-foot contacts

| batch | **LP constrained** | LP unconstrained | MJX fp64 no-contact | **MJX fp64 WITH contact** | MJX/LP (contact) |
|---:|---:|---:|---:|---:|---:|
|   1 |   3,730 |   263 |  7,297 |    21,049 |   **6×** |
|   8 |   3,841 |   267 |  7,584 |    50,496 |  **13×** |
|  32 |   3,843 |   268 |  9,017 |   186,768 |  **49×** |
| 128 |   6,037 |   268 | 22,561 |   564,976 |  **94×** |
| 256 |  12,075 |   519 | 30,795 | 1,306,160 | **108×** |

The tiled LP path pays a 14–23× overhead vs LP on the unconstrained H1 Jacobian (that's the cost of 235 chunk boundaries and 1.6 MB/instance spill-buffer traffic) — but MJX pays 3–5× going from contact-off to contact-on on top of its already-50× gap to LP, and the end result is LP winning by **6× to 108×** across the batch range.

## Headline

- **A1 constrained Jacobian, batch=128: LP 1.01 ms vs MJX-with-contact 1.19 s = 1,177× faster.**
- **H1 constrained Jacobian, batch=128: LP 6.0 ms vs MJX-with-contact 565 ms = 94× faster.**
- The MJX contact penalty is 17× on A1 (batch=128) and 25× on H1 (batch=128) — enabling contact costs MJX far more than it costs LP, which widens the original LP-vs-MJX gap rather than closing it.
- Enabling contact in MJX costs LP 6× on A1, 22× on H1 (measured as constrained-vs-unconstrained LP). Constraint-Jacobian overhead is real but bounded — the KKT solve in Pinocchio's `computeConstraintDynamicsDerivatives` is O(n²) not O(iteration × n²).

## MJX JIT compile time with contact

First-call (JIT + first run) wall-clock for the contact-on sweep, all fp64:

| batch | A1 MJX fp64+contact | H1 MJX fp64+contact |
|---:|---:|---:|
|   1 |  93 s |  96 s |
|   8 |  78 s |  72 s |
|  32 |  78 s |  88 s |
| 128 |  91 s | 107 s |
| 256 | 103 s | 123 s |
| 512 | 127 s | 148 s |

LP data-driven dispatch kernel: **1.5 s once**, reused for every function and every batch size.

## What's compiled / cherry-picked onto this branch

Four commits from `feature/smem-tiling` and `feature/unified-routing` were cherry-picked onto `feature/constrained-dynamics` — no conflict with master's single-sync optimization:

```
13dccc4 CusadiFunction2: unified routing across LP, tiled LP, and sequential kernels
6a51f0c Phase 3: full benchmark — sequential vs tiled level-parallel on all 5 functions
c612f38 Phase 2: tiled CUDA kernel with chunk boundaries and sub-batch spills
c359335 Phase 1: shared memory tiling with intra-level sub-batching
```

`CusadiFunction2` (in `src/level_parallel/unified.py`) queries `cudaDevAttrMaxSharedMemoryPerBlockOptin` at construction, runs DAG analysis + tile planning, and auto-selects `lp` if peak liveness fits or `tiled` otherwise. H1 constrained triggered the tiled path; A1 stayed on LP.

## Files

- `exploration/constrained_dynamics/build_constrained.py` — builds the CasADi `.casadi` files for A1 (4×3D) and H1 (2×6D)
- `exploration/constrained_dynamics/validate_constrained.py` — validates LP output against CasADi CPU oracle; column-major output layout matters for nnz-comparison
- `exploration/constrained_dynamics/bench_constrained.py` — timing sweep using `CusadiFunction2` auto-routing
- `exploration/constrained_dynamics/results/{a1,h1}_constrained_lp.json` — raw LP timings
- `benchmark/mjx-comparison/results/{a1,h1}_mjx_fp64_contact.{json,log}` — raw MJX-with-contact timings
- `benchmark/mjx-comparison/bench_mjx_jacobian.py` — added `--with_contact` flag; swaps CYLINDER → CAPSULE at load time
- `benchmark/mjx-comparison/run_contact.sh` — driver that chains the 4 contact-on configs
- `src/casadi_functions/{a1,h1}_constrained_aba_jac.casadi` — saved Pinocchio-derived symbolic functions
- `run_all_clean.sh` — end-to-end driver: LP rerun + MJX-with-contact rerun, one clean GPU run

## Rerun note (2026-04-16 evening)

Three separate rerun events during this measurement campaign changed the numbers materially, each time because GPU 5 was unexpectedly shared:

| Measurement | Contended | Clean | Delta |
|---|---:|---:|---:|
| A1 LP batch=1    |  1,529 µs |    991 µs |  1.5× |
| A1 LP batch=256  |  6,445 µs |  1,960 µs |  3.3× |
| H1 LP batch=1    | 19,305 µs |  3,730 µs | **5.2×** |
| H1 LP batch=256  | 32,133 µs | 12,075 µs |  2.7× |
| A1 MJX-contact batch=128 | 1,363 ms | 1,190 ms |  1.1× |

**Lesson:** check `nvidia-smi -i 5 --query-gpu=utilization.gpu,memory.used` returns something like `0 %, 6 MiB` before starting a benchmark. The LP kernel's shared-memory dominated evaluation is particularly sensitive to GPU co-tenancy because blocks compete for SM shared-memory slots; MJX is less sensitive because it's global-memory bandwidth bound.

## Suggested follow-ups

- **MJX fp32 with contact** — the fp32 column. Typically 1.5–2× faster than fp64 for MJX; won't flip the headline but rounds out the comparison. Each config ~15–30 min on clean GPU.
- **Tighten H1 peak liveness** below the 99 KB per-block cap to put H1 constrained on the LP fast path (would close most of the 14–23× tiled-vs-LP gap — H1 batch=128 would drop from 6.0 ms toward ~0.8 ms). The `peak_live_slots=18,939` suggests the SSA liveness analysis may be conservative; worth revisiting.
- **Alternative H1 contact models** — 4×3D heel+toe per foot instead of 2×6D may produce sparser KKT structure and smaller peak live.
- **MJX persistent compile cache** — `JAX_COMPILATION_CACHE_DIR` eliminates the 90–150 s JIT pause on reruns. Already implicit in our clean-GPU JIT numbers because the cache warmed up between configs.
